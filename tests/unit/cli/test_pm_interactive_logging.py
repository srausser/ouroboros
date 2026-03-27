"""Unit tests for interactive PM prompt behavior."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import typer

from ouroboros.bigbang.interview import InterviewRound
from ouroboros.cli.commands.pm import _multiline_prompt_async, _run_pm_interview, pm_command
from ouroboros.core.types import Result


def _build_ctx() -> MagicMock:
    """Build a Typer context mock for direct callback testing."""
    ctx = MagicMock()
    ctx.invoked_subcommand = None
    return ctx


def _build_state(*, pending_question: str | None = None) -> SimpleNamespace:
    """Create the minimal interview state shape used by the PM CLI."""
    rounds = []
    if pending_question is not None:
        rounds.append(
            InterviewRound(
                round_number=1,
                question=pending_question,
                user_response=None,
            )
        )

    return SimpleNamespace(
        is_complete=False,
        rounds=rounds,
        current_round_number=1,
        interview_id="interview_123",
        mark_updated=MagicMock(),
    )


def test_pm_command_exits_cleanly_on_keyboard_interrupt() -> None:
    """The PM command should surface Ctrl+C as a normal CLI exit."""

    def fake_run(coro: object) -> None:
        coro.close()  # type: ignore[union-attr]
        raise KeyboardInterrupt

    with patch("ouroboros.cli.commands.pm.asyncio.run", side_effect=fake_run):
        with pytest.raises(typer.Exit) as exc_info:
            pm_command(_build_ctx(), model="test-model", debug=False)

    assert exc_info.value.exit_code == 0


@pytest.mark.asyncio
async def test_multiline_prompt_async_uses_prompt_toolkit_with_patch_stdout() -> None:
    """PM input should use a multiline prompt that proxies stdout/stderr safely."""
    session = MagicMock()
    session.prompt_async = AsyncMock(return_value="line 1\nline 2")

    with (
        patch("ouroboros.cli.commands.pm.PromptSession", return_value=session) as mock_session,
        patch("ouroboros.cli.commands.pm.patch_stdout", return_value=nullcontext()) as mock_patch,
        patch("ouroboros.cli.commands.pm.console.print"),
    ):
        result = await _multiline_prompt_async("Prompt here")

    assert result == "line 1\nline 2"
    mock_patch.assert_called_once_with(raw=True)
    session.prompt_async.assert_awaited_once()

    kwargs = mock_session.call_args.kwargs
    assert kwargs["message"] == "> "
    assert kwargs["multiline"] is True
    assert kwargs["prompt_continuation"] == "  "

    key_bindings = kwargs["key_bindings"]
    bound_keys = {tuple(binding.keys) for binding in key_bindings.bindings}
    assert ("c-j",) in bound_keys
    assert ("c-m",) in bound_keys


@pytest.mark.asyncio
async def test_run_pm_interview_new_session_uses_multiline_prompt_and_shows_progress(
    tmp_path: Path,
) -> None:
    """New PM sessions should use the multiline prompt for both answers."""
    state = _build_state()
    engine = MagicMock()
    engine.get_opening_question.return_value = "What do you want to build?"
    engine.ask_opening_and_start = AsyncMock(return_value=Result.ok(state))
    engine.ask_next_question = AsyncMock(return_value=Result.ok("What is the first workflow?"))
    engine.save_state = AsyncMock(return_value=Result.ok(tmp_path / "state.json"))
    engine.record_response = AsyncMock(return_value=Result.ok(state))
    engine.complete_interview = AsyncMock(return_value=Result.ok(state))
    engine.format_decide_later_summary.return_value = ""

    with (
        patch.object(Path, "home", return_value=tmp_path),
        patch("ouroboros.providers.litellm_adapter.LiteLLMAdapter", return_value=object()),
        patch("ouroboros.bigbang.pm_interview.PMInterviewEngine.create", return_value=engine),
        patch("ouroboros.cli.commands.pm._check_existing_pm_seeds", return_value=True),
        patch("ouroboros.cli.commands.pm._load_brownfield_from_db", return_value=[]),
        patch("ouroboros.cli.commands.pm._select_repos", return_value=[]),
        patch("ouroboros.cli.commands.pm._save_cli_pm_meta"),
        patch(
            "ouroboros.cli.commands.pm._multiline_prompt_async",
            side_effect=["Initial context", "done"],
        ) as mock_prompt,
        patch("ouroboros.cli.commands.pm.console.input") as mock_console_input,
        patch("ouroboros.cli.commands.pm.print_info") as mock_print_info,
        patch("ouroboros.cli.commands.pm.print_success"),
    ):
        await _run_pm_interview(resume_id=None, model="test-model", debug=False)

    engine.ask_opening_and_start.assert_awaited_once()
    engine.ask_next_question.assert_awaited_once_with(state)
    assert mock_prompt.await_count == 2
    mock_console_input.assert_not_called()

    messages = [call.args[0] for call in mock_print_info.call_args_list]
    assert "Starting interview..." in messages
    assert "Generating next question..." in messages
    assert messages.index("Starting interview...") < messages.index("Generating next question...")


@pytest.mark.asyncio
async def test_run_pm_interview_resume_with_pending_question_skips_generation_message(
    tmp_path: Path,
) -> None:
    """Resume should reuse the saved pending question instead of generating a new one."""
    state = _build_state(pending_question="Saved question?")
    engine = MagicMock()
    engine.load_state = AsyncMock(return_value=Result.ok(state))
    engine.ask_next_question = AsyncMock()
    engine.save_state = AsyncMock(return_value=Result.ok(tmp_path / "state.json"))
    engine.complete_interview = AsyncMock(return_value=Result.ok(state))
    engine.format_decide_later_summary.return_value = ""
    engine._install_pm_steering = MagicMock()

    with (
        patch.object(Path, "home", return_value=tmp_path),
        patch("ouroboros.providers.litellm_adapter.LiteLLMAdapter", return_value=object()),
        patch("ouroboros.bigbang.pm_interview.PMInterviewEngine.create", return_value=engine),
        patch("ouroboros.cli.commands.pm._save_cli_pm_meta"),
        patch(
            "ouroboros.cli.commands.pm._multiline_prompt_async", return_value="done"
        ) as mock_prompt,
        patch("ouroboros.cli.commands.pm.console.input") as mock_console_input,
        patch("ouroboros.cli.commands.pm.print_info") as mock_print_info,
        patch("ouroboros.cli.commands.pm.print_success"),
    ):
        await _run_pm_interview(resume_id="resume_123", model="test-model", debug=False)

    engine.ask_next_question.assert_not_awaited()
    engine._install_pm_steering.assert_called_once()
    mock_prompt.assert_awaited_once()
    mock_console_input.assert_not_called()

    messages = [call.args[0] for call in mock_print_info.call_args_list]
    assert "Generating next question..." not in messages

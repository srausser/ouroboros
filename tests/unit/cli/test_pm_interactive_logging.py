"""Unit tests for interactive PM logging behavior."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import typer

from ouroboros.bigbang.interview import InterviewRound
from ouroboros.cli.commands.pm import _run_pm_interview, pm_command
from ouroboros.core.types import Result
from ouroboros.observability.logging import is_console_logging_enabled, set_console_logging


@pytest.fixture(autouse=True)
def reset_console_logging() -> None:
    """Reset console logging state around each test."""
    set_console_logging(True)
    yield
    set_console_logging(True)


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


def test_pm_command_disables_console_logging_by_default_and_restores_state() -> None:
    """Default PM sessions should suppress console logs only while they run."""
    observed: list[bool] = []

    def fake_run(coro: object) -> None:
        observed.append(is_console_logging_enabled())
        coro.close()  # type: ignore[union-attr]

    with patch("ouroboros.cli.commands.pm.asyncio.run", side_effect=fake_run):
        pm_command(_build_ctx(), model="test-model", debug=False)

    assert observed == [False]
    assert is_console_logging_enabled() is True


def test_pm_command_enables_console_logging_in_debug_and_restores_state() -> None:
    """Debug PM sessions should force console logs on during the run."""
    observed: list[bool] = []
    set_console_logging(False)

    def fake_run(coro: object) -> None:
        observed.append(is_console_logging_enabled())
        coro.close()  # type: ignore[union-attr]

    with patch("ouroboros.cli.commands.pm.asyncio.run", side_effect=fake_run):
        pm_command(_build_ctx(), model="test-model", debug=True)

    assert observed == [True]
    assert is_console_logging_enabled() is False


def test_pm_command_restores_console_logging_on_keyboard_interrupt() -> None:
    """Console logging should be restored even if the PM session is interrupted."""
    observed: list[bool] = []

    def fake_run(coro: object) -> None:
        observed.append(is_console_logging_enabled())
        coro.close()  # type: ignore[union-attr]
        raise KeyboardInterrupt

    with patch("ouroboros.cli.commands.pm.asyncio.run", side_effect=fake_run):
        with pytest.raises(typer.Exit) as exc_info:
            pm_command(_build_ctx(), model="test-model", debug=False)

    assert exc_info.value.exit_code == 0
    assert observed == [False]
    assert is_console_logging_enabled() is True


@pytest.mark.asyncio
async def test_run_pm_interview_new_session_shows_progress_messages(tmp_path: Path) -> None:
    """New PM sessions should show progress around startup and question generation."""
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
        patch("ouroboros.cli.commands.pm.console.input", side_effect=["Initial context", "done"]),
        patch("ouroboros.cli.commands.pm.print_info") as mock_print_info,
        patch("ouroboros.cli.commands.pm.print_success"),
    ):
        await _run_pm_interview(resume_id=None, model="test-model", debug=False)

    engine.ask_opening_and_start.assert_awaited_once()
    engine.ask_next_question.assert_awaited_once_with(state)
    messages = [call.args[0] for call in mock_print_info.call_args_list]
    assert "Starting interview..." in messages
    assert "Generating next question..." in messages
    assert messages.index("Starting interview...") < messages.index("Generating next question...")


@pytest.mark.asyncio
async def test_run_pm_interview_resume_with_pending_question_skips_generation_message(
    tmp_path: Path,
) -> None:
    """Resume should not show generation progress when a saved question already exists."""
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
        patch("ouroboros.cli.commands.pm.console.input", side_effect=["done"]),
        patch("ouroboros.cli.commands.pm.print_info") as mock_print_info,
        patch("ouroboros.cli.commands.pm.print_success"),
    ):
        await _run_pm_interview(resume_id="resume_123", model="test-model", debug=False)

    engine.ask_next_question.assert_not_awaited()
    engine._install_pm_steering.assert_called_once()
    messages = [call.args[0] for call in mock_print_info.call_args_list]
    assert "Generating next question..." not in messages

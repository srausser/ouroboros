"""Helpers for resolving interview initial context from text or files."""

from __future__ import annotations

from pathlib import Path

import yaml

from ouroboros.bigbang.pm_seed import PMSeed
from ouroboros.core.errors import ValidationError
from ouroboros.core.security import InputValidator
from ouroboros.core.types import Result

_PM_SEED_SUFFIXES = {".json", ".yaml", ".yml"}
_TEXT_CONTEXT_SUFFIXES = {".md", ".markdown", ".txt", ".rst"}


def load_pm_seed_as_context(seed_path: Path) -> Result[str, ValidationError]:
    """Load a PMSeed artifact and convert it into dev-interview context."""
    try:
        raw_data = yaml.safe_load(seed_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        return Result.err(
            ValidationError(
                f"Failed to load PM seed: {exc}",
                field="initial_context",
                value=str(seed_path),
            )
        )

    if not isinstance(raw_data, dict):
        return Result.err(
            ValidationError(
                "PM seed file must contain an object mapping",
                field="initial_context",
                value=str(seed_path),
            )
        )

    pm_seed = PMSeed.from_dict(raw_data)
    context = pm_seed.to_initial_context()
    is_valid, error_message = InputValidator.validate_initial_context(context)
    if not is_valid:
        return Result.err(
            ValidationError(
                error_message,
                field="initial_context",
                value=str(seed_path),
            )
        )

    return Result.ok(context)


def resolve_initial_context_input(
    initial_context: str,
    *,
    cwd: str | Path | None = None,
) -> Result[str, ValidationError]:
    """Resolve interview input from literal text or an existing file path."""
    context_candidate = initial_context.strip()
    if not context_candidate:
        return Result.err(
            ValidationError(
                "Initial context cannot be empty",
                field="initial_context",
                value=initial_context,
            )
        )

    candidate_path = Path(context_candidate).expanduser()
    if not candidate_path.is_absolute() and cwd is not None:
        candidate_path = Path(cwd).expanduser() / candidate_path

    if candidate_path.exists() and candidate_path.is_file():
        suffix = candidate_path.suffix.lower()
        if suffix in _PM_SEED_SUFFIXES:
            pm_seed_result = load_pm_seed_as_context(candidate_path)
            if pm_seed_result.is_ok:
                return pm_seed_result

        if suffix in _TEXT_CONTEXT_SUFFIXES or suffix in _PM_SEED_SUFFIXES:
            try:
                file_context = candidate_path.read_text(encoding="utf-8")
            except OSError as exc:
                return Result.err(
                    ValidationError(
                        f"Failed to read initial context file: {exc}",
                        field="initial_context",
                        value=str(candidate_path),
                    )
                )

            is_valid, error_message = InputValidator.validate_initial_context(file_context)
            if not is_valid:
                return Result.err(
                    ValidationError(
                        error_message,
                        field="initial_context",
                        value=str(candidate_path),
                    )
                )
            return Result.ok(file_context)

    is_valid, error_message = InputValidator.validate_initial_context(initial_context)
    if not is_valid:
        return Result.err(
            ValidationError(
                error_message,
                field="initial_context",
                value=initial_context,
            )
        )

    return Result.ok(initial_context)

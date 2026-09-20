"""JSONL ingestion with line-accurate error reporting.

Annotation exports arrive as JSONL and are routinely malformed. Every failure
here names the line number, because "invalid JSON in a 40k-line export" is not
an actionable bug report.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from .models import Annotation, Comparison, Verdict


class LoadError(ValueError):
    """A record could not be parsed. Carries the offending line number."""

    def __init__(self, path: Path, line_number: int, message: str) -> None:
        self.path = path
        self.line_number = line_number
        super().__init__(f"{path}:{line_number}: {message}")


def iter_jsonl(path: str | Path) -> Iterator[tuple[int, dict[str, object]]]:
    """Yield ``(line_number, record)`` for each non-blank line.

    Raises:
        LoadError: On malformed JSON or a non-object record.
        FileNotFoundError: If the path does not exist.
    """
    p = Path(path)
    with p.open(encoding="utf-8") as handle:
        for number, raw in enumerate(handle, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise LoadError(p, number, f"invalid JSON: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise LoadError(
                    p, number, f"expected an object, got {type(record).__name__}"
                )
            yield number, record


def load_annotations(path: str | Path) -> list[Annotation]:
    """Load annotations from JSONL.

    Expected per line::

        {"item_id": "task-1", "annotator_id": "ann-a",
         "scores": {"correctness": 4}, "notes": "optional"}
    """
    p = Path(path)
    out: list[Annotation] = []

    for number, record in iter_jsonl(p):
        try:
            scores = record["scores"]
            if not isinstance(scores, dict):
                raise LoadError(p, number, "'scores' must be an object")
            out.append(
                Annotation(
                    item_id=str(record["item_id"]),
                    annotator_id=str(record["annotator_id"]),
                    scores={k: float(v) for k, v in scores.items()},
                    notes=str(record.get("notes", "")),
                )
            )
        except KeyError as exc:
            raise LoadError(p, number, f"missing required field {exc}") from exc
        except (TypeError, ValueError) as exc:
            if isinstance(exc, LoadError):
                raise
            raise LoadError(p, number, str(exc)) from exc

    return out


def load_comparisons(path: str | Path) -> list[Comparison]:
    """Load pairwise comparisons from JSONL.

    Expected per line::

        {"item_id": "prompt-1", "annotator_id": "ann-a",
         "response_a": "model-x", "response_b": "model-y",
         "verdict": "a", "rationale": "optional"}
    """
    p = Path(path)
    out: list[Comparison] = []

    for number, record in iter_jsonl(p):
        try:
            raw_verdict = str(record["verdict"]).lower()
            try:
                verdict = Verdict(raw_verdict)
            except ValueError as exc:
                valid = ", ".join(v.value for v in Verdict)
                raise LoadError(
                    p, number, f"verdict must be one of [{valid}], got {raw_verdict!r}"
                ) from exc

            out.append(
                Comparison(
                    item_id=str(record["item_id"]),
                    annotator_id=str(record["annotator_id"]),
                    response_a=str(record["response_a"]),
                    response_b=str(record["response_b"]),
                    verdict=verdict,
                    rationale=str(record.get("rationale", "")),
                )
            )
        except KeyError as exc:
            raise LoadError(p, number, f"missing required field {exc}") from exc
        except ValueError as exc:
            if isinstance(exc, LoadError):
                raise
            raise LoadError(p, number, str(exc)) from exc

    return out


def load_rubric(path: str | Path) -> dict[str, object]:
    """Load a rubric definition from JSON."""
    p = Path(path)
    with p.open(encoding="utf-8") as handle:
        try:
            payload = json.load(handle)
        except json.JSONDecodeError as exc:
            raise LoadError(p, exc.lineno, f"invalid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise LoadError(p, 1, "rubric file must contain a JSON object")
    return payload

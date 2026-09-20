"""JSONL ingestion, with emphasis on error messages naming the bad line."""

from __future__ import annotations

from pathlib import Path

import pytest

from eval_harness.loaders import (
    LoadError,
    load_annotations,
    load_comparisons,
    load_rubric,
)
from eval_harness.models import Verdict


def write(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


class TestLoadAnnotations:
    def test_loads_valid_records(self, tmp_path: Path) -> None:
        path = write(
            tmp_path,
            "a.jsonl",
            '{"item_id":"t1","annotator_id":"ann-a","scores":{"correctness":4}}\n'
            '{"item_id":"t2","annotator_id":"ann-b","scores":{"correctness":2},"notes":"terse"}\n',
        )
        loaded = load_annotations(path)
        assert [a.item_id for a in loaded] == ["t1", "t2"]
        assert loaded[0].scores["correctness"] == 4.0
        assert loaded[1].notes == "terse"

    def test_skips_blank_lines(self, tmp_path: Path) -> None:
        path = write(
            tmp_path,
            "a.jsonl",
            '{"item_id":"t1","annotator_id":"a","scores":{"x":1}}\n'
            "\n   \n"
            '{"item_id":"t2","annotator_id":"a","scores":{"x":2}}\n',
        )
        assert len(load_annotations(path)) == 2

    def test_reports_the_offending_line_number(self, tmp_path: Path) -> None:
        path = write(
            tmp_path,
            "a.jsonl",
            '{"item_id":"t1","annotator_id":"a","scores":{"x":1}}\n'
            "{ not json\n",
        )
        with pytest.raises(LoadError) as excinfo:
            load_annotations(path)
        assert excinfo.value.line_number == 2
        assert "invalid JSON" in str(excinfo.value)

    def test_reports_missing_field_with_line(self, tmp_path: Path) -> None:
        path = write(tmp_path, "a.jsonl", '{"item_id":"t1","scores":{"x":1}}\n')
        with pytest.raises(LoadError, match=r"a\.jsonl:1.*annotator_id"):
            load_annotations(path)

    def test_rejects_non_object_record(self, tmp_path: Path) -> None:
        path = write(tmp_path, "a.jsonl", "[1, 2, 3]\n")
        with pytest.raises(LoadError, match="expected an object, got list"):
            load_annotations(path)

    def test_rejects_non_object_scores(self, tmp_path: Path) -> None:
        path = write(
            tmp_path, "a.jsonl", '{"item_id":"t","annotator_id":"a","scores":5}\n'
        )
        with pytest.raises(LoadError, match="'scores' must be an object"):
            load_annotations(path)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_annotations(tmp_path / "nope.jsonl")


class TestLoadComparisons:
    def test_loads_valid_records(self, tmp_path: Path) -> None:
        path = write(
            tmp_path,
            "c.jsonl",
            '{"item_id":"p1","annotator_id":"a","response_a":"m1",'
            '"response_b":"m2","verdict":"a"}\n'
            '{"item_id":"p2","annotator_id":"a","response_a":"m1",'
            '"response_b":"m2","verdict":"TIE"}\n',
        )
        loaded = load_comparisons(path)
        assert loaded[0].verdict is Verdict.A
        assert loaded[1].verdict is Verdict.TIE  # case-insensitive

    def test_rejects_unknown_verdict(self, tmp_path: Path) -> None:
        path = write(
            tmp_path,
            "c.jsonl",
            '{"item_id":"p1","annotator_id":"a","response_a":"m1",'
            '"response_b":"m2","verdict":"maybe"}\n',
        )
        with pytest.raises(LoadError, match="verdict must be one of"):
            load_comparisons(path)

    def test_rejects_self_comparison_with_line_number(self, tmp_path: Path) -> None:
        path = write(
            tmp_path,
            "c.jsonl",
            '{"item_id":"p1","annotator_id":"a","response_a":"m1",'
            '"response_b":"m1","verdict":"a"}\n',
        )
        with pytest.raises(LoadError, match=r"c\.jsonl:1.*against itself"):
            load_comparisons(path)


class TestLoadRubric:
    def test_loads_object(self, tmp_path: Path) -> None:
        path = write(tmp_path, "r.json", '{"name":"x","criteria":[]}')
        assert load_rubric(path)["name"] == "x"

    def test_rejects_non_object(self, tmp_path: Path) -> None:
        path = write(tmp_path, "r.json", "[]")
        with pytest.raises(LoadError, match="must contain a JSON object"):
            load_rubric(path)

    def test_rejects_invalid_json(self, tmp_path: Path) -> None:
        path = write(tmp_path, "r.json", "{oops")
        with pytest.raises(LoadError, match="invalid JSON"):
            load_rubric(path)

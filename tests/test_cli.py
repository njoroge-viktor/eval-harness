"""CLI behaviour, including exit codes and error routing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval_harness.cli import main

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


class TestScoreCommand:
    def test_table_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        code = main(
            [
                "score",
                "--rubric", str(EXAMPLES / "rubric.json"),
                "--annotations", str(EXAMPLES / "annotations.jsonl"),
            ]
        )
        out = capsys.readouterr().out
        assert code == 0
        assert "code-response-quality" in out
        assert "task-001" in out
        assert "mean normalised" in out

    def test_json_output_is_parseable(self, capsys: pytest.CaptureFixture[str]) -> None:
        code = main(
            [
                "score",
                "--rubric", str(EXAMPLES / "rubric.json"),
                "--annotations", str(EXAMPLES / "annotations.jsonl"),
                "--json",
            ]
        )
        payload = json.loads(capsys.readouterr().out)
        assert code == 0
        assert len(payload) == 6
        assert {"item_id", "annotator_id", "weighted_score", "normalized"} <= set(
            payload[0]
        )
        assert 0.0 <= payload[0]["normalized"] <= 1.0

    def test_missing_criteria_exits_one(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        partial = tmp_path / "partial.jsonl"
        partial.write_text(
            '{"item_id":"t","annotator_id":"a","scores":{"correctness":4}}\n',
            encoding="utf-8",
        )
        code = main(
            ["score", "--rubric", str(EXAMPLES / "rubric.json"),
             "--annotations", str(partial)]
        )
        assert code == 1
        assert "missing criteria" in capsys.readouterr().err

    def test_lenient_flag_accepts_partial_annotations(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        partial = tmp_path / "partial.jsonl"
        partial.write_text(
            '{"item_id":"t","annotator_id":"a","scores":{"correctness":4}}\n',
            encoding="utf-8",
        )
        code = main(
            ["score", "--rubric", str(EXAMPLES / "rubric.json"),
             "--annotations", str(partial), "--lenient", "--json"]
        )
        assert code == 0
        assert json.loads(capsys.readouterr().out)[0]["normalized"] == pytest.approx(0.75)


class TestAgreementCommand:
    def test_reports_both_coefficients(self, capsys: pytest.CaptureFixture[str]) -> None:
        code = main(
            ["agreement", "--annotations", str(EXAMPLES / "annotations.jsonl"),
             "--criterion", "correctness", "--json"]
        )
        payload = json.loads(capsys.readouterr().out)
        assert code == 0
        assert payload["criterion"] == "correctness"
        assert payload["items"] == 3
        assert payload["fleiss_kappa"] is not None

    def test_uneven_coverage_omits_fleiss(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        uneven = tmp_path / "uneven.jsonl"
        uneven.write_text(
            '{"item_id":"t1","annotator_id":"a","scores":{"x":1}}\n'
            '{"item_id":"t1","annotator_id":"b","scores":{"x":1}}\n'
            '{"item_id":"t2","annotator_id":"a","scores":{"x":2}}\n'
            '{"item_id":"t2","annotator_id":"b","scores":{"x":2}}\n'
            '{"item_id":"t2","annotator_id":"c","scores":{"x":2}}\n',
            encoding="utf-8",
        )
        code = main(["agreement", "--annotations", str(uneven),
                     "--criterion", "x", "--json"])
        payload = json.loads(capsys.readouterr().out)
        assert code == 0
        assert payload["fleiss_kappa"] is None
        assert payload["krippendorff_alpha"] == pytest.approx(1.0)

    def test_unknown_criterion_exits_two(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(
            ["agreement", "--annotations", str(EXAMPLES / "annotations.jsonl"),
             "--criterion", "nonexistent"]
        )
        assert code == 2
        assert "no annotations carry criterion" in capsys.readouterr().err


class TestRankCommand:
    def test_orders_models_by_strength(self, capsys: pytest.CaptureFixture[str]) -> None:
        code = main(
            ["rank", "--comparisons", str(EXAMPLES / "comparisons.jsonl"), "--json"]
        )
        payload = json.loads(capsys.readouterr().out)
        assert code == 0
        assert payload[0]["model"] == "claude-3-5-sonnet"
        assert payload[-1]["model"] == "llama-3-70b"
        # Smoothed by default, so even the winless model is ranked, not zeroed.
        assert payload[-1]["strength"] > 0


class TestErrorRouting:
    def test_malformed_jsonl_exits_two(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        bad = tmp_path / "bad.jsonl"
        bad.write_text("{ not json\n", encoding="utf-8")
        code = main(["rank", "--comparisons", str(bad)])
        assert code == 2
        assert "bad.jsonl:1" in capsys.readouterr().err

    def test_missing_file_exits_one(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(["rank", "--comparisons", str(tmp_path / "nope.jsonl")])
        assert code == 1
        assert "error:" in capsys.readouterr().err

    def test_requires_a_subcommand(self) -> None:
        with pytest.raises(SystemExit):
            main([])


class TestTableOutput:
    """The default human-readable paths, not just --json."""

    def test_agreement_table_reports_both_coefficients(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(
            ["agreement", "--annotations", str(EXAMPLES / "annotations.jsonl"),
             "--criterion", "correctness"]
        )
        out = capsys.readouterr().out
        assert code == 0
        assert "Krippendorff's alpha" in out
        assert "Fleiss' kappa" in out

    def test_agreement_table_marks_fleiss_unavailable(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        uneven = tmp_path / "uneven.jsonl"
        uneven.write_text(
            '{"item_id":"t1","annotator_id":"a","scores":{"x":1}}\n'
            '{"item_id":"t1","annotator_id":"b","scores":{"x":1}}\n'
            '{"item_id":"t2","annotator_id":"a","scores":{"x":2}}\n'
            '{"item_id":"t2","annotator_id":"b","scores":{"x":2}}\n'
            '{"item_id":"t2","annotator_id":"c","scores":{"x":2}}\n',
            encoding="utf-8",
        )
        code = main(["agreement", "--annotations", str(uneven), "--criterion", "x"])
        assert code == 0
        assert "n/a (uneven rater coverage)" in capsys.readouterr().out

    def test_rank_table_lists_every_model(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(["rank", "--comparisons", str(EXAMPLES / "comparisons.jsonl")])
        out = capsys.readouterr().out
        assert code == 0
        assert "strength" in out
        for model in ("gpt-4o", "claude-3-5-sonnet", "llama-3-70b"):
            assert model in out

    def test_score_table_omits_mean_for_empty_batch(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        empty = tmp_path / "empty.jsonl"
        empty.write_text("\n", encoding="utf-8")
        code = main(
            ["score", "--rubric", str(EXAMPLES / "rubric.json"),
             "--annotations", str(empty)]
        )
        out = capsys.readouterr().out
        assert code == 0
        assert "mean normalised" not in out

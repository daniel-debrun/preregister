"""Regression tests for inputs found by adversarial testing."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from preregister import Project, RunRecord, Verdict
from preregister.cli import main
from preregister.ledger import LedgerError
from preregister.runs import RunError, load_runs_csv, load_runs_json

from .conftest import FakeClock, make_runs
from .test_reconcile import pred, run_study, vals


@pytest.mark.parametrize(
    ("denominator", "numerator"),
    [(0.0, 3.0), (-1.8, 3.0), (-2.0, -10.0)],
)
def test_ratio_claims_need_a_positive_denominator(
    tmp_path: Path, clock: FakeClock, denominator: float, numerator: float
) -> None:
    project, _ = run_study(
        tmp_path,
        clock,
        pred("treatment / control >= 2", test="bootstrap"),
        {"control": vals(denominator), "treatment": vals(numerator)},
    )
    (result,) = project.analyze().predictions
    assert result.verdict == Verdict.INCONCLUSIVE
    assert result.p_confirm is None and result.estimate is None
    assert any("strictly positive denominator" in n for n in result.notes)


def test_ratio_claim_with_positive_denominator_still_works(
    tmp_path: Path, clock: FakeClock
) -> None:
    project, _ = run_study(
        tmp_path,
        clock,
        pred("treatment / control >= 2", test="bootstrap"),
        {"control": vals(1.0), "treatment": vals(3.0)},
    )
    assert project.analyze().predictions[0].verdict == Verdict.CONFIRMED


def _run(**overrides) -> dict:
    return {
        "run_id": "r1",
        "condition": "control",
        "seed": 0,
        "metrics": {"score": 1.0},
        "started_at": "2026-01-01T00:00:00Z",
        **overrides,
    }


@pytest.mark.parametrize("ts", ["2026-01-01T09:00:00", "2026-01-01 09:00:00"])
def test_naive_run_timestamps_are_rejected(ts: str) -> None:
    with pytest.raises(RunError, match="no UTC offset"):
        RunRecord.from_dict(_run(started_at=ts))
    with pytest.raises(RunError, match="no UTC offset"):
        RunRecord.from_dict(_run(started_at=datetime.fromisoformat(ts)))


def test_offset_timestamps_are_normalised_to_utc() -> None:
    run = RunRecord.from_dict(_run(started_at="2026-01-01T09:00:00+09:00"))
    assert run.to_dict()["started_at"] == "2026-01-01T00:00:00.000000Z"


@pytest.mark.parametrize("bad", ["yesterday", 12345])
def test_unparseable_timestamps_are_run_errors(bad) -> None:
    with pytest.raises(RunError):
        RunRecord.from_dict(_run(started_at=bad))


@pytest.mark.parametrize(("seed", "expected"), [(3, 3), ("3", 3), (3.0, 3), ("3.0", 3)])
def test_integral_seeds_are_accepted(seed, expected) -> None:
    assert RunRecord.from_dict(_run(seed=seed)).seed == expected


@pytest.mark.parametrize("seed", [3.9, "3.9", "x", None, True])
def test_non_integral_seeds_are_rejected(seed) -> None:
    with pytest.raises(RunError, match="not an integer"):
        RunRecord.from_dict(_run(seed=seed))


@pytest.mark.parametrize(
    "data", [["a"], {"foo": [1]}, [{**_run(), "metrics": [1, 2]}], [{**_run(), "run_id": None}]]
)
def test_malformed_run_json_is_a_run_error(tmp_path: Path, data) -> None:
    path = tmp_path / "runs.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(RunError):
        load_runs_json(path)


def test_invalid_json_is_a_run_error(tmp_path: Path) -> None:
    path = tmp_path / "runs.json"
    path.write_text("", encoding="utf-8")
    with pytest.raises(RunError, match="not valid JSON"):
        load_runs_json(path)


def test_csv_with_byte_order_mark(tmp_path: Path) -> None:
    path = tmp_path / "runs.csv"
    path.write_text(
        "﻿run_id,condition,seed,started_at,score\nb0,control,0,2026-01-01T00:00:00Z,1.5\n",
        encoding="utf-8",
    )
    (run,) = load_runs_csv(path)
    assert run.run_id == "b0" and run.metrics == {"score": 1.5}


def test_csv_missing_required_column(tmp_path: Path) -> None:
    path = tmp_path / "runs.csv"
    path.write_text("run_id,condition,started_at,score\nb0,control,2026-01-01T00:00:00Z,1\n")
    with pytest.raises(RunError, match="missing required columns"):
        load_runs_csv(path)


def test_analysis_is_refused_on_a_tampered_ledger(tmp_path: Path, clock: FakeClock) -> None:
    project, _ = run_study(
        tmp_path,
        clock,
        pred("treatment - control > 0"),
        {"control": vals(0.0), "treatment": vals(1.0)},
    )
    path = project.ledger.path
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[-1] = lines[-1].replace('"score": 1.', '"score": 9.', 1)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    n_before = len(lines)
    with pytest.raises(LedgerError, match="fails verification"):
        project.analyze()
    assert len(path.read_text(encoding="utf-8").splitlines()) == n_before
    # read-only analysis (as used by reports) still works and reports the failure
    assert project.analyze(record=False).integrity["ok"] is False


def test_malformed_anchor_line_is_a_clean_error(tmp_path: Path, clock: FakeClock) -> None:
    project = Project.init(tmp_path, clock=clock)
    project.anchors_path.write_text("garbage line x\n", encoding="utf-8")
    with pytest.raises(LedgerError, match="malformed anchor"):
        project.read_anchors()


def test_cli_reports_bad_yaml_without_a_traceback(tmp_path: Path, capsys) -> None:
    assert main(["--root", str(tmp_path), "init"]) == 0
    (tmp_path / "prereg.yaml").write_text("study: [unclosed\n", encoding="utf-8")
    assert main(["--root", str(tmp_path), "register"]) == 2
    assert "preregister: error:" in capsys.readouterr().err


def test_cli_reports_bad_runs_without_a_traceback(tmp_path: Path, clock: FakeClock, capsys) -> None:
    run_study(tmp_path, clock, pred("treatment - control > 0"), None)
    runs = tmp_path / "runs.json"
    runs.write_text(json.dumps([_run(started_at="2026-01-01T00:00:00")]), encoding="utf-8")
    assert main(["--root", str(tmp_path), "link", "--json", str(runs)]) == 2
    assert "no UTC offset" in capsys.readouterr().err


def test_make_runs_helper_uses_aware_times(clock: FakeClock) -> None:
    assert all(r.started_at.tzinfo for r in make_runs({"control": [1.0]}, clock()))

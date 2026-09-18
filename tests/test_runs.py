from __future__ import annotations

import math
from pathlib import Path

import pytest

from preregister import Exclusion, RunRecord, load_runs_csv, load_runs_json
from preregister.runs import RunError, exclusion_matches, first_matching_exclusion, write_runs_json

from .conftest import T0


def _run(**kw) -> RunRecord:
    base = dict(
        run_id="r1",
        condition="a",
        seed=3,
        metrics={"score": 1.5, "loss": math.nan},
        started_at=T0,
        meta={"host": "gpu1"},
    )
    base.update(kw)
    return RunRecord(**base)


def test_run_validation() -> None:
    with pytest.raises(RunError):
        _run(status="weird")
    with pytest.raises(RunError):
        _run(ended_at="2025-01-01T00:00:00Z")
    with pytest.raises(RunError):
        RunRecord.from_dict({"run_id": "x"})
    with pytest.raises(RunError):
        _run(metrics={"score": "high"})


def test_roundtrip_and_hash_stability() -> None:
    run = _run(started_at="2026-01-01T00:00:00+02:00")
    again = RunRecord.from_dict(run.to_dict())
    assert again.content_hash() == run.content_hash()
    assert math.isnan(again.metrics["loss"])
    assert run.to_dict()["started_at"] == "2025-12-31T22:00:00.000000Z"


@pytest.mark.parametrize(
    ("rule", "expected"),
    [
        ("status != finished", False),
        ("status == finished", True),
        ("seed >= 3", True),
        ("seed < 3", False),
        ("condition == a", True),
        ("metric < 2", True),
        ("metric:score > 2", False),
        ("metric:loss is_nan", True),
        ("metric:acc is_missing", True),
        ("metric:score is_missing", False),
        ("meta:host == gpu1", True),
        ("meta:host == 'gpu2'", False),
        ("meta:missing == x", False),
    ],
)
def test_exclusion_rules(rule: str, expected: bool) -> None:
    assert exclusion_matches(Exclusion(rule, "r"), _run(), "score") is expected


def test_first_matching_exclusion_order() -> None:
    rules = [Exclusion("seed > 10", "a"), Exclusion("metric < 5", "b"), Exclusion("seed == 3", "c")]
    assert first_matching_exclusion(rules, _run(), "score").reason == "b"


def test_json_and_csv_import(tmp_path: Path) -> None:
    runs = [_run(), _run(run_id="r2", seed=4, metric_versions={"score": "1"})]
    path = tmp_path / "runs.json"
    write_runs_json(runs, path)
    loaded = load_runs_json(path)
    assert [r.run_id for r in loaded] == ["r1", "r2"] and loaded[1].metric_versions == {
        "score": "1"
    }

    csv_path = tmp_path / "runs.csv"
    csv_path.write_text(
        "run_id,condition,seed,started_at,status,score,meta.host,version.score\n"
        "c1,a,0,2026-01-01T00:00:00Z,finished,0.9,gpu1,1\n"
        "c2,a,1,2026-01-01T00:00:00Z,failed,,gpu2,\n"
    )
    rows = load_runs_csv(csv_path)
    assert rows[0].metrics == {"score": 0.9} and rows[0].meta == {"host": "gpu1"}
    assert rows[0].metric_versions == {"score": "1"} and rows[1].metric_versions == {}
    assert rows[1].status == "failed" and math.isnan(rows[1].metrics["score"])

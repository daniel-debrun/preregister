from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from preregister import AnalysisPlan, Exclusion, Prediction, Project, Verdict
from preregister.reconcile import analyze_state

from .conftest import FakeClock, make_runs, make_study

NOISE = np.array([-1.5, -1.0, -0.6, -0.3, -0.1, 0.1, 0.3, 0.6, 1.0, 1.5])


def vals(mean: float, sd: float = 0.1) -> list[float]:
    return list(mean + sd * NOISE)


def run_study(tmp_path, clock, pred, data, gate=True, **study_kw):
    study = make_study(tmp_path, [pred] if isinstance(pred, Prediction) else pred, **study_kw)
    project = Project.init(tmp_path, clock=clock)
    project.register(study)
    if gate:
        project.run_gates(study)
    if data:
        project.link_runs(make_runs(data, clock.advance(hours=1)), force=not gate)
    return project, study


def pred(comparison: str, test: str = "welch", aggregation: str = "mean", **kw) -> Prediction:
    plan = AnalysisPlan(
        test=test,
        aggregation=aggregation,
        n_seeds=10,
        n_boot=2000,
        n_permutations=2000,
        **{k: kw.pop(k) for k in list(kw) if k in AnalysisPlan.__dataclass_fields__},
    )
    return Prediction(
        "P1", "h", "score", comparison, min_effect=kw.pop("min_effect", 0.5), analysis=plan, **kw
    )


VERDICT_CASES = [
    ("treatment - control > 0", {"treatment": vals(5), "control": vals(0)}, Verdict.CONFIRMED),
    ("treatment - control > 0", {"treatment": vals(0), "control": vals(5)}, Verdict.REFUTED),
    (
        "treatment - control > 0",
        {"treatment": vals(0, 1), "control": vals(0, 1)},
        Verdict.INCONCLUSIVE,
    ),
    ("treatment - control > 3", {"treatment": vals(5), "control": vals(0)}, Verdict.CONFIRMED),
    ("treatment - control > 3", {"treatment": vals(1), "control": vals(0)}, Verdict.REFUTED),
    (
        "treatment - control > 3",
        {"treatment": vals(3, 1), "control": vals(0, 1)},
        Verdict.INCONCLUSIVE,
    ),
    ("treatment < control", {"treatment": vals(1), "control": vals(2)}, Verdict.CONFIRMED),
    ("treatment - control < -2", {"treatment": vals(1), "control": vals(2)}, Verdict.REFUTED),
    ("treatment / control >= 1.5", {"treatment": vals(20), "control": vals(10)}, Verdict.CONFIRMED),
    ("treatment / control >= 1.5", {"treatment": vals(11), "control": vals(10)}, Verdict.REFUTED),
    ("treatment > 0.5", {"treatment": vals(0.9)}, Verdict.CONFIRMED),
    ("treatment <= 0.5", {"treatment": vals(0.9)}, Verdict.REFUTED),
]


@pytest.mark.parametrize(
    ("test", "comparison", "data", "expected"),
    [
        (test, *case)
        for test in ("welch", "bootstrap", "permutation", "mann_whitney")
        for case in VERDICT_CASES
        if not (test == "mann_whitney" and len(case[1]) == 1)
    ],
)
def test_verdict_table(tmp_path: Path, clock: FakeClock, test, comparison, data, expected) -> None:
    p = pred(comparison, test=test, aggregation="mean")
    project, _ = run_study(tmp_path, clock, p, data)
    result = project.analyze(record=False).result("P1")
    assert result.deviations == []
    assert result.verdict == expected, (result.p_confirm, result.p_refute)


@pytest.mark.parametrize(
    ("interval", "center", "sd", "test", "expected"),
    [
        ("treatment in [0.4, 0.6]", 0.5, 0.02, "welch", Verdict.CONFIRMED),
        ("treatment in [0.4, 0.6]", 0.5, 0.02, "bootstrap", Verdict.CONFIRMED),
        ("treatment in [0.4, 0.6]", 0.9, 0.02, "bootstrap", Verdict.REFUTED),
        ("treatment in [0.4, 0.6]", 0.6, 0.2, "welch", Verdict.INCONCLUSIVE),
    ],
)
def test_interval_verdicts(tmp_path, clock, interval, center, sd, test, expected) -> None:
    project, _ = run_study(
        tmp_path, clock, pred(interval, test=test, min_effect=0.05), {"treatment": vals(center, sd)}
    )
    assert project.analyze(record=False).result("P1").verdict == expected


def test_estimates_and_intervals(tmp_path: Path, clock: FakeClock) -> None:
    p = pred("treatment - control > 0", test="bootstrap", aggregation="iqm")
    project, _ = run_study(tmp_path, clock, p, {"treatment": vals(2), "control": vals(1)})
    r = project.analyze(record=False).result("P1")
    assert r.estimate == pytest.approx(1.0)
    assert r.ci_level == pytest.approx(0.9)
    assert r.ci_low < 1.0 < r.ci_high
    assert r.n_by_condition == {"treatment": 10, "control": 10}
    again = project.analyze(record=False).result("P1")
    assert (again.ci_low, again.p_confirm) == (r.ci_low, r.p_confirm)


def test_inadmissible_when_gates_fail_or_missing(tmp_path: Path, clock: FakeClock) -> None:
    failing = pred("treatment - control > 0", gates=({"type": "MinSeeds", "n": 50},))
    data = {"treatment": vals(5), "control": vals(0)}
    project, _ = run_study(tmp_path / "a", clock, failing, data)
    r = project.analyze(record=False).result("P1")
    assert r.verdict == Verdict.INADMISSIBLE and r.estimate is None and r.p_confirm is None
    assert not r.gate_results[-1]["passed"]

    project, _ = run_study(tmp_path / "b", clock, pred("treatment - control > 0"), data, gate=False)
    r = project.analyze(record=False).result("P1")
    assert r.verdict == Verdict.INADMISSIBLE and "not evaluated" in r.notes[0]


def test_seed_coverage_deviations(tmp_path: Path, clock: FakeClock) -> None:
    project, _ = run_study(tmp_path, clock, pred("treatment - control > 0"), None)
    start = clock.advance(hours=1)
    runs = make_runs(
        {"treatment": vals(5), "control": vals(0)},
        start,
        seeds={"treatment": [*range(9), 42], "control": range(10)},
    )
    runs += make_runs({"control": [0.5]}, start, seeds={"control": [3]})
    runs[-1] = runs[-1].__class__(**{**runs[-1].__dict__, "run_id": "control-3-rerun"})
    project.link_runs(runs)
    r = project.analyze(record=False).result("P1")
    assert r.verdict == Verdict.DEVIATED
    assert r.statistical_outcome == Verdict.CONFIRMED
    text = " | ".join(r.deviations)
    assert "treatment: declared seeds without a run: 9" in text
    assert "treatment: runs with undeclared seeds: 42" in text
    assert "control: multiple runs for seeds 3" in text


def test_declared_exclusions_are_not_deviations(tmp_path: Path, clock: FakeClock) -> None:
    p = pred("treatment - control > 0", exclusions=(Exclusion("metric is_nan", "diverged"),))
    project, _ = run_study(
        tmp_path, clock, p, None, exclusions=(Exclusion("status != finished", "crashed"),)
    )
    start = clock.advance(hours=1)
    runs = make_runs({"treatment": [*vals(5)[:9], math.nan], "control": vals(0)}, start)
    runs[3] = runs[3].__class__(**{**runs[3].__dict__, "status": "failed"})
    project.link_runs(runs)
    r = project.analyze(record=False).result("P1")
    assert r.verdict == Verdict.CONFIRMED
    assert {(x.run_id, x.reason.split(" (")[0]) for x in r.excluded} == {
        ("treatment-3", "crashed"),
        ("treatment-9", "diverged"),
    }
    assert r.n_by_condition["treatment"] == 8


def test_undeclared_nan_is_a_deviation(tmp_path: Path, clock: FakeClock) -> None:
    data = {"treatment": [*vals(5)[:9], math.nan], "control": vals(0)}
    project, _ = run_study(tmp_path, clock, pred("treatment - control > 0"), data)
    r = project.analyze(record=False).result("P1")
    assert r.verdict == Verdict.DEVIATED
    assert r.rejected[0].run_id == "treatment-9"


def test_override_is_a_deviation(tmp_path: Path, clock: FakeClock) -> None:
    project, _ = run_study(
        tmp_path, clock, pred("treatment - control > 0"), {"treatment": vals(5), "control": vals(0)}
    )
    r = project.analyze(record=False, overrides={"P1": {"test": "bootstrap"}}).result("P1")
    assert r.verdict == Verdict.DEVIATED and r.statistical_outcome == Verdict.CONFIRMED
    assert r.plan["test"] == "bootstrap"
    assert "analysis override: test 'welch' -> 'bootstrap'" in r.deviations
    with pytest.raises(KeyError):
        project.analyze(record=False, overrides={"P9": {"test": "welch"}})


def test_amendment_before_vs_after_linking(tmp_path: Path, clock: FakeClock) -> None:
    p = pred("treatment - control > 0")
    project, study = run_study(tmp_path, clock, p, None)
    amended = study.with_prediction(Prediction.from_dict({**p.to_dict(), "min_effect": 1.0}))
    project.amend(amended, "P1", reason="pilot variance was higher than expected")
    project.run_gates(amended)
    project.link_runs(make_runs({"treatment": vals(5), "control": vals(0)}, clock.advance(hours=1)))
    r = project.analyze(record=False).result("P1")
    assert r.verdict == Verdict.CONFIRMED
    assert r.amendments[0]["after_runs_linked"] is False

    late = amended.with_prediction(Prediction.from_dict({**p.to_dict(), "min_effect": 2.0}))
    project.amend(late, "P1", reason="changed after seeing data")
    r = project.analyze(record=False).result("P1")
    assert r.verdict == Verdict.INADMISSIBLE
    project.run_gates(late)
    r = project.analyze(record=False).result("P1")
    assert r.verdict == Verdict.DEVIATED and r.version == 3
    assert any("after runs were linked" in d for d in r.deviations)


def test_holm_family_counts_inadmissible_members(tmp_path: Path, clock: FakeClock) -> None:
    def member(pid: str, gates=()) -> Prediction:
        return Prediction(
            pid,
            "h",
            "score",
            "treatment - control > 0",
            0.5,
            gates=gates,
            analysis=AnalysisPlan(test="welch", n_seeds=10, correction="holm", family="f"),
        )

    preds = [member("A"), member("B", gates=({"type": "MinSeeds", "n": 99},)), member("C")]
    data = {"treatment": vals(1, 1), "control": vals(0, 1)}
    project, _ = run_study(tmp_path, clock, preds, data)
    analysis = project.analyze(record=False)
    a, b, c = (analysis.result(x) for x in "ABC")
    assert b.verdict == Verdict.INADMISSIBLE and b.p_confirm_adj is None
    assert a.p_confirm == c.p_confirm
    assert a.p_confirm < 0.05
    assert a.p_confirm_adj == pytest.approx(3 * a.p_confirm)
    assert c.p_confirm_adj == pytest.approx(3 * c.p_confirm)


def test_correction_can_change_verdict(tmp_path: Path, clock: FakeClock) -> None:
    def member(pid: str, correction: str) -> Prediction:
        return Prediction(
            pid,
            "h",
            "score",
            "treatment - control > 0",
            0.5,
            analysis=AnalysisPlan(
                test="welch", n_seeds=10, correction=correction, family=correction
            ),
        )

    data = {"treatment": vals(0.4, 0.5), "control": vals(0, 0.5)}
    preds = [member(f"H{i}", "holm") for i in range(6)] + [member("N", "none")]
    project, _ = run_study(tmp_path, clock, preds, data)
    analysis = project.analyze(record=False)
    assert analysis.result("N").p_confirm < 0.05
    assert analysis.result("N").verdict == Verdict.CONFIRMED
    assert analysis.result("H0").verdict == Verdict.INCONCLUSIVE


def test_linking_rules(tmp_path: Path, clock: FakeClock) -> None:
    project, _ = run_study(
        tmp_path,
        clock,
        pred("treatment - control > 0", gates=({"type": "MinSeeds", "n": 99},)),
        None,
    )
    start = clock.advance(hours=1)
    runs = make_runs({"treatment": vals(1)[:2]}, start)
    summary = project.link_runs(runs)
    assert summary.linked == [] and "passed its gates" in summary.refused[0][1]
    summary = project.link_runs(runs, force=True)
    assert summary.linked == ["treatment-0", "treatment-1"]
    assert project.link_runs(runs).unchanged == ["treatment-0", "treatment-1"]
    changed = make_runs({"treatment": [9.0]}, start)
    assert "different content" in project.link_runs(changed).refused[0][1]
    other = make_runs({"ghost": [1.0]}, start)
    assert "no registered prediction" in project.link_runs(other, force=True).refused[0][1]
    running = make_runs({"treatment": [1.0]}, start, seeds={"treatment": [5]}, status="running")
    assert "running" in project.link_runs(running, force=True).refused[0][1]
    assert project.state().links["treatment-0"].forced


def test_analysis_is_recorded_and_ledger_only(tmp_path: Path, clock: FakeClock) -> None:
    project, _ = run_study(
        tmp_path, clock, pred("treatment - control > 0"), {"treatment": vals(5), "control": vals(0)}
    )
    analysis = project.analyze()
    head = project.ledger.head()
    assert head.kind == "analysis" and head.payload["verdicts"] == {"P1": "CONFIRMED"}
    assert analysis.integrity["ok"]
    standalone = analyze_state(project.state())
    assert standalone.result("P1").verdict == Verdict.CONFIRMED


def test_exploratory(tmp_path: Path, clock: FakeClock) -> None:
    from preregister import ExploratorySpec

    project, _ = run_study(
        tmp_path, clock, pred("treatment - control > 0"), {"treatment": vals(3), "control": vals(1)}
    )
    specs = [
        ExploratorySpec(
            "E1", "score", "treatment / control > 1", aggregation="median", n_boot=1000
        ),
        ExploratorySpec("E2", "missing_metric", "treatment > control", n_boot=1000),
    ]
    e1, e2 = project.analyze(record=False, exploratory=specs).exploratory
    assert e1.estimate == pytest.approx(3.0) and e1.ci_low < 3.0 < e1.ci_high
    assert e2.estimate is None and "fewer than 2" in e2.note


def test_no_runs_yet_is_inconclusive(tmp_path: Path, clock: FakeClock) -> None:
    project, _ = run_study(tmp_path, clock, pred("treatment - control > 0"), None)
    r = project.analyze(record=False).result("P1")
    assert r.verdict == Verdict.INCONCLUSIVE and "no runs linked yet" in r.notes


def test_runs_planned_by_other_predictions_are_not_deviations(
    tmp_path: Path, clock: FakeClock
) -> None:
    def member(pid: str, seeds: tuple[int, ...]) -> Prediction:
        return Prediction(
            pid,
            "h",
            "score",
            "treatment - control > 0",
            0.5,
            analysis=AnalysisPlan(test="welch", n_seeds=len(seeds), seeds=seeds),
        )

    preds = [member("SMALL", tuple(range(5))), member("LARGE", tuple(range(10)))]
    data = {"treatment": [*vals(5), 9.0], "control": [*vals(0), 9.0]}
    seeds = {"treatment": [*range(10), 77], "control": [*range(10), 77]}
    study = make_study(tmp_path, preds)
    project = Project.init(tmp_path, clock=clock)
    project.register(study)
    project.run_gates(study)
    project.link_runs(make_runs(data, clock.advance(hours=1), seeds=seeds))
    analysis = project.analyze(record=False)
    small, large = analysis.result("SMALL"), analysis.result("LARGE")
    assert small.n_by_condition == {"treatment": 6, "control": 6}
    assert "10 linked run(s) use seeds planned by other predictions" in small.notes
    assert small.deviations == [
        "treatment: runs with undeclared seeds: 77",
        "control: runs with undeclared seeds: 77",
    ]
    assert large.n_by_condition == {"treatment": 11, "control": 11}
    assert small.verdict == large.verdict == Verdict.DEVIATED

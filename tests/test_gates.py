from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from preregister import (
    AnalysisPlan,
    Condition,
    GateResult,
    Metric,
    Prediction,
    Project,
    custom_gate,
)
from preregister.gates import GATE_REGISTRY, build_gate, gates_for
from preregister.registry import GateSpec, SpecError

from .conftest import FakeClock, make_runs, make_study


def _pred(gates, comparison="treatment - control > 0", **plan) -> Prediction:
    plan = {"test": "welch", "n_seeds": 10, **plan}
    return Prediction(
        "P1",
        "h",
        "score",
        comparison,
        min_effect=1.0,
        analysis=AnalysisPlan(**plan),
        gates=tuple(gates),
    )


def _gate(tmp_path: Path, clock: FakeClock, pred: Prediction, **study_kw) -> dict[str, GateResult]:
    study = make_study(tmp_path, [pred], **study_kw)
    project = Project.init(tmp_path, clock=clock)
    project.register(study)
    results = project.run_gates(study)["P1"]
    assert project.state().latest_gate("P1").passed == all(r.passed for r in results)
    return {r.gate: r for r in results}


def test_implicit_gates_always_present() -> None:
    names = [g.type for g in gates_for(_pred([{"type": "MinSeeds"}]))]
    assert names == ["MetricDefined", "ConfigFrozen", "NoPeeking", "MinSeeds"]
    names = [g.type for g in gates_for(_pred([{"type": "ConfigFrozen", "require_run_hash": True}]))]
    assert names.count("ConfigFrozen") == 1


@pytest.mark.parametrize(("minimum", "passed"), [(10, True), (11, False)])
def test_min_seeds(tmp_path: Path, clock: FakeClock, minimum: int, passed: bool) -> None:
    res = _gate(tmp_path, clock, _pred([{"type": "MinSeeds", "n": minimum}]))
    assert res["MinSeeds"].passed is passed


@pytest.mark.parametrize(
    ("params", "n_seeds", "passed"),
    [
        ({"sd": 1.0}, 13, True),
        ({"sd": 1.0}, 12, False),
        ({"sd_by_condition": {"treatment": 0.5, "control": 0.5}}, 5, True),
        ({"pilot": {"treatment": [0.0, 2.0], "control": [0.0, 2.0]}}, 10, False),
        ({"sd": 1.0, "method": "simulation", "n_sim": 400}, 20, True),
        ({"sd": 1.0, "method": "simulation", "n_sim": 400}, 4, False),
    ],
)
def test_power_analysis(
    tmp_path: Path, clock: FakeClock, params: dict, n_seeds: int, passed: bool
) -> None:
    res = _gate(tmp_path, clock, _pred([{"type": "PowerAnalysis", **params}], n_seeds=n_seeds))
    assert res["PowerAnalysis"].passed is passed, res["PowerAnalysis"].message


def test_power_analysis_bonferroni_for_corrected_family(tmp_path: Path, clock: FakeClock) -> None:
    gate = {"type": "PowerAnalysis", "sd": 1.0}
    preds = [
        Prediction(
            f"P{i}",
            "h",
            "score",
            "treatment - control > 0",
            1.0,
            analysis=AnalysisPlan(test="welch", n_seeds=13, correction="holm"),
            gates=(gate,),
        )
        for i in (1, 2, 3)
    ]
    study = make_study(tmp_path, preds)
    project = Project.init(tmp_path, clock=clock)
    project.register(study)
    power = next(r for r in project.run_gates(study, ["P1"])["P1"] if r.gate == "PowerAnalysis")
    assert not power.passed
    assert power.details["alpha"] == pytest.approx(0.05 / 3)


def test_power_analysis_parameter_errors(tmp_path: Path, clock: FakeClock) -> None:
    with pytest.raises(SpecError):
        build_gate(GateSpec("PowerAnalysis", {}))
    with pytest.raises(SpecError):
        build_gate(GateSpec("PowerAnalysis", {"sd": 1, "power": 0.3}))
    with pytest.raises(SpecError, match="invalid parameters"):
        build_gate(GateSpec("MinSeeds", {"minimum": 3}))
    res = _gate(
        tmp_path,
        clock,
        _pred([{"type": "PowerAnalysis", "sd": 1.0}], comparison="treatment / control > 1.1"),
    )
    assert not res["PowerAnalysis"].passed and "control_mean" in res["PowerAnalysis"].message


def test_control_present(tmp_path: Path, clock: FakeClock) -> None:
    assert _gate(tmp_path / "a", clock, _pred([{"type": "ControlPresent"}]))[
        "ControlPresent"
    ].passed
    no_control = {
        "control": Condition("control", role="treatment", config={"x": 1}),
        "treatment": Condition("treatment", role="treatment", config={"x": 2}),
    }
    res = _gate(tmp_path / "b", clock, _pred([{"type": "ControlPresent"}]), conditions=no_control)
    assert not res["ControlPresent"].passed
    unconfigured = {
        "control": Condition("control", role="control"),
        "treatment": Condition("treatment", role="treatment", config={"x": 2}),
    }
    res = _gate(tmp_path / "c", clock, _pred([{"type": "ControlPresent"}]), conditions=unconfigured)
    assert not res["ControlPresent"].passed and "config" in res["ControlPresent"].message
    res = _gate(
        tmp_path / "d", clock, _pred([{"type": "ControlPresent"}], comparison="treatment > 0.5")
    )
    assert res["ControlPresent"].passed and "not applicable" in res["ControlPresent"].message


def test_metric_defined_requires_version_or_function(tmp_path: Path, clock: FakeClock) -> None:
    res = _gate(tmp_path, clock, _pred([]), metrics={"score": Metric("score")})
    assert not res["MetricDefined"].passed


def metric_fn_v1(x: float) -> float:
    return x


def test_metric_function_change_detected(
    tmp_path: Path, clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    metrics = {"score": Metric("score", function="tests.test_gates:metric_fn_v1")}
    study = make_study(tmp_path, [_pred([])], metrics=metrics)
    project = Project.init(tmp_path, clock=clock)
    project.register(study)
    assert all(r.passed for r in project.run_gates(study)["P1"])
    import inspect

    monkeypatch.setattr(
        inspect, "getsource", lambda obj: "def metric_fn_v1(x):\n    return 2 * x\n"
    )
    res = {r.gate: r for r in project.run_gates(study)["P1"]}
    assert not res["MetricDefined"].passed and "changed" in res["MetricDefined"].message


def test_config_frozen_detects_drift(tmp_path: Path, clock: FakeClock) -> None:
    cfg = tmp_path / "control.yaml"
    cfg.write_text("lr: 0.1\n")
    conditions = {
        "control": Condition("control", role="control", config_file="control.yaml"),
        "treatment": Condition("treatment", role="treatment", config={"lr": 0.2}),
    }
    study = make_study(tmp_path, [_pred([])], conditions=conditions)
    project = Project.init(tmp_path, clock=clock)
    project.register(study)
    assert all(r.passed for r in project.run_gates(study)["P1"])
    cfg.write_text("lr: 0.5\n")
    res = {r.gate: r for r in project.run_gates(study)["P1"]}
    assert not res["ConfigFrozen"].passed and "control" in res["ConfigFrozen"].message


def test_per_run_checks(tmp_path: Path, clock: FakeClock) -> None:
    study = make_study(tmp_path, [_pred([{"type": "ConfigFrozen", "require_run_hash": True}])])
    project = Project.init(tmp_path, clock=clock)
    project.register(study)
    project.run_gates(study)
    registered = project.state().current("P1").frozen["conditions"]["control"]["config_hash"]
    start = clock.advance(hours=1)
    good = make_runs(
        {"control": [1.0]}, start, config_hash=registered, metric_versions={"score": "1"}
    )
    drift = make_runs({"control": [1.0]}, start, seeds={"control": [1]}, config_hash="ab" * 32)
    no_hash = make_runs({"control": [1.0]}, start, seeds={"control": [2]})
    wrong_version = make_runs(
        {"control": [1.0]},
        start,
        seeds={"control": [3]},
        config_hash=registered,
        metric_versions={"score": "2"},
    )
    no_metric = make_runs(
        {"control": [1.0]}, start, seeds={"control": [4]}, config_hash=registered, metric="other"
    )
    summary = project.link_runs(good + drift + no_hash + wrong_version + no_metric)
    assert len(summary.linked) == 5
    rejected = {rid: " ".join(v["P1"]) for rid, v in summary.rejected_for.items()}
    assert set(rejected) == {"control-1", "control-2", "control-3", "control-4"}
    assert "ConfigFrozen" in rejected["control-1"] and "required" in rejected["control-2"]
    assert "version" in rejected["control-3"] and "does not report" in rejected["control-4"]


def test_no_peeking_timestamps(tmp_path: Path, clock: FakeClock) -> None:
    study = make_study(tmp_path, [_pred([])])
    project = Project.init(tmp_path, clock=clock)
    before = clock.now - timedelta(seconds=30)
    project.register(study)
    registered_at = project.state().first("P1").time
    project.run_gates(study)
    runs = make_runs({"control": [1.0]}, before)
    runs += make_runs(
        {"control": [1.0]}, registered_at - timedelta(microseconds=1), seeds={"control": [1]}
    )
    runs += make_runs({"control": [1.0]}, registered_at, seeds={"control": [2]})
    summary = project.link_runs(runs)
    assert set(summary.rejected_for) == {"control-0", "control-1"}
    res = {r.gate: r for r in project.run_gates(study)["P1"]}
    assert not res["NoPeeking"].passed and res["NoPeeking"].details["runs"] == [
        "control-0",
        "control-1",
    ]


def test_custom_gate_decorator_variants(tmp_path: Path, clock: FakeClock) -> None:
    @custom_gate("AlwaysNo")
    def always_no(ctx, why: str = "nope"):
        return False, why

    @custom_gate
    def bare_true(ctx):
        return True

    @custom_gate("ReturnsResult")
    def returns_result(ctx):
        return GateResult("ReturnsResult", True, "explicit", {"k": 1})

    @custom_gate("Explodes")
    def explodes(ctx):
        raise RuntimeError("boom")

    assert "AlwaysNo" in GATE_REGISTRY and "bare_true" in GATE_REGISTRY
    assert always_no(None, why="direct call still works") == (False, "direct call still works")
    gates = [
        {"type": "AlwaysNo", "why": "layout broken"},
        {"type": "bare_true"},
        {"type": "ReturnsResult"},
        {"type": "Explodes"},
    ]
    res = _gate(tmp_path, clock, _pred(gates))
    assert (res["AlwaysNo"].passed, res["AlwaysNo"].message) == (False, "layout broken")
    assert res["bare_true"].passed
    assert res["ReturnsResult"].details == {"k": 1}
    assert not res["Explodes"].passed and "boom" in res["Explodes"].message


def test_gate_by_import_path(tmp_path: Path, clock: FakeClock) -> None:
    res = _gate(tmp_path, clock, _pred([{"type": "tests.test_gates:path_gate"}]))
    assert res["PathGate"].passed


@custom_gate("PathGate")
def path_gate(ctx):
    return ctx.prediction.id == "P1", "checked by import path"


def test_unknown_gate_fails_closed(tmp_path: Path, clock: FakeClock) -> None:
    res = _gate(tmp_path, clock, _pred([{"type": "DoesNotExist"}]))
    assert not res["DoesNotExist"].passed


def test_gating_requires_current_registration(tmp_path: Path, clock: FakeClock) -> None:
    from preregister.project import RegistrationError

    study = make_study(tmp_path, [_pred([])])
    project = Project.init(tmp_path, clock=clock)
    with pytest.raises(RegistrationError):
        project.run_gates(study)

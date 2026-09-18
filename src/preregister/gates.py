"""Admissibility gates.

A gate answers "can this study, as registered, answer its question?". Gates run in
two places:

- ``check(ctx)`` before runs are linked. Results are appended to the ledger and a
  prediction whose latest gate record failed receives the verdict ``INADMISSIBLE``.
- ``check_run(registration, run)`` for every linked run. A non-empty reason rejects
  the run for that prediction; rejected runs are still recorded and reported.

``MetricDefined``, ``ConfigFrozen`` and ``NoPeeking`` are protocol invariants and are
applied to every prediction even when not declared.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from preregister.analysis import (
    aggregation_efficiency_factor,
    normal_approx_power,
    normal_approx_seeds,
    simulate_power,
)
from preregister.registry import GateSpec, Prediction, SpecError, Study, import_object
from preregister.runs import RunRecord
from preregister.state import LedgerState, RegistrationVersion

IMPLICIT_GATES = ("MetricDefined", "ConfigFrozen", "NoPeeking")


@dataclass(frozen=True)
class GateResult:
    gate: str
    passed: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate": self.gate,
            "passed": self.passed,
            "message": self.message,
            "details": self.details,
        }


@dataclass
class GateContext:
    study: Study
    prediction: Prediction
    registration: RegistrationVersion
    state: LedgerState

    @property
    def root(self) -> Path:
        return self.study.root

    @property
    def family_size(self) -> int:
        fam = self.prediction.analysis.family
        return sum(1 for p in self.study.predictions if p.analysis.family == fam)


class Gate:
    name: ClassVar[str] = "Gate"

    def check(self, ctx: GateContext) -> GateResult:
        return GateResult(self.name, True, "no pre-run check")

    def check_run(self, registration: RegistrationVersion, run: RunRecord) -> str | None:
        return None

    def result(self, passed: bool, message: str, **details: Any) -> GateResult:
        return GateResult(self.name, passed, message, details)


GATE_REGISTRY: dict[str, Callable[..., Gate]] = {}


def register_gate(cls: type[Gate]) -> type[Gate]:
    GATE_REGISTRY[cls.name] = cls
    return cls


def build_gate(spec: GateSpec) -> Gate:
    factory = GATE_REGISTRY.get(spec.type)
    if factory is None:
        if ":" not in spec.type:
            raise SpecError(f"unknown gate {spec.type!r}; registered: {sorted(GATE_REGISTRY)}")
        obj = import_object(spec.type)
        factory = getattr(obj, "gate_factory", obj)
    try:
        gate = factory(**dict(spec.params))
    except TypeError as exc:
        raise SpecError(f"gate {spec.type}: invalid parameters: {exc}") from exc
    if not isinstance(gate, Gate):
        raise SpecError(f"gate {spec.type!r} did not produce a Gate instance")
    return gate


def gates_for(prediction: Prediction) -> list[GateSpec]:
    declared = list(prediction.gates)
    names = {g.type for g in declared}
    return [GateSpec(t) for t in IMPLICIT_GATES if t not in names] + declared


@register_gate
class MinSeeds(Gate):
    name = "MinSeeds"

    def __init__(self, n: int = 5) -> None:
        self.n = int(n)

    def check(self, ctx: GateContext) -> GateResult:
        planned = ctx.prediction.analysis.n_seeds
        return self.result(
            planned >= self.n,
            f"{planned} seeds per condition planned, minimum {self.n}",
            planned=planned,
            minimum=self.n,
        )


@register_gate
class PowerAnalysis(Gate):
    """Fail if the planned seeds cannot detect ``min_effect`` with the target power.

    Standard deviations come from ``sd`` (shared), ``sd_by_condition``, or ``pilot``
    (a mapping of condition to pilot metric values; pilot runs should use seeds that are
    not among the registered ones). ``method='normal'`` uses the normal approximation scaled by
    the efficiency of the planned aggregation (and pi/3 for Mann-Whitney);
    ``method='simulation'`` estimates the power of the planned test by Monte Carlo on
    Gaussian data. With a multiple-comparison correction, alpha is Bonferroni-split
    over the family, which is conservative for Holm and BH.
    """

    name = "PowerAnalysis"

    def __init__(
        self,
        sd: float | None = None,
        sd_by_condition: dict[str, float] | None = None,
        pilot: dict[str, list[float]] | None = None,
        power: float = 0.8,
        method: str = "normal",
        control_mean: float | None = None,
        n_sim: int = 300,
        seed: int = 0,
    ) -> None:
        if not 0.5 <= power < 1:
            raise SpecError("PowerAnalysis: power must be in [0.5, 1)")
        if method not in ("normal", "simulation"):
            raise SpecError("PowerAnalysis: method must be 'normal' or 'simulation'")
        if sd is None and not sd_by_condition and not pilot:
            raise SpecError("PowerAnalysis: provide sd, sd_by_condition, or pilot values")
        self.sd, self.sd_by_condition, self.pilot = sd, dict(sd_by_condition or {}), pilot
        self.power, self.method, self.control_mean = float(power), method, control_mean
        self.n_sim, self.seed = int(n_sim), int(seed)

    def _sd(self, condition: str) -> float:
        if self.pilot and condition in self.pilot:
            values = np.asarray(self.pilot[condition], dtype=float)
            if values.size < 2:
                raise SpecError(f"PowerAnalysis: pilot for {condition!r} needs >= 2 values")
            return float(values.std(ddof=1))
        if condition in self.sd_by_condition:
            return float(self.sd_by_condition[condition])
        if self.sd is None:
            raise SpecError(f"PowerAnalysis: no standard deviation for condition {condition!r}")
        return float(self.sd)

    def check(self, ctx: GateContext) -> GateResult:
        pred, plan = ctx.prediction, ctx.prediction.analysis
        cmp = pred.parsed
        try:
            sd_a = self._sd(cmp.left)
            sd_b = None if cmp.right is None else self._sd(cmp.right)
        except SpecError as exc:
            return self.result(False, str(exc))
        delta = pred.min_effect
        if cmp.kind == "ratio":
            if self.control_mean is None or self.control_mean <= 0:
                return self.result(False, "ratio comparisons need a positive control_mean")
            assert cmp.threshold is not None and sd_b is not None
            delta = pred.min_effect * self.control_mean
            sd_b = cmp.threshold * sd_b
        alpha = plan.alpha
        if plan.correction != "none" and ctx.family_size > 1:
            alpha = alpha / ctx.family_size
        n = plan.n_seeds
        if self.method == "normal":
            factor = aggregation_efficiency_factor(plan.aggregation, n)
            if plan.test == "mann_whitney":
                factor *= math.pi / 3.0
            required = normal_approx_seeds(delta, sd_a, sd_b, alpha, self.power, factor)
            achieved = normal_approx_power(delta, sd_a, sd_b, alpha, n, factor)
        else:
            achieved = simulate_power(
                plan, delta, sd_a, sd_b, alpha, n, n_sim=self.n_sim, seed=self.seed
            )
            required = None
        passed = achieved >= self.power
        req = "" if required is None else f", ~{required} required"
        return self.result(
            passed,
            f"power {achieved:.2f} at {n} seeds for margin {delta:g} "
            f"(target {self.power:.2f}{req}; alpha {alpha:.4g}, {self.method})",
            achieved_power=round(achieved, 4),
            required_seeds=required,
            planned_seeds=n,
            sd=[sd_a, sd_b],
            alpha=alpha,
            method=self.method,
        )


@register_gate
class ControlPresent(Gate):
    name = "ControlPresent"

    def check(self, ctx: GateContext) -> GateResult:
        cmp = ctx.prediction.parsed
        if cmp.right is None:
            return self.result(True, f"not applicable to {cmp.kind} comparisons")
        conds = [ctx.study.conditions[c] for c in cmp.conditions]
        controls = [c for c in conds if c.role in ("control", "baseline")]
        if not controls:
            return self.result(
                False, f"neither {cmp.left!r} nor {cmp.right!r} has role control/baseline"
            )
        unconfigured = [c.name for c in conds if not c.is_configured]
        if unconfigured:
            return self.result(False, f"conditions without a declared config: {unconfigured}")
        return self.result(True, f"control {controls[0].name!r} declared and configured")


@register_gate
class MetricDefined(Gate):
    """The metric is declared with a version or function, unchanged since registration,
    and every run reports that same version (when it reports one)."""

    name = "MetricDefined"

    def check(self, ctx: GateContext) -> GateResult:
        metric = ctx.study.metrics.get(ctx.prediction.metric)
        if metric is None:
            return self.result(False, f"metric {ctx.prediction.metric!r} not declared")
        if metric.version is None and metric.function is None:
            return self.result(False, f"metric {metric.name!r} has neither version nor function")
        registered = ctx.registration.frozen["metric"]
        current = {"name": metric.name, **metric.fingerprint()}
        if current != registered:
            return self.result(
                False,
                f"metric {metric.name!r} changed since registration",
                registered=registered,
                current=current,
            )
        return self.result(True, f"metric {metric.name!r} fingerprint matches registration")

    def check_run(self, registration: RegistrationVersion, run: RunRecord) -> str | None:
        metric = registration.frozen["metric"]
        name = metric["name"]
        if name not in run.metrics:
            return f"run does not report metric {name!r}"
        reported = run.metric_versions.get(name)
        if (
            reported is not None
            and metric.get("version") is not None
            and reported != metric["version"]
        ):
            return f"metric {name!r} version {reported!r} != registered {metric['version']!r}"
        return None


@register_gate
class ConfigFrozen(Gate):
    """Condition configs match the registered hashes, now and for every run."""

    name = "ConfigFrozen"

    def __init__(self, require_run_hash: bool = False) -> None:
        self.require_run_hash = bool(require_run_hash)

    def check(self, ctx: GateContext) -> GateResult:
        mismatched = []
        for cname, reg in ctx.registration.frozen["conditions"].items():
            current = ctx.study.conditions[cname].config_hash(ctx.root)
            if current != reg["config_hash"]:
                mismatched.append(cname)
        if mismatched:
            return self.result(False, f"config changed since registration: {mismatched}")
        return self.result(True, "condition configs match registered hashes")

    def check_run(self, registration: RegistrationVersion, run: RunRecord) -> str | None:
        expected = registration.frozen["conditions"].get(run.condition, {}).get("config_hash")
        if run.config_hash is None:
            if self.require_run_hash and expected is not None:
                return "run has no config_hash (required)"
            return None
        if expected is not None and run.config_hash != expected:
            return f"config_hash {run.config_hash[:12]} != registered {expected[:12]}"
        return None


@register_gate
class NoPeeking(Gate):
    """Runs that started before the prediction was first registered cannot count."""

    name = "NoPeeking"

    def check(self, ctx: GateContext) -> GateResult:
        cutoff = ctx.state.first(ctx.prediction.id).time
        early = [
            rec.run.run_id
            for rec in ctx.state.runs_for(ctx.prediction.parsed.conditions)
            if rec.run.started_at < cutoff
        ]
        if early:
            return self.result(False, f"{len(early)} linked runs predate registration", runs=early)
        return self.result(True, "no linked runs predate registration")

    @staticmethod
    def predates(first_registration: RegistrationVersion, run: RunRecord) -> bool:
        return run.started_at < first_registration.time


def custom_gate(
    name_or_fn: str | Callable[..., Any] | None = None,
) -> Any:
    """Turn a function into a gate usable from Python or YAML.

    The function receives a :class:`GateContext` plus the gate's declared parameters and
    returns ``bool``, ``(bool, message)``, or a :class:`GateResult`::

        @custom_gate("RecipesCompletable")
        def recipes_completable(ctx, layout: str) -> tuple[bool, str]:
            ...

    Reference it in YAML as ``{type: RecipesCompletable, layout: ...}`` once the defining
    module is imported, or by import path ``{type: "mypkg.gates:recipes_completable"}``.
    """

    def decorate(fn: Callable[..., Any], gate_name: str) -> Callable[..., Any]:
        class FunctionGate(Gate):
            name = gate_name

            def __init__(self, **params: Any) -> None:
                self.params = params

            def check(self, ctx: GateContext) -> GateResult:
                out = fn(ctx, **self.params)
                if isinstance(out, GateResult):
                    return out
                if isinstance(out, tuple):
                    passed, message = out
                    return self.result(bool(passed), str(message))
                return self.result(bool(out), "passed" if out else "failed")

        FunctionGate.__name__ = FunctionGate.__qualname__ = gate_name
        GATE_REGISTRY[gate_name] = FunctionGate

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return fn(*args, **kwargs)

        wrapper.gate_factory = FunctionGate  # type: ignore[attr-defined]
        wrapper.gate_name = gate_name  # type: ignore[attr-defined]
        return wrapper

    if callable(name_or_fn):
        return decorate(name_or_fn, name_or_fn.__name__)
    return lambda fn: decorate(fn, name_or_fn or fn.__name__)


def evaluate_gates(ctx: GateContext) -> list[GateResult]:
    results = []
    for spec in gates_for(ctx.prediction):
        try:
            results.append(build_gate(spec).check(ctx))
        except Exception as exc:
            results.append(GateResult(spec.type, False, f"gate raised {type(exc).__name__}: {exc}"))
    return results


def run_rejections(
    registration: RegistrationVersion, first: RegistrationVersion, run: RunRecord
) -> list[str]:
    """Reasons a linked run cannot count toward a prediction (empty if admissible)."""
    reasons = []
    if NoPeeking.predates(first, run):
        reasons.append("NoPeeking: run started before the prediction was registered")
    for spec in gates_for(registration.prediction):
        try:
            gate = build_gate(spec)
        except (SpecError, ImportError, AttributeError):
            continue
        reason = gate.check_run(registration, run)
        if reason:
            reasons.append(f"{gate.name}: {reason}")
    return reasons

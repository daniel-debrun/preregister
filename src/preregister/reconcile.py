"""Reconciliation: run exactly the registered analysis and assign verdicts.

Verdict precedence, highest first:

1. ``INADMISSIBLE``: gates were never evaluated for the current registered version, or
   the latest evaluation failed. No statistics are computed.
2. ``DEVIATED``: something differs from the registered plan (amendment after runs were
   linked, missing or undeclared seeds, runs rejected for peeking or config drift,
   non-finite values without a declared exclusion, or an explicit analysis override).
   The statistical outcome is still reported next to the verdict.
3. ``CONFIRMED`` / ``REFUTED`` / ``INCONCLUSIVE`` from the (adjusted) one-sided p-values:
   confirmed if the claim is significant at alpha, refuted if its negation is.
"""

from __future__ import annotations

import math
import zlib
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np

from preregister.analysis import (
    adjust_p_values,
    effect_bootstrap,
    effect_estimate,
    evaluate_claim,
    percentile_ci,
)
from preregister.gates import run_rejections
from preregister.ledger import format_timestamp, utcnow
from preregister.registry import AnalysisPlan, Comparison, ExploratorySpec
from preregister.runs import first_matching_exclusion
from preregister.state import LedgerState, LinkRecord

PROTOCOL_REJECTIONS = ("NoPeeking", "ConfigFrozen")


class Verdict(str, Enum):
    CONFIRMED = "CONFIRMED"
    REFUTED = "REFUTED"
    INCONCLUSIVE = "INCONCLUSIVE"
    INADMISSIBLE = "INADMISSIBLE"
    DEVIATED = "DEVIATED"


@dataclass(frozen=True)
class RunValue:
    run_id: str
    condition: str
    seed: int
    value: float


@dataclass(frozen=True)
class SetAside:
    run_id: str
    condition: str
    seed: int
    reason: str


@dataclass
class PredictionResult:
    prediction_id: str
    hypothesis: str
    metric: str
    comparison: str
    version: int
    prediction_hash: str
    registered_at: str
    first_registered_at: str
    plan: dict[str, Any]
    min_effect: float
    verdict: Verdict = Verdict.INCONCLUSIVE
    statistical_outcome: Verdict | None = None
    estimate: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    ci_level: float | None = None
    p_confirm: float | None = None
    p_refute: float | None = None
    p_confirm_adj: float | None = None
    p_refute_adj: float | None = None
    values: list[RunValue] = field(default_factory=list)
    excluded: list[SetAside] = field(default_factory=list)
    rejected: list[SetAside] = field(default_factory=list)
    deviations: list[str] = field(default_factory=list)
    amendments: list[dict[str, Any]] = field(default_factory=list)
    gate_results: list[dict[str, Any]] = field(default_factory=list)
    gates_evaluated_at: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def n_by_condition(self) -> dict[str, int]:
        counts = Counter(v.condition for v in self.values)
        return {c: counts.get(c, 0) for c in Comparison.parse(self.comparison).conditions}

    def samples(self) -> dict[str, np.ndarray]:
        cmp = Comparison.parse(self.comparison)
        return {
            c: np.array([v.value for v in self.values if v.condition == c], dtype=float)
            for c in cmp.conditions
        }

    def to_dict(self) -> dict[str, Any]:
        d = _finite(asdict(self))
        d["verdict"] = self.verdict.value
        d["statistical_outcome"] = (
            None if self.statistical_outcome is None else self.statistical_outcome.value
        )
        d["n_by_condition"] = self.n_by_condition
        return d


@dataclass
class ExploratoryResult:
    id: str
    metric: str
    comparison: str
    aggregation: str
    ci_level: float
    estimate: float | None
    ci_low: float | None
    ci_high: float | None
    values: list[RunValue]
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _finite(asdict(self))


@dataclass
class StudyAnalysis:
    study: str
    generated_at: str
    ledger_head: str | None
    n_entries: int
    integrity: dict[str, Any] | None
    predictions: list[PredictionResult]
    exploratory: list[ExploratoryResult]
    overrides: dict[str, dict[str, Any]]

    def verdicts(self) -> dict[str, str]:
        return {p.prediction_id: p.verdict.value for p in self.predictions}

    def result(self, prediction_id: str) -> PredictionResult:
        for p in self.predictions:
            if p.prediction_id == prediction_id:
                return p
        raise KeyError(prediction_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "study": self.study,
            "generated_at": self.generated_at,
            "ledger_head": self.ledger_head,
            "n_entries": self.n_entries,
            "integrity": self.integrity,
            "overrides": self.overrides,
            "predictions": [p.to_dict() for p in self.predictions],
            "exploratory": [e.to_dict() for e in self.exploratory],
        }


def _finite(obj: Any) -> Any:
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_finite(v) for v in obj]
    return obj


def _rng(seed: int, key: str) -> np.random.Generator:
    return np.random.default_rng([seed, zlib.crc32(key.encode("utf-8"))])


def _relevant_runs(state: LedgerState, pid: str, plan: AnalysisPlan) -> list[LinkRecord]:
    """Linked runs of the prediction's conditions, in ledger order.

    A run whose seed is not declared by this prediction but is declared by another
    registered prediction on the same condition belongs to that prediction's plan and
    is left out here. Runs with seeds no prediction declared stay in, and are reported.
    """
    declared = set(plan.planned_seeds)
    out = []
    for rec in sorted(
        state.runs_for(state.current(pid).comparison.conditions), key=lambda r: r.entry.index
    ):
        run = rec.run
        if run.seed not in declared and any(
            run.seed in state.current(other).prediction.analysis.planned_seeds
            for other in state.predictions_using(run.condition)
            if other != pid
        ):
            continue
        out.append(rec)
    return out


def _collect_runs(
    state: LedgerState, pid: str, plan: AnalysisPlan, result: PredictionResult
) -> None:
    current, first = state.current(pid), state.first(pid)
    metric = current.frozen["metric"]["name"]
    exclusions = current.exclusions
    relevant = _relevant_runs(state, pid, plan)
    others = len(state.runs_for(current.comparison.conditions)) - len(relevant)
    if others:
        result.notes.append(f"{others} linked run(s) use seeds planned by other predictions")
    for rec in relevant:
        run = rec.run
        reasons = run_rejections(current, first, run)
        if reasons:
            result.rejected.append(
                SetAside(run.run_id, run.condition, run.seed, "; ".join(reasons))
            )
            continue
        rule = first_matching_exclusion(exclusions, run, metric)
        if rule is not None:
            result.excluded.append(
                SetAside(run.run_id, run.condition, run.seed, f"{rule.reason} ({rule.rule})")
            )
            continue
        value = run.metrics[metric]
        if not math.isfinite(value):
            result.rejected.append(
                SetAside(
                    run.run_id,
                    run.condition,
                    run.seed,
                    "metric value is not finite and no declared exclusion covers it",
                )
            )
            result.deviations.append(
                f"run {run.run_id}: non-finite {metric} not covered by a declared exclusion"
            )
            continue
        result.values.append(RunValue(run.run_id, run.condition, run.seed, value))


def _check_deviations(
    state: LedgerState, pid: str, plan: AnalysisPlan, result: PredictionResult
) -> None:
    current = state.current(pid)
    cmp = current.comparison
    relevant = _relevant_runs(state, pid, plan)
    link_indices = [rec.entry.index for rec in relevant]
    first_link = min(link_indices) if link_indices else None
    for version in state.registrations[pid][1:]:
        after_data = first_link is not None and version.entry.index > first_link
        result.amendments.append(
            {
                "version": version.version,
                "timestamp": version.entry.timestamp,
                "reason": version.reason,
                "prediction_hash": version.prediction_hash,
                "after_runs_linked": after_data,
            }
        )
        if after_data:
            result.deviations.append(
                f"amended to v{version.version} after runs were linked: {version.reason}"
            )
    for gate in PROTOCOL_REJECTIONS:
        hit = [r.run_id for r in result.rejected if f"{gate}:" in r.reason]
        if hit:
            result.deviations.append(f"{len(hit)} run(s) rejected by {gate}: {', '.join(hit)}")
    if not relevant:
        result.notes.append("no runs linked yet")
        return
    declared = set(plan.planned_seeds)
    rejected_ids = {r.run_id for r in result.rejected}
    counted = [rec.run for rec in relevant if rec.run.run_id not in rejected_ids]
    for cond in cmp.conditions:
        seeds = [r.seed for r in counted if r.condition == cond]
        seen = set(seeds)
        missing = sorted(declared - seen)
        undeclared = sorted(seen - declared)
        dupes = sorted(s for s, n in Counter(seeds).items() if n > 1)
        if missing:
            result.deviations.append(f"{cond}: declared seeds without a run: {_seed_list(missing)}")
        if undeclared:
            result.deviations.append(
                f"{cond}: runs with undeclared seeds: {_seed_list(undeclared)}"
            )
        if dupes:
            result.deviations.append(f"{cond}: multiple runs for seeds {_seed_list(dupes)}")


def _seed_list(seeds: list[int], limit: int = 12) -> str:
    shown = ", ".join(str(s) for s in seeds[:limit])
    return shown + (f" (+{len(seeds) - limit} more)" if len(seeds) > limit else "")


def analyze_state(
    state: LedgerState,
    *,
    study_name: str | None = None,
    overrides: Mapping[str, Mapping[str, Any]] | None = None,
    exploratory: Iterable[ExploratorySpec] = (),
    integrity: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> StudyAnalysis:
    overrides = {k: dict(v) for k, v in (overrides or {}).items()}
    unknown = set(overrides) - set(state.registrations)
    if unknown:
        raise KeyError(f"overrides for unregistered predictions: {sorted(unknown)}")
    results: list[PredictionResult] = []
    plans: dict[str, AnalysisPlan] = {}
    for pid in state.prediction_ids:
        current = state.current(pid)
        pred = current.prediction
        plan = pred.analysis
        if pid in overrides:
            plan = replace(plan, **overrides[pid])
        plans[pid] = plan
        result = PredictionResult(
            prediction_id=pid,
            hypothesis=pred.hypothesis,
            metric=pred.metric,
            comparison=pred.comparison,
            version=current.version,
            prediction_hash=current.prediction_hash,
            registered_at=current.entry.timestamp,
            first_registered_at=state.first(pid).entry.timestamp,
            plan=plan.to_dict(),
            min_effect=pred.min_effect,
        )
        for key, value in overrides.get(pid, {}).items():
            result.deviations.append(
                f"analysis override: {key} {getattr(pred.analysis, key)!r} -> {value!r}"
            )
        _collect_runs(state, pid, plan, result)
        _check_deviations(state, pid, plan, result)
        gate = state.latest_gate(pid)
        if gate is not None:
            result.gate_results = gate.results
            result.gates_evaluated_at = gate.entry.timestamp
        if gate is None or not gate.passed:
            result.verdict = Verdict.INADMISSIBLE
            result.notes.append(
                "gates were not evaluated for the registered version"
                if gate is None
                else "admissibility gates failed; analysis withheld"
            )
            results.append(result)
            continue
        samples = result.samples()
        too_small = [c for c, arr in samples.items() if arr.size < 2]
        bad_denominator = _nonpositive_denominator(current.comparison, samples)
        if too_small:
            result.notes.append(f"fewer than 2 analyzable runs for {too_small}")
        elif bad_denominator:
            result.notes.append(bad_denominator)
        else:
            claim = evaluate_claim(current.comparison, samples, plan, _rng(plan.random_seed, pid))
            result.estimate = claim.estimate
            result.ci_low, result.ci_high, result.ci_level = (
                claim.ci_low,
                claim.ci_high,
                claim.ci_level,
            )
            result.p_confirm, result.p_refute = claim.p_confirm, claim.p_refute
        results.append(result)

    _apply_corrections(results, plans)
    for result in results:
        if result.verdict == Verdict.INADMISSIBLE:
            continue
        alpha = plans[result.prediction_id].alpha
        if result.p_confirm_adj is None or result.p_refute_adj is None:
            outcome = Verdict.INCONCLUSIVE
        elif result.p_confirm_adj < alpha:
            outcome = Verdict.CONFIRMED
        elif result.p_refute_adj < alpha:
            outcome = Verdict.REFUTED
        else:
            outcome = Verdict.INCONCLUSIVE
        result.statistical_outcome = outcome
        result.verdict = Verdict.DEVIATED if result.deviations else outcome

    ledger_head = state.entries[-1].hash if state.entries else None
    name = study_name
    if name is None and state.registrations:
        name = state.current(state.prediction_ids[0]).frozen.get("study")
    return StudyAnalysis(
        study=name or "study",
        generated_at=format_timestamp(now or utcnow()),
        ledger_head=ledger_head,
        n_entries=len(state.entries),
        integrity=integrity,
        predictions=results,
        exploratory=[_exploratory(state, spec) for spec in exploratory],
        overrides=overrides,
    )


def _nonpositive_denominator(cmp: Comparison, samples: Mapping[str, np.ndarray]) -> str | None:
    """Reason a ratio claim cannot be tested, or None.

    ``a / b >= r`` is tested as ``a - r * b >= 0``, which is only equivalent when every
    value of ``b`` (and so every resampled aggregate of it) is positive.
    """
    if cmp.kind != "ratio" or cmp.right is None:
        return None
    n_bad = int(np.count_nonzero(samples[cmp.right] <= 0))
    if not n_bad:
        return None
    return (
        f"ratio claim needs a strictly positive denominator; {cmp.right} has "
        f"{n_bad} value(s) <= 0, so no statistics were computed"
    )


def _apply_corrections(results: list[PredictionResult], plans: Mapping[str, AnalysisPlan]) -> None:
    """Adjust p-values within each family.

    The family size is the number of registered predictions in it, including those
    that are inadmissible or lack data (they enter with p = 1), so dropping a
    prediction cannot make the correction less strict.
    """
    families: dict[str, list[PredictionResult]] = {}
    for r in results:
        families.setdefault(plans[r.prediction_id].family, []).append(r)
    for members in families.values():
        method = plans[members[0].prediction_id].correction
        for attr in ("p_confirm", "p_refute"):
            raw = np.array([1.0 if getattr(r, attr) is None else getattr(r, attr) for r in members])
            adjusted = adjust_p_values(raw, method)
            for r, adj in zip(members, adjusted, strict=True):
                if getattr(r, attr) is not None:
                    setattr(r, f"{attr}_adj", float(adj))


def _exploratory(state: LedgerState, spec: ExploratorySpec) -> ExploratoryResult:
    cmp = Comparison.parse(spec.comparison)
    values = [
        RunValue(rec.run.run_id, rec.run.condition, rec.run.seed, rec.run.metrics[spec.metric])
        for rec in sorted(state.runs_for(cmp.conditions), key=lambda r: r.entry.index)
        if rec.run.status == "finished"
        and spec.metric in rec.run.metrics
        and math.isfinite(rec.run.metrics[spec.metric])
    ]
    result = ExploratoryResult(
        id=spec.id,
        metric=spec.metric,
        comparison=str(cmp),
        aggregation=spec.aggregation,
        ci_level=spec.ci_level,
        estimate=None,
        ci_low=None,
        ci_high=None,
        values=values,
    )
    samples = {
        c: np.array([v.value for v in values if v.condition == c], dtype=float)
        for c in cmp.conditions
    }
    if any(arr.size < 2 for arr in samples.values()):
        result.note = "fewer than 2 finished runs in a condition"
        return result
    bad_denominator = _nonpositive_denominator(cmp, samples)
    if bad_denominator:
        result.note = bad_denominator
        return result
    rng = _rng(spec.random_seed, spec.id)
    result.estimate = effect_estimate(cmp, samples, spec.aggregation)
    boot = effect_bootstrap(cmp, samples, spec.aggregation, spec.n_boot, rng)
    boot = boot[np.isfinite(boot)]
    if boot.size:
        result.ci_low, result.ci_high = percentile_ci(boot, spec.ci_level)
    return result


__all__ = [
    "ExploratoryResult",
    "PredictionResult",
    "RunValue",
    "SetAside",
    "StudyAnalysis",
    "Verdict",
    "analyze_state",
]

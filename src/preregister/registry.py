"""Study specification: conditions, metrics, predictions, and their frozen form.

A :class:`Study` is plain data. It can be built in Python or loaded from YAML
(``prereg.yaml``). Registration does not happen here; it happens when a
prediction's frozen payload is appended to the ledger (see :mod:`preregister.project`).
"""

from __future__ import annotations

import importlib
import inspect
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import yaml

from preregister.canonical import sha256_file, sha256_hex

SCHEMA_VERSION = 1

TestType = Literal["bootstrap", "welch", "mann_whitney", "permutation"]
Aggregation = Literal["mean", "median", "iqm"]
Correction = Literal["none", "holm", "bh"]

TESTS = ("bootstrap", "welch", "mann_whitney", "permutation")
AGGREGATIONS = ("mean", "median", "iqm")
CORRECTIONS = ("none", "holm", "bh")
ROLES = ("control", "baseline", "treatment", "other")


class SpecError(ValueError):
    """Raised when a study specification is invalid."""


_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_NAME = r"[A-Za-z_]\w*"
_INTERVAL_RE = re.compile(rf"^\s*({_NAME})\s+in\s+\[\s*({_NUM})\s*,\s*({_NUM})\s*\]\s*$")
_BINARY_RE = re.compile(
    rf"^\s*({_NAME})\s*(?:([-/])\s*({_NAME})\s*)?(<=|>=|<|>)\s*({_NUM}|{_NAME})\s*$"
)


@dataclass(frozen=True)
class Comparison:
    """A parsed directional claim about aggregated metric values.

    Supported forms (``a``/``b`` are condition names, ``t``/``r``/``lo``/``hi`` numbers)::

        a - b > t      difference with threshold
        a > b          difference with threshold 0
        a / b >= r     ratio of aggregates
        a > t          one-sample threshold
        a in [lo, hi]  aggregate lies within an interval
    """

    kind: Literal["difference", "ratio", "one_sample", "interval"]
    left: str
    right: str | None = None
    op: str | None = None
    threshold: float | None = None
    low: float | None = None
    high: float | None = None

    @classmethod
    def parse(cls, expr: str) -> Comparison:
        m = _INTERVAL_RE.match(expr)
        if m:
            lo, hi = float(m.group(2)), float(m.group(3))
            if not lo < hi:
                raise SpecError(f"interval bounds must satisfy lo < hi: {expr!r}")
            return cls(kind="interval", left=m.group(1), low=lo, high=hi)
        m = _BINARY_RE.match(expr)
        if not m:
            raise SpecError(
                f"cannot parse comparison {expr!r}; expected 'a - b > t', 'a > b', "
                "'a / b >= r', 'a > t', or 'a in [lo, hi]'"
            )
        left, arith, right, op, rhs = m.groups()
        rhs_is_num = re.fullmatch(_NUM, rhs) is not None
        if arith is None:
            if rhs_is_num:
                return cls(kind="one_sample", left=left, op=op, threshold=float(rhs))
            if rhs == left:
                raise SpecError(f"comparison compares a condition with itself: {expr!r}")
            return cls(kind="difference", left=left, right=rhs, op=op, threshold=0.0)
        if not rhs_is_num:
            raise SpecError(f"right-hand side of {expr!r} must be a number")
        if right == left:
            raise SpecError(f"comparison compares a condition with itself: {expr!r}")
        if arith == "-":
            return cls(kind="difference", left=left, right=right, op=op, threshold=float(rhs))
        if float(rhs) <= 0:
            raise SpecError(f"ratio threshold must be positive: {expr!r}")
        return cls(kind="ratio", left=left, right=right, op=op, threshold=float(rhs))

    @property
    def conditions(self) -> tuple[str, ...]:
        return (self.left,) if self.right is None else (self.left, self.right)

    @property
    def direction(self) -> str:
        if self.kind == "interval":
            return "within"
        return "increase" if self.op in (">", ">=") else "decrease"

    @property
    def boundary(self) -> float | tuple[float, float]:
        if self.kind == "interval":
            assert self.low is not None and self.high is not None
            return (self.low, self.high)
        assert self.threshold is not None
        return self.threshold

    def __str__(self) -> str:
        if self.kind == "interval":
            return f"{self.left} in [{_fmt(self.low)}, {_fmt(self.high)}]"
        if self.kind == "one_sample":
            return f"{self.left} {self.op} {_fmt(self.threshold)}"
        sym = "-" if self.kind == "difference" else "/"
        return f"{self.left} {sym} {self.right} {self.op} {_fmt(self.threshold)}"


def _fmt(x: float | None) -> str:
    if x is None:
        return "None"
    if float(x).is_integer() and abs(x) < 1e15:
        return str(int(x))
    return repr(float(x))


@dataclass(frozen=True)
class AnalysisPlan:
    test: str = "bootstrap"
    alpha: float = 0.05
    n_seeds: int = 10
    seeds: tuple[int, ...] | None = None
    aggregation: str = "mean"
    correction: str = "none"
    family: str = "default"
    n_boot: int = 10_000
    n_permutations: int = 10_000
    random_seed: int = 0

    def __post_init__(self) -> None:
        if self.test not in TESTS:
            raise SpecError(f"unknown test {self.test!r}; choose from {TESTS}")
        if self.aggregation not in AGGREGATIONS:
            raise SpecError(f"unknown aggregation {self.aggregation!r}; choose from {AGGREGATIONS}")
        if self.correction not in CORRECTIONS:
            raise SpecError(f"unknown correction {self.correction!r}; choose from {CORRECTIONS}")
        object.__setattr__(self, "alpha", float(self.alpha))
        if not 0 < self.alpha < 0.5:
            raise SpecError("alpha must be in (0, 0.5)")
        if self.n_seeds < 2:
            raise SpecError("n_seeds must be at least 2")
        if self.n_boot < 100 or self.n_permutations < 100:
            raise SpecError("n_boot and n_permutations must be at least 100")
        if self.seeds is not None:
            seeds = tuple(int(s) for s in self.seeds)
            if len(set(seeds)) != len(seeds):
                raise SpecError("declared seeds must be unique")
            if len(seeds) != self.n_seeds:
                raise SpecError(f"declared {len(seeds)} seeds but n_seeds={self.n_seeds}")
            object.__setattr__(self, "seeds", seeds)
        if self.test == "welch" and self.aggregation != "mean":
            raise SpecError("the Welch t-test is a test of means; use aggregation 'mean'")

    @property
    def planned_seeds(self) -> tuple[int, ...]:
        return self.seeds if self.seeds is not None else tuple(range(self.n_seeds))

    @property
    def ci_level(self) -> float:
        return 1.0 - 2.0 * self.alpha

    def to_dict(self) -> dict[str, Any]:
        return {
            "test": self.test,
            "alpha": self.alpha,
            "n_seeds": self.n_seeds,
            "seeds": list(self.planned_seeds),
            "aggregation": self.aggregation,
            "correction": self.correction,
            "family": self.family,
            "n_boot": self.n_boot,
            "n_permutations": self.n_permutations,
            "random_seed": self.random_seed,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> AnalysisPlan:
        _check_keys(d, cls.__dataclass_fields__, "analysis")
        d = dict(d)
        if d.get("seeds") is not None:
            d["seeds"] = tuple(d["seeds"])
            d.setdefault("n_seeds", len(d["seeds"]))
        return cls(**d)


_EXCL_RE = re.compile(r"^\s*([\w:.]+)\s*(==|!=|<=|>=|<|>|is_nan|is_missing)\s*(.*?)\s*$")


@dataclass(frozen=True)
class Exclusion:
    """A pre-declared exclusion criterion, e.g. ``status != finished``.

    Fields: ``status``, ``seed``, ``condition``, ``metric`` (the prediction's metric),
    ``metric:<name>``, ``meta:<key>``. Operators: ``== != < <= > >= is_nan is_missing``.
    """

    rule: str
    reason: str

    def __post_init__(self) -> None:
        m = _EXCL_RE.match(self.rule)
        if not m:
            raise SpecError(f"cannot parse exclusion rule {self.rule!r}")
        fld, op, value = m.groups()
        if op in ("is_nan", "is_missing") and value:
            raise SpecError(f"operator {op} takes no value: {self.rule!r}")
        if op not in ("is_nan", "is_missing") and not value:
            raise SpecError(f"operator {op} requires a value: {self.rule!r}")
        base = fld.split(":", 1)[0]
        if base not in ("status", "seed", "condition", "metric", "meta"):
            raise SpecError(f"unknown exclusion field {fld!r}")
        if not self.reason.strip():
            raise SpecError(f"exclusion {self.rule!r} must state a reason")

    @property
    def parts(self) -> tuple[str, str, str]:
        m = _EXCL_RE.match(self.rule)
        assert m is not None
        return m.group(1), m.group(2), m.group(3).strip("'\"")

    def to_dict(self) -> dict[str, str]:
        return {"rule": self.rule, "reason": self.reason}


@dataclass(frozen=True)
class Condition:
    name: str
    role: str = "other"
    config: Mapping[str, Any] | None = None
    config_file: str | None = None
    description: str = ""

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise SpecError(f"condition {self.name!r}: role must be one of {ROLES}")
        if self.config is not None and self.config_file is not None:
            raise SpecError(f"condition {self.name!r}: give config or config_file, not both")

    @property
    def is_configured(self) -> bool:
        return self.config is not None or self.config_file is not None

    def config_hash(self, root: Path) -> str | None:
        if self.config is not None:
            return sha256_hex(dict(self.config))
        if self.config_file is not None:
            path = root / self.config_file
            if not path.exists():
                raise SpecError(f"condition {self.name!r}: config file {path} not found")
            return sha256_file(path)
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "config": None if self.config is None else dict(self.config),
            "config_file": self.config_file,
            "description": self.description,
        }


@dataclass(frozen=True)
class Metric:
    name: str
    version: str | None = None
    function: str | None = None
    higher_is_better: bool | None = None
    description: str = ""

    def fingerprint(self) -> dict[str, str | None]:
        """Version string and, if a metric function is referenced, a hash of its source."""
        fn_hash = None
        if self.function is not None:
            fn_hash = sha256_hex(inspect.getsource(import_object(self.function)))
        return {"version": self.version, "function": self.function, "function_hash": fn_hash}

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "function": self.function,
            "higher_is_better": self.higher_is_better,
            "description": self.description,
        }


@dataclass(frozen=True)
class GateSpec:
    type: str
    params: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "params": dict(self.params)}

    @classmethod
    def from_any(cls, obj: Any) -> GateSpec:
        if isinstance(obj, GateSpec):
            return obj
        if isinstance(obj, str):
            return cls(type=obj)
        if isinstance(obj, Mapping):
            obj = dict(obj)
            gtype = obj.pop("type", None)
            if gtype is None:
                raise SpecError(f"gate spec missing 'type': {obj}")
            params = obj.pop("params", None)
            return cls(type=gtype, params=dict(params) if params is not None else obj)
        raise SpecError(f"invalid gate spec: {obj!r}")


@dataclass(frozen=True)
class Prediction:
    id: str
    hypothesis: str
    metric: str
    comparison: str
    min_effect: float
    analysis: AnalysisPlan = field(default_factory=AnalysisPlan)
    stopping_rule: str = "fixed number of seeds, no early stopping"
    direction: str | None = None
    gates: tuple[GateSpec, ...] = ()
    exclusions: tuple[Exclusion, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", self.id):
            raise SpecError(f"prediction id {self.id!r} must match [A-Za-z0-9_.-]+")
        object.__setattr__(self, "hypothesis", " ".join(self.hypothesis.split()))
        if not self.hypothesis:
            raise SpecError(f"prediction {self.id}: hypothesis text is required")
        object.__setattr__(self, "min_effect", float(self.min_effect))
        if not math.isfinite(self.min_effect) or self.min_effect <= 0:
            raise SpecError(f"prediction {self.id}: min_effect must be a positive number")
        cmp = self.parsed
        if self.direction is not None and self.direction != cmp.direction:
            raise SpecError(
                f"prediction {self.id}: declared direction {self.direction!r} contradicts "
                f"comparison {self.comparison!r} ({cmp.direction})"
            )
        object.__setattr__(self, "direction", cmp.direction)
        object.__setattr__(self, "comparison", str(cmp))
        object.__setattr__(self, "gates", tuple(GateSpec.from_any(g) for g in self.gates))
        test = self.analysis.test
        if cmp.kind == "interval" and test not in ("bootstrap", "welch"):
            raise SpecError(f"prediction {self.id}: interval claims support bootstrap or welch")
        if cmp.kind == "one_sample" and test == "mann_whitney":
            raise SpecError(f"prediction {self.id}: Mann-Whitney needs two conditions")

    @property
    def parsed(self) -> Comparison:
        return Comparison.parse(self.comparison)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "hypothesis": self.hypothesis,
            "metric": self.metric,
            "comparison": self.comparison,
            "direction": self.direction,
            "min_effect": self.min_effect,
            "analysis": self.analysis.to_dict(),
            "stopping_rule": self.stopping_rule,
            "gates": [g.to_dict() for g in self.gates],
            "exclusions": [e.to_dict() for e in self.exclusions],
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Prediction:
        _check_keys(d, cls.__dataclass_fields__, f"prediction {d.get('id')!r}")
        d = dict(d)
        d["analysis"] = AnalysisPlan.from_dict(d.get("analysis") or {})
        d["gates"] = tuple(GateSpec.from_any(g) for g in d.get("gates") or ())
        d["exclusions"] = tuple(Exclusion(**e) for e in d.get("exclusions") or ())
        return cls(**d)


@dataclass(frozen=True)
class ExploratorySpec:
    """An analysis that is explicitly not pre-registered. Reported separately, no verdict."""

    id: str
    metric: str
    comparison: str
    aggregation: str = "mean"
    ci_level: float = 0.95
    n_boot: int = 10_000
    random_seed: int = 0

    def __post_init__(self) -> None:
        Comparison.parse(self.comparison)
        if self.aggregation not in AGGREGATIONS:
            raise SpecError(f"exploratory {self.id}: unknown aggregation {self.aggregation!r}")
        if not 0 < self.ci_level < 1:
            raise SpecError(f"exploratory {self.id}: ci_level must be in (0, 1)")


@dataclass
class Study:
    name: str
    conditions: dict[str, Condition]
    metrics: dict[str, Metric]
    predictions: list[Prediction]
    description: str = ""
    exclusions: tuple[Exclusion, ...] = ()
    exploratory: list[ExploratorySpec] = field(default_factory=list)
    root: Path = field(default_factory=Path.cwd)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.validate()

    def validate(self) -> None:
        ids = [p.id for p in self.predictions]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise SpecError(f"duplicate prediction ids: {sorted(dupes)}")
        for name, cond in self.conditions.items():
            if cond.name != name:
                raise SpecError(f"condition key {name!r} does not match its name {cond.name!r}")
        families: dict[str, str] = {}
        for pred in self.predictions:
            for cname in pred.parsed.conditions:
                if cname not in self.conditions:
                    raise SpecError(f"prediction {pred.id}: unknown condition {cname!r}")
            if pred.metric not in self.metrics:
                raise SpecError(f"prediction {pred.id}: metric {pred.metric!r} is not declared")
            fam, corr = pred.analysis.family, pred.analysis.correction
            if families.setdefault(fam, corr) != corr:
                raise SpecError(f"family {fam!r} mixes corrections {families[fam]!r} and {corr!r}")
        for exp in self.exploratory:
            for cname in Comparison.parse(exp.comparison).conditions:
                if cname not in self.conditions:
                    raise SpecError(f"exploratory {exp.id}: unknown condition {cname!r}")

    def prediction(self, prediction_id: str) -> Prediction:
        for pred in self.predictions:
            if pred.id == prediction_id:
                return pred
        raise KeyError(prediction_id)

    def effective_exclusions(self, pred: Prediction) -> tuple[Exclusion, ...]:
        return tuple(self.exclusions) + tuple(pred.exclusions)

    def frozen_payload(self, prediction_id: str) -> dict[str, Any]:
        """Everything the prediction depends on, in the form that gets hashed and registered."""
        pred = self.prediction(prediction_id)
        conds = {}
        for cname in pred.parsed.conditions:
            cond = self.conditions[cname]
            conds[cname] = {"role": cond.role, "config_hash": cond.config_hash(self.root)}
        return {
            "schema": SCHEMA_VERSION,
            "study": self.name,
            "prediction": pred.to_dict(),
            "conditions": conds,
            "metric": {"name": pred.metric, **self.metrics[pred.metric].fingerprint()},
            "exclusions": [e.to_dict() for e in self.effective_exclusions(pred)],
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any], root: str | Path | None = None) -> Study:
        allowed = {
            "study",
            "name",
            "description",
            "conditions",
            "metrics",
            "predictions",
            "exclusions",
            "exploratory",
        }
        _check_keys(d, allowed, "study")
        name = d.get("study") or d.get("name")
        if not name:
            raise SpecError("study name is required ('study: <name>')")
        conditions = {}
        for cname, cdef in (d.get("conditions") or {}).items():
            conditions[cname] = Condition(name=cname, **(cdef or {}))
        metrics = {}
        for mname, mdef in (d.get("metrics") or {}).items():
            metrics[mname] = Metric(name=mname, **(mdef or {}))
        return cls(
            name=str(name),
            description=d.get("description", ""),
            conditions=conditions,
            metrics=metrics,
            predictions=[Prediction.from_dict(p) for p in d.get("predictions") or []],
            exclusions=tuple(Exclusion(**e) for e in d.get("exclusions") or ()),
            exploratory=[ExploratorySpec(**e) for e in d.get("exploratory") or []],
            root=Path(root) if root is not None else Path.cwd(),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> Study:
        path = Path(path)
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if not isinstance(data, Mapping):
            raise SpecError(f"{path}: top level must be a mapping")
        return cls.from_dict(data, root=path.resolve().parent)

    def with_prediction(self, pred: Prediction) -> Study:
        preds = [pred if p.id == pred.id else p for p in self.predictions]
        return replace(self, predictions=preds)


def _check_keys(d: Mapping[str, Any], allowed: Any, where: str) -> None:
    unknown = set(d) - set(allowed)
    if unknown:
        raise SpecError(f"{where}: unknown keys {sorted(unknown)}")


def import_object(path: str) -> Any:
    """Import ``package.module:attribute``."""
    if ":" not in path:
        raise SpecError(f"expected 'module:attribute', got {path!r}")
    module_name, attr = path.split(":", 1)
    obj: Any = importlib.import_module(module_name)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj

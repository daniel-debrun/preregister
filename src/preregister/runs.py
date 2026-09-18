"""Run records, importers, and exclusion evaluation."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from preregister.canonical import sha256_hex
from preregister.ledger import format_timestamp, parse_timestamp
from preregister.registry import Exclusion

RUN_STATUSES = ("finished", "failed", "killed", "crashed", "running")


class RunError(ValueError):
    pass


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    condition: str
    seed: int
    metrics: Mapping[str, float]
    started_at: datetime
    ended_at: datetime | None = None
    status: str = "finished"
    git_sha: str | None = None
    config_hash: str | None = None
    metric_versions: Mapping[str, str] = field(default_factory=dict)
    source: str = "manual"
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.run_id):
            raise RunError("run_id is required")
        if self.status not in RUN_STATUSES:
            raise RunError(f"run {self.run_id}: status must be one of {RUN_STATUSES}")
        object.__setattr__(self, "seed", _to_seed(self.run_id, self.seed))
        object.__setattr__(self, "started_at", _to_timestamp(self.run_id, self.started_at))
        if self.ended_at is not None:
            object.__setattr__(self, "ended_at", _to_timestamp(self.run_id, self.ended_at))
            if self.ended_at < self.started_at:
                raise RunError(f"run {self.run_id}: ended_at precedes started_at")
        metrics = {}
        for key, value in dict(self.metrics).items():
            metrics[str(key)] = _to_float(value)
        object.__setattr__(self, "metrics", metrics)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": str(self.run_id),
            "condition": self.condition,
            "seed": self.seed,
            "metrics": {k: (None if not math.isfinite(v) else v) for k, v in self.metrics.items()},
            "started_at": format_timestamp(self.started_at),
            "ended_at": None if self.ended_at is None else format_timestamp(self.ended_at),
            "status": self.status,
            "git_sha": self.git_sha,
            "config_hash": self.config_hash,
            "metric_versions": dict(self.metric_versions),
            "source": self.source,
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> RunRecord:
        known = set(cls.__dataclass_fields__)
        unknown = set(d) - known
        if unknown:
            raise RunError(f"run {d.get('run_id')}: unknown fields {sorted(unknown)}")
        missing = {"run_id", "condition", "seed", "metrics", "started_at"} - set(d)
        if missing:
            raise RunError(f"run {d.get('run_id')}: missing fields {sorted(missing)}")
        data = dict(d)
        if data["run_id"] is None or str(data["run_id"]).strip() == "":
            raise RunError("run_id is required")
        if not isinstance(data["metrics"] or {}, Mapping):
            raise RunError(f"run {d['run_id']}: metrics must be an object of name -> value")
        data["metrics"] = {k: _to_float(v) for k, v in (data["metrics"] or {}).items()}
        data["run_id"] = str(data["run_id"])
        return cls(**data)

    def content_hash(self) -> str:
        return sha256_hex(self.to_dict())


def _to_seed(run_id: str, value: Any) -> int:
    """Seeds must be integers; ``3`` and ``"3"`` and ``3.0`` are accepted, ``3.9`` is not."""
    try:
        number = float(value) if isinstance(value, str) else value
        seed = int(number)
    except (TypeError, ValueError) as exc:
        raise RunError(f"run {run_id}: seed {value!r} is not an integer") from exc
    if isinstance(value, bool) or seed != number:
        raise RunError(f"run {run_id}: seed {value!r} is not an integer")
    return seed


def _to_timestamp(run_id: str, value: Any) -> datetime:
    """Parse a run time, refusing values without a UTC offset.

    A naive time is ambiguous (local or UTC?), and guessing wrong moves runs across
    the registration time that ``NoPeeking`` checks against.
    """
    if isinstance(value, datetime):
        ts = value
    else:
        if not isinstance(value, str):
            raise RunError(f"run {run_id}: timestamp {value!r} is not an ISO 8601 string")
        text = value.strip()
        try:
            ts = datetime.fromisoformat(text[:-1] + "+00:00" if text.endswith("Z") else text)
        except ValueError as exc:
            raise RunError(f"run {run_id}: timestamp {value!r} is not ISO 8601") from exc
    if ts.tzinfo is None or ts.utcoffset() is None:
        raise RunError(
            f"run {run_id}: timestamp {value!s} has no UTC offset; "
            "write it as e.g. 2026-01-01T12:00:00Z or with +HH:MM"
        )
    return parse_timestamp(ts)


def _to_float(value: Any) -> float:
    if value is None or value == "":
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise RunError(f"metric value {value!r} is not numeric") from exc


def load_runs_json(path: str | Path, source: str = "json") -> list[RunRecord]:
    with open(path, encoding="utf-8-sig") as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise RunError(f"{path}: not valid JSON ({exc})") from exc
    if isinstance(data, Mapping):
        if "runs" not in data:
            raise RunError(f"{path}: expected a list of runs or {{'runs': [...]}}")
        data = data["runs"]
    if not isinstance(data, list) or not all(isinstance(d, Mapping) for d in data):
        raise RunError(f"{path}: expected a list of run objects or {{'runs': [...]}}")
    return [RunRecord.from_dict({"source": source, **d}) for d in data]


_RESERVED_COLUMNS = {
    "run_id",
    "condition",
    "seed",
    "status",
    "started_at",
    "ended_at",
    "git_sha",
    "config_hash",
}


def load_runs_csv(path: str | Path, source: str = "csv") -> list[RunRecord]:
    """Load runs from CSV.

    Reserved columns map to run fields. ``meta.<key>`` columns go to ``meta``,
    ``version.<metric>`` columns to ``metric_versions``; every other column is a metric.
    """
    runs = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        missing = {"run_id", "condition", "seed", "started_at"} - set(reader.fieldnames or [])
        if missing:
            raise RunError(f"{path}: missing required columns {sorted(missing)}")
        for row in reader:
            metrics: dict[str, float] = {}
            meta: dict[str, Any] = {}
            versions: dict[str, str] = {}
            for key, value in row.items():
                if key in _RESERVED_COLUMNS:
                    continue
                if key.startswith("meta."):
                    meta[key[5:]] = value
                elif key.startswith("version."):
                    if value:
                        versions[key[8:]] = value
                else:
                    metrics[key] = _to_float(value)
            runs.append(
                RunRecord(
                    run_id=row["run_id"],
                    condition=row["condition"],
                    seed=row["seed"],
                    metrics=metrics,
                    started_at=row["started_at"],
                    ended_at=row.get("ended_at") or None,
                    status=row.get("status") or "finished",
                    git_sha=row.get("git_sha") or None,
                    config_hash=row.get("config_hash") or None,
                    metric_versions=versions,
                    source=source,
                    meta=meta,
                )
            )
    return runs


def write_runs_json(runs: Iterable[RunRecord], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"runs": [r.to_dict() for r in runs]}, fh, indent=2)
        fh.write("\n")


_OPS = {
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
}


def exclusion_matches(rule: Exclusion, run: RunRecord, metric: str) -> bool:
    fld, op, raw = rule.parts
    base, _, sub = fld.partition(":")
    value: Any
    present = True
    if base == "status":
        value = run.status
    elif base == "seed":
        value = run.seed
    elif base == "condition":
        value = run.condition
    elif base == "metric":
        name = sub or metric
        present = name in run.metrics
        value = run.metrics.get(name, math.nan)
    else:
        present = sub in run.meta
        value = run.meta.get(sub)
    if op == "is_missing":
        return not present
    if op == "is_nan":
        return isinstance(value, float) and math.isnan(value)
    if not present:
        return False
    target: Any = raw
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            target = float(raw)
        except ValueError:
            return op == "!="
        if isinstance(value, float) and math.isnan(value):
            return False
    elif value is not None:
        value = str(value)
    return bool(_OPS[op](value, target))


def first_matching_exclusion(
    rules: Iterable[Exclusion], run: RunRecord, metric: str
) -> Exclusion | None:
    for rule in rules:
        if exclusion_matches(rule, run, metric):
            return rule
    return None

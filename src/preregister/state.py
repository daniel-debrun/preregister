"""A read model of the ledger: registrations, gate records, and linked runs."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from preregister.ledger import LedgerEntry
from preregister.registry import Comparison, Exclusion, Prediction
from preregister.runs import RunRecord


@dataclass(frozen=True)
class RegistrationVersion:
    prediction_id: str
    version: int
    entry: LedgerEntry
    frozen: dict[str, Any]
    prediction_hash: str
    reason: str | None = None
    supersedes: str | None = None

    @property
    def prediction(self) -> Prediction:
        return Prediction.from_dict(self.frozen["prediction"])

    @property
    def comparison(self) -> Comparison:
        return Comparison.parse(self.frozen["prediction"]["comparison"])

    @property
    def exclusions(self) -> tuple[Exclusion, ...]:
        return tuple(Exclusion(**e) for e in self.frozen.get("exclusions", []))

    @property
    def time(self) -> datetime:
        return self.entry.time


@dataclass(frozen=True)
class GateRecord:
    prediction_id: str
    prediction_hash: str
    passed: bool
    results: list[dict[str, Any]]
    entry: LedgerEntry


@dataclass(frozen=True)
class LinkRecord:
    run: RunRecord
    entry: LedgerEntry
    checks: dict[str, list[str]]
    forced: bool


@dataclass
class LedgerState:
    entries: list[LedgerEntry] = field(default_factory=list)
    registrations: dict[str, list[RegistrationVersion]] = field(default_factory=dict)
    gate_records: dict[str, list[GateRecord]] = field(default_factory=dict)
    links: dict[str, LinkRecord] = field(default_factory=dict)

    @classmethod
    def from_entries(cls, entries: Iterable[LedgerEntry]) -> LedgerState:
        state = cls()
        for entry in entries:
            state.entries.append(entry)
            p = entry.payload
            if entry.kind in ("register", "amend"):
                versions = state.registrations.setdefault(p["prediction_id"], [])
                versions.append(
                    RegistrationVersion(
                        prediction_id=p["prediction_id"],
                        version=len(versions) + 1,
                        entry=entry,
                        frozen=p["frozen"],
                        prediction_hash=p["prediction_hash"],
                        reason=p.get("reason"),
                        supersedes=p.get("supersedes"),
                    )
                )
            elif entry.kind == "gate":
                state.gate_records.setdefault(p["prediction_id"], []).append(
                    GateRecord(
                        prediction_id=p["prediction_id"],
                        prediction_hash=p["prediction_hash"],
                        passed=bool(p["passed"]),
                        results=list(p["results"]),
                        entry=entry,
                    )
                )
            elif entry.kind == "link":
                run = RunRecord.from_dict(p["run"])
                state.links[run.run_id] = LinkRecord(
                    run=run, entry=entry, checks=p.get("checks", {}), forced=bool(p.get("forced"))
                )
        return state

    @property
    def prediction_ids(self) -> list[str]:
        return list(self.registrations)

    def current(self, prediction_id: str) -> RegistrationVersion:
        try:
            return self.registrations[prediction_id][-1]
        except KeyError:
            raise KeyError(f"prediction {prediction_id!r} is not registered") from None

    def first(self, prediction_id: str) -> RegistrationVersion:
        return self.registrations[prediction_id][0]

    def latest_gate(self, prediction_id: str) -> GateRecord | None:
        """Most recent gate evaluation of the *current* registered version."""
        current_hash = self.current(prediction_id).prediction_hash
        for record in reversed(self.gate_records.get(prediction_id, [])):
            if record.prediction_hash == current_hash:
                return record
        return None

    def gates_passed(self, prediction_id: str) -> bool:
        record = self.latest_gate(prediction_id)
        return record is not None and record.passed

    def predictions_using(self, condition: str) -> list[str]:
        return [
            pid
            for pid in self.registrations
            if condition in self.current(pid).comparison.conditions
        ]

    def runs_for(self, conditions: Iterable[str]) -> list[LinkRecord]:
        wanted = set(conditions)
        return [rec for rec in self.links.values() if rec.run.condition in wanted]

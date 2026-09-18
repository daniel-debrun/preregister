"""Append-only, hash-chained ledger stored as JSON Lines.

Each entry commits to its payload (``payload_hash``) and to the previous entry
(``prev_hash``); ``hash`` is SHA-256 over the canonical encoding of every field
except itself. Editing, deleting, or reordering any line breaks the chain from
that point on, which :meth:`Ledger.verify` reports.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from preregister.canonical import CanonicalizationError, sha256_hex

GENESIS_HASH = "0" * 64
ENTRY_KINDS = ("init", "register", "amend", "gate", "link", "analysis")

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX
    fcntl = None  # type: ignore[assignment]


class LedgerError(RuntimeError):
    pass


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def format_timestamp(ts: datetime) -> str:
    if ts.tzinfo is None:
        raise LedgerError("timestamps must be timezone-aware")
    return ts.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def parse_timestamp(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    ts = datetime.fromisoformat(text)
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


@dataclass(frozen=True)
class LedgerEntry:
    index: int
    timestamp: str
    kind: str
    payload: dict[str, Any]
    payload_hash: str
    prev_hash: str
    git: dict[str, Any]
    author: str | None
    hash: str

    @property
    def time(self) -> datetime:
        return parse_timestamp(self.timestamp)

    def header(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "kind": self.kind,
            "payload_hash": self.payload_hash,
            "prev_hash": self.prev_hash,
            "git": self.git,
            "author": self.author,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.header(), "payload": self.payload, "hash": self.hash}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LedgerEntry:
        return cls(
            index=d["index"],
            timestamp=d["timestamp"],
            kind=d["kind"],
            payload=d["payload"],
            payload_hash=d["payload_hash"],
            prev_hash=d["prev_hash"],
            git=d.get("git") or {},
            author=d.get("author"),
            hash=d["hash"],
        )


@dataclass(frozen=True)
class Issue:
    index: int | None
    message: str
    severity: str = "error"

    def __str__(self) -> str:
        where = "ledger" if self.index is None else f"entry {self.index}"
        return f"[{self.severity}] {where}: {self.message}"


@dataclass
class VerifyReport:
    n_entries: int = 0
    head_hash: str | None = None
    issues: list[Issue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(i.severity == "error" for i in self.issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "n_entries": self.n_entries,
            "head_hash": self.head_hash,
            "issues": [
                {"index": i.index, "severity": i.severity, "message": i.message}
                for i in self.issues
            ],
        }


class Ledger:
    def __init__(self, path: str | Path, clock: Callable[[], datetime] = utcnow) -> None:
        self.path = Path(path)
        self.clock = clock

    def exists(self) -> bool:
        return self.path.exists()

    def _raw_lines(self) -> list[str]:
        if not self.path.exists():
            return []
        with open(self.path, encoding="utf-8") as fh:
            return [line for line in fh.read().splitlines() if line.strip()]

    def entries(self) -> list[LedgerEntry]:
        out = []
        for n, line in enumerate(self._raw_lines()):
            try:
                out.append(LedgerEntry.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise LedgerError(f"{self.path}: malformed line {n}: {exc}") from exc
        return out

    def head(self) -> LedgerEntry | None:
        entries = self.entries()
        return entries[-1] if entries else None

    def append(
        self,
        kind: str,
        payload: dict[str, Any],
        *,
        author: str | None = None,
        git: dict[str, Any] | None = None,
    ) -> LedgerEntry:
        return self.append_many([(kind, payload)], author=author, git=git)[0]

    def append_many(
        self,
        items: list[tuple[str, dict[str, Any]]],
        *,
        author: str | None = None,
        git: dict[str, Any] | None = None,
    ) -> list[LedgerEntry]:
        """Append entries atomically with respect to other writers, with one fsync."""
        for kind, _ in items:
            if kind not in ENTRY_KINDS:
                raise LedgerError(f"unknown entry kind {kind!r}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        written: list[LedgerEntry] = []
        with open(self.path, "a+", encoding="utf-8") as fh:
            if fcntl is not None:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
                fh.seek(0)
                lines = [ln for ln in fh.read().splitlines() if ln.strip()]
                prev = LedgerEntry.from_dict(json.loads(lines[-1])) if lines else None
                chunks = []
                for kind, payload in items:
                    header = {
                        "index": 0 if prev is None else prev.index + 1,
                        "timestamp": format_timestamp(self.clock()),
                        "kind": kind,
                        "payload_hash": sha256_hex(payload),
                        "prev_hash": GENESIS_HASH if prev is None else prev.hash,
                        "git": git or {},
                        "author": author,
                    }
                    entry = LedgerEntry(payload=payload, hash=sha256_hex(header), **header)
                    chunks.append(
                        json.dumps(entry.to_dict(), sort_keys=True, allow_nan=False) + "\n"
                    )
                    written.append(entry)
                    prev = entry
                fh.seek(0, os.SEEK_END)
                fh.write("".join(chunks))
                fh.flush()
                os.fsync(fh.fileno())
            finally:
                if fcntl is not None:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        return written

    def verify(self, anchors: list[tuple[int, str]] | None = None) -> VerifyReport:
        """Check chain integrity, payload hashes, and registration hashes.

        ``anchors`` is a list of ``(index, hash)`` pairs published outside the ledger;
        each must still be present, which detects rewriting of the anchored prefix.
        """
        report = VerifyReport()
        prev_hash = GENESIS_HASH
        prev_time: datetime | None = None
        by_hash: dict[str, LedgerEntry] = {}
        latest_version: dict[str, str] = {}
        entries: list[LedgerEntry] = []
        for n, line in enumerate(self._raw_lines()):
            try:
                entry = LedgerEntry.from_dict(json.loads(line))
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                report.issues.append(Issue(n, f"malformed entry: {exc}"))
                prev_hash = "<broken>"
                continue
            entries.append(entry)
            if entry.index != n:
                report.issues.append(Issue(n, f"index field is {entry.index}, expected {n}"))
            if entry.prev_hash != prev_hash:
                report.issues.append(
                    Issue(n, "prev_hash does not match previous entry (chain broken)")
                )
            try:
                if sha256_hex(entry.payload) != entry.payload_hash:
                    report.issues.append(Issue(n, "payload was modified (payload_hash mismatch)"))
                if sha256_hex(entry.header()) != entry.hash:
                    report.issues.append(Issue(n, "entry header was modified (hash mismatch)"))
            except CanonicalizationError as exc:
                report.issues.append(Issue(n, f"payload not canonicalizable: {exc}"))
            try:
                ts = entry.time
                if prev_time is not None and ts < prev_time:
                    report.issues.append(
                        Issue(n, "timestamp earlier than previous entry", severity="warning")
                    )
                prev_time = ts
            except ValueError:
                report.issues.append(Issue(n, f"unparseable timestamp {entry.timestamp!r}"))
            if entry.kind not in ENTRY_KINDS:
                report.issues.append(Issue(n, f"unknown entry kind {entry.kind!r}"))
            self._verify_registration(entry, by_hash, latest_version, report)
            by_hash[entry.hash] = entry
            prev_hash = entry.hash
        report.n_entries = len(entries)
        report.head_hash = entries[-1].hash if entries else None
        for index, anchored in anchors or []:
            if index >= len(entries) or entries[index].hash != anchored:
                report.issues.append(
                    Issue(
                        None, f"anchored hash {anchored[:12]} at entry {index} is not in the ledger"
                    )
                )
        return report

    @staticmethod
    def _verify_registration(
        entry: LedgerEntry,
        by_hash: dict[str, LedgerEntry],
        latest_version: dict[str, str],
        report: VerifyReport,
    ) -> None:
        if entry.kind not in ("register", "amend"):
            return
        payload = entry.payload
        pid = payload.get("prediction_id")
        frozen = payload.get("frozen")
        if not isinstance(frozen, dict) or pid is None:
            report.issues.append(Issue(entry.index, "registration missing prediction_id/frozen"))
            return
        try:
            if sha256_hex(frozen) != payload.get("prediction_hash"):
                report.issues.append(Issue(entry.index, f"{pid}: frozen prediction hash mismatch"))
        except CanonicalizationError as exc:
            report.issues.append(Issue(entry.index, f"{pid}: {exc}"))
        if entry.kind == "register":
            if pid in latest_version:
                report.issues.append(
                    Issue(entry.index, f"{pid}: registered twice without amendment")
                )
        else:
            target = payload.get("supersedes")
            if pid not in latest_version:
                report.issues.append(
                    Issue(entry.index, f"{pid}: amendment of unregistered prediction")
                )
            elif target != latest_version[pid] or target not in by_hash:
                report.issues.append(
                    Issue(entry.index, f"{pid}: amendment does not supersede the latest version")
                )
            if not str(payload.get("reason", "")).strip():
                report.issues.append(Issue(entry.index, f"{pid}: amendment without a reason"))
        latest_version[pid] = entry.hash

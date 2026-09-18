"""The protocol as a workflow: register, gate, link, analyze."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from preregister.canonical import sha256_hex
from preregister.gates import GateContext, GateResult, evaluate_gates, run_rejections
from preregister.gitinfo import git_commit_paths, git_state
from preregister.ledger import Issue, Ledger, LedgerEntry, LedgerError, VerifyReport, utcnow
from preregister.reconcile import StudyAnalysis, analyze_state
from preregister.registry import ExploratorySpec, Study
from preregister.runs import RunRecord
from preregister.state import LedgerState

PREREG_DIR = ".prereg"
LEDGER_FILE = "ledger.jsonl"
ANCHORS_FILE = "anchors.txt"


class RegistrationError(RuntimeError):
    pass


class LinkError(RuntimeError):
    pass


@dataclass
class RegistrationChange:
    prediction_id: str
    status: str
    entry: LedgerEntry | None = None


@dataclass
class LinkSummary:
    linked: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    refused: list[tuple[str, str]] = field(default_factory=list)
    rejected_for: dict[str, dict[str, list[str]]] = field(default_factory=dict)


class Project:
    """A directory holding ``.prereg/ledger.jsonl`` for one or more studies."""

    def __init__(self, root: str | Path, clock: Callable[[], datetime] = utcnow) -> None:
        self.root = Path(root).resolve()
        self.dir = self.root / PREREG_DIR
        self.ledger = Ledger(self.dir / LEDGER_FILE, clock=clock)
        self.anchors_path = self.dir / ANCHORS_FILE

    @classmethod
    def init(
        cls, root: str | Path, clock: Callable[[], datetime] = utcnow, author: str | None = None
    ) -> Project:
        project = cls(root, clock=clock)
        if project.ledger.exists() and project.ledger.entries():
            return project
        project.dir.mkdir(parents=True, exist_ok=True)
        project._append("init", {"prereg_version": _version()}, author)
        return project

    @classmethod
    def open(cls, root: str | Path, clock: Callable[[], datetime] = utcnow) -> Project:
        project = cls(root, clock=clock)
        if not project.ledger.exists():
            raise LedgerError(f"no ledger at {project.ledger.path}; run 'preregister init'")
        return project

    def _append(
        self,
        kind: str,
        payload: dict[str, Any],
        author: str | None,
        git: dict[str, Any] | None = None,
    ) -> LedgerEntry:
        git = git if git is not None else git_state(self.root).to_dict()
        return self.ledger.append(kind, payload, author=author, git=git)

    def state(self) -> LedgerState:
        return LedgerState.from_entries(self.ledger.entries())

    def diff(self, study: Study) -> list[RegistrationChange]:
        """Compare the study specification with what is registered."""
        state = self.state()
        changes = []
        for pred in study.predictions:
            if pred.id not in state.registrations:
                changes.append(RegistrationChange(pred.id, "new"))
                continue
            current = state.current(pred.id)
            same = sha256_hex(study.frozen_payload(pred.id)) == current.prediction_hash
            changes.append(
                RegistrationChange(pred.id, "unchanged" if same else "changed", current.entry)
            )
        return changes

    def register(self, study: Study, author: str | None = None) -> list[RegistrationChange]:
        """Register every new prediction. Changed predictions must be amended explicitly."""
        changes = self.diff(study)
        changed = [c.prediction_id for c in changes if c.status == "changed"]
        if changed:
            raise RegistrationError(
                f"predictions {changed} differ from their registered version; "
                "use amend() with a reason"
            )
        out = []
        for change in changes:
            if change.status != "new":
                out.append(change)
                continue
            frozen = study.frozen_payload(change.prediction_id)
            entry = self._append(
                "register",
                {
                    "prediction_id": change.prediction_id,
                    "prediction_hash": sha256_hex(frozen),
                    "frozen": frozen,
                },
                author,
            )
            out.append(RegistrationChange(change.prediction_id, "registered", entry))
        return out

    def amend(
        self, study: Study, prediction_id: str, reason: str, author: str | None = None
    ) -> LedgerEntry:
        if not reason.strip():
            raise RegistrationError("an amendment must state a reason")
        state = self.state()
        if prediction_id not in state.registrations:
            raise RegistrationError(f"{prediction_id} is not registered; use register()")
        current = state.current(prediction_id)
        frozen = study.frozen_payload(prediction_id)
        new_hash = sha256_hex(frozen)
        if new_hash == current.prediction_hash:
            raise RegistrationError(f"{prediction_id} is unchanged; nothing to amend")
        return self._append(
            "amend",
            {
                "prediction_id": prediction_id,
                "prediction_hash": new_hash,
                "frozen": frozen,
                "supersedes": current.entry.hash,
                "reason": reason,
            },
            author,
        )

    def run_gates(
        self, study: Study, prediction_ids: Iterable[str] | None = None, author: str | None = None
    ) -> dict[str, list[GateResult]]:
        """Evaluate gates for the registered version of each prediction and record them.

        Gates see the registered prediction and the current study files, so edits to
        configs or metric code after registration make ``ConfigFrozen`` or
        ``MetricDefined`` fail until the prediction is amended.
        """
        state = self.state()
        ids = (
            list(prediction_ids)
            if prediction_ids is not None
            else [p.id for p in study.predictions]
        )
        missing = [pid for pid in ids if pid not in state.registrations]
        if missing:
            raise RegistrationError(f"register before gating: {missing}")
        out = {}
        for pid in ids:
            current = state.current(pid)
            ctx = GateContext(
                study=study, prediction=current.prediction, registration=current, state=state
            )
            results = evaluate_gates(ctx)
            self._append(
                "gate",
                {
                    "prediction_id": pid,
                    "prediction_hash": current.prediction_hash,
                    "passed": all(r.passed for r in results),
                    "results": [r.to_dict() for r in results],
                },
                author,
            )
            out[pid] = results
        return out

    def link_runs(
        self, runs: Iterable[RunRecord], *, force: bool = False, author: str | None = None
    ) -> LinkSummary:
        """Append runs to the ledger.

        A run is refused (not recorded) if it is still running, belongs to a condition no
        registered prediction uses, conflicts with an already linked run of the same id,
        or, unless ``force``, no prediction using its condition has passed its gates.
        Everything else is recorded, including runs that per-run gates reject; those are
        reported, not hidden.
        """
        summary = LinkSummary()
        state = self.state()
        pending: list[RunRecord] = []
        items: list[tuple[str, dict[str, Any]]] = []
        seen: dict[str, str] = {}
        for run in runs:
            existing = state.links.get(run.run_id)
            if existing is not None or run.run_id in seen:
                prior = existing.run.content_hash() if existing is not None else seen[run.run_id]
                if prior == run.content_hash():
                    summary.unchanged.append(run.run_id)
                else:
                    summary.refused.append((run.run_id, "already linked with different content"))
                continue
            if run.status == "running":
                summary.refused.append((run.run_id, "run is still running"))
                continue
            pids = state.predictions_using(run.condition)
            if not pids:
                summary.refused.append(
                    (run.run_id, f"no registered prediction uses condition {run.condition!r}")
                )
                continue
            if not force and not any(state.gates_passed(pid) for pid in pids):
                summary.refused.append(
                    (run.run_id, f"no prediction using {run.condition!r} has passed its gates")
                )
                continue
            checks = {
                pid: run_rejections(state.current(pid), state.first(pid), run) for pid in pids
            }
            items.append(
                (
                    "link",
                    {
                        "run": run.to_dict(),
                        "run_hash": run.content_hash(),
                        "checks": checks,
                        "forced": force,
                    },
                )
            )
            pending.append(run)
            seen[run.run_id] = run.content_hash()
            summary.linked.append(run.run_id)
            rejected = {pid: reasons for pid, reasons in checks.items() if reasons}
            if rejected:
                summary.rejected_for[run.run_id] = rejected
        if items:
            self.ledger.append_many(items, author=author, git=git_state(self.root).to_dict())
        return summary

    def analyze(
        self,
        *,
        overrides: Mapping[str, Mapping[str, Any]] | None = None,
        exploratory: Iterable[ExploratorySpec] = (),
        record: bool = True,
        author: str | None = None,
    ) -> StudyAnalysis:
        verify = self.verify()
        if record and not verify.ok:
            raise LedgerError(
                "ledger fails verification; run 'preregister verify'. "
                "Refusing to record an analysis on top of it"
            )
        state = self.state()
        analysis = analyze_state(
            state, overrides=overrides, exploratory=exploratory, integrity=verify.to_dict()
        )
        if record:
            self._append(
                "analysis",
                {
                    "verdicts": analysis.verdicts(),
                    "overrides": analysis.overrides,
                    "ledger_head": analysis.ledger_head,
                    "result_hash": sha256_hex(
                        {
                            k: v
                            for k, v in analysis.to_dict().items()
                            if k not in ("generated_at", "integrity")
                        }
                    ),
                },
                author,
            )
        return analysis

    def read_anchors(self) -> list[tuple[int, str]]:
        if not self.anchors_path.exists():
            return []
        anchors = []
        for line in self.anchors_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if not parts or line.startswith("#"):
                continue
            if len(parts) < 3 or not parts[1].isdigit():
                raise LedgerError(f"{self.anchors_path}: malformed anchor line {line!r}")
            anchors.append((int(parts[1]), parts[2]))
        return anchors

    def verify(self, study: Study | None = None) -> VerifyReport:
        report = self.ledger.verify(anchors=self.read_anchors())
        if study is not None and report.ok:
            state = self.state()
            for pred in study.predictions:
                if pred.id not in state.registrations:
                    report.issues.append(
                        Issue(None, f"{pred.id}: in spec but not registered", "warning")
                    )
                    continue
                try:
                    current_hash = sha256_hex(study.frozen_payload(pred.id))
                except Exception as exc:
                    report.issues.append(Issue(None, f"{pred.id}: cannot freeze spec: {exc}"))
                    continue
                if current_hash != state.current(pred.id).prediction_hash:
                    report.issues.append(
                        Issue(
                            None,
                            f"{pred.id}: spec differs from registered version (unregistered edit)",
                        )
                    )
        return report

    def anchor(self, commit: bool = False) -> tuple[int, str]:
        """Append the ledger head to ``.prereg/anchors.txt`` (and optionally git-commit it).

        Publish the printed hash somewhere you do not control (a commit on a shared
        remote, an email, a post). Anyone holding it can later confirm that the ledger
        prefix up to that entry was not rewritten.
        """
        head = self.ledger.head()
        if head is None:
            raise LedgerError("ledger is empty")
        with open(self.anchors_path, "a", encoding="utf-8") as fh:
            fh.write(f"{head.timestamp} {head.index} {head.hash}\n")
        if commit:
            ok = git_commit_paths(
                self.root,
                [self.ledger.path, self.anchors_path],
                f"preregister anchor: entry {head.index} {head.hash[:16]}",
            )
            if not ok:
                raise LedgerError("git commit of the anchor failed (is this a git repository?)")
        return head.index, head.hash


def _version() -> str:
    from preregister import __version__

    return __version__

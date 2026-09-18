from __future__ import annotations

import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from preregister import Project
from preregister.canonical import sha256_hex
from preregister.ledger import GENESIS_HASH, Ledger, LedgerError, format_timestamp, parse_timestamp
from preregister.project import RegistrationError

from .conftest import FakeClock, make_study


def _ledger(tmp_path: Path, n: int = 4) -> Ledger:
    ledger = Ledger(tmp_path / "ledger.jsonl", clock=FakeClock())
    for i in range(n):
        ledger.append("link", {"i": i, "value": i * 1.5}, author="tester")
    return ledger


def _rewrite(ledger: Ledger, fn) -> None:
    lines = ledger.path.read_text().splitlines()
    ledger.path.write_text("\n".join(fn(lines)) + "\n")


def test_append_builds_chain(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    entries = ledger.entries()
    assert [e.index for e in entries] == [0, 1, 2, 3]
    assert entries[0].prev_hash == GENESIS_HASH
    for prev, cur in itertools.pairwise(entries):
        assert cur.prev_hash == prev.hash
        assert cur.time > prev.time
    assert entries[2].payload_hash == sha256_hex({"i": 2, "value": 3.0})
    report = ledger.verify()
    assert report.ok and report.n_entries == 4 and report.head_hash == entries[-1].hash


def test_detects_payload_edit(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)

    def edit(lines):
        d = json.loads(lines[1])
        d["payload"]["value"] = 99.0
        lines[1] = json.dumps(d)
        return lines

    _rewrite(ledger, edit)
    report = ledger.verify()
    assert not report.ok
    assert [i.index for i in report.issues] == [1]
    assert "payload" in report.issues[0].message


def test_detects_payload_edit_with_recomputed_payload_hash(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)

    def edit(lines):
        d = json.loads(lines[1])
        d["payload"]["value"] = 99.0
        d["payload_hash"] = sha256_hex(d["payload"])
        lines[1] = json.dumps(d)
        return lines

    _rewrite(ledger, edit)
    report = ledger.verify()
    assert not report.ok
    assert any(i.index == 1 and "header" in i.message for i in report.issues)


def test_detects_timestamp_edit_deletion_and_reordering(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)

    def backdate(lines):
        d = json.loads(lines[2])
        d["timestamp"] = "2020-01-01T00:00:00.000000Z"
        lines[2] = json.dumps(d)
        return lines

    _rewrite(ledger, backdate)
    assert not ledger.verify().ok

    ledger2 = _ledger(tmp_path / "b")
    _rewrite(ledger2, lambda lines: lines[:1] + lines[2:])
    messages = " ".join(i.message for i in ledger2.verify().issues)
    assert "chain broken" in messages

    ledger3 = _ledger(tmp_path / "c")
    _rewrite(ledger3, lambda lines: [lines[0], lines[2], lines[1], lines[3]])
    assert not ledger3.verify().ok


def test_malformed_line_reported(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    _rewrite(ledger, lambda lines: [*lines[:2], "{not json", *lines[2:]])
    report = ledger.verify()
    assert not report.ok
    with pytest.raises(LedgerError):
        ledger.entries()


def test_full_rewrite_caught_only_by_anchor(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    anchor = (3, ledger.head().hash)
    entries = [json.loads(line) for line in ledger.path.read_text().splitlines()]
    prev = GENESIS_HASH
    out = []
    for d in entries:
        if d["index"] == 1:
            d["payload"]["value"] = -1.0
        d["payload_hash"] = sha256_hex(d["payload"])
        d["prev_hash"] = prev
        d["hash"] = sha256_hex(
            {
                k: d[k]
                for k in (
                    "index",
                    "timestamp",
                    "kind",
                    "payload_hash",
                    "prev_hash",
                    "git",
                    "author",
                )
            }
        )
        prev = d["hash"]
        out.append(json.dumps(d))
    ledger.path.write_text("\n".join(out) + "\n")
    assert ledger.verify().ok
    report = ledger.verify(anchors=[anchor])
    assert not report.ok and "anchored" in report.issues[0].message


def test_truncation_caught_by_anchor(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    anchor = (3, ledger.head().hash)
    _rewrite(ledger, lambda lines: lines[:2])
    assert ledger.verify().ok
    assert not ledger.verify(anchors=[anchor]).ok


def test_non_monotonic_clock_is_a_warning(tmp_path: Path) -> None:
    times = iter(
        [datetime(2026, 1, 2, tzinfo=timezone.utc), datetime(2026, 1, 1, tzinfo=timezone.utc)]
    )
    ledger = Ledger(tmp_path / "l.jsonl", clock=lambda: next(times))
    ledger.append("init", {})
    ledger.append("link", {})
    report = ledger.verify()
    assert report.ok
    assert [i.severity for i in report.issues] == ["warning"]


def test_unknown_kind_and_naive_timestamps_rejected(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "l.jsonl")
    with pytest.raises(LedgerError):
        ledger.append("delete", {})
    with pytest.raises(LedgerError):
        format_timestamp(datetime(2026, 1, 1))
    assert parse_timestamp("2026-01-01T00:00:00Z") == datetime(2026, 1, 1, tzinfo=timezone.utc)


def test_registration_and_amendment_chain(tmp_path: Path, clock: FakeClock) -> None:
    study = make_study(tmp_path)
    project = Project.init(tmp_path, clock=clock)
    changes = project.register(study)
    assert [c.status for c in changes] == ["registered"]
    assert [c.status for c in project.register(study)] == ["unchanged"]

    pred = study.predictions[0]
    changed = study.with_prediction(pred.__class__.from_dict({**pred.to_dict(), "min_effect": 2.0}))
    with pytest.raises(RegistrationError, match="amend"):
        project.register(changed)
    with pytest.raises(RegistrationError, match="reason"):
        project.amend(changed, "P1", reason="  ")
    with pytest.raises(RegistrationError, match="unchanged"):
        project.amend(study, "P1", reason="no-op")
    entry = project.amend(changed, "P1", reason="pilot showed larger variance")
    state = project.state()
    versions = state.registrations["P1"]
    assert [v.version for v in versions] == [1, 2]
    assert entry.payload["supersedes"] == versions[0].entry.hash
    assert state.current("P1").prediction.min_effect == 2.0
    assert project.verify(changed).ok
    report = project.verify(study)
    assert not report.ok and "differs from registered" in report.issues[0].message


def test_verify_flags_forged_amendment(tmp_path: Path, clock: FakeClock) -> None:
    study = make_study(tmp_path)
    project = Project.init(tmp_path, clock=clock)
    project.register(study)
    frozen = study.frozen_payload("P1")
    frozen["prediction"]["min_effect"] = 3.0
    project.ledger.append(
        "amend",
        {
            "prediction_id": "P1",
            "prediction_hash": sha256_hex(frozen),
            "frozen": frozen,
            "supersedes": "f" * 64,
            "reason": "x",
        },
    )
    project.ledger.append(
        "register",
        {"prediction_id": "P1", "prediction_hash": "0" * 64, "frozen": frozen},
    )
    messages = [i.message for i in project.ledger.verify().issues]
    assert any("does not supersede" in m for m in messages)
    assert any("hash mismatch" in m for m in messages)
    assert any("registered twice" in m for m in messages)


def test_anchor_file(tmp_path: Path, clock: FakeClock) -> None:
    project = Project.init(tmp_path, clock=clock)
    index, head = project.anchor()
    assert project.read_anchors() == [(index, head)]
    assert project.verify().ok

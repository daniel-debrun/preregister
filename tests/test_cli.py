from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
import yaml

from preregister.canonical import sha256_hex
from preregister.cli import main


def cli(root: Path, *args: str) -> int:
    return main(["--root", str(root), *args])


def _runs(spec: dict, shift: float = 8.0) -> list[dict]:
    started = datetime.now(timezone.utc) + timedelta(seconds=1)
    rng = np.random.default_rng(0)
    runs = []
    for cond, mean in (("baseline", 100.0), ("candidate", 100.0 + shift)):
        cfg = spec["conditions"][cond]["config"]
        for seed in range(10):
            runs.append(
                {
                    "run_id": f"{cond}-{seed}",
                    "condition": cond,
                    "seed": seed,
                    "metrics": {"eval_return": float(rng.normal(mean, 4.0))},
                    "started_at": started.isoformat(),
                    "ended_at": (started + timedelta(minutes=3)).isoformat(),
                    "config_hash": sha256_hex(cfg),
                }
            )
    return runs


def test_cli_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "study"
    root.mkdir()
    assert cli(root, "status") == 2
    assert cli(root, "init") == 0
    spec_path = root / "prereg.yaml"
    spec = yaml.safe_load(spec_path.read_text())
    assert cli(root, "register") == 0
    assert "P1: registered" in capsys.readouterr().out
    assert cli(root, "register") == 0
    assert "P1: unchanged" in capsys.readouterr().out
    assert cli(root, "gate") == 0
    assert "PowerAnalysis" in capsys.readouterr().out

    runs = _runs(spec)
    (root / "runs.json").write_text(json.dumps(runs[:15]))
    header = "run_id,condition,seed,started_at,ended_at,config_hash,eval_return\n"
    rows = "".join(
        f"{r['run_id']},{r['condition']},{r['seed']},{r['started_at']},{r['ended_at']},{r['config_hash']},{r['metrics']['eval_return']}\n"
        for r in runs[15:]
    )
    (root / "runs.csv").write_text(header + rows)
    assert cli(root, "link", "--json", str(root / "runs.json")) == 0
    assert cli(root, "link", "--csv", str(root / "runs.csv")) == 0
    assert cli(root, "status") == 0
    out = capsys.readouterr().out
    assert "linked 15" in out and "linked 5" in out
    assert "P1 v1" in out and "gates passed" in out and "baseline 10/10" in out

    assert cli(root, "analyze", "--json", str(root / "analysis.json")) == 0
    out = capsys.readouterr().out
    assert "P1: CONFIRMED" in out
    saved = json.loads((root / "analysis.json").read_text())
    assert saved["predictions"][0]["verdict"] == "CONFIRMED"

    assert cli(root, "analyze", "--override", "P1:aggregation=median") == 0
    assert "DEVIATED" in capsys.readouterr().out

    assert cli(root, "report") == 0
    assert cli(root, "report", "--format", "html", "--output", str(root / "r.html")) == 0
    assert (root / "REPORT.md").read_text().startswith("# Reconciliation report: my-study")
    assert "<svg" in (root / "r.html").read_text()

    assert cli(root, "verify") == 0
    assert cli(root, "anchor") == 0
    out = capsys.readouterr().out
    assert "anchored entry" in out
    assert len((root / ".prereg" / "anchors.txt").read_text().splitlines()) == 1

    spec["conditions"]["candidate"]["config"]["lr"] = 0.01
    spec_path.write_text(yaml.safe_dump(spec, sort_keys=False))
    assert cli(root, "verify") == 1
    assert "unregistered edit" in capsys.readouterr().out
    assert cli(root, "register") == 2
    assert cli(root, "amend", "P1", "--reason", "fixed learning rate typo") == 0
    assert cli(root, "verify") == 0
    capsys.readouterr()
    assert cli(root, "analyze") == 0
    out = capsys.readouterr().out
    assert "INADMISSIBLE" in out

    ledger = root / ".prereg" / "ledger.jsonl"
    lines = ledger.read_text().splitlines()
    gate_line = next(i for i, ln in enumerate(lines) if '"kind": "gate"' in ln)
    lines[gate_line] = lines[gate_line].replace('"passed": true', '"passed": false', 1)
    ledger.write_text("\n".join(lines) + "\n")
    assert cli(root, "verify", "--ledger-only") == 1
    assert "FAILED" in capsys.readouterr().out
    assert cli(root, "anchor") == 1


def test_cli_plugin_gate_and_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path
    (root / "mygates.py").write_text(
        "from preregister import custom_gate\n\n"
        "@custom_gate('LayoutOk')\n"
        "def layout_ok(ctx, layout):\n"
        "    return layout != 'broken', f'layout {layout}'\n"
    )
    assert cli(root, "init") == 0
    spec = yaml.safe_load((root / "prereg.yaml").read_text())
    spec["predictions"][0]["gates"] = [{"type": "LayoutOk", "layout": "broken"}]
    (root / "prereg.yaml").write_text(yaml.safe_dump(spec))
    assert cli(root, "--plugin", "mygates", "register") == 0
    assert cli(root, "--plugin", "mygates", "gate") == 1
    assert "[FAIL] LayoutOk: layout broken" in capsys.readouterr().out
    assert cli(root, "analyze", "--override", "bogus") == 2
    assert cli(root, "link", "--wandb", "e/p") == 2


def test_module_entry_point(tmp_path: Path) -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "preregister.cli", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0 and "anchor" in proc.stdout

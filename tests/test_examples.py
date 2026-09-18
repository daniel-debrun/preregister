from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _load(path: Path) -> ModuleType:
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location(
            f"example_{path.parent.name}_{path.stem}", path
        )
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(path.parent))


def test_kitchen_gate_example(capsys) -> None:
    module = _load(EXAMPLES / "env_admissibility" / "check_layouts.py")
    assert module.main() == 1
    out = capsys.readouterr().out
    assert "OPEN: PASS" in out and "SPLIT: FAIL" in out
    assert "tomato_soup on split_kitchen needs unreachable ['tomato']" in out


def test_integrity_demo_catches_everything() -> None:
    text = _load(EXAMPLES / "integrity_demo" / "demo.py").run()
    assert "verdict: INCONCLUSIVE" in text
    assert "3 run(s) rejected by NoPeeking" in text
    assert "DEVIATED (statistically CONFIRMED)" in text and "runs with undeclared seeds" in text
    assert "payload was modified" in text
    assert (
        "verify without anchors ok=True" in text
        and "verify against published anchor ok=False" in text
    )
    committed = (EXAMPLES / "integrity_demo" / "OUTPUT.md").read_text()
    assert committed == text


def test_bandit_algorithms_run() -> None:
    bandit = _load(EXAMPLES / "bandit_study" / "bandit.py")
    for config in (
        {"algo": "eps_greedy", "epsilon": 0.1, "horizon": 200},
        {"algo": "ucb1", "c": 2.0, "horizon": 200},
        {"algo": "thompson", "prior": [1.0, 1.0], "horizon": 200},
    ):
        out = bandit.run_bandit(config, seed=0)
        assert 0 <= out["cumulative_regret"] <= 200 * 0.2
        assert 0 <= out["optimal_arm_rate"] <= 1
    assert bandit.run_bandit(config, 3) == bandit.run_bandit(config, 3)

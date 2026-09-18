"""Pilot on seeds 1000-1079, disjoint from the registered seeds 0-19.

Run before registration to pick plausible standard deviations for the power gates
and to sanity-check that the environment behaves. Its output is summarized in
prereg.yaml; pilot data is never linked to the study.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from bandit import run_bandit

PILOT_SEEDS = range(1000, 1080)


def main() -> None:
    spec = yaml.safe_load((Path(__file__).parent / "prereg.yaml").read_text())
    results = {
        name: [run_bandit(cond["config"], seed) for seed in PILOT_SEEDS]
        for name, cond in spec["conditions"].items()
    }
    for name, rows in results.items():
        for metric in ("cumulative_regret", "optimal_arm_rate"):
            vals = np.array([r[metric] for r in rows])
            print(f"{name:12s} {metric:18s} mean {vals.mean():8.3f} sd {vals.std(ddof=1):7.3f}")


if __name__ == "__main__":
    main()

"""Bernoulli multi-armed bandit and three classic algorithms (numpy only)."""

from __future__ import annotations

from typing import Any

import numpy as np

ARM_MEANS = (0.50, 0.45, 0.45, 0.40, 0.40, 0.40, 0.35, 0.35, 0.30, 0.30)


def run_bandit(config: dict[str, Any], seed: int) -> dict[str, float]:
    """Run one algorithm for ``config['horizon']`` steps; return summary metrics.

    ``cumulative_regret`` is pseudo-regret: the sum over steps of the gap between the
    best arm's mean and the chosen arm's mean.
    """
    means = np.asarray(config.get("arm_means", ARM_MEANS), dtype=float)
    horizon = int(config["horizon"])
    algo = config["algo"]
    rng = np.random.default_rng(seed)
    k = means.size
    pulls = np.zeros(k)
    successes = np.zeros(k)
    gaps = means.max() - means
    regret = 0.0
    optimal_late = 0
    late_start = horizon - horizon // 4
    for t in range(horizon):
        if algo == "eps_greedy":
            if t < k:
                arm = t
            elif rng.random() < config["epsilon"]:
                arm = int(rng.integers(k))
            else:
                arm = int(np.argmax(successes / np.maximum(pulls, 1)))
        elif algo == "ucb1":
            if t < k:
                arm = t
            else:
                bonus = np.sqrt(config.get("c", 2.0) * np.log(t + 1) / pulls)
                arm = int(np.argmax(successes / pulls + bonus))
        elif algo == "thompson":
            a0, b0 = config.get("prior", (1.0, 1.0))
            arm = int(np.argmax(rng.beta(successes + a0, pulls - successes + b0)))
        else:
            raise ValueError(f"unknown algorithm {algo!r}")
        reward = float(rng.random() < means[arm])
        pulls[arm] += 1
        successes[arm] += reward
        regret += gaps[arm]
        if t >= late_start and gaps[arm] == 0:
            optimal_late += 1
    return {
        "cumulative_regret": regret,
        "optimal_arm_rate": optimal_late / (horizon - late_start),
    }

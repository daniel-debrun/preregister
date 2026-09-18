from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from preregister import AnalysisPlan, Condition, Metric, Prediction, Project, RunRecord, Study

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


class FakeClock:
    def __init__(self, start: datetime = T0, step: timedelta = timedelta(seconds=1)) -> None:
        self.now = start
        self.step = step

    def __call__(self) -> datetime:
        self.now += self.step
        return self.now

    def advance(self, **kwargs: float) -> datetime:
        self.now += timedelta(**kwargs)
        return self.now


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


def make_study(root: Path, predictions: Iterable[Prediction] | None = None, **kwargs) -> Study:
    preds = (
        list(predictions)
        if predictions is not None
        else [
            Prediction(
                id="P1",
                hypothesis="treatment beats control",
                metric="score",
                comparison="treatment - control > 0",
                min_effect=1.0,
                analysis=AnalysisPlan(test="welch", n_seeds=10, n_boot=2000, n_permutations=2000),
                gates=({"type": "MinSeeds", "n": 5},),
            )
        ]
    )
    return Study(
        name="test-study",
        root=root,
        conditions=kwargs.pop(
            "conditions",
            {
                "control": Condition("control", role="control", config={"lr": 0.1}),
                "treatment": Condition("treatment", role="treatment", config={"lr": 0.2}),
            },
        ),
        metrics=kwargs.pop("metrics", {"score": Metric("score", version="1")}),
        predictions=preds,
        **kwargs,
    )


def make_runs(
    values: dict[str, Iterable[float]],
    started: datetime,
    seeds: dict[str, Iterable[int]] | None = None,
    metric: str = "score",
    **extra,
) -> list[RunRecord]:
    runs = []
    for cond, vals in values.items():
        cond_seeds = list(seeds[cond]) if seeds and cond in seeds else None
        for i, v in enumerate(vals):
            seed = cond_seeds[i] if cond_seeds else i
            runs.append(
                RunRecord(
                    run_id=f"{cond}-{seed}",
                    condition=cond,
                    seed=seed,
                    metrics={metric: v},
                    started_at=started,
                    ended_at=started + timedelta(minutes=1),
                    **extra,
                )
            )
    return runs


def normal_values(mean: float, sd: float, n: int, seed: int) -> list[float]:
    return list(np.random.default_rng(seed).normal(mean, sd, n))


@pytest.fixture
def project_factory(tmp_path: Path, clock: FakeClock) -> Callable[..., tuple[Project, Study]]:
    def factory(study: Study | None = None, gate: bool = True) -> tuple[Project, Study]:
        study = study or make_study(tmp_path)
        project = Project.init(tmp_path, clock=clock)
        project.register(study)
        if gate:
            project.run_gates(study)
        return project, study

    return factory

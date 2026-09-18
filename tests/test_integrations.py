from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pytest

from preregister import Project
from preregister.canonical import sha256_hex
from preregister.integrations import lookup, subset_config_hash
from preregister.integrations import mlflow as mlflow_adapter
from preregister.integrations import wandb as wandb_adapter

from .conftest import FakeClock, make_study


class FakeSummary:
    def __init__(self, data: dict[str, Any]) -> None:
        self._json_dict = data


@dataclass
class FakeWandbRun:
    id: str
    config: dict[str, Any]
    summary: FakeSummary
    state: str = "finished"
    created_at: str = "2026-03-01T10:00:00"
    name: str = "run"
    url: str = "https://wandb.ai/x"
    metadata: dict[str, Any] = field(default_factory=lambda: {"git": {"commit": "abc123"}})


class FakeWandbApi:
    def __init__(self, runs: list[FakeWandbRun]) -> None:
        self._runs = runs
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def runs(self, path: str, filters: dict[str, Any] | None = None) -> list[FakeWandbRun]:
        self.calls.append((path, filters or {}))
        return self._runs


def test_lookup_and_subset_hash() -> None:
    assert lookup({"a": {"b": 1}}, "a.b") == 1
    assert lookup({"a.b": 2, "a": {"b": 1}}, "a.b") == 2
    assert lookup({"a": 1}, "a.b") is None
    assert subset_config_hash({"lr": 0.1, "other": 3}, ["lr"]) == sha256_hex({"lr": 0.1})
    assert subset_config_hash({"lr": 0.1}, None) is None


def test_wandb_fetch_maps_runs() -> None:
    runs = [
        FakeWandbRun(
            "a1",
            {"condition": "treatment", "seed": 0, "lr": {"value": 0.2, "desc": None}},
            FakeSummary({"score": 0.9, "_runtime": 120.0}),
        ),
        FakeWandbRun(
            "a2",
            {"condition": "control", "seed": 1, "lr": 0.1},
            FakeSummary({"loss": 1.0}),
            state="crashed",
        ),
    ]
    api = FakeWandbApi(runs)
    records = wandb_adapter.fetch_runs(
        "ent/proj", ["score"], group="g1", tags=["main"], api=api, config_keys=["lr"]
    )
    assert api.calls == [("ent/proj", {"group": "g1", "tags": {"$in": ["main"]}})]
    r1, r2 = records
    assert (r1.run_id, r1.condition, r1.seed, r1.metrics, r1.status) == (
        "wandb:a1",
        "treatment",
        0,
        {"score": 0.9},
        "finished",
    )
    assert r1.started_at == datetime(2026, 3, 1, 10, tzinfo=timezone.utc)
    assert (r1.ended_at - r1.started_at).total_seconds() == 120
    assert r1.git_sha == "abc123" and r1.config_hash == sha256_hex({"lr": 0.2})
    assert r2.status == "crashed" and r2.metrics == {} and r2.ended_at is None


def test_wandb_missing_condition_raises() -> None:
    api = FakeWandbApi([FakeWandbRun("x", {"seed": 1}, FakeSummary({}))])
    with pytest.raises(ValueError, match="condition"):
        wandb_adapter.fetch_runs("e/p", ["score"], api=api)


def test_wandb_import_is_lazy() -> None:
    def boom() -> Any:
        raise ImportError("no wandb")

    with pytest.raises(ImportError):
        wandb_adapter.fetch_runs("e/p", ["score"], api_factory=boom)


@dataclass
class FakeInfo:
    run_id: str
    status: str
    start_time: int
    end_time: int | None
    experiment_id: str = "7"
    run_name: str = "r"


@dataclass
class FakeData:
    params: dict[str, str]
    metrics: dict[str, float]
    tags: dict[str, str]


@dataclass
class FakeMlflowRun:
    info: FakeInfo
    data: FakeData


class FakePage(list):
    def __init__(self, items: list[FakeMlflowRun], token: str | None) -> None:
        super().__init__(items)
        self.token = token


@dataclass
class FakeExperiment:
    experiment_id: str


class FakeMlflowClient:
    def __init__(self, pages: list[FakePage]) -> None:
        self.pages = pages
        self.calls: list[dict[str, Any]] = []

    def get_experiment_by_name(self, name: str) -> FakeExperiment | None:
        return FakeExperiment("7") if name == "study" else None

    def search_runs(self, **kwargs: Any) -> FakePage:
        self.calls.append(kwargs)
        return self.pages[len(self.calls) - 1]


def _mlrun(i: int, cond: str, status: str = "FINISHED") -> FakeMlflowRun:
    start = int(datetime(2026, 3, 1, tzinfo=timezone.utc).timestamp() * 1000) + i * 1000
    return FakeMlflowRun(
        FakeInfo(f"id{i}", status, start, start + 60_000),
        FakeData(
            {"condition": cond, "seed": str(i), "lr": "0.1"},
            {"score": float(i)},
            {"mlflow.source.git.commit": "deadbeef"},
        ),
    )


def test_mlflow_fetch_paginates_and_maps() -> None:
    client = FakeMlflowClient(
        [
            FakePage([_mlrun(0, "control"), _mlrun(1, "treatment")], "tok"),
            FakePage([_mlrun(2, "control", "KILLED")], None),
        ]
    )
    records = mlflow_adapter.fetch_runs(
        "study", ["score"], filter_string="tags.group = 'a'", client=client, config_keys=["lr"]
    )
    assert [c["page_token"] for c in client.calls] == [None, "tok"]
    assert (
        client.calls[0]["experiment_ids"] == ["7"]
        and client.calls[0]["filter_string"] == "tags.group = 'a'"
    )
    assert [r.run_id for r in records] == ["mlflow:id0", "mlflow:id1", "mlflow:id2"]
    assert records[2].status == "killed" and records[0].git_sha == "deadbeef"
    assert (records[0].ended_at - records[0].started_at).total_seconds() == 60
    assert records[1].seed == 1 and records[1].metrics == {"score": 1.0}
    assert records[0].config_hash == sha256_hex({"lr": "0.1"})
    with pytest.raises(ValueError, match="not found"):
        mlflow_adapter.fetch_runs("nope", ["score"], client=client)


def test_adapter_runs_link_end_to_end(tmp_path, clock: FakeClock) -> None:
    clock.now = datetime(2026, 2, 1, tzinfo=timezone.utc)
    study = make_study(tmp_path)
    project = Project.init(tmp_path, clock=clock)
    project.register(study)
    project.run_gates(study)
    client = FakeMlflowClient(
        [FakePage([_mlrun(i, c) for i in range(2) for c in ("control", "treatment")], None)]
    )
    records = mlflow_adapter.fetch_runs("study", ["score"], client=client)
    summary = project.link_runs(records)
    assert summary.refused == [
        ("mlflow:id0", "already linked with different content"),
        ("mlflow:id1", "already linked with different content"),
    ]
    assert len(summary.linked) == 2

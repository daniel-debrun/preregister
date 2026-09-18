"""Pull runs from Weights & Biases via the public API.

Install with ``pip install 'preregister[wandb]'``. The adapter reads, it never writes:
the condition and seed come from run config keys, metrics from the run summary.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta, timezone
from typing import Any

from preregister.integrations import lookup, subset_config_hash
from preregister.ledger import parse_timestamp
from preregister.runs import RunRecord

_STATUS = {
    "finished": "finished",
    "failed": "failed",
    "crashed": "crashed",
    "killed": "killed",
    "running": "running",
}


def _default_api() -> Any:
    try:
        import wandb
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError("the wandb adapter needs 'pip install preregister[wandb]'") from exc
    return wandb.Api()


def build_filters(group: str | None = None, tags: Sequence[str] | None = None) -> dict[str, Any]:
    filters: dict[str, Any] = {}
    if group is not None:
        filters["group"] = group
    if tags:
        filters["tags"] = {"$in": list(tags)}
    return filters


def _summary_dict(summary: Any) -> dict[str, Any]:
    raw = getattr(summary, "_json_dict", None)
    if isinstance(raw, Mapping):
        return dict(raw)
    return dict(summary) if summary is not None else {}


def _config_dict(config: Any) -> dict[str, Any]:
    out = {}
    for key, value in dict(config or {}).items():
        if isinstance(value, Mapping) and set(value) == {"value", "desc"}:
            value = value["value"]
        out[key] = value
    return out


def run_to_record(
    run: Any,
    metrics: Sequence[str],
    condition_key: str = "condition",
    seed_key: str = "seed",
    config_keys: Sequence[str] | None = None,
    metric_versions: Mapping[str, str] | None = None,
) -> RunRecord:
    config = _config_dict(run.config)
    summary = _summary_dict(run.summary)
    started = parse_timestamp(run.created_at) if run.created_at else datetime.now(timezone.utc)
    runtime = summary.get("_runtime")
    ended = started + timedelta(seconds=float(runtime)) if runtime is not None else None
    metadata = getattr(run, "metadata", None) or {}
    git_sha = (metadata.get("git") or {}).get("commit") or getattr(run, "commit", None)
    condition = lookup(config, condition_key)
    seed = lookup(config, seed_key)
    if condition is None or seed is None:
        raise ValueError(f"wandb run {run.id}: config lacks {condition_key!r} or {seed_key!r}")
    return RunRecord(
        run_id=f"wandb:{run.id}",
        condition=str(condition),
        seed=int(seed),
        metrics={m: summary.get(m) for m in metrics if m in summary},
        started_at=started,
        ended_at=ended,
        status=_STATUS.get(str(run.state), "failed"),
        git_sha=git_sha,
        config_hash=subset_config_hash(config, config_keys),
        metric_versions=dict(metric_versions or {}),
        source="wandb",
        meta={"name": getattr(run, "name", None), "url": getattr(run, "url", None)},
    )


def fetch_runs(
    path: str,
    metrics: Sequence[str],
    *,
    group: str | None = None,
    tags: Sequence[str] | None = None,
    condition_key: str = "condition",
    seed_key: str = "seed",
    config_keys: Sequence[str] | None = None,
    metric_versions: Mapping[str, str] | None = None,
    api: Any = None,
    api_factory: Callable[[], Any] = _default_api,
) -> list[RunRecord]:
    """Fetch every run in ``entity/project`` matching group/tags, in any state.

    ``config_keys`` selects the config entries that make up a condition's registered
    config, so that ``ConfigFrozen`` can compare hashes.
    """
    api = api if api is not None else api_factory()
    runs = api.runs(path, filters=build_filters(group, tags))
    return [
        run_to_record(run, metrics, condition_key, seed_key, config_keys, metric_versions)
        for run in runs
    ]

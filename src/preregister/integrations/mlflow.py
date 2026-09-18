"""Pull runs from MLflow tracking.

Install with ``pip install 'preregister[mlflow]'``. Condition and seed come from run
params (or tags), metrics from the latest logged metric values.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from preregister.integrations import subset_config_hash
from preregister.runs import RunRecord

_STATUS = {
    "FINISHED": "finished",
    "FAILED": "failed",
    "KILLED": "killed",
    "RUNNING": "running",
    "SCHEDULED": "running",
}
GIT_COMMIT_TAG = "mlflow.source.git.commit"


def _default_client() -> Any:
    try:
        from mlflow.tracking import MlflowClient
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError("the mlflow adapter needs 'pip install preregister[mlflow]'") from exc
    return MlflowClient()


def _ms(value: int | None) -> datetime | None:
    return None if value is None else datetime.fromtimestamp(value / 1000.0, tz=timezone.utc)


def run_to_record(
    run: Any,
    metrics: Sequence[str],
    condition_key: str = "condition",
    seed_key: str = "seed",
    config_keys: Sequence[str] | None = None,
    metric_versions: Mapping[str, str] | None = None,
) -> RunRecord:
    info, data = run.info, run.data
    params, tags = dict(data.params), dict(data.tags)
    condition = params.get(condition_key, tags.get(condition_key))
    seed = params.get(seed_key, tags.get(seed_key))
    if condition is None or seed is None:
        raise ValueError(f"mlflow run {info.run_id}: missing {condition_key!r} or {seed_key!r}")
    started = _ms(info.start_time)
    if started is None:
        raise ValueError(f"mlflow run {info.run_id}: no start_time")
    return RunRecord(
        run_id=f"mlflow:{info.run_id}",
        condition=str(condition),
        seed=int(float(seed)),
        metrics={m: data.metrics[m] for m in metrics if m in data.metrics},
        started_at=started,
        ended_at=_ms(info.end_time),
        status=_STATUS.get(str(info.status), "failed"),
        git_sha=tags.get(GIT_COMMIT_TAG),
        config_hash=subset_config_hash(params, config_keys),
        metric_versions=dict(metric_versions or {}),
        source="mlflow",
        meta={"experiment_id": info.experiment_id, "run_name": getattr(info, "run_name", None)},
    )


def fetch_runs(
    experiment: str,
    metrics: Sequence[str],
    *,
    filter_string: str = "",
    condition_key: str = "condition",
    seed_key: str = "seed",
    config_keys: Sequence[str] | None = None,
    metric_versions: Mapping[str, str] | None = None,
    client: Any = None,
    client_factory: Callable[[], Any] = _default_client,
    page_size: int = 1000,
) -> list[RunRecord]:
    """Fetch all runs of an experiment (by name) matching ``filter_string``, following pagination.

    Note that MLflow params are strings; a condition config declared in YAML with
    numbers will not hash equal to the params unless declared as strings.
    """
    client = client if client is not None else client_factory()
    exp = client.get_experiment_by_name(experiment)
    if exp is None:
        raise ValueError(f"mlflow experiment {experiment!r} not found")
    records: list[RunRecord] = []
    token = None
    while True:
        page = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string=filter_string,
            max_results=page_size,
            page_token=token,
        )
        records.extend(
            run_to_record(r, metrics, condition_key, seed_key, config_keys, metric_versions)
            for r in page
        )
        token = getattr(page, "token", None)
        if not token:
            return records

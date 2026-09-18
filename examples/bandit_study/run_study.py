"""End-to-end pre-registered study: register -> gate -> run -> link -> analyze -> report.

    python run_study.py --fresh

``--fresh`` deletes the existing ledger first. Without it, the script refuses to
overwrite a completed study, because re-running after registration is exactly the
kind of thing the ledger is meant to make visible.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from bandit import run_bandit  # noqa: E402

from preregister import Project, RunRecord, Study  # noqa: E402
from preregister.canonical import sha256_hex  # noqa: E402
from preregister.report import write_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true", help="delete .prereg/ and start over")
    parser.add_argument("--author", default="Daniel De Brun")
    args = parser.parse_args()

    if args.fresh:
        shutil.rmtree(HERE / ".prereg", ignore_errors=True)
    elif (HERE / ".prereg").exists():
        sys.exit("study already has a ledger; pass --fresh to start a new one")

    study = Study.from_yaml(HERE / "prereg.yaml")
    project = Project.init(HERE, author=args.author)

    for change in project.register(study, author=args.author):
        print(f"registered {change.prediction_id}")

    for pid, results in project.run_gates(study, author=args.author).items():
        status = "PASS" if all(r.passed for r in results) else "FAIL"
        print(f"gates {pid}: {status}")
        for r in results:
            print(f"    [{'pass' if r.passed else 'FAIL'}] {r.gate}: {r.message}")

    seeds = sorted({s for p in study.predictions for s in p.analysis.planned_seeds})
    runs = []
    for name, cond in study.conditions.items():
        config = dict(cond.config or {})
        for seed in seeds:
            started = datetime.now(timezone.utc)
            metrics = run_bandit(config, seed)
            runs.append(
                RunRecord(
                    run_id=f"{name}-s{seed}",
                    condition=name,
                    seed=seed,
                    metrics=metrics,
                    started_at=started,
                    ended_at=datetime.now(timezone.utc),
                    config_hash=sha256_hex(config),
                    metric_versions={"cumulative_regret": "1", "optimal_arm_rate": "1"},
                    source="run_study.py",
                )
            )
        print(f"ran {name} on seeds {seeds[0]}-{seeds[-1]}")

    summary = project.link_runs(runs, author=args.author)
    print(f"linked {len(summary.linked)} runs, refused {len(summary.refused)}")

    analysis = project.analyze(exploratory=study.exploratory, author=args.author)
    for result in analysis.predictions:
        print(f"{result.prediction_id}: {result.verdict.value}")

    state = project.state()
    write_report(analysis, state, HERE / "REPORT.md")
    write_report(analysis, state, HERE / "REPORT.html")
    print(f"verify: {'OK' if project.verify(study).ok else 'FAILED'}")


if __name__ == "__main__":
    main()

"""Command-line interface: ``preregister <command>``."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from preregister.ledger import LedgerError
from preregister.project import Project, RegistrationError
from preregister.registry import SpecError, Study
from preregister.report import write_report
from preregister.runs import RunError, RunRecord, load_runs_csv, load_runs_json

TEMPLATE = """\
study: my-study
description: What question this study answers.

conditions:
  baseline:
    role: control
    config: {lr: 0.001, algo: baseline}
  candidate:
    role: treatment
    config: {lr: 0.001, algo: candidate}

metrics:
  eval_return:
    version: "1"
    higher_is_better: true
    description: Mean undiscounted return over 100 evaluation episodes.

exclusions:
  - rule: "status != finished"
    reason: run crashed or was killed before evaluation

predictions:
  - id: P1
    hypothesis: The candidate improves evaluation return over the baseline.
    metric: eval_return
    comparison: "candidate - baseline > 0"
    min_effect: 5.0
    stopping_rule: exactly 10 seeds per condition, no early stopping
    analysis:
      test: bootstrap
      aggregation: iqm
      alpha: 0.05
      n_seeds: 10
      correction: none
    gates:
      - {type: MinSeeds, n: 5}
      - {type: ControlPresent}
      - {type: PowerAnalysis, sd: 4.0, power: 0.8}
"""


def _author(args: argparse.Namespace) -> str | None:
    return args.author or os.environ.get("PREREG_AUTHOR")


def _spec_path(args: argparse.Namespace) -> Path:
    return Path(args.spec) if args.spec else Path(args.root) / "prereg.yaml"


def _load_study(args: argparse.Namespace) -> Study:
    if args.plugin:
        sys.path.insert(0, str(Path(args.root).resolve()))
        for module in args.plugin:
            __import__(module)
    return Study.from_yaml(_spec_path(args))


def _project(args: argparse.Namespace) -> Project:
    return Project.open(args.root)


def cmd_init(args: argparse.Namespace) -> int:
    project = Project.init(args.root, author=_author(args))
    spec = _spec_path(args)
    if not spec.exists():
        spec.write_text(TEMPLATE, encoding="utf-8")
        print(f"wrote template {spec}")
    print(f"ledger at {project.ledger.path}")
    return 0


def cmd_register(args: argparse.Namespace) -> int:
    changes = _project(args).register(_load_study(args), author=_author(args))
    for c in changes:
        detail = (
            f" hash {c.entry.payload['prediction_hash'][:16]}"
            if c.status == "registered" and c.entry
            else ""
        )
        print(f"{c.prediction_id}: {c.status}{detail}")
    return 0


def cmd_amend(args: argparse.Namespace) -> int:
    project, study = _project(args), _load_study(args)
    for pid in args.prediction:
        entry = project.amend(study, pid, args.reason, author=_author(args))
        print(
            f"{pid}: amended, hash {entry.payload['prediction_hash'][:16]} supersedes {entry.payload['supersedes'][:16]}"
        )
    return 0


def cmd_gate(args: argparse.Namespace) -> int:
    results = _project(args).run_gates(
        _load_study(args), args.prediction or None, author=_author(args)
    )
    failed = False
    for pid, gate_results in results.items():
        passed = all(r.passed for r in gate_results)
        failed |= not passed
        print(f"{pid}: {'PASS' if passed else 'FAIL'}")
        for r in gate_results:
            print(f"  [{'pass' if r.passed else 'FAIL'}] {r.gate}: {r.message}")
    return 1 if failed else 0


def _fetch_runs(args: argparse.Namespace) -> list[RunRecord]:
    if args.json:
        return load_runs_json(args.json)
    if args.csv:
        return load_runs_csv(args.csv)
    if not args.metric:
        raise RunError("--metric is required for tracker imports")
    config_keys = args.config_key or None
    if args.wandb:
        from preregister.integrations import wandb as wb

        return wb.fetch_runs(
            args.wandb,
            args.metric,
            group=args.group,
            tags=args.tag,
            condition_key=args.condition_key,
            seed_key=args.seed_key,
            config_keys=config_keys,
        )
    from preregister.integrations import mlflow as mf

    return mf.fetch_runs(
        args.mlflow,
        args.metric,
        filter_string=args.filter or "",
        condition_key=args.condition_key,
        seed_key=args.seed_key,
        config_keys=config_keys,
    )


def cmd_link(args: argparse.Namespace) -> int:
    runs = _fetch_runs(args)
    summary = _project(args).link_runs(runs, force=args.force, author=_author(args))
    print(
        f"linked {len(summary.linked)}, already linked {len(summary.unchanged)}, refused {len(summary.refused)}"
    )
    for run_id, reason in summary.refused:
        print(f"  refused {run_id}: {reason}")
    for run_id, per_pred in summary.rejected_for.items():
        for pid, reasons in per_pred.items():
            print(f"  {run_id} will not count for {pid}: {'; '.join(reasons)}")
    return 1 if summary.refused and not summary.linked else 0


def _parse_overrides(items: Sequence[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for item in items:
        try:
            pid, assignment = item.split(":", 1)
            key, raw = assignment.split("=", 1)
        except ValueError:
            raise SpecError(f"override must look like P1:test=welch, got {item!r}") from None
        out.setdefault(pid, {})[key] = yaml.safe_load(raw)
    return out


def _exploratory(args: argparse.Namespace) -> list[Any]:
    spec = _spec_path(args)
    if not spec.exists():
        return []
    return _load_study(args).exploratory


def cmd_analyze(args: argparse.Namespace) -> int:
    project = _project(args)
    analysis = project.analyze(
        overrides=_parse_overrides(args.override or []),
        exploratory=_exploratory(args),
        author=_author(args),
    )
    for r in analysis.predictions:
        outcome = (
            f" (statistically {r.statistical_outcome.value})"
            if r.verdict.value == "DEVIATED" and r.statistical_outcome
            else ""
        )
        est = (
            ""
            if r.estimate is None
            else f" estimate {r.estimate:.4g} [{r.ci_low:.4g}, {r.ci_high:.4g}]"
        )
        print(f"{r.prediction_id}: {r.verdict.value}{outcome}{est}")
        for d in r.deviations:
            print(f"  deviation: {d}")
    if args.json:
        Path(args.json).write_text(json.dumps(analysis.to_dict(), indent=2), encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    project = _project(args)
    analysis = project.analyze(exploratory=_exploratory(args), record=False)
    fmt = args.format
    output = (
        Path(args.output)
        if args.output
        else Path(args.root) / f"REPORT.{'html' if fmt == 'html' else 'md'}"
    )
    path = write_report(analysis, project.state(), output, fmt)
    print(f"wrote {path}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    project = _project(args)
    spec = _spec_path(args)
    study = _load_study(args) if spec.exists() and not args.ledger_only else None
    report = project.verify(study)
    for issue in report.issues:
        print(issue)
    head = report.head_hash[:16] if report.head_hash else "none"
    print(f"{'OK' if report.ok else 'FAILED'}: {report.n_entries} entries, head {head}")
    return 0 if report.ok else 1


def cmd_anchor(args: argparse.Namespace) -> int:
    project = _project(args)
    report = project.verify()
    if not report.ok:
        print("refusing to anchor a ledger that fails verification", file=sys.stderr)
        return 1
    index, head = project.anchor(commit=args.commit)
    print(f"anchored entry {index}: {head}")
    print(f"recorded in {project.anchors_path}; publish this hash somewhere outside your control")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    project = _project(args)
    state = project.state()
    spec = _spec_path(args)
    if spec.exists():
        for change in project.diff(_load_study(args)):
            if change.status != "unchanged":
                print(f"{change.prediction_id}: spec {change.status} (not registered)")
    if not state.registrations:
        print("no registered predictions")
    for pid in state.prediction_ids:
        current = state.current(pid)
        gate = state.latest_gate(pid)
        gate_text = (
            "gates not run" if gate is None else ("gates passed" if gate.passed else "gates FAILED")
        )
        plan = current.prediction.analysis
        counts = []
        for cond in current.comparison.conditions:
            n = sum(1 for rec in state.runs_for([cond]))
            counts.append(f"{cond} {n}/{plan.n_seeds}")
        print(
            f"{pid} v{current.version} [{current.prediction_hash[:12]}] {gate_text}; runs: {', '.join(counts)}"
        )
    print(f"ledger: {len(state.entries)} entries")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="preregister", description="Pre-registration for ML experiments."
    )
    parser.add_argument(
        "--root", default=".", help="project directory containing .prereg/ (default: .)"
    )
    parser.add_argument("--spec", help="study spec (default: <root>/prereg.yaml)")
    parser.add_argument("--author", help="author recorded in ledger entries (or PREREG_AUTHOR)")
    parser.add_argument("--plugin", action="append", help="module to import first (custom gates)")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser(
        "init", help="create .prereg/ledger.jsonl and a template prereg.yaml"
    ).set_defaults(func=cmd_init)
    sub.add_parser("register", help="register new predictions from the spec").set_defaults(
        func=cmd_register
    )

    p = sub.add_parser("amend", help="register a changed prediction as an amendment")
    p.add_argument("prediction", nargs="+")
    p.add_argument("--reason", required=True)
    p.set_defaults(func=cmd_amend)

    p = sub.add_parser("gate", help="evaluate admissibility gates and record the results")
    p.add_argument("prediction", nargs="*")
    p.set_defaults(func=cmd_gate)

    p = sub.add_parser("link", help="link run results")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--json")
    src.add_argument("--csv")
    src.add_argument("--wandb", metavar="ENTITY/PROJECT")
    src.add_argument("--mlflow", metavar="EXPERIMENT")
    p.add_argument("--group")
    p.add_argument("--tag", action="append")
    p.add_argument("--filter", help="MLflow filter string")
    p.add_argument("--metric", action="append", help="metric to import (repeatable)")
    p.add_argument("--condition-key", default="condition")
    p.add_argument("--seed-key", default="seed")
    p.add_argument("--config-key", action="append", help="config key hashed for ConfigFrozen")
    p.add_argument("--force", action="store_true", help="link even if no gate has passed")
    p.set_defaults(func=cmd_link)

    p = sub.add_parser("analyze", help="run the registered analysis and record verdicts")
    p.add_argument("--override", action="append", help="deviate from the plan, e.g. P1:test=welch")
    p.add_argument("--json", help="write full results as JSON")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("report", help="write a reconciliation report")
    p.add_argument("--format", choices=("md", "html"), default="md")
    p.add_argument("--output")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("verify", help="verify ledger integrity and spec-vs-registration hashes")
    p.add_argument("--ledger-only", action="store_true")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("anchor", help="record the ledger head hash in .prereg/anchors.txt")
    p.add_argument("--commit", action="store_true", help="git-commit the ledger and anchors file")
    p.set_defaults(func=cmd_anchor)

    sub.add_parser("status", help="show registrations, gates, and linked runs").set_defaults(
        func=cmd_status
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (
        SpecError,
        RegistrationError,
        LedgerError,
        RunError,
        KeyError,
        OSError,
        ImportError,
        yaml.YAMLError,
    ) as exc:
        print(f"preregister: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

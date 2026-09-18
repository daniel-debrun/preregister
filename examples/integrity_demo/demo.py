"""What the ledger catches: peeking, seed-picking, and tampering.

A toy study registers "the variant beats the baseline" with seeds 0-9. The true
effect is zero. Three things then go wrong, and preregister reports each of them:

1. Runs started before registration are linked. They are recorded but rejected.
2. The variant is run on 30 seeds and only the best 10 are linked. The declared
   seed list exposes it: undeclared seeds are present and declared seeds are missing.
3. The ledger is edited after the fact, first naively, then by recomputing every
   hash. The first is caught by the hash chain, the second by a published anchor.

Writes OUTPUT.md next to this file.
"""

from __future__ import annotations

import io
import json
import tempfile
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from preregister import AnalysisPlan, Condition, Metric, Prediction, Project, RunRecord, Study
from preregister.canonical import sha256_hex
from preregister.ledger import GENESIS_HASH

HERE = Path(__file__).resolve().parent
T0 = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)


class Clock:
    """Deterministic clock so OUTPUT.md is reproducible."""

    def __init__(self) -> None:
        self.now = T0

    def __call__(self) -> datetime:
        self.now += timedelta(seconds=1)
        return self.now


def score(condition: str, seed: int) -> float:
    rng = np.random.default_rng([seed, 7 if condition == "variant" else 3])
    return float(rng.normal(0.70, 0.05))


def make_run(condition: str, seed: int, started: datetime) -> RunRecord:
    return RunRecord(
        run_id=f"{condition}-s{seed}",
        condition=condition,
        seed=seed,
        metrics={"accuracy": score(condition, seed)},
        started_at=started,
        ended_at=started + timedelta(minutes=5),
    )


def make_study(root: Path) -> Study:
    return Study(
        name="variant-vs-baseline",
        root=root,
        conditions={
            "baseline": Condition("baseline", role="control", config={"arch": "base"}),
            "variant": Condition("variant", role="treatment", config={"arch": "variant"}),
        },
        metrics={"accuracy": Metric("accuracy", version="1")},
        predictions=[
            Prediction(
                id="H1",
                hypothesis="The variant improves accuracy over the baseline.",
                metric="accuracy",
                comparison="variant - baseline > 0",
                min_effect=0.05,
                analysis=AnalysisPlan(test="welch", n_seeds=10),
                gates=({"type": "MinSeeds", "n": 5},),
            )
        ],
    )


def setup(tmp: Path, name: str) -> tuple[Project, Clock]:
    clock = Clock()
    root = tmp / name
    project = Project.init(root, clock=clock)
    study = make_study(root)
    project.register(study)
    project.run_gates(study)
    return project, clock


def show(analysis, pid: str = "H1") -> None:
    r = analysis.result(pid)
    outcome = (
        f" (statistically {r.statistical_outcome.value})"
        if r.statistical_outcome and r.verdict.value == "DEVIATED"
        else ""
    )
    print(f"verdict: {r.verdict.value}{outcome}")
    if r.estimate is not None:
        print(
            f"estimate {r.estimate:+.4f}, {r.ci_level:.0%} interval [{r.ci_low:+.4f}, {r.ci_high:+.4f}], p(claim) {r.p_confirm:.4f}"
        )
    for d in r.deviations:
        print(f"  deviation: {d}")
    for x in r.rejected:
        print(f"  rejected {x.run_id}: {x.reason}")


def run() -> str:
    buf = io.StringIO()
    with tempfile.TemporaryDirectory() as tmpdir, redirect_stdout(buf):
        tmp = Path(tmpdir)

        print("## 1. Honest study: seeds 0-9 for both conditions\n")
        print("```")
        project, clock = setup(tmp, "honest")
        start = clock.now + timedelta(hours=1)
        runs = [make_run(c, s, start) for c in ("baseline", "variant") for s in range(10)]
        project.link_runs(runs)
        show(project.analyze(record=False))
        print("```\n")

        print("## 2. Peeking: three variant runs started before registration\n")
        print("```")
        project, clock = setup(tmp, "peeking")
        early = [make_run("variant", s, T0 - timedelta(days=2)) for s in range(3)]
        start = clock.now + timedelta(hours=1)
        late = [make_run("baseline", s, start) for s in range(10)]
        late += [make_run("variant", s, start) for s in range(3, 10)]
        project.link_runs(early + late)
        show(project.analyze(record=False))
        print("```\n")

        print("## 3. Seed-picking: variant run on seeds 0-29, best 10 linked\n")
        print("```")
        project, clock = setup(tmp, "picked")
        start = clock.now + timedelta(hours=1)
        best = sorted(range(30), key=lambda s: score("variant", s), reverse=True)[:10]
        print(f"linked variant seeds: {sorted(best)}")
        runs = [make_run("baseline", s, start) for s in range(10)]
        runs += [make_run("variant", s, start) for s in best]
        project.link_runs(runs)
        show(project.analyze(record=False))
        print("```\n")

        print("## 4. Tampering with the ledger\n")
        print("```")
        project, clock = setup(tmp, "tamper")
        start = clock.now + timedelta(hours=1)
        project.link_runs(
            [make_run(c, s, start) for c in ("baseline", "variant") for s in range(10)]
        )
        index, anchored = project.anchor()
        print(f"anchored head: entry {index} {anchored[:16]} (published elsewhere)")
        path = project.ledger.path
        lines = path.read_text().splitlines()
        target = next(i for i, ln in enumerate(lines) if '"variant-s0"' in ln)

        entry = json.loads(lines[target])
        entry["payload"]["run"]["metrics"]["accuracy"] += 0.2
        naive = lines.copy()
        naive[target] = json.dumps(entry, sort_keys=True)
        path.write_text("\n".join(naive) + "\n")
        print("\n(a) edit one metric value in place")
        report = project.ledger.verify()
        print(f"verify ok={report.ok}")
        for issue in report.issues[:3]:
            print(f"  {issue}")

        prev = GENESIS_HASH
        rewritten = []
        for i, ln in enumerate(lines):
            e = json.loads(ln)
            if i == target:
                e["payload"]["run"]["metrics"]["accuracy"] += 0.2
            e["payload_hash"] = sha256_hex(e["payload"])
            e["prev_hash"] = prev
            header = {
                k: e[k]
                for k in (
                    "index",
                    "timestamp",
                    "kind",
                    "payload_hash",
                    "prev_hash",
                    "git",
                    "author",
                )
            }
            e["hash"] = sha256_hex(header)
            prev = e["hash"]
            rewritten.append(json.dumps(e, sort_keys=True))
        path.write_text("\n".join(rewritten) + "\n")
        print("\n(b) edit the value and recompute every hash in the chain")
        print(f"verify without anchors ok={project.ledger.verify().ok}")
        report = project.ledger.verify(anchors=[(index, anchored)])
        print(f"verify against published anchor ok={report.ok}")
        for issue in report.issues:
            print(f"  {issue}")
        print("```")

    return "# Integrity demo output\n\nGenerated by `python demo.py`.\n\n" + buf.getvalue()


def main() -> None:
    text = run()
    (HERE / "OUTPUT.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()

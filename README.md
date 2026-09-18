# preregister

Pre-registration for machine learning experiments. You write down your predictions,
check that the experiment can actually answer them, run it, and then compare what
happened with what you predicted. Every step goes into a tamper-evident ledger.

`preregister` is a plugin for the experiment trackers you already use (JSON/CSV, Weights &
Biases, MLflow). It does not replace them. Trackers record what happened; `preregister`
records what you expected before it happened and holds the analysis to that record.

```
pip install preregister            # import name and CLI: preregister
pip install 'preregister[wandb]'   # optional tracker adapters
pip install 'preregister[mlflow]'
```

Status: v0.1.0, alpha. Python 3.10+. Runtime dependencies: numpy and PyYAML.
Not yet published to PyPI; install from source with `pip install -e .`.

## Contents

1. [Problem](#problem)
2. [Protocol](#protocol)
3. [Quickstart](#quickstart)
4. [Concepts](#concepts)
5. [Integrations](#integrations)
6. [Worked example](#worked-example)
7. [Related tools](#related-tools)
8. [Limitations](#limitations)
9. [Roadmap](#roadmap)

## Problem

Most ML papers report results without any record of what the authors expected before
the runs. That leaves several failure modes invisible:

- the hypothesis gets rewritten after the results are in;
- the test, aggregation (mean, median, IQM, best seed), or exclusion rule gets chosen
  after looking at the data;
- seeds get added until the result looks right, or runs that disagree get dropped;
- the setup could never have answered the question: an environment layout where a
  recipe cannot be cooked, a reward that credits deliveries to the wrong agent, a
  "best seed" metric that is biased upward by construction, or too few seeds to detect
  any effect worth caring about.

This project came out of a multi-agent RL research project where I wrote dated
predictions before each experimental step, audited environments and metrics before
running anything, and compared outcomes against predictions afterwards. The audits
caught real bugs, and several of those comparisons ended in retractions. Doing all of
that by hand in documents was slow and easy to get wrong, and there was no tooling for
it. `preregister` turns that practice into a library.

## Protocol

```
  register  -->  gate  -->  run & link  -->  reconcile
     |            |              |               |
  frozen       recorded      every run       verdicts from the
  prediction   gate results  recorded        registered plan only
     \____________\______________\_______________/
          .prereg/ledger.jsonl (append-only, SHA-256 hash chain)
```

1. **Register.** Each prediction is converted to canonical JSON and hashed with
   SHA-256, then appended to the ledger along with a UTC timestamp, the git commit
   and dirty flag, and the hashes of the condition configs and the metric. A
   prediction includes its analysis plan, seed list, stopping rule, and exclusion
   criteria. To change it later, you add an amendment that gives a reason.
2. **Gate.** Admissibility checks (enough seeds, enough power, a control is present,
   the metric definition and configs haven't changed, no runs started before
   registration, plus your own domain checks) run before any runs are linked. Their
   results are recorded in the ledger too.
3. **Link.** Runs are recorded against conditions. Every linked run is accounted for:
   it is counted in the analysis, excluded by a criterion you declared up front, or
   rejected with a reason. Nothing is dropped without a record.
4. **Reconcile.** The registered analysis runs exactly as planned, and each
   prediction gets a verdict: `CONFIRMED`, `REFUTED`, `INCONCLUSIVE`, `INADMISSIBLE`,
   or `DEVIATED`. Analyses you didn't register go in a separate section with a clear
   label.

The methodology, including what each gate prevents and the exact verdict semantics,
is in [docs/protocol.md](docs/protocol.md).

## Quickstart

```bash
preregister init                       # creates .prereg/ledger.jsonl and a template prereg.yaml
$EDITOR prereg.yaml                    # write predictions before running anything
preregister register                   # freeze them
preregister gate                       # admissibility checks; exit code 1 if any fail
# ... run experiments, log to your tracker ...
preregister link --json runs.json      # or --csv, --wandb ENTITY/PROJECT, --mlflow EXPERIMENT
preregister analyze                    # verdicts, recorded in the ledger
preregister report --format html       # REPORT.html (or REPORT.md)
preregister verify                     # hash chain, anchors, and spec-vs-registration check
preregister anchor --commit            # record the head hash in git; publish it
preregister status
```

A minimal `prereg.yaml`:

```yaml
study: lr-warmup
conditions:
  baseline:  {role: control,   config: {warmup: 0,    lr: 3.0e-4}}
  warmup:    {role: treatment, config: {warmup: 1000, lr: 3.0e-4}}
metrics:
  eval_return: {version: "1", higher_is_better: true}
exclusions:
  - {rule: "status != finished", reason: run crashed or was killed}
predictions:
  - id: P1
    hypothesis: Warmup improves evaluation return by at least 5.
    metric: eval_return
    comparison: "warmup - baseline > 5"
    min_effect: 5
    stopping_rule: exactly 10 seeds (0-9) per condition
    analysis: {test: bootstrap, aggregation: iqm, alpha: 0.05, n_seeds: 10}
    gates:
      - {type: MinSeeds, n: 5}
      - {type: ControlPresent}
      - {type: PowerAnalysis, sd: 4.0, power: 0.8}
```

The same workflow in Python:

```python
from preregister import Project, Study, load_runs_json

study = Study.from_yaml("prereg.yaml")
project = Project.init(".")
project.register(study, author="Daniel De Brun")
project.run_gates(study)
# ... run experiments ...
project.link_runs(load_runs_json("runs.json"))
analysis = project.analyze(exploratory=study.exploratory)
print(analysis.verdicts())
```

The JSON run format is a list of objects with `run_id`, `condition`, `seed`, `metrics`,
and `started_at`, plus these optional fields: `ended_at`, `status`, `git_sha`,
`config_hash`, `metric_versions`, `meta`. Times must be ISO 8601 with a UTC offset
(`2026-01-01T12:00:00Z` or `+09:00`). Times without an offset are rejected: a local time
read as UTC can move a run to the wrong side of the registration time.

## Concepts

**Comparisons.** Each claim is directional and compares aggregates across seeds:

| Form | Meaning |
|---|---|
| `a - b > t` | the difference exceeds `t` (`<`, `<=`, `>=` also work) |
| `a > b` | shorthand for `a - b > 0` |
| `a / b >= r` | ratio of aggregates; every value of `b` must be positive, otherwise the prediction is `INCONCLUSIVE` with no statistics |
| `a > t` | a single condition against a threshold |
| `a in [lo, hi]` | the aggregate lies within an interval (two one-sided tests) |

**Analysis plan.** The test is `bootstrap` (percentile), `welch`, `mann_whitney`
(exact for small samples without ties, otherwise normal approximation with tie
correction), or `permutation` (sign-flip for one-sample claims). Aggregation is `mean`,
`median`, or `iqm`, the interquartile mean from Agarwal et al., "Deep Reinforcement
Learning at the Edge of the Statistical Precipice", NeurIPS 2021. The plan also sets
`alpha`, `n_seeds` and an explicit `seeds` list (defaults to `0..n_seeds-1`), and
`correction` (`none`, `holm`, `bh`) within a `family`. Invalid combinations are
rejected when you register. For example, Welch with IQM is rejected because the t-test
is a test of means. All statistics use numpy only, and the test suite checks them
against scipy.

**Verdicts.** Each claim gets two one-sided p-values: one for the claim and one for
its opposite. Both are adjusted within the family, where the family size counts every
registered member, including inadmissible ones.

| Verdict | Condition |
|---|---|
| `INADMISSIBLE` | gates failed or were never run for this version; no statistics are shown |
| `DEVIATED` | analysis differs from the plan (late amendment, missing or undeclared seeds, peeking, config drift, undeclared non-finite values, analysis override); the statistical outcome is shown next to it |
| `CONFIRMED` | adjusted p for the claim < alpha |
| `REFUTED` | adjusted p for the opposite < alpha |
| `INCONCLUSIVE` | neither |

Intervals are percentile bootstrap intervals at level 1 - 2 alpha (the interval that
matches a one-sided test) and are not adjusted for multiple comparisons.

**Gates.**

| Gate | Checks |
|---|---|
| `MinSeeds(n)` | planned seeds per condition >= n |
| `PowerAnalysis(sd \| sd_by_condition \| pilot, power, method)` | planned seeds can detect `min_effect`; `normal` approximation adjusted for the aggregation's efficiency, or `simulation` of the planned test |
| `ControlPresent` | a comparison has a configured control or baseline |
| `MetricDefined` | metric has a version or function; fingerprint unchanged; each run reports the same version |
| `ConfigFrozen(require_run_hash)` | condition config hashes unchanged; run config hashes match |
| `NoPeeking` | no counted run started before the prediction's first registration |

`MetricDefined`, `ConfigFrozen`, and `NoPeeking` always apply. Custom gates are plain
functions:

```python
from preregister import custom_gate


@custom_gate("RecipesCompletable")
def recipes_completable(ctx, **params):
    layout = ctx.study.conditions["open_comm"].config["layout"]
    ok = all_recipes_reachable(layout)
    return ok, "every recipe is completable" if ok else "unreachable ingredient"
```

In YAML you can refer to a gate by name once its module is imported (`--plugin module`
on the CLI) or by import path (`type: "mypkg.gates:recipes_completable"`). If a gate
raises an exception, it counts as failed. See
[examples/env_admissibility](examples/env_admissibility).

**Amendments.** `preregister register` refuses to overwrite a changed prediction.
`preregister amend P1 --reason "..."` appends a new version that points to the one it
replaces. Amendments made before any run is linked are listed in the report. Amendments
made after runs are linked make the verdict `DEVIATED`. A new version needs its gates
re-run.

**Anti seed-picking.** The seed list is part of the registration, and every linked run
is recorded. If you link the best 10 of 30 seeds, the report shows undeclared seeds
present and declared seeds missing. Runs whose seeds belong to another prediction
using the same condition are not counted against this one.

**Integrity.** `preregister verify` recomputes every payload hash, entry hash, chain link,
and registration hash. It also compares the current `prereg.yaml` with what was
registered, so an edit you never registered gets reported. If someone recomputes the
entire chain, only an anchor can catch it: `preregister anchor` appends the head hash to
`.prereg/anchors.txt`, and `--commit` commits it to git. Publish that hash somewhere
you don't control, such as a pushed commit, an email, or a message to collaborators.
See [examples/integrity_demo/OUTPUT.md](examples/integrity_demo/OUTPUT.md).

**Exploratory analyses.** Put them under `exploratory:` in `prereg.yaml`. They appear
in a separate report section with estimates and intervals, but no p-values and no
verdicts.

## Integrations

Both adapters only read from the tracker and are imported lazily. The tests use fake
clients and never touch the network.

**Weights & Biases.** Condition and seed come from run config keys; metrics come from
the run summary; the start time comes from `created_at`; the end time comes from
`_runtime`; the commit comes from run metadata.

```bash
preregister link --wandb my-entity/my-project --group sweep-3 --tag registered \
  --metric eval_return --condition-key condition --seed-key seed --config-key lr --config-key warmup
```

```python
from preregister.integrations import wandb

runs = wandb.fetch_runs(
    "my-entity/my-project", ["eval_return"], group="sweep-3", config_keys=["lr", "warmup"]
)
```

**MLflow.** Condition and seed come from params (or tags); metrics are the latest
logged values; status, times, and the `mlflow.source.git.commit` tag come from the run;
results are paginated.

```bash
preregister link --mlflow my-experiment --filter "tags.phase = 'registered'" --metric eval_return
```

`--config-key` picks the config entries whose hash `ConfigFrozen` compares against the
registered condition config. MLflow params are strings, so for condition configs that
come from MLflow, declare the values as strings.

## Worked example

[examples/bandit_study](examples/bandit_study) compares epsilon-greedy (epsilon = 0.1),
UCB1, and Thompson sampling on a fixed 10-armed Bernoulli bandit over 2000 steps, with
20 seeds per algorithm. `run_study.py` registers, gates, runs, links, analyzes, and
writes the report in about 15 seconds on a CPU. The committed
[`.prereg/ledger.jsonl`](examples/bandit_study/.prereg/ledger.jsonl),
[`REPORT.md`](examples/bandit_study/REPORT.md), and
[`REPORT.html`](examples/bandit_study/REPORT.html) come from an actual run.

Before registering, I ran `pilot.py` on seeds 1000-1079, which are not among the
registered seeds, to pick the standard deviations for the power gates. As a result,
the split between outcomes below was expected. The example is here to show the
workflow, not to report a discovery.

| Prediction | Claim (cumulative regret, lower is better) | Plan | Verdict |
|---|---|---|---|
| P1 | `thompson - ucb < -20` | bootstrap, IQM, Holm | **CONFIRMED**: IQM difference -76.3, 90% interval [-83.2, -67.7] |
| P2 | `ucb - eps_greedy < 0` | Welch, mean, Holm | **REFUTED**: UCB1 has *more* regret, +80.1 [61.8, 98.0]. The textbook asymptotic ordering does not hold at this horizon with these arm gaps. |
| P3 | `thompson - eps_greedy < 0` | bootstrap, mean, Holm | **INCONCLUSIVE**: +6.6 [-12.4, 25.3]. Powered for a 45-unit difference; the true difference is much smaller. |
| P4 | `thompson - eps_greedy > 0.1` (optimal-arm rate, 5 seeds) | Mann-Whitney, median | **INADMISSIBLE**: `PowerAnalysis` failed (power 0.09 at 5 seeds, about 278 needed). No statistics reported. |

The optimal-arm-rate comparison that P4 could not support appears in the exploratory
section instead (E1, median difference -0.09, 95% interval [-0.28, 0.67]), with no
verdict. Epsilon-greedy's per-seed distribution is bimodal: it either locks onto the
best arm or onto the 0.45 arm. You can see this in the report's per-seed plots, and a
single summary number hides it.

Two more examples:

- [examples/integrity_demo](examples/integrity_demo): linking runs that started before
  registration (rejected, `DEVIATED`), linking the best 10 of 30 seeds (`DEVIATED`,
  even though the statistics alone would say `CONFIRMED`), editing a metric value in
  the ledger (caught by the hash chain), and rewriting the whole chain (caught only by
  the published anchor).
- [examples/env_admissibility](examples/env_admissibility): a custom gate for a toy
  cooking gridworld that uses BFS to check that every recipe's ingredients can be
  reached. The layout with a walled-off tomato station makes its prediction
  `INADMISSIBLE` before any agent is trained.

**Note on the recorded commit.** The bandit ledger records git commit `e65d6b8ad9`, from
this project's development history. That history was squashed into a single commit before
release, so the commit no longer exists here. The ledger itself is unchanged and still
verifies: `preregister --root examples/bandit_study verify`.

## Related tools

Hash-anchored pre-registration is not new. What `preregister` adds is the ML-specific
loop: checks that run before any results exist (power, seeds, a control), accounting
for every seed, verdicts computed only from the registered plan, and read-only links to
the trackers where runs already live.

| Tool | What it does | Difference from `preregister` |
|---|---|---|
| [OSF Registries](https://osf.io/registries), [AsPredicted](https://aspredicted.org) | Timestamped pre-registration documents held by a third party | Documents only; nothing connects the plan to the runs or the analysis. A good place to publish an anchor hash. |
| [`prereg`](https://pypi.org/project/prereg/) (reproducible-science) | Freezes a `PREREG.md` by commit and hash, keeps an append-only amendment and deviation log, and can push a draft to OSF | Works at the document level: no gates, statistics, run linking, or verdicts. It installs a `prereg` command and module, which is why this package is named `preregister`. |
| [`sealedlab`](https://pypi.org/project/sealedlab/) | Pre-registration hashes, hash-chained ledger, sealed truth panels, mutation-tested gates, and a closed verdict vocabulary | Closest in spirit. General-purpose, with no power or seed-count gates, no seed accounting, and no W&B or MLflow linking. |
| MLflow, Weights & Biases, DVC, Sacred | Record configs, metrics, and artifacts of runs | Record what happened, not what was predicted. `preregister` reads from them. |
| [rliable](https://github.com/google-research/rliable) | Reliable aggregate metrics (IQM, performance profiles) for RL | A statistics library. `preregister` borrows IQM from it and holds the choice of statistic to the plan. |
| OpenTimestamps, RFC 3161 | Timestamps from a trusted third party | A complement: on the roadmap as an anchor backend. |

## Limitations

- **Nobody is forced to register.** `preregister` makes pre-registration cheap and
  verifiable, not mandatory. A reader has to ask for the ledger and its anchors.
- **Timestamps come from the local clock.** They can be trusted only as far as an
  anchor published outside the author's control. Without one, an entire ledger can be
  fabricated after the fact.
- **Run metadata is reported by whoever links the runs.** A determined bad actor can
  relabel seeds, fake start times, or run the whole study privately before
  registering. Tracker adapters narrow this gap but do not close it.
- **Statistical assumptions.** Seeds are treated as independent replicates. Percentile
  bootstrap intervals can undercover with very few seeds. The normal-approximation
  power gate is approximate, and the simulation method assumes Gaussian data. Bimodal
  per-seed distributions, which are common in RL (see P3 above), call for the
  simulation method, a rank-based test, or more seeds.
- **Power with a correction uses Bonferroni's alpha / m.** For Holm and BH this is
  conservative.
- **The implemented set is limited.** There are no stratified or hierarchical
  bootstraps across tasks, no sequential designs with alpha spending, and no two-sided
  claims.
- **Schema stability.** The frozen payload format is version 1. Future changes will
  need migrations to keep old ledgers verifiable.

## Roadmap

- Stratified bootstrap across tasks and environments, and performance profiles
  (Agarwal et al. 2021).
- Sequential designs with pre-registered alpha spending, so studies can stop early
  legitimately.
- A `preregister diff` command that shows field-level changes between registered versions.
- External timestamping through RFC 3161 or OpenTimestamps as an optional anchor
  backend.
- Tracker-side tagging: write the prediction hash back to W&B or MLflow runs when they
  are linked.
- A pytest-style runner for custom gates on environments, with reusable gates for
  common MARL pitfalls such as credit assignment and reachability.

## Development

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'
ruff check . && ruff format --check . && pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md). MIT licensed.

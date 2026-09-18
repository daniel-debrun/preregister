# Pre-registration for machine learning experiments: a protocol

Daniel De Brun

This note describes the protocol that `preregister` implements. It is written to stand on
its own, independent of the library.

## 1. Why pre-register ML experiments

Experiment trackers record what happened. They do not record what anyone expected to
happen, so nothing distinguishes a prediction made before a run from a story fitted
to it afterwards. Several common failure modes in empirical ML come from exactly that
gap:

- **Post-hoc hypotheses.** After seeing results, the claim is rewritten to match them
  ("method A helps on the harder layouts").
- **Analysis flexibility.** The test, aggregation (mean, median, IQM, best seed), metric
  window, or exclusion rule is chosen after the data is in. Each choice looks
  reasonable in isolation; together they form a garden of forking paths.
- **Seed-picking and optional stopping.** More seeds are run until the result looks
  right, or runs that disagree are dropped as "unstable".
- **Inadmissible setups.** The experiment cannot answer the question at all: an
  environment layout where a recipe is uncookable, a reward function that
  mis-credits deliveries to the wrong agent, a "best seed" metric that is biased
  upward by construction, or a comparison with too few seeds to detect any effect
  of interest. The result is then interpreted anyway.

Pre-registration in the clinical and social sciences addresses the first three by
fixing the hypothesis and analysis plan before data collection. ML differs in two
ways that shape this protocol. First, the unit of replication is usually the seed, runs
are cheap to repeat, and results are easy to regenerate after the fact, so the record
needs to bind runs to a registration in time. Second, a large share of the failures
we have encountered in practice were not statistical at all but setups that could not
answer the question. The protocol therefore makes admissibility checks a first-class
stage rather than a footnote.

The protocol grew out of a multi-agent RL research project in which predictions were
written down with timestamps before each experimental step, admissibility audits were
run on environments and metrics before experiments, and each step ended with a
reconciliation against the prediction. Several of those reconciliations were
retractions, and several admissibility audits caught bugs that would otherwise have
been reported as findings.

## 2. The four stages

```
  register  -->  gate  -->  run & link  -->  reconcile
     |            |              |               |
  frozen       recorded      every run       verdicts from the
  prediction   gate results  recorded        registered plan only
     \____________\______________\_______________/
                 one append-only, hash-chained ledger
```

### 2.1 Registration

A prediction states:

- a hypothesis in words;
- a metric, with a version string or a reference to the code that computes it;
- a directional comparison between conditions: `a - b > t`, `a > b`, `a / b >= r`,
  `a > t`, or `a in [lo, hi]`;
- a minimum effect of interest, the smallest effect the study should reliably detect;
- the analysis plan: test (percentile bootstrap, Welch t, Mann-Whitney U, permutation),
  alpha, the number of seeds and the exact seed list, aggregation over seeds (mean,
  median, or interquartile mean), and the multiple-comparison correction (Holm or
  Benjamini-Hochberg) with its family;
- a stopping rule;
- exclusion criteria, each with a reason (for example `status != finished`);
- the gates that must pass.

Registration serializes everything the prediction depends on (including hashes of each
condition's configuration and the metric's fingerprint) to canonical JSON, hashes it
with SHA-256, and appends it to the ledger with a UTC timestamp, the git commit, and
whether the working tree was dirty.

Changing a registered prediction is allowed only as an **amendment**: a new ledger
entry that references the entry it supersedes and states a reason. Amendments made
before any run is linked are ordinary protocol revisions. Amendments made after runs
are linked are reported as deviations.

### 2.2 Admissibility gates

Gates are checks that decide whether the registered experiment can answer its question
at all. They are evaluated after registration and before runs are linked, and their
results are appended to the ledger. A prediction whose latest gate evaluation failed
is **inadmissible**; its data is still recorded, but no statistics are reported for it.

| Gate | What it prevents |
|---|---|
| `MinSeeds` | Claims from a handful of seeds. |
| `PowerAnalysis` | Studies that cannot detect the declared minimum effect with the planned seeds and test. Uses pilot or declared standard deviations; normal approximation adjusted for the aggregation's efficiency, or Monte Carlo simulation of the planned test. Alpha is Bonferroni-split when a correction is declared. |
| `ControlPresent` | Comparisons with no declared, configured control or baseline. |
| `MetricDefined` | Metrics that change meaning between registration and analysis, or differ between conditions. Checked against a version string or a hash of the metric function's source, and per run against the reported metric version. |
| `ConfigFrozen` | Silent configuration drift: condition configs must hash to the registered values, both at gate time and for each run that reports a config hash. |
| `NoPeeking` | Using runs that began before the prediction was registered. |
| custom gates | Domain-specific admissibility: every recipe in a layout is completable, a reward is credited to the correct agent, an evaluation set does not overlap training data. |

`MetricDefined`, `ConfigFrozen`, and `NoPeeking` are protocol invariants and apply to
every prediction.

### 2.3 Run linking

Runs are linked to conditions, with seed, metric values, start and end time, git commit,
config hash, and status. Every linked run is written to the ledger. A linked run can be:

- **counted** in the analysis;
- **excluded** by a pre-declared criterion, and listed with that criterion's reason;
- **rejected** by a per-run gate (started before registration, config hash mismatch,
  metric missing or at a different version), and listed with the reason.

Nothing is dropped silently. The seed list is part of the registration, so linking the
best 10 of 30 seeds shows up as undeclared seeds present and declared seeds missing.
A run whose metric is not finite and is not covered by a declared exclusion is a
deviation, not a quiet omission.

### 2.4 Reconciliation

The analysis executes the registered plan. Each claim is rewritten as
`agg(x) - agg(y) > 0` (shifts and positive scalings commute with mean, median, and IQM),
and two one-sided p-values are computed: one for the claim and one for its negation.
Interval claims use two one-sided tests. Within each family the p-values are adjusted
with the declared correction, where the family size is the number of *registered*
predictions in it: inadmissible or empty members enter with p = 1, so dropping a
prediction can never make the correction less strict.

Effect estimates are reported with percentile bootstrap intervals at level 1 - 2 alpha,
the interval dual to a one-sided test at alpha. These intervals are not
multiplicity-adjusted.

## 3. Verdicts

Verdicts are assigned in order of precedence:

1. **INADMISSIBLE**: gates failed, or were never evaluated for the current version.
   No estimate is reported, because a number from an experiment that cannot answer the
   question invites interpretation.
2. **DEVIATED**: the executed analysis differs from the registered one. Causes: an
   amendment after runs were linked, missing or undeclared seeds, duplicate runs per
   seed, runs rejected for peeking or config drift, undeclared non-finite values, or an
   explicit analysis override. The statistical outcome is reported alongside, labelled
   as such.
3. **CONFIRMED**: the adjusted p-value for the claim is below alpha. For bootstrap
   tests without correction this is equivalent to the interval lying entirely on the
   predicted side of the declared boundary.
4. **REFUTED**: the adjusted p-value for the negation of the claim is below alpha: the
   evidence is on the wrong side of the boundary. Note that for `a - b > 0.05` a
   significant effect of 0.02 refutes the claim as registered, even though the effect
   is positive.
5. **INCONCLUSIVE**: neither.

INCONCLUSIVE is not a failure of the protocol. A study powered for a 40-unit effect
that observes a 6-unit one has learned that the effect, if any, is smaller than
expected.

## 4. Exploratory analysis

Honest exploratory work is valuable and supported. Analyses that were not registered
are computed from the same linked runs and reported in a separate section labelled
"not pre-registered", with estimates and intervals but no p-values and no verdicts.
A useful exploratory finding becomes a new registered prediction tested on new seeds.

## 5. Tamper evidence and anchoring

Each ledger entry commits to its payload and to the previous entry. Editing, deleting,
or reordering entries breaks the chain, and `verify` reports the first entry affected.
A motivated editor can rewrite the whole chain with fresh hashes; that is caught only
by comparing against a head hash published before the rewrite. `preregister anchor` writes
the head hash to a git-tracked file (optionally committing it). The anchor is only as
trustworthy as where it is published: a commit pushed to a shared remote, an email to
a collaborator or advisor, a message in a team channel, or a line in a paper
submission are all outside the author's unilateral control.

## 6. Limitations

- **It cannot stop someone who never registers.** The protocol makes pre-registration
  cheap and verifiable, not mandatory. Its value to a reader depends on seeing the
  ledger and its anchors.
- **Timestamps are only as trustworthy as the anchor.** Ledger timestamps come from the
  local clock. Without an anchor published outside the author's control, a whole
  ledger can be fabricated after the fact.
- **Run metadata is self-reported.** A determined bad actor can relabel seeds, fake
  start times, or run the study privately before registering it. Tracker adapters
  reduce this surface (start times and states come from the tracker), but do not
  remove it.
- **Statistical assumptions remain.** Seeds are treated as independent replicates;
  percentile bootstrap intervals can undercover with few seeds; the normal
  approximation in the power gate is approximate, and the simulation assumes Gaussian
  data. Heavy-tailed or multimodal per-seed distributions, which are common in RL,
  deserve the simulation method or a rank-based test.
- **Gates are only as good as their authors.** A custom gate that checks the wrong
  thing produces false confidence.

## References

- R. Agarwal, M. Schwarzer, P. S. Castro, A. Courville, M. G. Bellemare. "Deep
  Reinforcement Learning at the Edge of the Statistical Precipice." NeurIPS 2021.
  (Interquartile mean and stratified bootstrap intervals for RL evaluation.)
- S. Holm. "A Simple Sequentially Rejective Multiple Test Procedure." Scandinavian
  Journal of Statistics 6(2), 1979.
- Y. Benjamini, Y. Hochberg. "Controlling the False Discovery Rate: A Practical and
  Powerful Approach to Multiple Testing." JRSS Series B 57(1), 1995.
- B. L. Welch. "The Generalization of 'Student's' Problem when Several Different
  Population Variances are Involved." Biometrika 34, 1947.
- H. B. Mann, D. R. Whitney. "On a Test of Whether One of Two Random Variables is
  Stochastically Larger than the Other." Annals of Mathematical Statistics 18, 1947.
- D. J. Schuirmann. "A Comparison of the Two One-Sided Tests Procedure and the Power
  Approach for Assessing the Equivalence of Average Bioavailability." Journal of
  Pharmacokinetics and Biopharmaceutics 15, 1987.
- B. A. Nosek, C. R. Ebersole, A. C. DeHaven, D. T. Mellor. "The Preregistration
  Revolution." PNAS 115(11), 2018.
- P. Henderson, R. Islam, P. Bachman, J. Pineau, D. Precup, D. Meger. "Deep
  Reinforcement Learning that Matters." AAAI 2018.

# Reconciliation report: bandit-algorithms

Generated 2026-09-16T23:40:32.338802Z by prereg 0.1.0. Ledger: 69 entries, head `643a8fe31504a563`. Integrity check: **PASSED**.

Verdicts are computed from the registered analysis plan only. Effect intervals are
percentile bootstrap intervals at level 1 - 2 alpha (the interval dual to the one-sided
test) and are not multiplicity-adjusted; verdicts use the adjusted p-values.

## Summary

| Prediction | Metric | Claim | Verdict | Estimate | Interval | p (claim, adj.) |
|---|---|---|---|---|---|---|
| P1 | cumulative_regret | `thompson - ucb < -20` | CONFIRMED | -76.3 | 90% [-83.18, -67.71] | 0.0003 |
| P2 | cumulative_regret | `ucb - eps_greedy < 0` | REFUTED | 80.12 | 90% [61.83, 98.03] | 1.0000 |
| P3 | cumulative_regret | `thompson - eps_greedy < 0` | INCONCLUSIVE | 6.628 | 90% [-12.36, 25.33] | 1.0000 |
| P4 | optimal_arm_rate | `thompson - eps_greedy > 0.1` | INADMISSIBLE | n/a | n/a | n/a |

## Timeline

| Entry | Time (UTC) | Event | Detail | Commit |
|---|---|---|---|---|
| 0 | 2026-09-16T23:40:12.551785Z | init | prereg 0.1.0 | e65d6b8ad9 |
| 1 | 2026-09-16T23:40:12.588980Z | register | P1 hash e05499515c3c | e65d6b8ad9 |
| 2 | 2026-09-16T23:40:12.620167Z | register | P2 hash 3735615c961e | e65d6b8ad9 |
| 3 | 2026-09-16T23:40:12.656981Z | register | P3 hash 7ae985801ddb | e65d6b8ad9 |
| 4 | 2026-09-16T23:40:12.694466Z | register | P4 hash d8da1126d126 | e65d6b8ad9 |
| 5 | 2026-09-16T23:40:12.881742Z | gate | P1: passed | e65d6b8ad9 |
| 6 | 2026-09-16T23:40:12.918858Z | gate | P2: passed | e65d6b8ad9 |
| 7 | 2026-09-16T23:40:12.950546Z | gate | P3: passed | e65d6b8ad9 |
| 8 | 2026-09-16T23:40:13.075068Z | gate | P4: FAILED | e65d6b8ad9 |
| 9-68 | 2026-09-16T23:40:31.852202Z | link 60 run(s) | eps_greedy: 20, ucb: 20, thompson: 20 | e65d6b8ad9 |
| 69 | 2026-09-16T23:40:32.635016Z | analysis | P1=CONFIRMED, P2=REFUTED, P3=INCONCLUSIVE, P4=INADMISSIBLE | e65d6b8ad9 |

## Predictions

### P1: Thompson sampling accumulates at least 20 less regret than UCB1.

- Claim: `thompson - ucb < -20` on `cumulative_regret` (direction: decrease, minimum effect of interest 25)
- Registered: 2026-09-16T23:40:12.588980Z; current version v1 at 2026-09-16T23:40:12.588980Z, hash `e05499515c3c50d6`
- Plan: bootstrap test, iqm over 20 seeds (0..19), alpha 0.05, correction holm (family `regret`)

**Verdict: CONFIRMED** - effect in the predicted direction; claim significant at the registered alpha.

- Estimate of iqm(thompson) - iqm(ucb): -76.3, interval 90% [-83.18, -67.71]
- p(claim) = <0.0001 (adjusted 0.0003); p(opposite) = 1.0000 (adjusted 1.0000)
- Analyzed runs: thompson=20, ucb=20

**Gates** (evaluated 2026-09-16T23:40:12.881742Z)

| Gate | Result | Message |
|---|---|---|
| MetricDefined | pass | metric 'cumulative_regret' fingerprint matches registration |
| ConfigFrozen | pass | condition configs match registered hashes |
| NoPeeking | pass | no linked runs predate registration |
| MinSeeds | pass | 20 seeds per condition planned, minimum 10 |
| PowerAnalysis | pass | power 0.91 at 20 seeds for margin 25 (target 0.80, ~15 required; alpha 0.01667, normal) |

**Per-seed values**

```
thompson [oo  o      oo@o@|   o @ o  o  o       o                     ] n=20
     ucb [                                        oo o  oo oo@| @@ o@o] n=20
          52.85                                                  177.4
```

| seed | thompson | ucb |
|---|---|---|
| 0 | 85.15 | 150.85 |
| 1 | 98.6 | 177.4 |
| 2 | 132.2 | 163.6 |
| 3 | 55.75 | 169.05 |
| 4 | 115.95 | 167.7 |
| 5 | 80.95 | 159.05 |
| 6 | 52.85 | 156.1 |
| 7 | 95.45 | 161.25 |
| 8 | 75.85 | 151.9 |
| 9 | 77.7 | 163.5 |
| 10 | 61.2 | 144.2 |
| 11 | 110.8 | 174.4 |
| 12 | 85.1 | 172.6 |
| 13 | 84.5 | 168.65 |
| 14 | 81.85 | 161.35 |
| 15 | 99.8 | 166.9 |
| 16 | 79.95 | 137.25 |
| 17 | 87.55 | 161.45 |
| 18 | 103.6 | 175 |
| 19 | 87.5 | 140 |

**Deviations**

None.

**Excluded runs (pre-declared criteria)**

None.

**Rejected runs**

None.

### P2: UCB1 accumulates less regret than epsilon-greedy at this horizon.

- Claim: `ucb - eps_greedy < 0` on `cumulative_regret` (direction: decrease, minimum effect of interest 40)
- Registered: 2026-09-16T23:40:12.620167Z; current version v1 at 2026-09-16T23:40:12.620167Z, hash `3735615c961e520b`
- Plan: welch test, mean over 20 seeds (0..19), alpha 0.05, correction holm (family `regret`)

**Verdict: REFUTED** - the opposite of the claim is significant at the registered alpha.

- Estimate of mean(ucb) - mean(eps_greedy): 80.12, interval 90% [61.83, 98.03]
- p(claim) = 1.0000 (adjusted 1.0000); p(opposite) = <0.0001 (adjusted <0.0001)
- Analyzed runs: ucb=20, eps_greedy=20

**Gates** (evaluated 2026-09-16T23:40:12.918858Z)

| Gate | Result | Message |
|---|---|---|
| MetricDefined | pass | metric 'cumulative_regret' fingerprint matches registration |
| ConfigFrozen | pass | condition configs match registered hashes |
| NoPeeking | pass | no linked runs predate registration |
| MinSeeds | pass | 20 seeds per condition planned, minimum 10 |
| ControlPresent | pass | control 'eps_greedy' declared and configured |
| PowerAnalysis | pass | power 0.87 at 20 seeds for margin 40 (target 0.80, ~17 required; alpha 0.01667, normal) |

**Per-seed values**

```
       ucb [                                        oo o @ oo| @ @@     ] n=20
eps_greedy [@oo@    oo   o      |      o  o @@o  oo                    o] n=20
            25.4                                                   190.2
```

| seed | ucb | eps_greedy |
|---|---|---|
| 0 | 150.85 | 118.45 |
| 1 | 177.4 | 116.05 |
| 2 | 163.6 | 25.4 |
| 3 | 169.05 | 34.3 |
| 4 | 167.7 | 100.25 |
| 5 | 159.05 | 32.15 |
| 6 | 156.1 | 127.4 |
| 7 | 161.25 | 61.35 |
| 8 | 151.9 | 32.4 |
| 9 | 163.5 | 109.1 |
| 10 | 144.2 | 120.75 |
| 11 | 174.4 | 26.2 |
| 12 | 172.6 | 35.05 |
| 13 | 168.65 | 190.2 |
| 14 | 161.35 | 50.45 |
| 15 | 166.9 | 114.3 |
| 16 | 137.25 | 118.8 |
| 17 | 161.45 | 131.75 |
| 18 | 175 | 48.05 |
| 19 | 140 | 27.35 |

**Deviations**

None.

**Excluded runs (pre-declared criteria)**

None.

**Rejected runs**

None.

### P3: Thompson sampling accumulates less regret than epsilon-greedy.

- Claim: `thompson - eps_greedy < 0` on `cumulative_regret` (direction: decrease, minimum effect of interest 45)
- Registered: 2026-09-16T23:40:12.656981Z; current version v1 at 2026-09-16T23:40:12.656981Z, hash `7ae985801ddbada0`
- Plan: bootstrap test, mean over 20 seeds (0..19), alpha 0.05, correction holm (family `regret`)

**Verdict: INCONCLUSIVE** - neither the claim nor its opposite is significant.

- Estimate of mean(thompson) - mean(eps_greedy): 6.628, interval 90% [-12.36, 25.33]
- p(claim) = 0.7207 (adjusted 1.0000); p(opposite) = 0.2794 (adjusted 0.5587)
- Analyzed runs: thompson=20, eps_greedy=20

**Gates** (evaluated 2026-09-16T23:40:12.950546Z)

| Gate | Result | Message |
|---|---|---|
| MetricDefined | pass | metric 'cumulative_regret' fingerprint matches registration |
| ConfigFrozen | pass | condition configs match registered hashes |
| NoPeeking | pass | no linked runs predate registration |
| MinSeeds | pass | 20 seeds per condition planned, minimum 10 |
| ControlPresent | pass | control 'eps_greedy' declared and configured |
| PowerAnalysis | pass | power 0.87 at 20 seeds for margin 45 (target 0.80, ~17 required; alpha 0.01667, normal) |

**Per-seed values**

```
  thompson [          oo o    oo@@|  oooo  oo     o                     ] n=20
eps_greedy [@oo@    oo   o      |      o  o @@o  oo                    o] n=20
            25.4                                                   190.2
```

| seed | thompson | eps_greedy |
|---|---|---|
| 0 | 85.15 | 118.45 |
| 1 | 98.6 | 116.05 |
| 2 | 132.2 | 25.4 |
| 3 | 55.75 | 34.3 |
| 4 | 115.95 | 100.25 |
| 5 | 80.95 | 32.15 |
| 6 | 52.85 | 127.4 |
| 7 | 95.45 | 61.35 |
| 8 | 75.85 | 32.4 |
| 9 | 77.7 | 109.1 |
| 10 | 61.2 | 120.75 |
| 11 | 110.8 | 26.2 |
| 12 | 85.1 | 35.05 |
| 13 | 84.5 | 190.2 |
| 14 | 81.85 | 50.45 |
| 15 | 99.8 | 114.3 |
| 16 | 79.95 | 118.8 |
| 17 | 87.55 | 131.75 |
| 18 | 103.6 | 48.05 |
| 19 | 87.5 | 27.35 |

**Deviations**

None.

**Excluded runs (pre-declared criteria)**

None.

**Rejected runs**

None.

### P4: Quick check on 5 seeds: Thompson sampling spends at least 10 percentage points more of the late phase on the best arm than epsilon-greedy.

- Claim: `thompson - eps_greedy > 0.1` on `optimal_arm_rate` (direction: increase, minimum effect of interest 0.1)
- Registered: 2026-09-16T23:40:12.694466Z; current version v1 at 2026-09-16T23:40:12.694466Z, hash `d8da1126d126452f`
- Plan: mann_whitney test, median over 5 seeds (0..4), alpha 0.05, correction none (family `default`)

**Verdict: INADMISSIBLE** - admissibility gates failed or were not evaluated; analysis withheld.

- Note: 30 linked run(s) use seeds planned by other predictions
- Note: admissibility gates failed; analysis withheld

**Gates** (evaluated 2026-09-16T23:40:13.075068Z)

| Gate | Result | Message |
|---|---|---|
| MetricDefined | pass | metric 'optimal_arm_rate' fingerprint matches registration |
| ConfigFrozen | pass | condition configs match registered hashes |
| NoPeeking | pass | no linked runs predate registration |
| MinSeeds | pass | 5 seeds per condition planned, minimum 3 |
| PowerAnalysis | FAIL | power 0.09 at 5 seeds for margin 0.1 (target 0.80, ~278 required; alpha 0.05, normal) |

**Deviations**

None.

**Excluded runs (pre-declared criteria)**

None.

**Rejected runs**

None.

## Exploratory analyses (not pre-registered)

These analyses were not registered before the runs. They carry no verdict, no
p-value, and should be read as hypothesis-generating only.

| Id | Metric | Comparison | Aggregation | Estimate | Interval | n | Note |
|---|---|---|---|---|---|---|---|
| E1 | optimal_arm_rate | `thompson - eps_greedy > 0` | median | -0.094 | 95% [-0.281, 0.668] | 40 |  |
| E2 | cumulative_regret | `ucb / thompson > 1` | iqm | 1.88 | 95% [1.702, 2.056] | 40 |  |

## Ledger integrity

All 69 entries verified: hash chain intact, payload and registration hashes match, anchors present.

# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed

- Renamed the distribution, import package, and CLI from `prereg-ml` / `prereg` to
  `preregister`, to avoid clashing with the unrelated `prereg` package on PyPI. Project
  files (`.prereg/`, `prereg.yaml`) and the ledger schema are unchanged, so existing
  ledgers still verify.
- Run times without a UTC offset are rejected instead of being read as UTC, so local
  times can no longer move runs across the registration time that `NoPeeking` checks.
- Seeds must be integers; `3.9` is rejected instead of being truncated to 3.
- `analyze` refuses to record an analysis on a ledger that fails verification.

### Fixed

- Ratio claims (`a / b >= r`) with a zero or negative denominator gave wrong verdicts,
  because the test rewrites the claim as `a - r*b >= 0`. Such predictions are now
  `INCONCLUSIVE` with a note, and no statistics are computed.
- CSV run files with a byte-order mark (as Excel writes them) are read correctly.
- Malformed YAML, run files, timestamps, and anchor lines now give a one-line error
  instead of a traceback. A JSON object without a `runs` key is an error rather than
  zero runs.

## [0.1.0] - 2026-09-16

### Added

- Study specification in Python or YAML: conditions, versioned metrics, directional
  comparisons (`a - b > t`, `a > b`, `a / b >= r`, `a > t`, `a in [lo, hi]`), analysis
  plans, stopping rules, exclusion criteria, and gates.
- Append-only, SHA-256 hash-chained ledger (`.prereg/ledger.jsonl`) with canonical JSON
  hashing, git commit and dirty-flag capture, amendments with reasons, verification,
  and head-hash anchoring.
- Admissibility gates: `MinSeeds`, `PowerAnalysis` (normal approximation or simulation),
  `ControlPresent`, `MetricDefined`, `ConfigFrozen`, `NoPeeking`, and the `custom_gate`
  decorator.
- Run linking from JSON, CSV, Weights & Biases, and MLflow; per-run rejection and
  pre-declared exclusions, all recorded.
- Statistics without a scipy runtime dependency: percentile bootstrap, Welch and
  one-sample t-tests, Mann-Whitney U (exact and asymptotic), permutation and sign-flip
  tests, IQM, Holm and Benjamini-Hochberg corrections.
- Reconciliation with verdicts CONFIRMED, REFUTED, INCONCLUSIVE, INADMISSIBLE, and
  DEVIATED; exploratory analyses reported separately.
- Markdown and self-contained HTML reports with per-seed plots.
- CLI: `init`, `register`, `amend`, `gate`, `link`, `analyze`, `report`, `verify`,
  `anchor`, `status`.
- Examples: a bandit study with its committed ledger and report, an integrity demo
  (peeking, seed-picking, tampering), and a custom environment admissibility gate.

# Contributing

Thanks for considering a contribution. Issues describing a failure mode that the
protocol does not catch are as welcome as code.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

## Checks

```bash
ruff check .
ruff format --check .
pytest
```

The full test suite should stay well under a minute. Tests must not touch the network;
tracker adapters are tested against fake clients.

## Guidelines

- Statistical code is checked against scipy reference values in the tests. New tests or
  estimators need the same, or a hand-computed case with the derivation in the test.
- Anything that changes the canonical form of a registration (the frozen payload, the
  ledger entry header, or canonical JSON) breaks verification of existing ledgers. Such
  changes need a schema version bump and a migration note in `CHANGELOG.md`.
- New gates should state in their docstring what failure they prevent, and fail closed:
  an exception inside a gate is a failed gate.
- Keep runtime dependencies minimal. Tracker integrations are optional extras and must
  import lazily.

## Releasing

Update the version in `pyproject.toml` and `src/preregister/__init__.py`, add a
`CHANGELOG.md` entry, and tag the release.

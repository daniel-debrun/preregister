"""Register the kitchen study in a temporary directory and evaluate its gates.

Equivalent CLI, run from this directory:

    preregister --root /tmp/kitchen --spec prereg.yaml --plugin kitchen_gate init
    preregister --root /tmp/kitchen --spec prereg.yaml --plugin kitchen_gate register
    preregister --root /tmp/kitchen --spec prereg.yaml --plugin kitchen_gate gate
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import kitchen_gate  # noqa: E402,F401  (registers the RecipesCompletable gate)

from preregister import Project, Study  # noqa: E402


def main() -> int:
    study = Study.from_yaml(HERE / "prereg.yaml")
    with tempfile.TemporaryDirectory() as tmp:
        project = Project.init(tmp)
        project.register(study)
        all_passed = True
        for pid, results in project.run_gates(study).items():
            passed = all(r.passed for r in results)
            all_passed &= passed
            print(f"{pid}: {'PASS' if passed else 'FAIL'}")
            for r in results:
                print(f"    [{'pass' if r.passed else 'FAIL'}] {r.gate}: {r.message}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

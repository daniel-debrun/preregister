from __future__ import annotations

import math
from pathlib import Path

from preregister import AnalysisPlan, ExploratorySpec, Prediction, Project
from preregister.reconcile import RunValue
from preregister.report import render_html, render_markdown, text_dot_plot, write_report

from .conftest import FakeClock, make_runs, make_study, normal_values


def _project(tmp_path: Path, clock: FakeClock) -> Project:
    preds = [
        Prediction(
            "P1",
            "treatment <beats> control",
            "score",
            "treatment - control > 0",
            0.5,
            analysis=AnalysisPlan(test="bootstrap", aggregation="iqm", n_seeds=10, n_boot=1000),
        ),
        Prediction(
            "P2",
            "underpowered",
            "score",
            "treatment > control",
            0.5,
            analysis=AnalysisPlan(n_seeds=10, n_boot=1000),
            gates=({"type": "MinSeeds", "n": 50},),
        ),
    ]
    study = make_study(tmp_path, preds)
    project = Project.init(tmp_path, clock=clock)
    project.register(study)
    project.run_gates(study)
    changed = study.with_prediction(
        Prediction.from_dict(
            {**preds[0].to_dict(), "hypothesis": "treatment <clearly> beats | control"}
        )
    )
    project.amend(changed, "P1", reason="clarified wording")
    project.run_gates(changed)
    runs = make_runs(
        {"treatment": normal_values(2, 1, 10, 1), "control": normal_values(0, 1, 10, 2)},
        clock.advance(hours=1),
    )
    runs[0] = runs[0].__class__(**{**runs[0].__dict__, "metrics": {"score": math.nan}})
    project.link_runs(runs)
    return project


def test_markdown_report_sections(tmp_path: Path, clock: FakeClock) -> None:
    project = _project(tmp_path, clock)
    analysis = project.analyze(
        record=False,
        exploratory=[ExploratorySpec("E1", "score", "treatment > control", n_boot=500)],
    )
    md = render_markdown(analysis, project.state())
    for heading in (
        "## Summary",
        "## Timeline",
        "## Predictions",
        "## Exploratory analyses (not pre-registered)",
        "## Ledger integrity",
    ):
        assert heading in md
    assert "Integrity check: **PASSED**" in md
    assert "DEVIATED (statistically CONFIRMED)" in md
    assert "INADMISSIBLE" in md and "analysis withheld" in md
    assert "clarified wording" in md
    assert "### P1: treatment <clearly> beats | control" in md
    assert "metric value is not finite" in md
    assert "link 20 run(s)" in md
    assert "**Per-seed values**" in md and "n=9" in md


def test_html_report_is_self_contained(tmp_path: Path, clock: FakeClock) -> None:
    project = _project(tmp_path, clock)
    analysis = project.analyze(record=False)
    page = render_html(analysis, project.state())
    assert page.startswith("<!DOCTYPE html>") and page.rstrip().endswith("</html>")
    assert "<script" not in page and "http://" not in page and "https://" not in page
    assert "<svg" in page and "<circle" in page
    assert "treatment &lt;clearly&gt; beats | control" in page
    path = write_report(analysis, project.state(), tmp_path / "out" / "report.html")
    assert path.read_text().startswith("<!DOCTYPE html>")
    md_path = write_report(analysis, project.state(), tmp_path / "report.md")
    assert md_path.read_text().startswith("# Reconciliation report")


def test_report_shows_failed_integrity(tmp_path: Path, clock: FakeClock) -> None:
    project = _project(tmp_path, clock)
    lines = project.ledger.path.read_text().splitlines()
    lines[-1] = lines[-1].replace('"forced": false', '"forced": true')
    project.ledger.path.write_text("\n".join(lines) + "\n")
    md = render_markdown(project.analyze(record=False), project.state())
    assert "Integrity check: **FAILED**" in md and "payload was modified" in md


def test_text_dot_plot() -> None:
    values = [RunValue(f"r{i}", "a", i, float(i)) for i in range(5)] + [
        RunValue("b0", "bb", 0, 4.0),
        RunValue("b1", "bb", 1, 4.0),
    ]
    plot = text_dot_plot(values, ["a", "bb"], "mean", width=21)
    lines = plot.splitlines()
    assert lines[0].startswith(" a [o") and lines[0].endswith("] n=5")
    assert "|" in lines[1] and lines[1].endswith("n=2")
    assert text_dot_plot([], ["a"], "mean") == "(no analyzable runs)"

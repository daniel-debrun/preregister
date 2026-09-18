"""Markdown and single-file HTML reconciliation reports."""

from __future__ import annotations

import html
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from preregister.analysis import aggregate
from preregister.reconcile import (
    ExploratoryResult,
    PredictionResult,
    RunValue,
    StudyAnalysis,
    Verdict,
)
from preregister.registry import Comparison
from preregister.state import LedgerState

VERDICT_TEXT = {
    Verdict.CONFIRMED: "effect in the predicted direction; claim significant at the registered alpha",
    Verdict.REFUTED: "the opposite of the claim is significant at the registered alpha",
    Verdict.INCONCLUSIVE: "neither the claim nor its opposite is significant",
    Verdict.INADMISSIBLE: "admissibility gates failed or were not evaluated; analysis withheld",
    Verdict.DEVIATED: "the executed analysis differs from the registered plan",
}


def _num(x: float | None, digits: int = 4) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{x:.{digits}g}"


def _p(x: float | None) -> str:
    if x is None:
        return "n/a"
    return "<0.0001" if x < 1e-4 else f"{x:.4f}"


def _short(h: str | None, n: int = 12) -> str:
    return "n/a" if not h else h[:n]


def _effect_label(result: PredictionResult) -> str:
    cmp = Comparison.parse(result.comparison)
    agg = result.plan["aggregation"]
    if cmp.kind == "difference":
        return f"{agg}({cmp.left}) - {agg}({cmp.right})"
    if cmp.kind == "ratio":
        return f"{agg}({cmp.left}) / {agg}({cmp.right})"
    return f"{agg}({cmp.left})"


def _ci(result: PredictionResult | ExploratoryResult) -> str:
    if result.ci_low is None or result.ci_level is None:
        return "n/a"
    return f"{result.ci_level:.0%} [{_num(result.ci_low)}, {_num(result.ci_high)}]"


def _verdict_cell(result: PredictionResult) -> str:
    if result.verdict == Verdict.DEVIATED and result.statistical_outcome is not None:
        return f"DEVIATED (statistically {result.statistical_outcome.value})"
    return result.verdict.value


def _timeline(state: LedgerState) -> list[tuple[str, str, str, str, str]]:
    rows: list[tuple[str, str, str, str, str]] = []
    pending: list[Any] = []

    def flush() -> None:
        if not pending:
            return
        counts: dict[str, int] = {}
        for e in pending:
            cond = e.payload["run"]["condition"]
            counts[cond] = counts.get(cond, 0) + 1
        detail = ", ".join(f"{c}: {n}" for c, n in counts.items())
        span = (
            f"{pending[0].index}-{pending[-1].index}" if len(pending) > 1 else str(pending[0].index)
        )
        rows.append(
            (
                span,
                pending[0].timestamp,
                f"link {len(pending)} run(s)",
                detail,
                _commit(pending[-1]),
            )
        )
        pending.clear()

    for e in state.entries:
        if e.kind == "link":
            pending.append(e)
            continue
        flush()
        p = e.payload
        if e.kind == "init":
            detail = f"preregister {p.get('prereg_version')}"
        elif e.kind == "register":
            detail = f"{p['prediction_id']} hash {_short(p['prediction_hash'])}"
        elif e.kind == "amend":
            detail = f"{p['prediction_id']} hash {_short(p['prediction_hash'])}; reason: {p.get('reason')}"
        elif e.kind == "gate":
            detail = f"{p['prediction_id']}: {'passed' if p['passed'] else 'FAILED'}"
        elif e.kind == "analysis":
            detail = ", ".join(f"{k}={v}" for k, v in p.get("verdicts", {}).items())
        else:
            detail = ""
        rows.append((str(e.index), e.timestamp, e.kind, detail, _commit(e)))
    flush()
    return rows


def _commit(entry: Any) -> str:
    commit = entry.git.get("commit")
    if not commit:
        return "n/a"
    return commit[:10] + (" (dirty)" if entry.git.get("dirty") else "")


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    def cell(x: Any) -> str:
        return str(x).replace("|", "\\|").replace("\n", " ")

    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(cell(c) for c in row) + " |" for row in rows]
    return out


def text_dot_plot(
    values: Sequence[RunValue], conditions: Sequence[str], aggregation: str, width: int = 60
) -> str:
    """Per-seed strip plot as plain text: ``o`` one run, ``@`` several, ``|`` the aggregate."""
    all_vals = [v.value for v in values]
    if not all_vals:
        return "(no analyzable runs)"
    lo, hi = min(all_vals), max(all_vals)
    span = hi - lo or 1.0
    label_w = max(len(c) for c in conditions)

    def pos(x: float) -> int:
        return round((x - lo) / span * (width - 1))

    lines = []
    for cond in conditions:
        vals = [v.value for v in values if v.condition == cond]
        row = [" "] * width
        for x in vals:
            i = pos(x)
            row[i] = "@" if row[i] in ("o", "@") else "o"
        if vals:
            row[pos(float(aggregate(np.array(vals), aggregation)))] = "|"
        lines.append(f"{cond.rjust(label_w)} [{''.join(row)}] n={len(vals)}")
    lines.append(
        f"{' ' * label_w}  {_num(lo)}{' ' * max(1, width - len(_num(lo)) - len(_num(hi)))}{_num(hi)}"
    )
    return "\n".join(lines)


def _per_seed_rows(values: Sequence[RunValue], conditions: Sequence[str]) -> list[list[str]]:
    seeds = sorted({v.seed for v in values})
    rows = []
    for seed in seeds:
        row = [str(seed)]
        for cond in conditions:
            vals = [_num(v.value, 5) for v in values if v.condition == cond and v.seed == seed]
            row.append(", ".join(vals) if vals else "-")
        rows.append(row)
    return rows


def render_markdown(analysis: StudyAnalysis, state: LedgerState) -> str:
    from preregister import __version__

    integ = analysis.integrity or {}
    out = [
        f"# Reconciliation report: {analysis.study}",
        "",
        f"Generated {analysis.generated_at} by preregister {__version__}. "
        f"Ledger: {analysis.n_entries} entries, head `{_short(analysis.ledger_head, 16)}`. "
        f"Integrity check: **{'PASSED' if integ.get('ok', False) else 'FAILED'}**"
        + (f" ({len(integ.get('issues', []))} issue(s))" if integ.get("issues") else "")
        + ".",
        "",
        "Verdicts are computed from the registered analysis plan only. Effect intervals are",
        "percentile bootstrap intervals at level 1 - 2 alpha (the interval dual to the one-sided",
        "test) and are not multiplicity-adjusted; verdicts use the adjusted p-values.",
        "",
        "## Summary",
        "",
    ]
    out += _md_table(
        ["Prediction", "Metric", "Claim", "Verdict", "Estimate", "Interval", "p (claim, adj.)"],
        [
            [
                r.prediction_id,
                r.metric,
                f"`{r.comparison}`",
                _verdict_cell(r),
                _num(r.estimate),
                _ci(r),
                _p(r.p_confirm_adj),
            ]
            for r in analysis.predictions
        ],
    )
    out += ["", "## Timeline", ""]
    out += _md_table(["Entry", "Time (UTC)", "Event", "Detail", "Commit"], _timeline(state))
    out += ["", "## Predictions", ""]
    for r in analysis.predictions:
        out += _md_prediction(r)
    out += ["## Exploratory analyses (not pre-registered)", ""]
    if analysis.exploratory:
        out += [
            "These analyses were not registered before the runs. They carry no verdict, no",
            "p-value, and should be read as hypothesis-generating only.",
            "",
        ]
        out += _md_table(
            ["Id", "Metric", "Comparison", "Aggregation", "Estimate", "Interval", "n", "Note"],
            [
                [
                    e.id,
                    e.metric,
                    f"`{e.comparison}`",
                    e.aggregation,
                    _num(e.estimate),
                    _ci(e),
                    len(e.values),
                    e.note,
                ]
                for e in analysis.exploratory
            ],
        )
    else:
        out.append("None.")
    out += ["", "## Ledger integrity", ""]
    issues = integ.get("issues", [])
    if not issues:
        out.append(
            f"All {integ.get('n_entries', 0)} entries verified: hash chain intact, payload and "
            "registration hashes match, anchors present."
        )
    else:
        out += [f"- [{i['severity']}] entry {i['index']}: {i['message']}" for i in issues]
    out.append("")
    return "\n".join(out)


def _md_prediction(r: PredictionResult) -> list[str]:
    plan = r.plan
    cmp = Comparison.parse(r.comparison)
    seeds = plan["seeds"]
    seed_text = (
        f"{seeds[0]}..{seeds[-1]}"
        if seeds == list(range(seeds[0], seeds[0] + len(seeds)))
        else str(seeds)
    )
    out = [
        f"### {r.prediction_id}: {r.hypothesis}",
        "",
        f"- Claim: `{r.comparison}` on `{r.metric}` (direction: {cmp.direction}, minimum effect of interest {_num(r.min_effect)})",
        f"- Registered: {r.first_registered_at}; current version v{r.version} at {r.registered_at}, hash `{_short(r.prediction_hash, 16)}`",
        f"- Plan: {plan['test']} test, {plan['aggregation']} over {plan['n_seeds']} seeds ({seed_text}), "
        f"alpha {plan['alpha']}, correction {plan['correction']} (family `{plan['family']}`)",
        "",
        f"**Verdict: {_verdict_cell(r)}** - {VERDICT_TEXT[r.verdict]}.",
        "",
    ]
    if r.estimate is not None or r.p_confirm is not None:
        out += [
            f"- Estimate of {_effect_label(r)}: {_num(r.estimate)}, interval {_ci(r)}",
            f"- p(claim) = {_p(r.p_confirm)} (adjusted {_p(r.p_confirm_adj)}); "
            f"p(opposite) = {_p(r.p_refute)} (adjusted {_p(r.p_refute_adj)})",
            f"- Analyzed runs: {', '.join(f'{c}={n}' for c, n in r.n_by_condition.items())}",
            "",
        ]
    for note in r.notes:
        out.append(f"- Note: {note}")
    if r.notes:
        out.append("")
    out += [f"**Gates** (evaluated {r.gates_evaluated_at or 'never'})", ""]
    if r.gate_results:
        out += _md_table(
            ["Gate", "Result", "Message"],
            [[g["gate"], "pass" if g["passed"] else "FAIL", g["message"]] for g in r.gate_results],
        )
    else:
        out.append("No gate evaluation recorded for this version.")
    out.append("")
    if r.values and r.verdict != Verdict.INADMISSIBLE:
        out += [
            "**Per-seed values**",
            "",
            "```",
            text_dot_plot(r.values, cmp.conditions, plan["aggregation"]),
            "```",
            "",
        ]
        out += _md_table(["seed", *cmp.conditions], _per_seed_rows(r.values, cmp.conditions))
        out.append("")
    out += ["**Deviations**", ""]
    out += [f"- {d}" for d in r.deviations] or ["None."]
    out.append("")
    if r.amendments:
        out += ["**Amendments**", ""]
        out += _md_table(
            ["Version", "Time (UTC)", "After runs linked", "Reason"],
            [
                [
                    f"v{a['version']}",
                    a["timestamp"],
                    "yes" if a["after_runs_linked"] else "no",
                    a["reason"],
                ]
                for a in r.amendments
            ],
        )
        out.append("")
    for title, rows in (
        ("Excluded runs (pre-declared criteria)", r.excluded),
        ("Rejected runs", r.rejected),
    ):
        out += [f"**{title}**", ""]
        if rows:
            out += _md_table(
                ["Run", "Condition", "Seed", "Reason"],
                [[x.run_id, x.condition, x.seed, x.reason] for x in rows],
            )
        else:
            out.append("None.")
        out.append("")
    return out


_CSS = """
.viz-root{color-scheme:light;--surface-1:#fcfcfb;--text-primary:#0b0b0b;--text-secondary:#52514e;
--rule:#dddcd6;--series-1:#2a78d6;--good:#1f7a3a;--bad:#b3261e;--warn:#8a5a00;
font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif;background:var(--surface-1);color:var(--text-primary);
max-width:980px;margin:0 auto;padding:24px}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{color-scheme:dark;
--surface-1:#1a1a19;--text-primary:#ffffff;--text-secondary:#c3c2b7;--rule:#3a3a37;--series-1:#3987e5;
--good:#6fcf8a;--bad:#f28b82;--warn:#e0b050}}
.viz-root table{border-collapse:collapse;margin:8px 0 16px;width:100%}
.viz-root th,.viz-root td{border-bottom:1px solid var(--rule);padding:4px 8px;text-align:left;vertical-align:top}
.viz-root code{font-size:13px}.viz-root .muted{color:var(--text-secondary)}
.viz-root .verdict{font-weight:600}.viz-root .CONFIRMED{color:var(--good)}.viz-root .REFUTED{color:var(--bad)}
.viz-root .DEVIATED,.viz-root .INADMISSIBLE{color:var(--warn)}
.viz-root section{border-top:1px solid var(--rule);margin-top:24px}
.viz-root svg text{fill:var(--text-secondary);font-size:12px}
"""


def svg_dot_plot(
    values: Sequence[RunValue], conditions: Sequence[str], aggregation: str, width: int = 640
) -> str:
    """Inline SVG strip plot: one labelled row per condition, one dot per run, a tick at the aggregate."""
    all_vals = [v.value for v in values]
    if not all_vals:
        return "<p class='muted'>No analyzable runs.</p>"
    lo, hi = min(all_vals), max(all_vals)
    pad = (hi - lo) * 0.05 or 1.0
    lo, hi = lo - pad, hi + pad
    left, right, row_h = 120, 20, 44
    height = row_h * len(conditions) + 30
    plot_w = width - left - right

    def x(v: float) -> float:
        return left + (v - lo) / (hi - lo) * plot_w

    parts = [
        f'<svg role="img" aria-label="Per-seed values" viewBox="0 0 {width} {height}" width="100%" '
        f'style="max-width:{width}px">'
    ]
    for i, cond in enumerate(conditions):
        cy = 16 + i * row_h + row_h / 2
        parts.append(
            f'<text x="{left - 10}" y="{cy + 4:.1f}" text-anchor="end">{html.escape(cond)}</text>'
        )
        parts.append(
            f'<line x1="{left}" x2="{width - right}" y1="{cy:.1f}" y2="{cy:.1f}" stroke="var(--rule)" stroke-width="1"/>'
        )
        vals = [v for v in values if v.condition == cond]
        for j, v in enumerate(vals):
            jitter = ((j * 7919) % 11 - 5) * 1.6
            parts.append(
                f'<circle cx="{x(v.value):.1f}" cy="{cy + jitter:.1f}" r="4" fill="var(--series-1)" '
                f'fill-opacity="0.75" stroke="var(--surface-1)" stroke-width="1">'
                f"<title>{html.escape(v.run_id)} seed {v.seed}: {v.value:.6g}</title></circle>"
            )
        if vals:
            agg = float(aggregate(np.array([v.value for v in vals]), aggregation))
            parts.append(
                f'<line x1="{x(agg):.1f}" x2="{x(agg):.1f}" y1="{cy - 14:.1f}" y2="{cy + 14:.1f}" '
                f'stroke="var(--text-primary)" stroke-width="2"><title>{aggregation} = {agg:.6g}</title></line>'
            )
    axis_y = height - 8
    for frac in (0.0, 0.5, 1.0):
        val = lo + frac * (hi - lo)
        anchor = {0.0: "start", 0.5: "middle", 1.0: "end"}[frac]
        parts.append(
            f'<text x="{left + frac * plot_w:.1f}" y="{axis_y}" text-anchor="{anchor}">{val:.4g}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _html_table(
    headers: Sequence[str], rows: Sequence[Sequence[Any]], raw_cols: Sequence[int] = ()
) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join(
        "<tr>"
        + "".join(
            f"<td>{c if i in raw_cols else html.escape(str(c))}</td>" for i, c in enumerate(row)
        )
        + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def render_html(analysis: StudyAnalysis, state: LedgerState) -> str:
    from preregister import __version__

    esc = html.escape
    integ = analysis.integrity or {}
    ok = integ.get("ok", False)
    parts = [
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>",
        f"<title>Reconciliation report: {esc(analysis.study)}</title>",
        f"<style>{_CSS}</style></head><body class='viz-root'>",
        f"<h1>Reconciliation report: {esc(analysis.study)}</h1>",
        f"<p class='muted'>Generated {esc(analysis.generated_at)} by preregister {__version__}. "
        f"Ledger: {analysis.n_entries} entries, head <code>{esc(_short(analysis.ledger_head, 16))}</code>. "
        f"Integrity check: <strong class='{'CONFIRMED' if ok else 'REFUTED'}'>{'PASSED' if ok else 'FAILED'}</strong>.</p>",
        "<p class='muted'>Intervals are percentile bootstrap intervals at level 1 - 2 alpha, not "
        "multiplicity-adjusted; verdicts use adjusted p-values.</p>",
        "<h2>Summary</h2>",
        _html_table(
            ["Prediction", "Metric", "Claim", "Verdict", "Estimate", "Interval", "p (claim, adj.)"],
            [
                [
                    esc(r.prediction_id),
                    esc(r.metric),
                    f"<code>{esc(r.comparison)}</code>",
                    f"<span class='verdict {r.verdict.value}'>{esc(_verdict_cell(r))}</span>",
                    _num(r.estimate),
                    esc(_ci(r)),
                    esc(_p(r.p_confirm_adj)),
                ]
                for r in analysis.predictions
            ],
            raw_cols=(0, 1, 2, 3, 4, 5, 6),
        ),
        "<h2>Timeline</h2>",
        _html_table(["Entry", "Time (UTC)", "Event", "Detail", "Commit"], _timeline(state)),
    ]
    for r in analysis.predictions:
        cmp = Comparison.parse(r.comparison)
        plan = r.plan
        parts += [
            f"<section><h3>{esc(r.prediction_id)}: {esc(r.hypothesis)}</h3>",
            f"<p>Claim <code>{esc(r.comparison)}</code> on <code>{esc(r.metric)}</code>; "
            f"{esc(plan['test'])} test, {esc(plan['aggregation'])} over {plan['n_seeds']} seeds, alpha {plan['alpha']}, "
            f"correction {esc(plan['correction'])}. Registered {esc(r.first_registered_at)}, "
            f"v{r.version} hash <code>{esc(_short(r.prediction_hash, 16))}</code>.</p>",
            f"<p class='verdict {r.verdict.value}'>Verdict: {esc(_verdict_cell(r))}</p>",
            f"<p class='muted'>{esc(VERDICT_TEXT[r.verdict])}.</p>",
        ]
        if r.estimate is not None or r.p_confirm is not None:
            parts.append(
                f"<p>Estimate of {esc(_effect_label(r))}: <strong>{_num(r.estimate)}</strong>, interval {esc(_ci(r))}. "
                f"p(claim) {esc(_p(r.p_confirm))} (adj. {esc(_p(r.p_confirm_adj))}); "
                f"p(opposite) {esc(_p(r.p_refute))} (adj. {esc(_p(r.p_refute_adj))}).</p>"
            )
        parts += [f"<p class='muted'>Note: {esc(n)}</p>" for n in r.notes]
        parts.append("<h4>Gates</h4>")
        parts.append(
            _html_table(
                ["Gate", "Result", "Message"],
                [
                    [g["gate"], "pass" if g["passed"] else "FAIL", g["message"]]
                    for g in r.gate_results
                ],
            )
            if r.gate_results
            else "<p>No gate evaluation recorded for this version.</p>"
        )
        if r.values and r.verdict != Verdict.INADMISSIBLE:
            parts += [
                "<h4>Per-seed values</h4>",
                svg_dot_plot(r.values, cmp.conditions, plan["aggregation"]),
                "<details><summary>Table</summary>",
                _html_table(["seed", *cmp.conditions], _per_seed_rows(r.values, cmp.conditions)),
                "</details>",
            ]
        parts.append("<h4>Deviations</h4>")
        parts.append(
            "<ul>" + "".join(f"<li>{esc(d)}</li>" for d in r.deviations) + "</ul>"
            if r.deviations
            else "<p>None.</p>"
        )
        if r.amendments:
            parts.append("<h4>Amendments</h4>")
            parts.append(
                _html_table(
                    ["Version", "Time (UTC)", "After runs linked", "Reason"],
                    [
                        [
                            f"v{a['version']}",
                            a["timestamp"],
                            "yes" if a["after_runs_linked"] else "no",
                            a["reason"],
                        ]
                        for a in r.amendments
                    ],
                )
            )
        for title, rows in (("Excluded runs", r.excluded), ("Rejected runs", r.rejected)):
            parts.append(f"<h4>{title}</h4>")
            parts.append(
                _html_table(
                    ["Run", "Condition", "Seed", "Reason"],
                    [[x.run_id, x.condition, x.seed, x.reason] for x in rows],
                )
                if rows
                else "<p>None.</p>"
            )
        parts.append("</section>")
    parts.append("<section><h2>Exploratory analyses (not pre-registered)</h2>")
    if analysis.exploratory:
        parts.append(
            "<p class='muted'>Not registered before the runs; no verdicts or p-values.</p>"
        )
        parts.append(
            _html_table(
                ["Id", "Metric", "Comparison", "Aggregation", "Estimate", "Interval", "n", "Note"],
                [
                    [
                        e.id,
                        e.metric,
                        e.comparison,
                        e.aggregation,
                        _num(e.estimate),
                        _ci(e),
                        len(e.values),
                        e.note,
                    ]
                    for e in analysis.exploratory
                ],
            )
        )
    else:
        parts.append("<p>None.</p>")
    parts.append("</section><section><h2>Ledger integrity</h2>")
    issues = integ.get("issues", [])
    parts.append(
        "<ul>"
        + "".join(
            f"<li>[{esc(i['severity'])}] entry {i['index']}: {esc(i['message'])}</li>"
            for i in issues
        )
        + "</ul>"
        if issues
        else f"<p>All {integ.get('n_entries', 0)} entries verified.</p>"
    )
    parts.append("</section></body></html>")
    return "\n".join(parts)


def write_report(
    analysis: StudyAnalysis, state: LedgerState, path: str | Path, fmt: str | None = None
) -> Path:
    path = Path(path)
    fmt = fmt or ("html" if path.suffix.lower() in (".html", ".htm") else "md")
    text = render_html(analysis, state) if fmt == "html" else render_markdown(analysis, state)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path

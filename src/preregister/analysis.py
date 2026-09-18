"""Statistical procedures used by the registered analysis.

Everything here is implemented on numpy and the standard library so that the
runtime does not depend on scipy; the test suite checks these functions against
scipy reference values.

References
----------
- IQM: R. Agarwal, M. Schwarzer, P. S. Castro, A. Courville, M. G. Bellemare.
  "Deep Reinforcement Learning at the Edge of the Statistical Precipice." NeurIPS 2021.
- Holm: S. Holm. "A Simple Sequentially Rejective Multiple Test Procedure."
  Scandinavian Journal of Statistics, 1979.
- BH: Y. Benjamini, Y. Hochberg. "Controlling the False Discovery Rate."
  JRSS-B, 1995.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from statistics import NormalDist

import numpy as np

from preregister.registry import AnalysisPlan, Comparison

_MAX_BLOCK = 4_000_000
_STD_NORMAL = NormalDist()


def iqm(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Interquartile mean: mean of the middle 50% (``scipy.stats.trim_mean(x, 0.25)``)."""
    arr = np.sort(np.asarray(x, dtype=float), axis=axis)
    n = arr.shape[axis]
    k = int(0.25 * n)
    return np.take(arr, np.arange(k, n - k), axis=axis).mean(axis=axis)


AGGREGATORS: dict[str, Callable[..., np.ndarray]] = {
    "mean": lambda x, axis=-1: np.mean(x, axis=axis),
    "median": lambda x, axis=-1: np.median(x, axis=axis),
    "iqm": iqm,
}


def aggregate(x: np.ndarray, method: str, axis: int = -1) -> np.ndarray:
    try:
        return AGGREGATORS[method](np.asarray(x, dtype=float), axis=axis)
    except KeyError:
        raise ValueError(f"unknown aggregation {method!r}") from None


def _blocks(total: int, width: int) -> list[int]:
    per = max(1, _MAX_BLOCK // max(width, 1))
    sizes = [per] * (total // per)
    if total % per:
        sizes.append(total % per)
    return sizes


def bootstrap_aggregates(
    x: np.ndarray, aggregation: str, n_boot: int, rng: np.random.Generator
) -> np.ndarray:
    """Aggregate of ``n_boot`` resamples-with-replacement of ``x`` (vectorized, chunked)."""
    x = np.asarray(x, dtype=float)
    out = [
        aggregate(x[rng.integers(0, x.size, size=(size, x.size))], aggregation, axis=1)
        for size in _blocks(n_boot, x.size)
    ]
    return np.concatenate(out)


def percentile_ci(samples: np.ndarray, level: float) -> tuple[float, float]:
    tail = (1.0 - level) / 2.0
    lo, hi = np.quantile(samples, [tail, 1.0 - tail])
    return float(lo), float(hi)


@dataclass(frozen=True)
class TestResult:
    statistic: float
    p_value: float
    df: float | None = None


def _betacf(a: float, b: float, x: float) -> float:
    fpmin = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > fpmin else fpmin)
    h = d
    for m in range(1, 10_000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > fpmin else fpmin)
        c = 1.0 + aa / c
        c = c if abs(c) > fpmin else fpmin
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > fpmin else fpmin)
        c = 1.0 + aa / c
        c = c if abs(c) > fpmin else fpmin
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b) via Lentz's continued fraction."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_bt = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    bt = math.exp(log_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def t_sf(t: float, df: float) -> float:
    """Survival function of Student's t distribution."""
    if math.isnan(t):
        return math.nan
    if math.isinf(t):
        return 0.0 if t > 0 else 1.0
    tail = 0.5 * betainc(df / 2.0, 0.5, df / (df + t * t))
    return tail if t > 0 else 1.0 - tail


def norm_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _check_alternative(alternative: str) -> None:
    if alternative not in ("greater", "less"):
        raise ValueError("alternative must be 'greater' or 'less'")


def welch_t_test(x: np.ndarray, y: np.ndarray, alternative: str = "greater") -> TestResult:
    """One-sided Welch t-test of mean(x) vs mean(y)."""
    _check_alternative(alternative)
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    n1, n2 = x.size, y.size
    if n1 < 2 or n2 < 2:
        raise ValueError("Welch t-test needs at least 2 observations per group")
    v1, v2 = x.var(ddof=1) / n1, y.var(ddof=1) / n2
    diff = x.mean() - y.mean()
    if v1 + v2 == 0.0:
        stat = math.copysign(math.inf, diff) if diff != 0 else 0.0
        p = 0.5 if diff == 0 else float((diff > 0) != (alternative == "greater"))
        return TestResult(stat, p, None)
    stat = float(diff / math.sqrt(v1 + v2))
    df = float((v1 + v2) ** 2 / (v1**2 / (n1 - 1) + v2**2 / (n2 - 1)))
    p = t_sf(stat, df) if alternative == "greater" else t_sf(-stat, df)
    return TestResult(stat, p, df)


def one_sample_t_test(x: np.ndarray, mu: float = 0.0, alternative: str = "greater") -> TestResult:
    _check_alternative(alternative)
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        raise ValueError("one-sample t-test needs at least 2 observations")
    diff = x.mean() - mu
    se = x.std(ddof=1) / math.sqrt(x.size)
    df = float(x.size - 1)
    if se == 0.0:
        stat = math.copysign(math.inf, diff) if diff != 0 else 0.0
        p = 0.5 if diff == 0 else float((diff > 0) != (alternative == "greater"))
        return TestResult(stat, p, df)
    stat = float(diff / se)
    p = t_sf(stat, df) if alternative == "greater" else t_sf(-stat, df)
    return TestResult(stat, p, df)


def rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks (1-based), ties receive the mean of their positions."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    ranks = np.empty(x.size, dtype=float)
    boundaries = np.flatnonzero(np.diff(sorted_x)) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [x.size]))
    for s, e in zip(starts, ends, strict=True):
        ranks[order[s:e]] = (s + e + 1) / 2.0
    return ranks


@lru_cache(maxsize=256)
def _mwu_counts(m: int, n: int) -> tuple[float, ...]:
    """Number of orderings giving each value of U for sample sizes (m, n), no ties."""
    rows = [np.ones(1) for _ in range(n + 1)]
    for i in range(1, m + 1):
        new = [np.ones(1)]
        for j in range(1, n + 1):
            counts = np.zeros(i * j + 1)
            above = rows[j]
            counts[j : j + above.size] += above
            counts[: new[j - 1].size] += new[j - 1]
            new.append(counts)
        rows = new
    return tuple(rows[n])


def mann_whitney_u(
    x: np.ndarray, y: np.ndarray, alternative: str = "greater", method: str = "auto"
) -> TestResult:
    """One-sided Mann-Whitney U test; ``greater`` means x tends to exceed y.

    ``auto`` uses the exact null distribution when either sample has at most 8
    observations and there are no ties, otherwise the tie-corrected normal
    approximation with continuity correction (the same rule as scipy).
    """
    _check_alternative(alternative)
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    n1, n2 = x.size, y.size
    if n1 == 0 or n2 == 0:
        raise ValueError("Mann-Whitney U needs non-empty samples")
    combined = np.concatenate([x, y])
    ranks = rankdata(combined)
    u1 = float(ranks[:n1].sum() - n1 * (n1 + 1) / 2.0)
    u = u1 if alternative == "greater" else n1 * n2 - u1
    _, counts = np.unique(combined, return_counts=True)
    ties = bool((counts > 1).any())
    if method == "auto":
        method = "asymptotic" if (n1 > 8 and n2 > 8) or ties else "exact"
    if method == "exact":
        if ties:
            raise ValueError("exact Mann-Whitney distribution requires no ties")
        dist = np.asarray(_mwu_counts(n1, n2))
        p = float(dist[round(u) :].sum() / dist.sum())
    elif method == "asymptotic":
        n = n1 + n2
        tie_term = float((counts**3 - counts).sum())
        sigma = math.sqrt(n1 * n2 / 12.0 * ((n + 1) - tie_term / (n * (n - 1))))
        p = 1.0 if sigma == 0.0 else norm_sf((u - n1 * n2 / 2.0 - 0.5) / sigma)
    else:
        raise ValueError(f"unknown method {method!r}")
    return TestResult(u1, min(1.0, max(0.0, p)))


def permutation_test(
    x: np.ndarray,
    y: np.ndarray | None,
    aggregation: str,
    n_resamples: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Monte Carlo permutation test of agg(x) - agg(y) (or sign-flip test of agg(x) vs 0).

    Returns ``(observed, p_greater, p_less)`` with the (1 + count) / (1 + n) correction.
    """
    x = np.asarray(x, dtype=float)
    stats = []
    if y is None:
        observed = float(aggregate(x, aggregation))
        for size in _blocks(n_resamples, x.size):
            signs = rng.choice(np.array([-1.0, 1.0]), size=(size, x.size))
            stats.append(aggregate(signs * x, aggregation, axis=1))
    else:
        y = np.asarray(y, dtype=float)
        pooled = np.concatenate([x, y])
        observed = float(aggregate(x, aggregation) - aggregate(y, aggregation))
        for size in _blocks(n_resamples, pooled.size):
            perm = pooled[np.argsort(rng.random((size, pooled.size)), axis=1)]
            stats.append(
                aggregate(perm[:, : x.size], aggregation, axis=1)
                - aggregate(perm[:, x.size :], aggregation, axis=1)
            )
    null = np.concatenate(stats)
    tol = 1e-12 * max(1.0, abs(observed))
    p_greater = (1.0 + np.count_nonzero(null >= observed - tol)) / (n_resamples + 1.0)
    p_less = (1.0 + np.count_nonzero(null <= observed + tol)) / (n_resamples + 1.0)
    return observed, float(p_greater), float(p_less)


def holm(p_values: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values (FWER)."""
    p = np.asarray(p_values, dtype=float)
    m = p.size
    if m == 0:
        return p
    order = np.argsort(p, kind="mergesort")
    adjusted = np.maximum.accumulate((m - np.arange(m)) * p[order])
    out = np.empty(m)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg step-up adjusted p-values (FDR)."""
    p = np.asarray(p_values, dtype=float)
    m = p.size
    if m == 0:
        return p
    order = np.argsort(p, kind="mergesort")
    scaled = p[order] * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(scaled[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def adjust_p_values(p_values: np.ndarray, method: str) -> np.ndarray:
    if method == "none":
        return np.asarray(p_values, dtype=float)
    if method == "holm":
        return holm(p_values)
    if method == "bh":
        return benjamini_hochberg(p_values)
    raise ValueError(f"unknown correction {method!r}")


@dataclass(frozen=True)
class ClaimResult:
    estimate: float
    ci_low: float
    ci_high: float
    ci_level: float
    p_confirm: float
    p_refute: float


def effect_estimate(
    comparison: Comparison, samples: Mapping[str, np.ndarray], aggregation: str
) -> float:
    a = float(aggregate(samples[comparison.left], aggregation))
    if comparison.kind == "difference":
        return a - float(aggregate(samples[comparison.right], aggregation))
    if comparison.kind == "ratio":
        b = float(aggregate(samples[comparison.right], aggregation))
        return a / b if b != 0 else math.nan
    return a


def effect_bootstrap(
    comparison: Comparison,
    samples: Mapping[str, np.ndarray],
    aggregation: str,
    n_boot: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Bootstrap distribution of the effect, resampling each condition independently."""
    a = bootstrap_aggregates(samples[comparison.left], aggregation, n_boot, rng)
    if comparison.kind in ("one_sample", "interval"):
        return a
    b = bootstrap_aggregates(samples[comparison.right], aggregation, n_boot, rng)
    if comparison.kind == "difference":
        return a - b
    with np.errstate(divide="ignore", invalid="ignore"):
        return a / b


def oriented_claims(
    comparison: Comparison, samples: Mapping[str, np.ndarray]
) -> list[tuple[np.ndarray, np.ndarray | None]]:
    """Rewrite the comparison as claims of the form ``agg(x) - agg(y) > 0`` (or ``agg(x) > 0``).

    Shifts and positive scalings commute with mean, median and IQM, so the rewritten
    claims are equivalent to the declared one.
    """
    a = np.asarray(samples[comparison.left], dtype=float)
    b = None if comparison.right is None else np.asarray(samples[comparison.right], dtype=float)
    greater = comparison.op in (">", ">=")
    t = comparison.threshold
    if comparison.kind == "interval":
        assert comparison.low is not None and comparison.high is not None
        return [(a - comparison.low, None), (comparison.high - a, None)]
    assert t is not None
    if comparison.kind == "one_sample":
        return [(a - t, None) if greater else (t - a, None)]
    assert b is not None
    if comparison.kind == "difference":
        return [(a - t, b) if greater else (b, a - t)]
    return [(a, t * b) if greater else (t * b, a)]


def claim_pvalues(
    x: np.ndarray,
    y: np.ndarray | None,
    plan: AnalysisPlan,
    rng: np.random.Generator,
    n_boot: int | None = None,
) -> tuple[float, float]:
    """One-sided p-values for ``agg(x) - agg(y) > 0`` (confirm) and ``< 0`` (refute)."""
    test = plan.test
    if test == "bootstrap":
        B = n_boot or plan.n_boot
        boot = bootstrap_aggregates(x, plan.aggregation, B, rng)
        if y is not None:
            boot = boot - bootstrap_aggregates(y, plan.aggregation, B, rng)
        p_c = (1.0 + np.count_nonzero(boot <= 0.0)) / (B + 1.0)
        p_r = (1.0 + np.count_nonzero(boot >= 0.0)) / (B + 1.0)
        return float(p_c), float(p_r)
    if test == "welch":
        if y is None:
            return (
                one_sample_t_test(x, 0.0, "greater").p_value,
                one_sample_t_test(x, 0.0, "less").p_value,
            )
        return welch_t_test(x, y, "greater").p_value, welch_t_test(x, y, "less").p_value
    if test == "mann_whitney":
        if y is None:
            raise ValueError("Mann-Whitney U needs two samples")
        return mann_whitney_u(x, y, "greater").p_value, mann_whitney_u(x, y, "less").p_value
    if test == "permutation":
        _, p_g, p_l = permutation_test(x, y, plan.aggregation, n_boot or plan.n_permutations, rng)
        return p_g, p_l
    raise ValueError(f"unknown test {test!r}")


def evaluate_claim(
    comparison: Comparison,
    samples: Mapping[str, np.ndarray],
    plan: AnalysisPlan,
    rng: np.random.Generator,
) -> ClaimResult:
    estimate = effect_estimate(comparison, samples, plan.aggregation)
    boot = effect_bootstrap(comparison, samples, plan.aggregation, plan.n_boot, rng)
    finite = boot[np.isfinite(boot)]
    lo, hi = percentile_ci(finite, plan.ci_level) if finite.size else (math.nan, math.nan)
    confirms, refutes = [], []
    for x, y in oriented_claims(comparison, samples):
        p_c, p_r = claim_pvalues(x, y, plan, rng)
        confirms.append(p_c)
        refutes.append(p_r)
    return ClaimResult(
        estimate=estimate,
        ci_low=lo,
        ci_high=hi,
        ci_level=plan.ci_level,
        p_confirm=max(confirms),
        p_refute=min(refutes),
    )


def z_quantile(p: float) -> float:
    return _STD_NORMAL.inv_cdf(p)


def aggregation_efficiency_factor(
    aggregation: str, n: int, reps: int = 20_000, seed: int = 0
) -> float:
    """Variance of the aggregate relative to the mean for Gaussian samples of size ``n``.

    Multiplying a mean-based sample size by this factor approximates the sample size
    needed when the analysis aggregates with median or IQM.
    """
    if aggregation == "mean":
        return 1.0
    rng = np.random.default_rng(seed)
    draws = rng.standard_normal((reps, max(n, 4)))
    return float(np.var(aggregate(draws, aggregation, axis=1)) / np.var(draws.mean(axis=1)))


def normal_approx_seeds(
    delta: float,
    sd_a: float,
    sd_b: float | None,
    alpha: float,
    power: float,
    factor: float = 1.0,
) -> int:
    """Seeds per condition for a one-sided test to detect margin ``delta`` at ``power``.

    Two-sample: n = (z_{1-alpha} + z_power)^2 (sd_a^2 + sd_b^2) / delta^2.
    One-sample: n = (z_{1-alpha} + z_power)^2 sd_a^2 / delta^2.
    """
    if delta <= 0:
        raise ValueError("delta must be positive")
    z = z_quantile(1.0 - alpha) + z_quantile(power)
    variance = sd_a**2 + (sd_b**2 if sd_b is not None else 0.0)
    return max(2, math.ceil(factor * z * z * variance / delta**2))


def normal_approx_power(
    delta: float, sd_a: float, sd_b: float | None, alpha: float, n: int, factor: float = 1.0
) -> float:
    variance = sd_a**2 + (sd_b**2 if sd_b is not None else 0.0)
    se = math.sqrt(factor * variance / n)
    if se == 0:
        return 1.0
    return 1.0 - _STD_NORMAL.cdf(z_quantile(1.0 - alpha) - delta / se)


def simulate_power(
    plan: AnalysisPlan,
    delta: float,
    sd_a: float,
    sd_b: float | None,
    alpha: float,
    n: int,
    n_sim: int = 300,
    n_boot: int = 1000,
    seed: int = 0,
) -> float:
    """Monte Carlo power of the planned test on Gaussian data with true margin ``delta``."""
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_sim):
        x = delta + sd_a * rng.standard_normal(n)
        y = None if sd_b is None else sd_b * rng.standard_normal(n)
        p_c, _ = claim_pvalues(x, y, plan, rng, n_boot=n_boot)
        hits += p_c < alpha
    return hits / n_sim

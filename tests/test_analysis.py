from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import special, stats

from preregister.analysis import (
    aggregate,
    aggregation_efficiency_factor,
    benjamini_hochberg,
    betainc,
    bootstrap_aggregates,
    claim_pvalues,
    evaluate_claim,
    holm,
    iqm,
    mann_whitney_u,
    normal_approx_seeds,
    one_sample_t_test,
    oriented_claims,
    permutation_test,
    rankdata,
    simulate_power,
    t_sf,
    welch_t_test,
)
from preregister.registry import AnalysisPlan, Comparison

RNG = np.random.default_rng(12345)


@pytest.mark.parametrize("n", [4, 5, 7, 10, 20, 21, 101])
def test_iqm_matches_trim_mean(n: int) -> None:
    x = RNG.normal(size=n)
    assert iqm(x) == pytest.approx(stats.trim_mean(x, 0.25), abs=1e-12)
    mat = RNG.normal(size=(6, n))
    np.testing.assert_allclose(iqm(mat, axis=1), stats.trim_mean(mat, 0.25, axis=1), atol=1e-12)


def test_iqm_known_case() -> None:
    assert iqm(np.array([1, 2, 3, 4, 5, 6, 7, 100.0])) == pytest.approx(4.5)


@pytest.mark.parametrize(
    ("a", "b", "x"),
    [(0.5, 0.5, 0.3), (2, 3, 0.9), (10, 0.5, 0.99), (50, 60, 0.45), (0.1, 30, 1e-4)],
)
def test_betainc_matches_scipy(a: float, b: float, x: float) -> None:
    assert betainc(a, b, x) == pytest.approx(special.betainc(a, b, x), rel=1e-10, abs=1e-14)


@pytest.mark.parametrize(
    ("t", "df"), [(0.0, 5), (1.3, 2.5), (-2.2, 17.3), (4.0, 3), (12.0, 40), (-0.4, 1)]
)
def test_t_sf_matches_scipy(t: float, df: float) -> None:
    assert t_sf(t, df) == pytest.approx(stats.t.sf(t, df), rel=1e-9, abs=1e-14)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("alternative", ["greater", "less"])
def test_welch_matches_scipy(seed: int, alternative: str) -> None:
    rng = np.random.default_rng(seed)
    x = rng.normal(0.3, 1.0, 8 + seed)
    y = rng.normal(0.0, 2.0, 12)
    ours = welch_t_test(x, y, alternative)
    ref = stats.ttest_ind(x, y, equal_var=False, alternative=alternative)
    assert ours.statistic == pytest.approx(ref.statistic, rel=1e-10)
    assert ours.p_value == pytest.approx(ref.pvalue, rel=1e-8)


def test_one_sample_t_matches_scipy() -> None:
    x = RNG.normal(0.4, 1.0, 9)
    for alt in ("greater", "less"):
        ours = one_sample_t_test(x, 0.1, alt)
        ref = stats.ttest_1samp(x, 0.1, alternative=alt)
        assert ours.p_value == pytest.approx(ref.pvalue, rel=1e-8)


def test_t_tests_degenerate_variance() -> None:
    assert welch_t_test(np.ones(3), np.zeros(3)).p_value == 0.0
    assert welch_t_test(np.ones(3), np.zeros(3), "less").p_value == 1.0
    assert one_sample_t_test(np.full(4, 2.0), 2.0).p_value == 0.5
    with pytest.raises(ValueError):
        welch_t_test(np.ones(1), np.ones(3))


def test_rankdata_matches_scipy() -> None:
    x = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5.0])
    np.testing.assert_allclose(rankdata(x), stats.rankdata(x))


@pytest.mark.parametrize(("n1", "n2"), [(3, 4), (5, 5), (8, 3), (6, 8)])
@pytest.mark.parametrize("alternative", ["greater", "less"])
def test_mann_whitney_exact_matches_scipy(n1: int, n2: int, alternative: str) -> None:
    rng = np.random.default_rng(n1 * 10 + n2)
    x, y = rng.normal(0.5, 1, n1), rng.normal(0, 1, n2)
    ours = mann_whitney_u(x, y, alternative, method="exact")
    ref = stats.mannwhitneyu(x, y, alternative=alternative, method="exact")
    assert ours.statistic == pytest.approx(ref.statistic)
    assert ours.p_value == pytest.approx(ref.pvalue, rel=1e-10)


@pytest.mark.parametrize("alternative", ["greater", "less"])
def test_mann_whitney_asymptotic_with_ties_matches_scipy(alternative: str) -> None:
    rng = np.random.default_rng(3)
    x = rng.integers(0, 6, 15).astype(float)
    y = rng.integers(1, 7, 18).astype(float)
    ours = mann_whitney_u(x, y, alternative)
    ref = stats.mannwhitneyu(
        x, y, alternative=alternative, method="asymptotic", use_continuity=True
    )
    assert ours.p_value == pytest.approx(ref.pvalue, rel=1e-10)
    with pytest.raises(ValueError):
        mann_whitney_u(x, y, alternative, method="exact")


def test_mann_whitney_auto_rule() -> None:
    x, y = np.arange(20.0) + 0.5, np.arange(20.0)
    auto = mann_whitney_u(x, y).p_value
    assert auto == pytest.approx(stats.mannwhitneyu(x, y, alternative="greater").pvalue, rel=1e-10)


def test_permutation_test_against_exact_enumeration() -> None:
    x = np.array([1.1, 2.3, 3.0, 4.2])
    y = np.array([0.2, 0.5, 1.0, 1.4])
    obs, p_g, p_l = permutation_test(x, y, "mean", 20000, np.random.default_rng(0))
    ref = stats.permutation_test(
        (x, y),
        lambda a, b: np.mean(a) - np.mean(b),
        permutation_type="independent",
        alternative="greater",
        n_resamples=np.inf,
    )
    assert obs == pytest.approx(ref.statistic)
    assert p_g == pytest.approx(ref.pvalue, abs=0.005)
    assert p_l > 0.9


def test_sign_flip_permutation_one_sample() -> None:
    rng = np.random.default_rng(1)
    _, p_g, p_l = permutation_test(np.abs(rng.normal(1, 0.2, 10)), None, "mean", 5000, rng)
    assert p_g < 0.01 and p_l > 0.99


def test_bootstrap_is_seeded_and_chunked(monkeypatch: pytest.MonkeyPatch) -> None:
    x = RNG.normal(size=30)
    a = bootstrap_aggregates(x, "iqm", 5000, np.random.default_rng(7))
    b = bootstrap_aggregates(x, "iqm", 5000, np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)
    import preregister.analysis as mod

    monkeypatch.setattr(mod, "_MAX_BLOCK", 1000)
    c = bootstrap_aggregates(x, "mean", 5000, np.random.default_rng(7))
    assert c.shape == (5000,)
    assert c.mean() == pytest.approx(x.mean(), abs=0.02)
    assert c.std() == pytest.approx(x.std() / math.sqrt(30), rel=0.1)


def test_holm_hand_computed() -> None:
    p = np.array([0.01, 0.04, 0.03, 0.005])
    np.testing.assert_allclose(holm(p), [0.03, 0.06, 0.06, 0.02])
    np.testing.assert_allclose(holm(np.array([0.5, 0.4])), [0.8, 0.8])


def test_bh_hand_computed() -> None:
    p = np.array([0.01, 0.04, 0.03, 0.005])
    np.testing.assert_allclose(benjamini_hochberg(p), [0.02, 0.04, 0.04, 0.02])
    np.testing.assert_allclose(benjamini_hochberg(np.array([0.9, 0.8, 0.02])), [0.9, 0.9, 0.06])


def test_bh_matches_scipy_false_discovery_control() -> None:
    p = RNG.uniform(size=25) ** 2
    np.testing.assert_allclose(benjamini_hochberg(p), stats.false_discovery_control(p, method="bh"))


@pytest.mark.parametrize(
    ("expr", "a", "b", "expected"),
    [
        ("a - b > 1", [5.0, 6.0], [1.0, 2.0], ([4.0, 5.0], [1.0, 2.0])),
        ("a - b < 1", [5.0, 6.0], [1.0, 2.0], ([1.0, 2.0], [4.0, 5.0])),
        ("a / b >= 2", [5.0, 6.0], [1.0, 2.0], ([5.0, 6.0], [2.0, 4.0])),
        ("a > 3", [5.0, 6.0], None, ([2.0, 3.0], None)),
        ("a < 3", [5.0, 6.0], None, ([-2.0, -3.0], None)),
    ],
)
def test_oriented_claims(expr, a, b, expected) -> None:
    samples = {"a": np.array(a)} | ({"b": np.array(b)} if b is not None else {})
    [(x, y)] = oriented_claims(Comparison.parse(expr), samples)
    np.testing.assert_allclose(x, expected[0])
    if expected[1] is None:
        assert y is None
    else:
        np.testing.assert_allclose(y, expected[1])


@pytest.mark.parametrize("test", ["bootstrap", "welch", "mann_whitney", "permutation"])
def test_claim_pvalues_direction(test: str) -> None:
    rng = np.random.default_rng(0)
    plan = AnalysisPlan(test=test, n_seeds=20, n_boot=2000, n_permutations=2000)
    x, y = rng.normal(2, 1, 20), rng.normal(0, 1, 20)
    p_c, p_r = claim_pvalues(x, y, plan, rng)
    assert p_c < 0.001 and p_r > 0.99


def test_evaluate_claim_interval_tost() -> None:
    rng = np.random.default_rng(0)
    samples = {"a": rng.normal(0.5, 0.05, 30)}
    plan = AnalysisPlan(test="welch", n_seeds=30, n_boot=2000)
    inside = evaluate_claim(Comparison.parse("a in [0.4, 0.6]"), samples, plan, rng)
    assert inside.p_confirm < 0.001
    assert inside.ci_low > 0.4 and inside.ci_high < 0.6
    outside = evaluate_claim(Comparison.parse("a in [0.6, 0.9]"), samples, plan, rng)
    assert outside.p_refute < 0.001 and outside.p_confirm > 0.99


def test_power_formulas() -> None:
    n = normal_approx_seeds(delta=1.0, sd_a=1.0, sd_b=1.0, alpha=0.05, power=0.8)
    z = stats.norm.ppf(0.95) + stats.norm.ppf(0.8)
    assert n == math.ceil(2 * z**2)
    assert aggregation_efficiency_factor("mean", 20) == 1.0
    assert aggregation_efficiency_factor("median", 200) == pytest.approx(math.pi / 2, rel=0.08)
    assert 1.0 < aggregation_efficiency_factor("iqm", 40) < math.pi / 2


def test_simulated_power_agrees_with_normal_approximation() -> None:
    plan = AnalysisPlan(test="welch", n_seeds=13)
    power = simulate_power(
        plan, delta=1.0, sd_a=1.0, sd_b=1.0, alpha=0.05, n=13, n_sim=1500, seed=1
    )
    assert power == pytest.approx(0.8, abs=0.06)


def test_aggregate_unknown() -> None:
    with pytest.raises(ValueError):
        aggregate(np.ones(3), "mode")

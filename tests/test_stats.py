from src.stats import mcnemar_exact, wilson, holm, cochran_armitage, flips

def test_mcnemar_exact_small():
    # b=9,c=2: two-sided exact = 2*P(X<=2 | n=11) = 2*(1+11+55)/2048
    assert abs(mcnemar_exact(9, 2) - 134 / 2048) < 1e-12
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == 1.0

def test_mcnemar_paper_numbers():
    assert 0.008 < mcnemar_exact(75, 46) < 0.014     # paper headline, exact version
    assert 0.065 < mcnemar_exact(66, 46) < 0.080     # clean-item reanalysis

def test_wilson_known():
    pct, lo, hi = wilson(5, 10)
    assert abs(pct - 50.0) < 1e-9 and abs(lo - 23.66) < 0.05 and abs(hi - 76.34) < 0.05

def test_holm():
    assert holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]

def test_cochran_armitage_trend_vs_flat():
    z, p = cochran_armitage([10, 20, 30, 40], [50, 50, 50, 50])
    assert p < 1e-3 and z > 0
    _, pflat = cochran_armitage([25, 25, 25, 25], [50, 50, 50, 50])
    assert pflat > 0.9

def test_flips():
    assert flips([(1, 0), (1, 0), (0, 1), (1, 1), (0, 0)]) == (2, 1)


# --- heterogeneity across strata -------------------------------------------------

from src.stats import chisq_sf, heterogeneity, dersimonian_laird, log_or_strata


def test_chisq_sf_matches_published_critical_values():
    """The 5% critical points of chi-square. No scipy here, so the implementation is
    ours and has to be pinned against values that can be looked up."""
    for x, df in ((3.841, 1), (5.991, 2), (7.815, 3), (11.070, 5), (18.307, 10)):
        assert abs(chisq_sf(x, df) - 0.05) < 0.001
    assert chisq_sf(0, 3) == 1.0
    assert chisq_sf(-5, 3) == 1.0


def test_identical_strata_show_no_heterogeneity():
    """Six copies of the same 2x2: all variation is sampling error, so Q ~ 0 and
    I^2 pins at 0. tau^2 must be exactly 0, not a small positive number."""
    h = heterogeneity([(40, 20, 20, 40)] * 6)
    assert h["k"] == 6 and h["df"] == 5
    assert h["Q"] < 1e-9
    assert h["I2"] == 0.0
    assert h["tau2"] == 0.0


def test_opposed_strata_show_high_heterogeneity():
    """Three strata favouring one direction and three the other. Q must be large,
    I^2 high, and Q significant."""
    h = heterogeneity([(80, 20, 20, 80)] * 3 + [(20, 80, 80, 20)] * 3)
    assert h["I2"] > 90
    assert h["p"] < 0.001


def test_dl_reduces_to_fixed_effect_when_homogeneous():
    """With tau^2 = 0 the random-effects weights ARE the inverse-variance weights, so
    the DL point estimate must equal the fixed-effect one."""
    strata = [(40, 20, 20, 40)] * 6
    dl = dersimonian_laird(strata)
    yv = log_or_strata(strata)
    w = [1 / v for _, v in yv]
    fixed = sum(wi * yi for wi, (yi, _) in zip(w, yv)) / sum(w)
    assert dl["tau2"] == 0.0
    assert abs(dl["or"] - 2.718281828 ** fixed) < 1e-9


def test_dl_interval_widens_under_heterogeneity():
    """The whole point of random effects: disagreement between strata must COST
    precision. The heterogeneous interval has to be wider on the log scale."""
    homo = dersimonian_laird([(40, 20, 20, 40)] * 6)
    hetero = dersimonian_laird([(80, 20, 20, 80)] * 3 + [(20, 80, 80, 20)] * 3)
    assert hetero["tau2"] > 0
    assert hetero["se"] > homo["se"]


def test_zero_cell_is_corrected_not_infinite():
    """A zero cell would send log OR to infinity; Haldane-Anscombe keeps it finite.
    Strata with no information at all are dropped rather than corrected into one."""
    yv = log_or_strata([(10, 0, 5, 5)])
    assert len(yv) == 1 and abs(yv[0][0]) != float("inf")
    assert log_or_strata([(0, 0, 5, 5)]) == []

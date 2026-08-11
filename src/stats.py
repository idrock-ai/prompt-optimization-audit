"""Exact paired statistics for the erosion analyses. Pure stdlib, unit-tested."""
from __future__ import annotations
from math import comb, erfc, exp, lgamma, log, sqrt


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact binomial McNemar p for discordant counts (b, c)."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * p, 100 * (centre - half), 100 * (centre + half)


def holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def cochran_armitage(ks: list[int], ns: list[int], scores: list[float] | None = None):
    """Trend test for proportions ks/ns over ordered groups. Returns (z, two-sided p)."""
    s = scores or list(range(len(ks)))
    N, K = sum(ns), sum(ks)
    if N == 0 or K in (0, N):
        return 0.0, 1.0
    pbar = K / N
    t = sum(si * (ki - ni * pbar) for si, ki, ni in zip(s, ks, ns))
    var = pbar * (1 - pbar) * (sum(ni * si * si for si, ni in zip(s, ns))
                               - sum(ni * si for si, ni in zip(s, ns)) ** 2 / N)
    if var <= 0:
        return 0.0, 1.0
    z = t / sqrt(var)
    return z, erfc(abs(z) / sqrt(2))


def flips(pairs: list[tuple[int, int]]) -> tuple[int, int]:
    """(b, c): b = first right & second wrong; c = first wrong & second right."""
    b = sum(1 for a, x in pairs if a and not x)
    c = sum(1 for a, x in pairs if not a and x)
    return b, c


def fisher_exact_2x2(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p for [[a, b], [c, d]] by summing every hypergeometric
    table at most as probable as the observed one."""
    n, r1, c1 = a + b + c + d, a + b, a + c
    if n == 0 or r1 in (0, n) or c1 in (0, n):
        return 1.0

    def prob(x):
        return comb(r1, x) * comb(n - r1, c1 - x) / comb(n, c1)

    p_obs = prob(a)
    lo, hi = max(0, c1 - (n - r1)), min(r1, c1)
    return min(1.0, sum(p for x in range(lo, hi + 1)
                        if (p := prob(x)) <= p_obs * (1 + 1e-9)))


def mantel_haenszel_or(strata: list[tuple[int, int, int, int]]) -> float:
    """Mantel-Haenszel common odds ratio over strata of 2x2 tables [[a, b], [c, d]].
    Pools the association while allowing each stratum its own baseline rate. Returns
    inf when no stratum contributes to the denominator."""
    num = den = 0.0
    for a, b, c, d in strata:
        n = a + b + c + d
        if n:
            num += a * d / n
            den += b * c / n
    return num / den if den else float("inf")


# --- heterogeneity across strata ------------------------------------------------
# Mantel-Haenszel pools an association under the assumption that every stratum shares
# ONE underlying odds ratio. When strata disagree, that assumption is wrong and the MH
# interval describes a common effect that does not exist. These functions test the
# assumption (Cochran's Q, I^2) and provide the estimator that does not make it
# (DerSimonian-Laird random effects).


def _gser(a: float, x: float, itmax: int = 500, eps: float = 1e-12) -> float:
    """Regularized LOWER incomplete gamma P(a,x) by series; use for x < a+1."""
    if x <= 0:
        return 0.0
    ap, total, term = a, 1.0 / a, 1.0 / a
    for _ in range(itmax):
        ap += 1
        term *= x / ap
        total += term
        if abs(term) < abs(total) * eps:
            break
    return total * exp(-x + a * log(x) - lgamma(a))


def _gcf(a: float, x: float, itmax: int = 500, eps: float = 1e-12) -> float:
    """Regularized UPPER incomplete gamma Q(a,x) by modified Lentz continued
    fraction; use for x >= a+1."""
    tiny = 1e-300
    b = x + 1.0 - a
    c, d = 1.0 / tiny, 1.0 / b if b else 1.0 / tiny
    h = d
    for i in range(1, itmax + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h * exp(-x + a * log(x) - lgamma(a))


def chisq_sf(x: float, df: int) -> float:
    """P(chi^2_df > x). Implemented here because the project carries no scipy."""
    if df <= 0 or x <= 0:
        return 1.0
    a, xx = df / 2.0, x / 2.0
    return 1.0 - _gser(a, xx) if xx < a + 1.0 else _gcf(a, xx)


def log_or_strata(strata: list[tuple[int, int, int, int]]) -> list[tuple[float, float]]:
    """[(log OR, variance of log OR)] per stratum of [[a, b], [c, d]].

    A Haldane-Anscombe 0.5 is added to every cell ONLY when some cell is zero, which
    would otherwise send the log odds ratio to +-inf. Applying it unconditionally would
    shift the point estimates away from the uncorrected ones the paper reports, so we
    correct only where the estimate is undefined."""
    out = []
    for a, b, c, d in strata:
        if a + b == 0 or c + d == 0:
            continue                      # stratum carries no information at all
        if min(a, b, c, d) == 0:
            a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
        out.append((log((a * d) / (b * c)), 1 / a + 1 / b + 1 / c + 1 / d))
    return out


def heterogeneity(strata: list[tuple[int, int, int, int]]) -> dict:
    """Cochran's Q, its p-value, I^2 and the DerSimonian-Laird tau^2.

    I^2 is the share of total variation across strata attributable to real between-
    stratum differences rather than sampling error. Conventional reading: 25% low,
    50% moderate, 75% high. Q is notoriously underpowered with few strata, so a
    non-significant Q with a large I^2 is evidence of heterogeneity, not against it."""
    yv = log_or_strata(strata)
    k = len(yv)
    if k < 2:
        return {"k": k, "Q": None, "df": 0, "p": None, "I2": None, "tau2": 0.0}
    w = [1 / v for _, v in yv]
    sw = sum(w)
    ybar = sum(wi * yi for wi, (yi, _) in zip(w, yv)) / sw
    q = sum(wi * (yi - ybar) ** 2 for wi, (yi, _) in zip(w, yv))
    df = k - 1
    denom = sw - sum(wi * wi for wi in w) / sw
    return {"k": k, "Q": q, "df": df, "p": chisq_sf(q, df),
            "I2": max(0.0, (q - df) / q) * 100 if q > 0 else 0.0,
            "tau2": max(0.0, (q - df) / denom) if denom > 0 else 0.0}


def dersimonian_laird(strata: list[tuple[int, int, int, int]]) -> dict:
    """Random-effects pooled odds ratio: does NOT assume a common effect.

    Each stratum is weighted by 1/(v_i + tau^2), so between-stratum variance widens
    the interval instead of being ignored. Where MH answers "what is the common odds
    ratio", this answers "what is the mean of the distribution of odds ratios" -- the
    right question once Q/I^2 say the strata disagree."""
    yv = log_or_strata(strata)
    if not yv:
        return {"or": float("nan"), "ci95": (float("nan"), float("nan")),
                "se": float("nan"), "tau2": 0.0}
    tau2 = heterogeneity(strata)["tau2"]
    w = [1 / (v + tau2) for _, v in yv]
    sw = sum(w)
    ybar = sum(wi * yi for wi, (yi, _) in zip(w, yv)) / sw
    se = sqrt(1 / sw)
    return {"or": exp(ybar), "ci95": (exp(ybar - 1.96 * se), exp(ybar + 1.96 * se)),
            "se": se, "tau2": tau2}

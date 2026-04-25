"""
Junike-style adaptive COS pricing with improved truncation and grid selection.

Key improvements over the plain Fang–Oosterlee COS method:

  1. Adaptive domain truncation — vary L until a Chernoff two-sided tail
     bound on P(|Z| > w) drops below ``eps_trunc``.
  2. Centered pricing — price in Z = X − c1 space on a symmetric interval
     [−w, w]; the CF is shifted by exp(−i·u·c1) and the forward by exp(c1).
  3. Resolution-coupled N — N = ceil(2w / dx_target), rounded to the next
     power of two, so support width and spectral resolution move together.
  4. Fallback routing — when 2w exceeds ``width_fallback``, automatically
     route to Lewis / Carr–Madan / FrFT instead of COS.
  5. Default put-pricing with call recovery via put–call parity, avoiding
     the cancellation instability that plagues direct-call COS on wide grids.

Reference:
    Junike G (2024) On the number of terms in the COS method for European
    option pricing.  Numerische Mathematik 156(3):867-898.
    https://doi.org/10.1007/s00211-024-01402-1

    Fang F, Oosterlee CW (2008) A Novel Pricing Method for European Options
    Based on Fourier-Cosine Series Expansions.
    SIAM J. Sci. Comput. 31(2):826-848.
    https://doi.org/10.1137/080718061
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .cos_method import cos_price

# ── Model family → sensible defaults ─────────────────────────────────────────
# gaussian_like : BSM, Heston (light-tailed log-return)
# semi_heavy    : VG, NIG (moderate tails from subordination)
# heavy         : CGMY with Y close to 2 (near-Cauchy), extreme maturities

_FAMILY_DEFAULTS: dict[str, dict] = {
    "gaussian_like": {"L": 12.0, "dx_target": 0.05},
    # VG/NIG CFs decay algebraically in u, so adequate N requires finer dx than Gaussian.
    # dx_target = pi/u_max; for algebraic decay |phi(u)| ~ u^{-alpha} we need
    # u_max ~ O(1000) to reach 1e-7 accuracy, giving dx ~ 0.003.
    "semi_heavy":    {"L": 10.0, "dx_target": 0.003},
    "heavy":         {"L": 12.0, "dx_target": 0.10},
}


@dataclass
class COSGridPolicy:
    """
    Configuration object for the improved COS grid builder.

    Parameters
    ----------
    mode
        ``"benchmark"`` (single-strike accuracy focus) or
        ``"surface"``   (multi-strike throughput focus; default).
    truncation
        ``"tolerance"`` — adapt L until Chernoff tail < eps_trunc (default).
        ``"heuristic"`` — fixed L * sqrt(c2 + sqrt(|c4|)) directly.
        ``"paper"``     — Fang & Oosterlee fixed-L rule using paper_L.
    centered
        If ``True`` (default) build symmetric interval [−w, w] for Z = X − c1
        and shift the CF and forward accordingly.
    dx_target
        Target x-space resolution.  ``None`` → model-family default.
    fixed_N
        Override the adaptive N when set.
    L
        Starting half-width multiplier.  ``None`` → model-family default.
    paper_L
        L value used when ``truncation="paper"``.
    eps_trunc
        Tail-probability tolerance for ``truncation="tolerance"``.
    min_N
        Lower clamp on the adaptive N.
    max_N
        Upper clamp on the adaptive N.
    width_fallback
        If 2w > width_fallback, route to the fallback engine instead of COS.
    fallback_method
        ``"lewis"`` (default), ``"carr_madan"``, or ``"frft"``.
    """

    mode:            str            = "surface"
    truncation:      str            = "tolerance"
    centered:        bool           = True
    dx_target:       Optional[float] = None
    fixed_N:         Optional[int]   = None
    L:               Optional[float] = None
    paper_L:         float           = 12.0
    eps_trunc:       float           = 1e-8
    min_N:           int             = 32
    max_N:           int             = 4096
    width_fallback:  float           = 150.0
    fallback_method: str             = "lewis"


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def cos_improved_grid(
    char_func,
    cumulants: Tuple[float, float, float],
    policy: Optional[COSGridPolicy] = None,
    model_family: str = "gaussian_like",
) -> Tuple[float, float, int, float]:
    """
    Compute an improved (a, b, N, center) COS grid.

    Parameters
    ----------
    char_func
        Callable ``u → complex array``, CF of ``log(S_T / F)`` at real or
        imaginary frequency ``u``.
    cumulants
        ``(c1, c2, c4)`` — first, second, and fourth cumulant of
        ``log(S_T / F)``.
    policy
        :class:`COSGridPolicy`.  Defaults to ``COSGridPolicy()``.
    model_family
        ``"gaussian_like"``, ``"semi_heavy"``, or ``"heavy"``.

    Returns
    -------
    a, b
        Truncation-interval endpoints for the variable that ``cos_price``
        sees (Z-space if ``policy.centered`` else X-space).
    N
        Number of COS terms.
    center
        ``c1`` when ``centered=True``; 0.0 otherwise.  The caller must shift
        the CF by ``exp(−i·u·center)`` and the forward by ``exp(center)``.
    """
    if policy is None:
        policy = COSGridPolicy()

    c1, c2, c4 = float(cumulants[0]), float(cumulants[1]), float(cumulants[2])
    defaults  = _FAMILY_DEFAULTS[model_family]
    L_start   = policy.L         if policy.L         is not None else defaults["L"]
    dx_target = policy.dx_target if policy.dx_target is not None else defaults["dx_target"]

    # ── 1. Half-width w ───────────────────────────────────────────────────────
    if policy.truncation == "paper":
        w = policy.paper_L * np.sqrt(abs(c2) + np.sqrt(abs(c4)))
    elif policy.truncation == "heuristic":
        w = L_start * np.sqrt(abs(c2) + np.sqrt(abs(c4)))
    else:  # "tolerance"  (default)
        w = _adaptive_half_width(char_func, c1, c2, c4, L_start, policy.eps_trunc)

    # ── 2. Interval ───────────────────────────────────────────────────────────
    if policy.centered:
        a, b   = -w, +w
        center = c1
    else:
        a, b   = c1 - w, c1 + w
        center = 0.0

    # ── 3. N ──────────────────────────────────────────────────────────────────
    if policy.fixed_N is not None:
        N = int(policy.fixed_N)
    else:
        N_raw = (b - a) / dx_target
        N = _next_pow2(max(int(np.ceil(N_raw)), policy.min_N))
        N = min(N, policy.max_N)

    return float(a), float(b), int(N), float(center)


def cos_improved_price(
    char_func,
    texp: float,
    strike,
    fwd: float,
    df: float,
    cumulants: Tuple[float, float, float],
    cp=1,
    policy: Optional[COSGridPolicy] = None,
    model_family: str = "gaussian_like",
    use_parity: bool = True,
):
    """
    Price European options via the Junike-style improved COS method.

    Full production path
    --------------------
    1. Build an adaptive grid with :func:`cos_improved_grid`.
    2. Route to a fallback engine when the interval is too wide.
    3. Shift the CF by ``exp(−i·u·c1)`` and the forward by ``exp(c1)`` so
       the pricing operates on the centered variable Z = X − c1.
    4. Default to put pricing; recover calls via put–call parity
       ``C = P + df·(F − K)`` using the *original* forward F.

    Parameters
    ----------
    char_func
        Callable ``u → complex array``, CF of ``log(S_T / F)``.
    texp
        Time to expiry.
    strike
        Float or array of strikes.
    fwd
        Forward price ``F = S_0 · exp((r − q) · T)``.
    df
        Discount factor ``exp(−r · T)``.
    cumulants
        ``(c1, c2, c4)`` of ``log(S_T / F)``.
    cp
        ``+1`` call / ``−1`` put (scalar or array).
    policy
        :class:`COSGridPolicy`.  Defaults to ``COSGridPolicy()``.
    model_family
        Tail-type classifier; controls default L and dx_target.
    use_parity
        If ``True`` (default) compute puts and recover calls by PCP,
        avoiding direct-call instability on wide intervals.

    Returns
    -------
    float or np.ndarray
    """
    if policy is None:
        policy = COSGridPolicy()

    a, b, N, center = cos_improved_grid(char_func, cumulants, policy, model_family)

    # ── Fallback routing ──────────────────────────────────────────────────────
    if (b - a) > policy.width_fallback:
        return _fallback_price(policy.fallback_method, char_func, texp, strike, fwd, df, cp)

    # ── Centered CF and shifted forward ──────────────────────────────────────
    # Z = X − center   where  X = log(S_T / F)
    # CF of Z: phi_Z(u) = exp(−i·u·center) · phi_X(u)
    # Payoff in Z-space: (fwd_z · exp(Z) − K)^+  with  fwd_z = fwd · exp(center)
    if center != 0.0:
        _c = center
        def char_func_z(u):
            u_arr = np.asarray(u)
            return char_func(u_arr) * np.exp(-1j * u_arr * _c)
        fwd_z = fwd * np.exp(center)
    else:
        char_func_z = char_func
        fwd_z       = fwd

    strike_arr = np.atleast_1d(np.asarray(strike, dtype=float))
    scalar_out = np.isscalar(strike) and np.isscalar(cp)

    # ── Pricing ───────────────────────────────────────────────────────────────
    if use_parity:
        # Price the OTM option in Z-space (zero intrinsic → more accurate COS).
        # After centering, fwd_z = fwd*exp(c1). The call is OTM when K > fwd_z;
        # the put is OTM when K <= fwd_z. The ITM sibling is recovered via PCP
        # using the *original* fwd (centering does not affect the true forward).
        otm_cp = np.where(strike_arr > fwd_z, 1, -1).astype(float)
        otm_prices  = cos_price(char_func_z, texp, strike_arr, fwd_z, df,
                                cp=otm_cp, n_cos=N, trunc_range=(a, b))
        call_prices = np.where(otm_cp > 0, otm_prices,
                               otm_prices + df * (fwd - strike_arr))
        put_prices  = np.where(otm_cp < 0, otm_prices,
                               otm_prices - df * (fwd - strike_arr))

        if np.ndim(cp) == 0:
            result = call_prices if float(cp) > 0 else put_prices
        else:
            cp_arr = np.asarray(cp, dtype=float)
            result = np.where(cp_arr > 0, call_prices, put_prices)
    else:
        result = cos_price(char_func_z, texp, strike_arr, fwd_z, df,
                           cp=cp, n_cos=N, trunc_range=(a, b))

    if scalar_out:
        return float(np.squeeze(result))
    return result.reshape(np.broadcast_shapes(np.shape(strike), np.shape(cp)))


# ─────────────────────────────────────────────────────────────────────────────
# Numerical cumulants  (usable by any model as a fallback)
# ─────────────────────────────────────────────────────────────────────────────

def numerical_cumulants(char_func, eps: float = 1e-3) -> Tuple[float, float, float]:
    """
    Estimate (c1, c2, c4) via finite differences on log(MGF).

    Parameters
    ----------
    char_func
        CF of ``log(S_T / F)`` evaluated at *imaginary* arguments
        (i.e. the MGF:  ``mgf(v) = char_func(−i·v)``).
    eps
        Finite-difference step; 1e-3 balances truncation and round-off.

    Returns
    -------
    (c1, c2, c4)
    """
    mgf  = lambda v: float(char_func(-1j * v).real)
    lm0  = np.log(mgf(0.0))
    lmp1 = np.log(mgf( eps));    lmm1 = np.log(mgf(-eps))
    lmp2 = np.log(mgf(2 * eps)); lmm2 = np.log(mgf(-2 * eps))
    c1   = (lmp1 - lmm1) / (2 * eps)
    c2   = (lmp1 + lmm1 - 2 * lm0) / eps ** 2
    c4   = (lmp2 - 4 * lmp1 + 6 * lm0 - 4 * lmm1 + lmm2) / eps ** 4
    return float(c1), float(c2), float(c4)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _adaptive_half_width(
    char_func,
    c1: float,
    c2: float,
    c4: float,
    L_start: float,
    eps_trunc: float,
    max_iter: int = 15,
    L_cap: float = 30.0,
) -> float:
    """
    Grow L until the two-sided Chernoff tail bound on P(|Z| > w) < eps_trunc.

    w = L · sqrt(|c2| + sqrt(|c4|)) is the Fang–Oosterlee half-width.
    The Chernoff bound is evaluated via the CF at imaginary arguments.
    """
    L = L_start
    for _ in range(max_iter):
        w    = L * np.sqrt(abs(c2) + np.sqrt(abs(c4)))
        tail = _chernoff_tail_bound(char_func, c1, w)
        if tail <= eps_trunc:
            break
        if L >= L_cap:
            break
        L = min(L * 1.25, L_cap)
    return L * np.sqrt(abs(c2) + np.sqrt(abs(c4)))


def _chernoff_tail_bound(
    char_func,
    c1: float,
    w: float,
    n_grid: int = 40,
) -> float:
    """
    Two-sided Chernoff bound on P(|Z| > w) where Z = X − c1.

    Upper tail:  P(Z > w)   ≤  inf_{s>0} M_X(s) · exp(−s·(c1 + w))
    Lower tail:  P(Z < −w)  ≤  inf_{s>0} M_X(−s) · exp(s·(c1 − w))

    where  M_X(s) = char_func(−i·s)  and  M_X(−s) = char_func(i·s).

    For a Gaussian Z ~ N(0, c2) the optimal saddlepoint lies at s* = w/c2,
    which can be large (e.g. s*=30 for sigma=0.2, T=1, L=6).  We therefore
    search on a log-spaced grid from 0.01 to 100 and skip any s where the
    MGF diverges (as happens for CGMY outside its strip of analyticity).
    """
    # Log-spaced grid covers the saddlepoint s* = w/c2 for typical models.
    s_grid = np.logspace(-2, 2, n_grid)   # 0.01 … 100
    try:
        # Upper tail
        mgf_pos = char_func(-1j * s_grid).real
        ok_pos  = np.isfinite(mgf_pos) & (mgf_pos > 0)
        if ok_pos.any():
            log_up = np.log(mgf_pos[ok_pos]) - s_grid[ok_pos] * (c1 + w)
            upper  = float(np.exp(np.min(log_up)))
        else:
            upper  = 1.0

        # Lower tail
        mgf_neg = char_func(1j * s_grid).real
        ok_neg  = np.isfinite(mgf_neg) & (mgf_neg > 0)
        if ok_neg.any():
            log_lo = np.log(mgf_neg[ok_neg]) + s_grid[ok_neg] * (c1 - w)
            lower  = float(np.exp(np.min(log_lo)))
        else:
            lower  = 1.0

        return max(0.0, upper + lower)
    except Exception:
        # Fallback: standard Gaussian tail approximation using w directly
        from scipy.special import erfc
        return float(erfc(w / np.sqrt(2.0)))


def _next_pow2(n: int) -> int:
    """Smallest power of two ≥ n."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _fallback_price(method: str, char_func, texp, strike, fwd, df, cp):
    """Route to an alternative Fourier engine."""
    if method == "carr_madan":
        from .carr_madan import carr_madan_price
        return carr_madan_price(char_func, texp, strike, fwd, df, cp=cp)
    if method == "frft":
        from .frft import frft_price
        return frft_price(char_func, texp, strike, fwd, df, cp=cp)
    # default: lewis
    from .lewis import lewis_price
    return lewis_price(char_func, texp, strike, fwd, df, cp=cp)

"""
Core COS pricing engine.

Implements the Fourier-Cosine (COS) method of Fang & Oosterlee (2008)
as a standalone, dependency-free module.

Reference:
    Fang F, Oosterlee CW (2008) A Novel Pricing Method for European Options
    Based on Fourier-Cosine Series Expansions.
    SIAM J. Sci. Comput. 31(2):826-848.
    https://doi.org/10.1137/080718061
"""

import numpy as np


def cos_price(
    char_func,
    texp,
    strike,
    fwd,
    df,
    cp=1,
    n_cos=128,
    trunc_range=None,
):
    """
    Price a European call or put via the COS method (Eq. 2 / Eq. 21 of the paper).

    Fully vectorised over *strike* and *cp* using NumPy broadcasting.
    The dominant cost is one (M × N) matrix-vector product where
    M = number of strikes and N = n_cos.

    Parameters
    ----------
    char_func : callable
        Characteristic function of log(S_T / F).
        Signature: ``char_func(u: np.ndarray) -> np.ndarray``
        where *u* is a real-valued frequency array and the return value
        is complex.
    texp : float
        Time to expiry T.
    strike : float or array-like, shape (M,)
        Strike price(s).
    fwd : float
        Forward price F = S · exp((r − q) · T).
    df : float
        Discount factor exp(−r · T).
    cp : int or array-like, shape (M,)
        +1 for call, −1 for put.
    n_cos : int
        Number of Fourier-cosine terms N.  Default 128.
        Increase for very long maturities or extreme vol parameters.
    trunc_range : tuple (a, b) or None
        Explicit integration range.  If None, it is computed from the
        cumulants of the characteristic function via Eq. (5.2).

    Returns
    -------
    float or np.ndarray
        Option price(s) matching the broadcast shape of (strike, cp).
    """
    scalar_out = np.isscalar(strike) and np.isscalar(cp)
    kk   = np.atleast_1d(np.asarray(strike / fwd, dtype=float))   # (M,)
    cp_a = np.broadcast_to(
        np.atleast_1d(np.asarray(cp, dtype=float)), kk.shape
    ).copy()

    if trunc_range is None:
        a, b = _truncation_range_from_cf(char_func, n_cos)
    else:
        a, b = trunc_range

    ba    = b - a
    k_arr = np.arange(n_cos)          # (N,)
    u_arr = k_arr * np.pi / ba        # (N,)

    # CF values shifted by exp(−i u a) so the cosine basis starts at x=a
    cf   = char_func(u_arr)            # (N,) complex
    cf_s = cf * np.exp(-1j * u_arr * a)
    cf_s[0] *= 0.5                     # prime-sum: k=0 term gets factor ½
    cf_re = cf_s.real                  # (N,)

    # Payoff-coefficient arrays  –  shape (M, N)
    log_kk = np.clip(np.log(kk), a, b)[:, None]  # (M, 1)
    u  = u_arr[None, :]                           # (1, N)
    k  = k_arr[None, :]                           # (1, N)
    kk_c = kk[:, None]                            # (M, 1)

    # Call: V_k = (2/ba) * [chi(log_kk, b) − kk · psi(log_kk, b)]
    W_call = (2.0 / ba) * (
        _chi(k, u, a, log_kk, b) - kk_c * _psi(k, u, a, log_kk, b)
    )
    # Put:  V_k = (2/ba) * [kk · psi(a, log_kk) − chi(a, log_kk)]
    W_put = (2.0 / ba) * (
        kk_c * _psi(k, u, a, a, log_kk) - _chi(k, u, a, a, log_kk)
    )

    W         = np.where(cp_a[:, None] > 0, W_call, W_put)  # (M, N)
    price_arr = df * fwd * (W @ cf_re)                        # (M,)

    if scalar_out:
        return float(price_arr[0])
    return price_arr.reshape(
        np.broadcast_shapes(np.shape(strike), np.shape(cp))
    )


# ─────────────────────────────────────────────────────────────────────────────
# Truncation-range helper  (Eq. 5.2 of Fang & Oosterlee 2008)
# ─────────────────────────────────────────────────────────────────────────────

def _truncation_range_from_cf(char_func, n_cos, L=12.0, eps=1e-3):
    """
    Estimate the integration interval [a, b] from cumulants derived by
    numerically differentiating the cumulant generating function
    log(MGF(v)) at v = 0.

    Not called when the model provides analytic cumulants
    (e.g. BsmModel, HestonModel).
    """
    # MGF at real arguments: MGF(v) = CF(−iv)
    mgf = lambda v: char_func(-1j * v).real
    lm0  = np.log(mgf(0.0))
    lmp1 = np.log(mgf( eps));  lmm1 = np.log(mgf(-eps))
    lmp2 = np.log(mgf(2*eps)); lmm2 = np.log(mgf(-2*eps))
    c1 = (lmp1 - lmm1) / (2*eps)
    c2 = (lmp1 + lmm1 - 2*lm0) / eps**2
    c4 = (lmp2 - 4*lmp1 + 6*lm0 - 4*lmm1 + lmm2) / eps**4
    half = L * np.sqrt(abs(c2) + np.sqrt(abs(c4)))
    return c1 - half, c1 + half


# ─────────────────────────────────────────────────────────────────────────────
# Payoff coefficient helpers  (Eqs. 22-23)
# ─────────────────────────────────────────────────────────────────────────────

def _chi(k, u, a, c, d):
    """
    Eq. (22):  ∫_c^d  exp(x) · cos(k π (x−a)/(b−a))  dx

    Broadcasting: u ~ (1, N), c and d ~ (M, 1) → result ~ (M, N).
    """
    exp_d, exp_c = np.exp(d), np.exp(c)
    cos_d = np.cos(u * (d - a));  cos_c = np.cos(u * (c - a))
    sin_d = np.sin(u * (d - a));  sin_c = np.sin(u * (c - a))
    num   = (cos_d * exp_d - cos_c * exp_c
             + u * (sin_d * exp_d - sin_c * exp_c))
    return num / (1.0 + u**2)


def _psi(k, u, a, c, d):
    """
    Eq. (23):  ∫_c^d  cos(k π (x−a)/(b−a))  dx

    k = 0 handled exactly to avoid division by zero.
    """
    safe_u = np.where(k == 0, 1.0, u)
    return np.where(
        k == 0,
        d - c,
        (np.sin(u * (d - a)) - np.sin(u * (c - a))) / safe_u,
    )
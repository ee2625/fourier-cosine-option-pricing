"""
Core COS pricing engine — optimized.

Implements the Fourier-Cosine (COS) method of Fang & Oosterlee (2008).

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
    Price a European call or put via the COS method (Eq. 2 / Eq. 21).

    Vectorised over *strike* and *cp*.  Dominant cost: one (M x N)
    matrix-vector product (M = number of strikes, N = n_cos).

    Optimisations vs a naive implementation
    ----------------------------------------
    1. Trig sharing  — cos/sin of u*(log_kk − a) are computed once and
       reused for both chi and psi.  At the upper boundary x = b,
       sin(k*pi) = 0 and cos(k*pi) = (−1)^k so no trig is needed there.

    2. Put-call parity  — only the needed payoff matrix is built.
       For mixed cp, W_put is derived from W_call via an O(N) correction
       (see docstring of _chi_psi_call / _chi_psi_put).

    3. Scalar cp fast-path  — avoids numpy reduction overhead for the
       common all-call / all-put case (cp = +1 / −1 scalar).

    Parameters
    ----------
    char_func    : callable u -> complex array, CF of log(S_T/F)
    texp         : float
    strike       : float or array (M,)
    fwd          : float
    df           : float
    cp           : +1 call / -1 put (scalar or array)
    n_cos        : int (default 128)
    trunc_range  : (a, b) or None

    Returns
    -------
    float or np.ndarray
    """
    scalar_out = np.isscalar(strike) and np.isscalar(cp)
    kk   = np.atleast_1d(np.asarray(strike / fwd, dtype=float))   # (M,)

    # Fast-path: scalar cp avoids all numpy reduction overhead
    if np.ndim(cp) == 0:
        cp_val    = float(cp)
        all_calls = cp_val > 0
        all_puts  = cp_val < 0
        cp_a      = None                 # not needed in scalar path
    else:
        cp_a      = np.asarray(cp, dtype=float)
        all_calls = bool(np.all(cp_a > 0))
        all_puts  = bool(np.all(cp_a < 0))

    if trunc_range is None:
        a, b = _truncation_range_from_cf(char_func, n_cos)
    else:
        a, b = trunc_range

    ba    = b - a
    k_arr = np.arange(n_cos)            # (N,)
    u_arr = k_arr * (np.pi / ba)        # (N,)

    # CF with phase shift exp(-i*u*a); k=0 term gets factor 1/2 (prime-sum)
    cf       = char_func(u_arr)
    cf_s     = cf * np.exp(-1j * u_arr * a)
    cf_s[0] *= 0.5
    cf_re    = cf_s.real                # (N,)

    # Shared pre-computations
    log_kk  = np.clip(np.log(kk), a, b)[:, None]    # (M, 1)
    u       = u_arr[None, :]                          # (1, N)
    k       = k_arr[None, :]                          # (1, N)
    kk_c    = kk[:, None]                             # (M, 1)
    phase   = u * (log_kk - a)                        # (M, N)
    cos_ph  = np.cos(phase)
    sin_ph  = np.sin(phase)
    safe_u  = np.where(k == 0, 1.0, u)
    # At x=b: cos(u*(b-a))=cos(k*pi)=(-1)^k, sin(u*(b-a))=0
    sign_k  = (-1.0) ** k_arr                         # (N,)
    exp_b   = np.exp(b)                               # scalar
    exp_a   = np.exp(a)                               # scalar
    inv_1u2 = 1.0 / (1.0 + u_arr**2)                 # (N,)

    if all_calls:
        # chi(log_kk, b): upper boundary simplifies via sin(k*pi)=0
        chi = (sign_k * exp_b - kk_c * (cos_ph + u * sin_ph)) * inv_1u2
        # psi(log_kk, b): sin(k*pi) = 0
        psi = np.where(k == 0, b - log_kk, -sin_ph / safe_u)
        W   = (2.0 / ba) * (chi - kk_c * psi)

    elif all_puts:
        # chi(a, log_kk): lower boundary simplifies via cos(0)=1, sin(0)=0
        chi = (kk_c * (cos_ph + u * sin_ph) - exp_a) * inv_1u2
        # psi(a, log_kk): sin(0) = 0
        psi = np.where(k == 0, log_kk - a, sin_ph / safe_u)
        W   = (2.0 / ba) * (kk_c * psi - chi)

    else:
        # Mixed: build W_call, derive W_put via COS put-call parity.
        # W_call − W_put = (2/ba)*chi_full − 2*kk*delta_{k,0}
        # chi_full[k] = ((-1)^k*e^b − e^a)/(1+u^2)  (sin(k*pi)=0)
        chi = (sign_k * exp_b - kk_c * (cos_ph + u * sin_ph)) * inv_1u2
        psi = np.where(k == 0, b - log_kk, -sin_ph / safe_u)
        W_call = (2.0 / ba) * (chi - kk_c * psi)
        chi_full  = (sign_k * exp_b - exp_a) * inv_1u2         # (N,)
        delta     = np.broadcast_to(
            (2.0 / ba) * chi_full, (len(kk), n_cos)
        ).copy()
        delta[:, 0] -= 2.0 * kk
        W_put  = W_call - delta
        cp_a2  = cp_a if cp_a is not None else np.full(kk.shape, cp_val)
        W      = np.where(cp_a2[:, None] > 0, W_call, W_put)

    price_arr = df * fwd * (W @ cf_re)   # (M,)

    if scalar_out:
        return float(price_arr[0])
    return price_arr.reshape(
        np.broadcast_shapes(np.shape(strike), np.shape(cp))
    )


# ─────────────────────────────────────────────────────────────────────────────
# Truncation-range helper  (Eq. 5.2)
# ─────────────────────────────────────────────────────────────────────────────

def _truncation_range_from_cf(char_func, n_cos, L=12.0, eps=1e-3):
    """Integration interval [a, b] from numerical cumulants of the CF."""
    mgf  = lambda v: char_func(-1j * v).real
    lm0  = np.log(mgf(0.0))
    lmp1 = np.log(mgf( eps));  lmm1 = np.log(mgf(-eps))
    lmp2 = np.log(mgf(2*eps)); lmm2 = np.log(mgf(-2*eps))
    c1   = (lmp1 - lmm1) / (2 * eps)
    c2   = (lmp1 + lmm1 - 2 * lm0) / eps**2
    c4   = (lmp2 - 4*lmp1 + 6*lm0 - 4*lmm1 + lmm2) / eps**4
    half = L * np.sqrt(abs(c2) + np.sqrt(abs(c4)))
    return c1 - half, c1 + half


# ─────────────────────────────────────────────────────────────────────────────
# Payoff coefficient helpers kept as public API (used externally / in tests)
# ─────────────────────────────────────────────────────────────────────────────

def _chi(k, u, a, c, d):
    """Eq. (22): integral of exp(x)*cos(k*pi*(x-a)/ba) from c to d."""
    exp_d, exp_c = np.exp(d), np.exp(c)
    cos_d = np.cos(u*(d-a)); cos_c = np.cos(u*(c-a))
    sin_d = np.sin(u*(d-a)); sin_c = np.sin(u*(c-a))
    return (cos_d*exp_d - cos_c*exp_c + u*(sin_d*exp_d - sin_c*exp_c)) / (1.+u**2)


def _psi(k, u, a, c, d):
    """Eq. (23): integral of cos(k*pi*(x-a)/ba) from c to d."""
    safe_u = np.where(k == 0, 1.0, u)
    return np.where(k==0, d-c, (np.sin(u*(d-a))-np.sin(u*(c-a)))/safe_u)
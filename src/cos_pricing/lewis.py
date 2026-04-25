"""
Lewis (2001) Fourier-inversion option pricing.

Single-integral representation of European call/put prices in terms of
the characteristic function of log(S_T/F).  The integrand uses the fixed
contour shift u - i/2 (equivalent to Carr-Madan with damping alpha = 1/2),
so there is no damping parameter to tune.

Reference:
    Lewis A (2001) A Simple Option Formula for General Jump-Diffusion and
    other Exponential Levy Processes.
    Albrecher H, Mayer P, Schoutens W, Tistaert J (2007) The Little Heston
    Trap. Wilmott Magazine, Eq. 2.7.
"""
import numpy as np
from scipy.special import roots_legendre


def lewis_price(char_func, texp, strike, fwd, df, cp=1, n_quad=64, u_max=200.0):
    """
    Price European calls/puts via the Lewis (2001) single-integral formula.

    In our (forward, discount-factor) convention the formula reads
        C(K) = df * ( F  -  sqrt(K F)/pi * I(k) )
        I(k) = integral_0^inf  Re[ exp(i u k) * phi_F(u - i/2) ] / (u^2 + 1/4) du
        k    = log(F / K)
    where phi_F(u) = E[ exp(i u log(S_T/F)) ] is the CF of log-moneyness from
    the forward (the same CF used by ``cos_price`` and ``carr_madan_price``).

    The semi-infinite integral is evaluated by an n_quad-node Gauss-Legendre
    rule on [0, u_max].  u_max defaults to 200, large enough to capture the
    integrand support for typical short-to-medium-maturity calibrations of
    BSM, Heston, VG, and CGMY; for very low vol or very long maturity bump
    n_quad before u_max.

    Parameters
    ----------
    char_func : callable  CF of log(S_T/F); must accept complex u.
    texp      : float
    strike    : float or array
    fwd       : float     F = S0 * exp((r-q)*T)
    df        : float     exp(-r*T)
    cp        : +1 call / -1 put  (scalar or array)
    n_quad    : int       number of Gauss-Legendre nodes  (default 64)
    u_max     : float     upper truncation of the frequency integral

    Returns
    -------
    float or np.ndarray
    """
    scalar_in = np.isscalar(strike)
    K_arr     = np.atleast_1d(np.asarray(strike, dtype=float))

    # Gauss-Legendre nodes on [-1, 1] mapped to [0, u_max].
    xi, wi  = roots_legendre(n_quad)
    u       = (xi + 1.0) * (0.5 * u_max)
    weights = wi * (0.5 * u_max)

    # Evaluate (CF on the shifted contour) / (u^2 + 1/4) once, reuse per strike.
    cf_over_denom = char_func(u - 0.5j) / (u * u + 0.25)

    calls = np.empty_like(K_arr)
    for i, K in enumerate(K_arr):
        k        = np.log(fwd / K)
        integral = float(((np.exp(1j * u * k) * cf_over_denom) * weights).real.sum())
        calls[i] = df * (fwd - np.sqrt(K * fwd) / np.pi * integral)

    # Put via parity: P = C - df * (F - K)
    if np.ndim(cp) == 0:
        result = calls if float(cp) > 0 else (calls - df * (fwd - K_arr))
    else:
        result = np.where(np.asarray(cp) > 0, calls, calls - df * (fwd - K_arr))

    return float(result[0]) if scalar_in else result

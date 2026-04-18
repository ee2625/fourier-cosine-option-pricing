"""
Carr-Madan FFT option pricing.

Reference:
    Carr P, Madan D (1999) Option valuation using the fast Fourier transform.
    J. Computational Finance 2(4):61-73.
"""
import numpy as np
from scipy.fft import fft
from scipy.interpolate import CubicSpline


def carr_madan_price(char_func, texp, strike, fwd, df, cp=1,
                     N=2**12, alpha=0.75, eta_grid=0.05):
    """
    Price European options via the Carr-Madan FFT method.

    Uses Simpson's rule weights for 4th-order accuracy (per the paper).
    char_func must be the CF of log(S_T/F) and accept complex arguments.

    Parameters
    ----------
    char_func : callable  CF of log(S_T/F); must handle complex u.
    texp      : float
    strike    : float or array
    fwd       : float     F = S0 * exp((r-q)*T)
    df        : float     exp(-r*T)
    cp        : +1 call / -1 put
    N         : int       FFT grid size (power of 2, default 2^12).
    alpha     : float     Damping exponent (default 0.75, per Remark 5.1).
    eta_grid  : float     Frequency grid spacing (default 0.25).
    """
    v  = np.arange(N) * eta_grid
    dk = 2.0 * np.pi / (N * eta_grid)
    b  = 0.5 * N * dk                       # log-strike half-range

    psi = df * char_func(v - (alpha + 1.0) * 1j) / (
        alpha ** 2 + alpha - v ** 2 + 1j * (2.0 * alpha + 1.0) * v
    )

    w          = np.ones(N)          # ones so the un-covered w[N-2] slot stays 1.0
    w[0]       = w[-1]     = 1.0 / 3.0
    w[1:-1:2]  = 4.0 / 3.0
    w[2:-2:2]  = 2.0 / 3.0

    y         = fft(np.exp(1j * b * v) * psi * w * eta_grid).real
    k_grid    = -b + dk * np.arange(N)
    call_grid = fwd * np.exp(-alpha * k_grid) / np.pi * y

    scalar_in  = np.isscalar(strike)
    strike_arr = np.atleast_1d(np.asarray(strike, dtype=float))
    calls      = CubicSpline(k_grid, call_grid)(np.log(strike_arr / fwd))
    puts       = calls - fwd * df + strike_arr * df

    if np.ndim(cp) == 0:
        result = calls if float(cp) > 0 else puts
    else:
        result = np.where(np.asarray(cp) > 0, calls, puts)

    return float(result[0]) if scalar_in else result

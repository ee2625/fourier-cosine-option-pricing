"""
Carr-Madan option pricing via the fractional FFT (FrFT).

Plain Carr-Madan FFT ties the frequency grid spacing eta to the strike
grid spacing lambda by  eta * lambda = 2*pi / N.  Picking eta large enough
to capture the CF tail forces the strike grid coarse, and vice versa.

The FrFT (Bailey & Swarztrauber 1991) computes
    y_k = sum_{j=0}^{N-1} x_j exp(-2 pi i beta j k),     k = 0, ..., N-1
for any real beta, via three length-2N FFTs.  Plugging this into the
Carr-Madan integrand lets eta and lambda be chosen independently
(beta = eta * lambda / (2 pi) replaces the plain-FFT 1/N), so a small
lambda gives a fine strike grid for accurate spline interpolation
without forcing eta down.

References:
    Carr P, Madan D (1999) J. Computational Finance 2(4):61-73.
    Bailey DH, Swarztrauber PN (1991) The Fractional Fourier Transform
        and Applications. SIAM Review 33(3):389-404.
    Chourdakis K (2005) Option pricing using the fractional FFT.
        J. Computational Finance 8(2):1-18.
"""
import numpy as np
from scipy.fft import fft, ifft
from scipy.interpolate import CubicSpline


def _frft(x, beta):
    """Bailey-Swarztrauber FrFT: y_k = sum_j x_j exp(-2 pi i beta j k).

    Three length-2N FFTs.  Identity used:  jk = (j^2 + k^2 - (j-k)^2) / 2
    so  exp(-2 pi i beta j k) = e^{-i pi beta j^2} e^{-i pi beta k^2} e^{i pi beta (j-k)^2}
    which factors the sum into a convolution of two chirped sequences.
    """
    N    = x.shape[-1]
    j    = np.arange(N)
    chirp_neg = np.exp(-1j * np.pi * beta * j * j)            # e^{-i pi beta j^2}
    chirp_pos = np.exp( 1j * np.pi * beta * j * j)            # e^{ i pi beta j^2}

    # Length-2N padded sequences.
    y = np.zeros(2 * N, dtype=complex)
    z = np.zeros(2 * N, dtype=complex)
    y[:N] = x * chirp_neg
    z[:N] = chirp_pos
    # The convolution kernel must extend to negative indices: use the symmetry
    # exp(i pi beta j^2) = exp(i pi beta (-j)^2), so the wrapped tail mirrors.
    j_tail = np.arange(N, 2 * N) - 2 * N                       # -N, ..., -1
    z[N:]  = np.exp(1j * np.pi * beta * j_tail * j_tail)

    conv = ifft(fft(y) * fft(z))[:N]
    return chirp_neg * conv


def frft_price(char_func, texp, strike, fwd, df, cp=1,
               N=2**12, alpha=0.75, eta_grid=0.25, lambda_grid=0.005):
    """Price European calls/puts via the Carr-Madan inversion using the FrFT.

    Independent control of the frequency grid spacing ``eta_grid`` and the
    log-strike grid spacing ``lambda_grid``: the FrFT scales their product
    via ``beta = eta_grid * lambda_grid / (2 pi)``, so the two no longer have
    to satisfy ``eta * lambda = 2 pi / N`` (which is what plain Carr-Madan
    enforces).  Pick a small ``lambda_grid`` for a fine strike grid (good for
    interpolating to exact target strikes) regardless of ``eta_grid``.

    Parameters
    ----------
    char_func : callable  CF of log(S_T/F); must accept complex u.
    texp      : float
    strike    : float or array
    fwd       : float     F = S0 * exp((r-q)*T)
    df        : float     exp(-r*T)
    cp        : +1 call / -1 put (scalar or array)
    N         : int       FFT length (default 2**12)
    alpha     : float     Damping exponent (default 0.75, per Carr-Madan)
    eta_grid  : float     Frequency grid spacing
    lambda_grid : float   Log-strike grid spacing (independent of eta_grid)

    Returns
    -------
    float or np.ndarray
    """
    j  = np.arange(N)
    v  = j * eta_grid                                          # frequencies
    b  = 0.5 * N * lambda_grid                                  # log-strike half-range

    psi = df * char_func(v - (alpha + 1.0) * 1j) / (
        alpha ** 2 + alpha - v ** 2 + 1j * (2.0 * alpha + 1.0) * v
    )

    # Simpson weights, same as carr_madan_price.
    w        = np.ones(N)
    w[0]     = w[-1] = 1.0 / 3.0
    w[1:-1:2] = 4.0 / 3.0
    w[2:-2:2] = 2.0 / 3.0

    x       = np.exp(1j * b * v) * psi * w * eta_grid
    beta    = eta_grid * lambda_grid / (2.0 * np.pi)
    y       = _frft(x, beta).real

    k_grid    = -b + lambda_grid * j
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

"""
Carr-Madan option pricing via the fractional FFT (FrFT).

The plain Carr-Madan FFT (``FftABC.price_fft``) ties the frequency-grid
spacing ``eta`` to the log-strike grid spacing ``lambda`` via
``eta * lambda = 2*pi / N``.  Picking ``eta`` large enough to capture the
characteristic-function tail forces the strike grid coarse, and vice
versa.

The FrFT (Bailey & Swarztrauber 1991) computes
    y_k = sum_{j=0}^{N-1} x_j exp(-2*pi*i * beta * j * k),  k = 0, ..., N-1
for any real ``beta`` via three length-2N FFTs.  Plugging this into the
Carr-Madan integrand lets ``eta`` and ``lambda`` be chosen independently
(``beta = eta*lambda/(2*pi)`` replaces the plain-FFT ``1/N``), so a small
``lambda`` gives a fine strike grid for accurate spline interpolation
without forcing ``eta`` down (Chourdakis 2005).

Provides ``FrftMixin`` plus four pre-mixed concrete classes:

    BsmFrft, VarGammaFrft, CgmyFrft, HestonFrft

Each composes the existing ``*Fft`` class with the FrFT mixin and
exposes ``price_frft(strike, spot, texp, cp=1)`` alongside the
inherited ``price`` (Lewis single integral) and ``price_fft``
(plain Carr-Madan FFT).

References:
    Carr P, Madan D (1999) Option valuation using the fast Fourier
        transform. J. Computational Finance 2(4):61-73.
    Bailey DH, Swarztrauber PN (1991) The Fractional Fourier Transform
        and Applications. SIAM Review 33(3):389-404.
    Chourdakis K (2005) Option pricing using the fractional FFT.
        J. Computational Finance 8(2):1-18.
"""

import numpy as np
from scipy.fft import fft, ifft
from scipy.interpolate import CubicSpline

from .sv_fft import BsmFft, VarGammaFft, CgmyFft, HestonFft


__all__ = [
    "FrftMixin",
    "BsmFrft",
    "VarGammaFrft",
    "CgmyFrft",
    "HestonFrft",
]


def _frft(x, beta):
    """Bailey-Swarztrauber FrFT: ``y_k = sum_j x_j exp(-2*pi*i*beta*j*k)``.

    Three length-2N FFTs.  Identity ``j*k = (j^2 + k^2 - (j-k)^2) / 2``
    factors the sum into a convolution of two chirped sequences.
    """
    N = x.shape[-1]
    j = np.arange(N)
    chirp_neg = np.exp(-1j * np.pi * beta * j * j)
    chirp_pos = np.exp( 1j * np.pi * beta * j * j)

    # Length-2N padded sequences for the convolution.
    y = np.zeros(2 * N, dtype=complex)
    z = np.zeros(2 * N, dtype=complex)
    y[:N] = x * chirp_neg
    z[:N] = chirp_pos
    # Symmetry exp(i*pi*beta*j^2) = exp(i*pi*beta*(-j)^2): wrapped tail mirrors.
    j_tail = np.arange(N, 2 * N) - 2 * N                       # -N, ..., -1
    z[N:]  = np.exp(1j * np.pi * beta * j_tail * j_tail)

    conv = ifft(fft(y) * fft(z))[:N]
    return chirp_neg * conv


class FrftMixin:
    """
    Mixin adding Carr-Madan fractional-FFT pricing on top of any
    ``FftABC`` subclass.

    Combine with a model class that provides ``mgf_logprice(uu, texp)``
    (every ``FftABC`` descendant does).  The mixin adds no state and no
    abstract methods; concrete pre-mixed classes (``BsmFrft`` etc.) are
    one-line subclasses.

    Attributes:
        n_frft (int):       FFT length N (default 2**12).
        alpha_frft (float): Carr-Madan damping exponent (default 0.75).
        eta_frft (float):   Frequency-grid spacing.
        lambda_frft (float): Log-strike grid spacing -- chosen
            independently of ``eta_frft`` (this is the FrFT advantage).

    Examples:
        >>> import numpy as np
        >>> import pyfeng as pf
        >>> m = pf.BsmFrft(sigma=0.2, intr=0.05, divr=0.1)
        >>> m.price_frft(np.array([90, 100, 110]), 100.0, 1.0)
    """

    n_frft: int       = 2 ** 12
    alpha_frft: float = 0.75
    eta_frft: float   = 0.25
    lambda_frft: float = 0.005

    def price_frft(self, strike, spot, texp, cp=1):
        """European call/put price via Carr-Madan + Bailey-Swarztrauber FrFT.

        Args:
            strike: scalar or array of strikes.
            spot:   scalar spot.
            texp:   time to expiry.
            cp:     +1 call, -1 put. Scalar or array broadcastable with ``strike``.

        Returns:
            Scalar (when both ``strike`` and ``cp`` are scalar) or array.
        """
        fwd, df, _ = self._fwd_factor(spot, texp)
        N      = int(self.n_frft)
        alpha  = float(self.alpha_frft)
        eta    = float(self.eta_frft)
        lam    = float(self.lambda_frft)

        j = np.arange(N)
        v = j * eta                                              # frequencies
        b = 0.5 * N * lam                                         # log-strike half-range

        # Carr-Madan damped CF.  charfunc_logprice(u) = mgf_logprice(i*u),
        # so char_func(v - (alpha+1)i) == mgf_logprice(i*v + (alpha+1)).
        psi = df * self.mgf_logprice(1j * v + (alpha + 1.0), texp) / (
            alpha ** 2 + alpha - v ** 2 + 1j * (2.0 * alpha + 1.0) * v
        )

        # Simpson weights.
        w        = np.ones(N)
        w[0]     = w[-1] = 1.0 / 3.0
        w[1:-1:2] = 4.0 / 3.0
        w[2:-2:2] = 2.0 / 3.0

        x    = np.exp(1j * b * v) * psi * w * eta
        beta = eta * lam / (2.0 * np.pi)
        y    = _frft(x, beta).real

        k_grid    = -b + lam * j
        call_grid = float(fwd) * np.exp(-alpha * k_grid) / np.pi * y

        scalar_in  = np.isscalar(strike) and np.isscalar(cp)
        strike_arr = np.atleast_1d(np.asarray(strike, dtype=float))
        calls      = CubicSpline(k_grid, call_grid)(np.log(strike_arr / float(fwd)))
        puts       = calls - float(fwd) * float(df) + strike_arr * float(df)

        if np.ndim(cp) == 0:
            result = calls if float(cp) > 0 else puts
        else:
            result = np.where(np.asarray(cp) > 0, calls, puts)

        return float(result[0]) if scalar_in else result


# ─────────────────────────────────────────────────────────────────────────────
# Concrete pre-mixed classes -- one line each.
# ─────────────────────────────────────────────────────────────────────────────


class BsmFrft(BsmFft, FrftMixin):
    """BSM European option pricing via Carr-Madan fractional FFT."""


class VarGammaFrft(VarGammaFft, FrftMixin):
    """Variance Gamma European option pricing via Carr-Madan fractional FFT."""


class CgmyFrft(CgmyFft, FrftMixin):
    """CGMY European option pricing via Carr-Madan fractional FFT."""


class HestonFrft(HestonFft, FrftMixin):
    """Heston European option pricing via Carr-Madan fractional FFT."""

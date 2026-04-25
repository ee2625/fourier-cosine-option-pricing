"""
European option pricing via the Fourier-Cosine (COS) method.

References:
    Fang F, Oosterlee CW (2008) A Novel Pricing Method for European Options
    Based on Fourier-Cosine Series Expansions.
    SIAM Journal on Scientific Computing 31(2):826-848.
    https://doi.org/10.1137/080718061
"""

import abc
import numpy as np

from . import opt_abc as opt

__all__ = ["CosABC", "BsmCos", "HestonCos"]


class CosABC(opt.OptABC, abc.ABC):
    """
    Abstract base class for European vanilla pricing by the COS method.

    Subclasses implement ``mgf_logprice(uu, texp)``, the moment generating
    function of log(S_T / F), where F is the forward price supplied by
    ``OptABC._fwd_factor``.  Fang-Oosterlee write the vanilla derivation
    with x = log(S0/K) and y = log(S_T/K).  This PyFENG integration uses
    z = log(S_T/F); the density coefficients, payoff coefficients, prime
    summation, and final dot product are the same COS method after this
    change of variables.

    Attributes:
        n_cos (int): Number of Fourier-cosine terms N (default 128).
        L (float): Generic truncation half-width multiplier (default 12).
    """

    n_cos: int = 128
    L: float = 12.0

    @abc.abstractmethod
    def mgf_logprice(self, uu, texp):
        """
        Moment generating function of log(S_T / F).

        Args:
            uu: scalar or array, real or complex.
            texp: time to expiry.

        Returns:
            MGF values with the same shape as ``uu``.
        """
        raise NotImplementedError

    def charfunc_logprice(self, u, texp):
        """Characteristic function phi(u) = MGF(i*u)."""
        return self.mgf_logprice(1j * u, texp)

    # ------------------------------------------------------------------
    # Cumulants and truncation interval
    # ------------------------------------------------------------------

    def _cumulants(self, texp):
        """
        First four cumulants of log(S_T/F) by finite differences.

        Subclasses should override this with analytic cumulants when
        available.

        Returns:
            (c1, c2, c3, c4)
        """
        eps = 1e-3
        lm = lambda v: float(np.log(self.mgf_logprice(v, texp)).real)
        lm0 = lm(0.0)
        lmp1, lmm1 = lm(eps), lm(-eps)
        lmp2, lmm2 = lm(2 * eps), lm(-2 * eps)
        c1 = (lmp1 - lmm1) / (2 * eps)
        c2 = (lmp1 + lmm1 - 2 * lm0) / eps**2
        c4 = (lmp2 - 4 * lmp1 + 6 * lm0 - 4 * lmm1 + lmm2) / eps**4
        return c1, c2, 0.0, c4

    def _truncation_range(self, texp):
        """
        Generic cumulant-based interval [a, b] for log(S_T/F).

        This is the common COS range c1 +/- L*sqrt(|c2| + sqrt(|c4|)).
        Fang-Oosterlee's paper gives model-specific domain choices in its
        numerical sections, so models with a sharper paper-prescribed range
        should override ``_integration_range`` or this method.
        """
        c1, c2, _, c4 = self._cumulants(texp)
        half = self.L * np.sqrt(abs(c2) + np.sqrt(abs(c4)))
        return c1 - half, c1 + half

    def _integration_range(self, strike, spot, texp):
        """
        COS interval used by ``price``.

        The default is strike-independent in log(S_T/F).  Subclasses can
        override this hook for model-specific intervals.
        """
        return self._truncation_range(texp)

    # ------------------------------------------------------------------
    # Payoff coefficient helpers (F&O Eqs. 22-23)
    # ------------------------------------------------------------------

    @staticmethod
    def _chi(k, u, a, c, d):
        """
        Integral of exp(x)*cos(k*pi*(x-a)/(b-a)) from c to d.

        ``u`` is k*pi/(b-a).  Broadcasting convention: u has shape
        (1, N), and c/d can have shape (M, 1).
        """
        exp_d, exp_c = np.exp(d), np.exp(c)
        cos_d = np.cos(u * (d - a))
        cos_c = np.cos(u * (c - a))
        sin_d = np.sin(u * (d - a))
        sin_c = np.sin(u * (c - a))
        num = cos_d * exp_d - cos_c * exp_c
        num += u * (sin_d * exp_d - sin_c * exp_c)
        return num / (1.0 + u**2)

    @staticmethod
    def _psi(k, u, a, c, d):
        """Integral of cos(k*pi*(x-a)/(b-a)) from c to d."""
        safe_u = np.where(k == 0, 1.0, u)
        return np.where(
            k == 0,
            d - c,
            (np.sin(u * (d - a)) - np.sin(u * (c - a))) / safe_u,
        )

    # ------------------------------------------------------------------
    # Fang-Oosterlee pricing stages
    # ------------------------------------------------------------------

    def _cos_grid(self, a, b):
        """COS mode indices and frequencies u_k = k*pi/(b-a)."""
        k_arr = np.arange(int(self.n_cos))
        u_arr = k_arr * np.pi / (b - a)
        return k_arr, u_arr

    def _density_coefficients(self, u_arr, a, texp):
        """
        Real density-side coefficients in the COS dot product.

        This is the characteristic-function phase factor from F&O Eqs.
        8-9 and 19, expressed in log(S_T/F).  The first term is half
        weighted for the prime sum.
        """
        coeff = self.charfunc_logprice(u_arr, texp) * np.exp(-1j * u_arr * a)
        coeff[0] *= 0.5
        return coeff.real

    def _vanilla_payoff_coefficients(self, strike, fwd, a, b, cp):
        """
        Vanilla call/put payoff coefficients from F&O Eqs. 20-25.

        In the paper's y = log(S_T/K), the payoff boundary is y=0.  In
        PyFENG's z = log(S_T/F), that boundary is z = log(K/F), while
        the final scale factor is df*F.
        """
        kk = np.atleast_1d(np.asarray(strike / fwd, dtype=float))
        cp_a = np.broadcast_to(
            np.atleast_1d(np.asarray(cp, dtype=float)), kk.shape
        ).copy()

        k_arr, u_arr = self._cos_grid(a, b)
        log_kk = np.clip(np.log(kk), a, b)[:, None]
        u = u_arr[None, :]
        k = k_arr[None, :]
        kk_c = kk[:, None]

        w_call = (2.0 / (b - a)) * (
            self._chi(k, u, a, log_kk, b)
            - kk_c * self._psi(k, u, a, log_kk, b)
        )
        w_put = (2.0 / (b - a)) * (
            kk_c * self._psi(k, u, a, a, log_kk)
            - self._chi(k, u, a, a, log_kk)
        )
        return np.where(cp_a[:, None] > 0, w_call, w_put)

    # ------------------------------------------------------------------
    # Pricing
    # ------------------------------------------------------------------

    def price(self, strike, spot, texp, cp=1):
        """
        European call/put price via the COS method.

        This method satisfies ``OptABC.price`` and intentionally uses
        ``OptABC._fwd_factor`` for forward, discount, dividend, and
        ``is_fwd`` handling.
        """
        fwd, df, _ = self._fwd_factor(spot, texp)

        scalar_out = np.isscalar(strike) and np.isscalar(cp)

        a, b = self._integration_range(strike, spot, texp)
        _, u_arr = self._cos_grid(a, b)
        cf_re = self._density_coefficients(u_arr, a, texp)
        w_payoff = self._vanilla_payoff_coefficients(strike, fwd, a, b, cp)

        price_arr = df * fwd * (w_payoff @ cf_re)

        if scalar_out:
            return float(price_arr[0])
        return price_arr.reshape(
            np.broadcast_shapes(np.shape(strike), np.shape(cp))
        )


class BsmCos(CosABC):
    """
    Black-Scholes-Merton European option pricing via the COS method.

    Uses analytic BSM cumulants with c4 = 0.
    """

    def mgf_logprice(self, uu, texp):
        """BSM log-price MGF: exp(-0.5*sigma^2*T*u*(1-u))."""
        return np.exp(-0.5 * self.sigma**2 * texp * uu * (1.0 - uu))

    def _cumulants(self, texp):
        """Exact BSM cumulants of log(S_T/F)."""
        s2t = self.sigma**2 * texp
        return -0.5 * s2t, s2t, 0.0, 0.0


def __getattr__(name):
    """Lazy re-export to avoid a circular import with sv_heston_cos."""
    if name == "HestonCos":
        from .sv_heston_cos import HestonCos

        return HestonCos
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

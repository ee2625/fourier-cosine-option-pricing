"""
Black-Scholes-Merton pricing model for use with the generic COS engine.

Exposes:
    • char_func(texp)   →  callable  u ↦ φ(u)
    • trunc_range(texp) →  (a, b)
    • price(strike, spot, texp, cp)

Heston pricing lives in ``heston_cos_pricer.py`` (paper-faithful Section 4 form).

Reference:
    Fang F, Oosterlee CW (2008) SIAM J. Sci. Comput. 31(2):826-848.
"""

import numpy as np
from .cos_method import cos_price


class BsmModel:
    """
    Black-Scholes-Merton (GBM) model.

    Parameters
    ----------
    sigma : float  Constant volatility σ.
    intr  : float  Continuously compounded risk-free rate r.  Default 0.
    divr  : float  Continuous dividend yield q.  Default 0.

    Examples
    --------
    >>> import numpy as np
    >>> from cos_pricing.models import BsmModel
    >>> m = BsmModel(sigma=0.2, intr=0.05, divr=0.1)
    >>> m.price(np.arange(80, 121, 10), spot=100, texp=1.2)
    array([15.71361973,  9.69250803,  5.52948546,  2.94558338,  1.48139131])
    """

    def __init__(self, sigma, intr=0.0, divr=0.0):
        self.sigma = sigma
        self.intr  = intr
        self.divr  = divr

    def char_func(self, texp):
        """
        BSM characteristic function of log(S_T / F).

        φ(u) = E[exp(i·u·X)]  where X = log(S_T/F) ~ N(−½σ²T, σ²T)
             = exp(−½σ²T · u · (u + i))

        Derived by substituting uu = i·u into the MGF
            M(uu) = exp(−½σ²T · uu · (1 − uu))
        """
        sig2t = self.sigma**2 * texp
        def cf(u):
            uu = 1j * u                            # uu = i·u  (complex)
            return np.exp(-0.5 * sig2t * uu * (1.0 - uu))
        return cf

    def trunc_range(self, texp, L=12.0):
        """
        Exact BSM truncation interval.  Cumulants: c1 = −½σ²T, c2 = σ²T, c4 = 0.
        """
        s2t  = self.sigma**2 * texp
        c1   = -0.5 * s2t
        half = L * np.sqrt(s2t)
        return c1 - half, c1 + half

    def _fwd_df(self, spot, texp):
        df  = np.exp(-self.intr * texp)
        fwd = spot * np.exp((self.intr - self.divr) * texp)
        return fwd, df

    def price(self, strike, spot, texp, cp=1, n_cos=128):
        """
        European option price via the COS method.

        Parameters
        ----------
        strike : float or array
        spot   : float
        texp   : float
        cp     : +1 call / −1 put
        n_cos  : int (default 128)
        """
        fwd, df = self._fwd_df(spot, texp)
        return cos_price(self.char_func(texp), texp, strike, fwd, df,
                         cp=cp, n_cos=n_cos, trunc_range=self.trunc_range(texp))

    @staticmethod
    def price_analytic(strike, spot, sigma, texp, intr=0.0, divr=0.0, cp=1):
        """Closed-form BSM for cross-validation."""
        from scipy.stats import norm
        fwd = spot * np.exp((intr - divr) * texp)
        df  = np.exp(-intr * texp)
        sqt = sigma * np.sqrt(texp)
        d1  = np.log(fwd / strike) / sqt + 0.5 * sqt
        d2  = d1 - sqt
        return df * cp * (fwd * norm.cdf(cp * d1) - strike * norm.cdf(cp * d2))

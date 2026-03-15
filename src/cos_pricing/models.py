"""
Option pricing models for use with the COS method.

Each model exposes:
    • char_func(texp)   →  callable  u ↦ φ(u)  (characteristic function, u real)
    • trunc_range(texp) →  (a, b)               (analytic truncation interval)
    • price(strike, spot, texp, cp)              (convenience one-liner)

Reference:
    Fang F, Oosterlee CW (2008) SIAM J. Sci. Comput. 31(2):826-848.
"""

import numpy as np
from .cos_method import cos_price


# ─────────────────────────────────────────────────────────────────────────────
# Black-Scholes-Merton
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Heston stochastic-volatility model
# ─────────────────────────────────────────────────────────────────────────────

class HestonModel:
    """
    Heston (1993) stochastic-volatility model.

    Dynamics (risk-neutral)::

        dS/S = (r − q) dt + √V dW_S
        dV   = κ(V̄ − V) dt + η √V dW_V
        dW_S dW_V = ρ dt

    Parameters
    ----------
    v0    : float  Initial variance V₀.
    kappa : float  Mean-reversion speed κ.
    theta : float  Long-run variance V̄.
    eta   : float  Vol-of-vol η.
    rho   : float  Correlation ρ ∈ (−1, 1).
    intr  : float  Risk-free rate r.  Default 0.
    divr  : float  Dividend yield q.  Default 0.

    Notes
    -----
    Feller condition: 2κV̄ > η² ensures V stays strictly positive.
    For calibrated parameters with Feller violated, increase n_cos
    (e.g. 256 or 512) for better accuracy.

    References
    ----------
    - Heston SL (1993) Rev. Financial Studies 6:327-343.
    - Lord R, Kahl C (2010) Mathematical Finance 20:671-694.
      (branch-cut-safe CF formulation used here)
    - Fang F, Oosterlee CW (2008) SIAM J. Sci. Comput. 31:826-848.
      (Appendix A cumulant formulas for truncation range)

    Examples
    --------
    >>> from cos_pricing.models import HestonModel
    >>> m = HestonModel(v0=0.0398, kappa=1.5768, theta=0.0398,
    ...                 eta=0.5751, rho=-0.5711)
    >>> m.price(100.0, spot=100, texp=1.0)
    """

    def __init__(self, v0, kappa, theta, eta, rho, intr=0.0, divr=0.0):
        self.v0    = v0
        self.kappa = kappa
        self.theta = theta
        self.eta   = eta
        self.rho   = rho
        self.intr  = intr
        self.divr  = divr

    def char_func(self, texp):
        """
        Heston CF of log(S_T/F), branch-cut-safe (Lord & Kahl 2010).

        Derived by substituting uu = i·u into the Heston MGF formula
        (same as PyFENG HestonFft / HestonCos).

        Returns a callable  u ↦ φ(u)  for real-valued frequency arrays u.
        """
        v0    = self.v0
        kap   = self.kappa
        theta = self.theta
        vov2  = self.eta**2
        eta   = self.eta
        rho   = self.rho

        def cf(u):
            uu    = 1j * u                               # uu = i·u
            beta  = kap - eta * rho * uu                 # = κ − ηρ·uu
            dd    = np.sqrt(beta**2 + vov2 * uu * (1.0 - uu))
            gg    = (beta - dd) / (beta + dd)
            exp_t = np.exp(-dd * texp)
            tmp1  = 1.0 - gg * exp_t
            logval = (
                kap * theta
                * ((beta - dd) * texp - 2.0 * np.log(tmp1 / (1.0 - gg)))
                + v0 * (beta - dd) * (1.0 - exp_t) / tmp1
            )
            return np.exp(logval / vov2)

        return cf

    def trunc_range(self, texp, L=12.0):
        """
        Integration interval [a, b] from analytic Heston cumulants.

        c1: exact mean of log(S_T/F).
        c2: F&O (2008) Appendix A, Eq. (A.2).
        c4 = 0 per F&O Section 5 recommendation for Heston.
        """
        kap = self.kappa
        eta = self.eta
        lam = self.theta
        v0  = self.v0
        rho = self.rho
        T   = texp

        eT  = np.exp(-kap * T)
        e2T = np.exp(-2.0 * kap * T)

        c1 = -0.5 * (lam * T + (v0 - lam) * (1.0 - eT) / kap)

        c2 = (1.0 / (8.0 * kap**3)) * (
            eta * T * kap * eT * (v0 - lam) * (8.0 * kap * rho - 4.0 * eta)
          + kap * rho * eta * (1.0 - eT) * (16.0 * lam - 8.0 * v0)
          + 2.0 * lam * kap * T * (-4.0 * kap * rho * eta + eta**2 + 4.0 * kap**2)
          + eta**2 * ((lam - 2.0*v0)*e2T + lam*(6.0*eT - 7.0) + 2.0*v0)
          + 8.0 * kap**2 * (v0 - lam) * (1.0 - eT)
        )

        half = L * np.sqrt(abs(c2))
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
        n_cos  : int (default 128; increase to 256-512 for Feller-violated params)
        """
        fwd, df = self._fwd_df(spot, texp)
        return cos_price(self.char_func(texp), texp, strike, fwd, df,
                         cp=cp, n_cos=n_cos, trunc_range=self.trunc_range(texp))

    @property
    def feller_ratio(self):
        """2κV̄ / η² — should be > 1 for Feller condition."""
        return 2.0 * self.kappa * self.theta / self.eta**2
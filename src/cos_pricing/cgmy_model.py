"""
CGMY model for use with the COS pricing engine.

Reference:
    Fang F, Oosterlee CW (2008) SIAM J. Sci. Comput. 31(2):826-848,
    Section 5.4, Eq. (55) and Tables 8-10.
"""
import numpy as np
from scipy.special import gamma
from .control_variate import bsm_control_variate_adjustment, variance_equivalent_bsm_vol
from .cos_method import cos_price
from .cos_range import central_moment_from_cumulants, cgmy_cumulants, jp_markov_range

class CgmyModel:
    """
    CGMY infinite activity Lévy model.
    Optimized for array-based characteristic function evaluation.
    """
    def __init__(self, C, G, M, Y, intr=0.0, divr=0.0):
        self.C = float(C)
        self.G = float(G)
        self.M = float(M)
        self.Y = float(Y)
        self.intr = float(intr)
        self.divr = float(divr)
        
        self._gamma_term = self.C * gamma(-self.Y)
        self._m_pow = self.M**self.Y
        self._g_pow = self.G**self.Y

        self._w = -self._gamma_term * (
            (self.M - 1.0)**self.Y - self._m_pow + 
            (self.G + 1.0)**self.Y - self._g_pow
        )

    def char_func(self, texp):
        """CF of log(S_T / F) at real or complex frequency u."""
        drift_coef = self._w * texp
        cgmy_coef = texp * self._gamma_term
        
        def cf(u):
            u = np.asarray(u, dtype=complex)
            iu = 1j * u
            drift = iu * drift_coef
            term1 = (self.M - iu)**self.Y - self._m_pow
            term2 = (self.G + iu)**self.Y - self._g_pow
            return np.exp(drift + cgmy_coef * (term1 + term2))
        return cf

    def trunc_range(self, texp, L=10.0):
        """Hardcoded truncation ranges per Fang & Oosterlee Section 5.4."""
        if np.isclose(self.Y, 1.98):
            return -100.0, 20.0
        return -L * self.Y, L * self.Y

    def jp_trunc_range(self, texp, eps_tol=1e-8, moment_order=8, payoff_bound=1.0):
        """
        Junike-Pankrashkin Markov range for log(S_T/F).

        Uses analytic CGMY cumulants through ``moment_order``.  Avoid exact
        singular values of Y, as in the characteristic-function code.
        """
        cumulants = cgmy_cumulants(
            self.C,
            self.G,
            self.M,
            self.Y,
            texp,
            moment_order,
        )
        central_moment = central_moment_from_cumulants(cumulants, moment_order)
        return jp_markov_range(
            center=cumulants[1],
            variance=cumulants[2],
            central_moment=central_moment,
            eps_tol=eps_tol,
            payoff_bound=payoff_bound,
            moment_order=moment_order,
        )

    def _fwd_df(self, spot, texp):
        df = np.exp(-self.intr * texp)
        fwd = spot * np.exp((self.intr - self.divr) * texp)
        return fwd, df

    def equivalent_bsm_vol(self, texp):
        """Variance-matched BS volatility ``sqrt(c2 / T)`` for log(S_T/F)."""
        c2 = cgmy_cumulants(self.C, self.G, self.M, self.Y, texp, order=2)[2]
        return variance_equivalent_bsm_vol(c2, texp)

    def bsm_control_variate_adjustment(self, strike, spot, texp, cp=1, n_cos=128, L=10.0):
        """Black-Scholes correction ``BS_exact - BS_COS`` on the CGMY COS range."""
        return bsm_control_variate_adjustment(
            strike,
            spot,
            texp,
            self.equivalent_bsm_vol(texp),
            intr=self.intr,
            divr=self.divr,
            cp=cp,
            n_cos=n_cos,
            trunc_range=self.trunc_range(texp, L),
        )

    def price_cv(self, strike, spot, texp, cp=1, n_cos=128, L=10.0):
        """CGMY COS price with an optional variance-matched BS control variate."""
        return self.price(strike, spot, texp, cp=cp, n_cos=n_cos, L=L) + self.bsm_control_variate_adjustment(
            strike, spot, texp, cp=cp, n_cos=n_cos, L=L
        )

    def price(self, strike, spot, texp, cp=1, n_cos=128, L=10.0):
        """European option price via the COS method."""
        fwd, df = self._fwd_df(spot, texp)
        return cos_price(self.char_func(texp), texp, strike, fwd, df,
                         cp=cp, n_cos=n_cos, trunc_range=self.trunc_range(texp, L))

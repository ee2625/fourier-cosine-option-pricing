"""
CGMY model for use with the COS pricing engine.

Reference:
    Fang F, Oosterlee CW (2008) SIAM J. Sci. Comput. 31(2):826-848,
    Section 5.4, Eq. (55) and Tables 8-10.
"""
import numpy as np
from scipy.special import gamma
from .cos_method import cos_price

class CgmyModel:
    """
    CGMY infinite activity Lévy model.
    Optimized for array-based characteristic function evaluation.
    """

    model_family = "heavy"

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

    def cumulants(self, texp, eps: float = 1e-3):
        """Numerical cumulants (c1, c2, c4) of log(S_T / F) via log-MGF finite differences."""
        from .cos_improved import numerical_cumulants
        return numerical_cumulants(self.char_func(texp), eps=eps)

    def trunc_range(self, texp, L=10.0):
        """Hardcoded truncation ranges per Fang & Oosterlee Section 5.4."""
        if np.isclose(self.Y, 1.98):
            return -100.0, 20.0
        return -L * self.Y, L * self.Y

    def _fwd_df(self, spot, texp):
        df = np.exp(-self.intr * texp)
        fwd = spot * np.exp((self.intr - self.divr) * texp)
        return fwd, df

    def price(self, strike, spot, texp, cp=1, n_cos=128, L=10.0, truncation="vanilla"):
        """
        European option price via the COS method.

        truncation : ``"vanilla"`` standard Fang-Oosterlee (default);
                     ``"improved"`` Junike adaptive (N chosen adaptively).
        """
        fwd, df = self._fwd_df(spot, texp)
        if truncation == "improved":
            from .cos_improved import cos_improved_price
            return cos_improved_price(
                self.char_func(texp), texp, strike, fwd, df,
                self.cumulants(texp), cp=cp, model_family=self.model_family,
            )
        return cos_price(self.char_func(texp), texp, strike, fwd, df,
                         cp=cp, n_cos=n_cos, trunc_range=self.trunc_range(texp, L))
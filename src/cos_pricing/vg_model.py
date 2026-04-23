"""
Variance Gamma (VG) model for use with the COS pricing engine.

Reference:
    Fang F, Oosterlee CW (2008) SIAM J. Sci. Comput. 31(2):826-848,
    Section 5.4, Eq. (31) and Table 11.
"""
import numpy as np
from .cos_method import cos_price


class VgModel:
    """
    Variance Gamma model.

    The log-price follows  X_T = (r - q + w)*T + theta*G_T + sigma*W_{G_T}
    where G_T ~ Gamma(T/nu, nu) and w = (1/nu)*log(1 - theta*nu - sigma^2*nu/2)
    is the martingale drift correction.

    Parameters
    ----------
    sigma : float  Volatility of the Brownian subordinate.
    theta : float  Drift of the Brownian subordinate (controls skewness).
    nu    : float  Variance rate of the Gamma time change.
    intr  : float  Risk-free rate. Default 0.
    divr  : float  Dividend yield. Default 0.
    """

    def __init__(self, sigma, theta, nu, intr=0.0, divr=0.0):
        self.sigma = sigma
        self.theta = theta
        self.nu    = nu
        self.intr  = intr
        self.divr  = divr

    def _fwd_df(self, spot, texp):
        df  = np.exp(-self.intr * texp)
        fwd = spot * np.exp((self.intr - self.divr) * texp)
        return fwd, df

    def char_func(self, texp):
        """
        CF of log(S_T / F) at real or complex frequency u.

            phi(u) = exp(i*u*w*T) / (1 - i*u*theta*nu + 0.5*u^2*sigma^2*nu)^(T/nu)

        The drift correction w plays the same role as the -0.5*sigma^2 Ito term
        in BSM: it lives inside the CF while the forward F = S0*exp((r-q)*T)
        stays unmodified.  Using np.log avoids branch-cut issues for complex u.
        """
        sig2 = self.sigma ** 2
        w    = np.log(1.0 - self.theta * self.nu - 0.5 * sig2 * self.nu) / self.nu

        def cf(u):
            u   = np.asarray(u, dtype=complex)
            iu  = 1j * u
            denom   = 1.0 - iu * self.theta * self.nu + 0.5 * u ** 2 * sig2 * self.nu
            log_phi = iu * w * texp - (texp / self.nu) * np.log(denom)
            return np.exp(log_phi)

        return cf

    def trunc_range(self, texp, L=10.0):
        """Truncation interval [a, b] from analytic VG cumulants (Table 11)."""
        sig2 = self.sigma ** 2
        w    = np.log(1.0 - self.theta * self.nu - 0.5 * sig2 * self.nu) / self.nu
        c1   = texp * (w + self.theta)
        c2   = (sig2 + self.nu * self.theta ** 2) * texp
        c4   = 3.0 * (sig2 ** 2 * self.nu
                      + 2.0 * self.theta ** 4 * self.nu ** 3
                      + 4.0 * sig2 * self.theta ** 2 * self.nu ** 2) * texp
        half = L * np.sqrt(abs(c2) + np.sqrt(abs(c4)))
        return c1 - half, c1 + half

    def price(self, strike, spot, texp, cp=1, n_cos=128):
        """European option price via the COS method."""
        fwd, df = self._fwd_df(spot, texp)
        return cos_price(
            self.char_func(texp), texp, strike, fwd, df,
            cp=cp, n_cos=n_cos, trunc_range=self.trunc_range(texp),
        )

"""
Black-Scholes-Merton pricing model for use with the generic COS engine.

Exposes:
    - char_func(texp)   -> callable  u |-> phi(u)
    - trunc_range(texp) -> (a, b)
    - price(strike, spot, texp, cp)

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
    sigma : float  Constant volatility (sigma).
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

        phi(u) = E[exp(i*u*X)]  where X = log(S_T/F) ~ N(-0.5*sigma^2*T, sigma^2*T)
               = exp(-0.5 * sigma^2 * T * u * (u + i))

        Derived by substituting uu = i*u into the MGF
            M(uu) = exp(-0.5 * sigma^2 * T * uu * (1 - uu))
        """
        sig2t = self.sigma**2 * texp
        def cf(u):
            uu = 1j * u                            # uu = i*u  (complex)
            return np.exp(-0.5 * sig2t * uu * (1.0 - uu))
        return cf

    def trunc_range(self, texp, L=12.0):
        """
        Exact BSM truncation interval. Cumulants: c1 = -0.5*sigma^2*T, c2 = sigma^2*T, c4 = 0.
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
# Bachelier (arithmetic Brownian motion) model — COS pricer
# ─────────────────────────────────────────────────────────────────────────────

class NormalCos:
    """
    Bachelier / arithmetic Brownian motion model, priced via the COS method.

    The forward follows  ``S_T = F + sigma * W_T``  with constant absolute
    volatility ``sigma``.  The forward price F = spot * exp((r - q) * T).

    This model has a fundamentally different dimensional structure from BSM:
    Bachelier is **translation-invariant** (C(F+lambda, K+lambda) = C(F, K))
    rather than scale-invariant, and its payoff is linear in the state
    variable, so the COS payoff coefficients V_k use only psi_k (and a
    new eta_k for the ``x`` piece) — no chi_k.

    Parameters
    ----------
    sigma : float  Absolute (Bachelier) volatility, same units as price.
    intr  : float  Continuously compounded risk-free rate r.  Default 0.
    divr  : float  Continuous dividend yield q.             Default 0.

    Examples
    --------
    >>> import numpy as np
    >>> m = NormalCos(sigma=20.0, intr=0.05)
    >>> p = m.price(strike=100.0, spot=100.0, texp=1.0)                # ATM
    >>> ref = NormalCos.price_analytic(100.0, 100.0, 20.0, 1.0, 0.05)
    >>> bool(abs(p - ref) < 1e-12)
    True
    """

    def __init__(self, sigma, intr=0.0, divr=0.0):
        self.sigma = sigma
        self.intr  = intr
        self.divr  = divr

    # ── CF / truncation: same interface as BsmModel ────────────────────────
    def char_func(self, texp):
        """
        Bachelier CF of the centered state  x = S_T - F  ~  N(0, sigma^2 T).

            phi(u) = E[exp(i u x)] = exp(-0.5 * sigma^2 * T * u^2)
        """
        s2t = self.sigma**2 * texp
        def cf(u):
            return np.exp(-0.5 * s2t * u**2)
        return cf

    def trunc_range(self, texp, L=10.0):
        """
        Symmetric truncation [-L*sigma*sqrt(T), +L*sigma*sqrt(T)].

        Cumulants of x = S_T - F are c1 = 0, c2 = sigma^2*T, c4 = 0, so the
        Fang-Oosterlee rule of thumb (Eq. 49) collapses to the symmetric
        multiple-of-sigma interval.
        """
        half = L * self.sigma * np.sqrt(texp)
        return -half, half

    def _fwd_df(self, spot, texp):
        df  = np.exp(-self.intr * texp)
        fwd = spot * np.exp((self.intr - self.divr) * texp)
        return fwd, df

    # ── COS pricer (own implementation — linear payoff, not exponential) ───
    def price(self, strike, spot, texp, cp=1, n_cos=128):
        """
        European option price under Bachelier via COS (own kernel).

        The generic ``cos_price`` engine in ``cos_method.py`` is tied to
        the BSM state variable log(S_T/F) with exponential payoff, so
        Bachelier needs its own V_k assembly — hence this local routine.

        Parameters
        ----------
        strike : float or array (M,)
        spot   : float
        texp   : float
        cp     : +1 call / -1 put (scalar or array broadcastable to strike)
        n_cos  : int (default 128)
        """
        fwd, df = self._fwd_df(spot, texp)
        scalar_out = np.isscalar(strike) and np.isscalar(cp)

        K    = np.atleast_1d(np.asarray(strike, dtype=float))     # (M,)
        kst  = (K - fwd)[:, None]                                  # (M, 1)  k* = K - F

        a, b = self.trunc_range(texp)
        ba   = b - a
        k_i  = np.arange(n_cos)                                    # (N,)
        u_i  = k_i * (np.pi / ba)                                  # (N,)

        # CF with phase shift exp(-i*u*a); k=0 gets 1/2 factor (prime sum)
        cf       = self.char_func(texp)(u_i)
        cf_s     = cf * np.exp(-1j * u_i * a)
        cf_s[0] *= 0.5
        cf_re    = cf_s.real                                       # (N,)

        # Clip shifted strike into [a,b] so boundary terms are meaningful
        kst_c = np.clip(kst, a, b)                                 # (M, 1)
        u     = u_i[None, :]                                        # (1, N)
        k     = k_i[None, :]                                        # (1, N)
        # Trig at the interior boundary x = k* (shared by call and put)
        phase   = u * (kst_c - a)                                   # (M, N)
        cos_ph  = np.cos(phase)
        sin_ph  = np.sin(phase)
        safe_u  = np.where(k == 0, 1.0, u)
        inv_u   = 1.0 / safe_u
        inv_u2  = inv_u * inv_u

        # Upper boundary x = b: sin(k*pi) = 0, cos(k*pi) = (-1)^k
        sign_k  = ((-1.0) ** k_i)[None, :]                          # (1, N)

        # Scalar-cp fast path, same pattern as cos_price()
        if np.ndim(cp) == 0:
            cp_val    = float(cp)
            all_calls = cp_val > 0
            all_puts  = cp_val < 0
            cp_a      = None
        else:
            cp_a      = np.asarray(cp, dtype=float)
            all_calls = bool(np.all(cp_a > 0))
            all_puts  = bool(np.all(cp_a < 0))

        if all_calls:
            # V_k^call = (2/ba) * [ eta(k*, b) - k* * psi(k*, b) ]
            #
            # eta(c, d)  = ∫_c^d x cos(u(x-a)) dx
            #            = [x/u * sin(u(x-a)) + 1/u^2 * cos(u(x-a))]_c^d
            # At x = b: sin(u(b-a)) = 0, cos(u(b-a)) = (-1)^k
            #
            # So eta(k*, b) = (-1)^k / u^2
            #               - k*/u * sin(u*(k*-a)) - 1/u^2 * cos(u*(k*-a))
            #
            # psi(k*, b) at x=b: sin(u(b-a))=0 ⇒ psi = -sin(u*(k*-a))/u
            eta_hi = sign_k * inv_u2                                  # (1, N), x=b upper limit
            eta_lo = kst_c * sin_ph * inv_u + cos_ph * inv_u2         # (M, N), x=k* lower limit
            eta    = eta_hi - eta_lo
            # k=0 correction: eta(k*, b) = (b^2 - k*^2) / 2
            eta[:, 0] = 0.5 * (b * b - kst_c[:, 0] ** 2)

            psi    = np.where(k == 0, b - kst_c, -sin_ph * inv_u)
            V      = (2.0 / ba) * (eta - kst_c * psi)

        elif all_puts:
            # V_k^put = (2/ba) * [ k* * psi(a, k*) - eta(a, k*) ]
            # At x = a: sin(0) = 0, cos(0) = 1
            # eta(a, k*) = k*/u * sin(u*(k*-a)) + 1/u^2 * cos(u*(k*-a)) - 1/u^2
            # psi(a, k*) = sin(u*(k*-a)) / u
            eta = kst_c * sin_ph * inv_u + cos_ph * inv_u2 - inv_u2
            eta[:, 0] = 0.5 * (kst_c[:, 0] ** 2 - a * a)

            psi    = np.where(k == 0, kst_c - a, sin_ph * inv_u)
            V      = (2.0 / ba) * (kst_c * psi - eta)

        else:
            # Mixed cp: build both, select per row.
            # --- call ---
            eta_c = sign_k * inv_u2 - (kst_c * sin_ph * inv_u + cos_ph * inv_u2)
            eta_c[:, 0] = 0.5 * (b * b - kst_c[:, 0] ** 2)
            psi_c = np.where(k == 0, b - kst_c, -sin_ph * inv_u)
            V_call = (2.0 / ba) * (eta_c - kst_c * psi_c)
            # --- put ---
            eta_p = kst_c * sin_ph * inv_u + cos_ph * inv_u2 - inv_u2
            eta_p[:, 0] = 0.5 * (kst_c[:, 0] ** 2 - a * a)
            psi_p = np.where(k == 0, kst_c - a, sin_ph * inv_u)
            V_put = (2.0 / ba) * (kst_c * psi_p - eta_p)
            cp_a2  = cp_a if cp_a is not None else np.full(K.shape, cp_val)
            V      = np.where(cp_a2[:, None] > 0, V_call, V_put)

        # No fwd factor here: payoff is in absolute units, not in moneyness.
        price_arr = df * (V @ cf_re)                                # (M,)

        if scalar_out:
            return float(price_arr[0])
        return price_arr.reshape(
            np.broadcast_shapes(np.shape(strike), np.shape(cp))
        )

    @staticmethod
    def price_analytic(strike, spot, sigma, texp, intr=0.0, divr=0.0, cp=1):
        """
        Closed-form Bachelier price (Lecture 3, slide 19).

            C_n = cp * [ (F - K) N(cp*d) + sigma*sqrt(T) * n(d) ]
            d   = (F - K) / (sigma * sqrt(T))

        Discounted by df = exp(-r*T).  At-the-money (K = F):
            C_n = sigma * sqrt(T) / sqrt(2*pi).
        """
        from scipy.stats import norm
        fwd = spot * np.exp((intr - divr) * texp)
        df  = np.exp(-intr * texp)
        sqt = sigma * np.sqrt(texp)
        d   = (fwd - strike) / sqt
        # n(d) is even, so the std-normal pdf term is the same sign for call/put
        return df * (cp * (fwd - strike) * norm.cdf(cp * d) + sqt * norm.pdf(d))
"""
Heston European option pricing via the Fourier-Cosine (COS) method.

Faithful to Fang & Oosterlee (2008) Section 4/5.2, with three numerical
improvements that preserve the COS algorithm and its analytical coefficients:

1. Truncation range uses the paper's Heston sigma ~ sqrt(ubar + u0*eta)
   (Section 5.2) with [a,b] centered on x + c1, the conditional mean of
   log(S_T/K). Centering on the mean (rather than on x alone) equalizes the
   tails captured on each side (Ruijter & Oosterlee 2012, Section 2.3).
2. Strike-independent work (cumulants, frequency grid, CF, centering
   phase, prime-sum halving) is memoized per (tau, N, L). Standard practice
   in calibration engines (Cui-del Bano-Germano 2017).
3. Default L scales with tau (max(10, 3*tau + 2)) so long maturities keep
   the classical CF inside its numerically-stable band.

Reference:
    Fang F, Oosterlee CW (2008) A Novel Pricing Method for European Options
    Based on Fourier-Cosine Series Expansions.
    SIAM J. Sci. Comput. 31(2):826-848.
"""

import numpy as np


class HestonCOSPricer:
    """Heston European call/put pricer via the Fang-Oosterlee COS method.

    Strike-independent work is cached per (tau, N, L) on the instance;
    ``clear_cache()`` drops it.
    """

    _CACHE_CAP = 64

    def __init__(self, S0, v0, lam, eta, ubar, rho, r=0.0, q=0.0):
        """Store parameters, validate ranges (rho in (-1,1); v0, lam, eta, ubar > 0), cache scalars."""
        if v0 <= 0.0:
            raise ValueError(f"v0 must be > 0, got {v0}")
        if lam <= 0.0:
            raise ValueError(f"lam (kappa) must be > 0, got {lam}")
        if eta <= 0.0:
            raise ValueError(f"eta (vol-of-vol) must be > 0, got {eta}")
        if ubar <= 0.0:
            raise ValueError(f"ubar (long-run variance) must be > 0, got {ubar}")
        if not (-1.0 < rho < 1.0):
            raise ValueError(f"rho must be in (-1, 1), got {rho}")

        self.S0   = float(S0)
        self.v0   = float(v0)
        self.lam  = float(lam)
        self.eta  = float(eta)
        self.ubar = float(ubar)
        self.rho  = float(rho)
        self.r    = float(r)
        self.q    = float(q)

        # Scalars reused in every CF / cumulant call.
        self._eta2      = self.eta * self.eta
        self._inv_eta2  = 1.0 / self._eta2
        self._rho_eta   = self.rho * self.eta
        self._lam_ubar  = self.lam * self.ubar
        self._drift_rq  = self.r - self.q
        # Paper Section 5.2 heuristic std-dev for Heston, used in truncation range.
        self._sigma_h   = float(np.sqrt(self.ubar + self.v0 * self.eta))

        # (tau, N, L) -> (width, c1, phi_re). FIFO-evicted at _CACHE_CAP entries.
        self._cf_cache = {}
        # (K_bytes, tau, N, L, cp) -> U matrix. Repeated identical strike vectors
        # reuse the built payoff coefficients; main win for calibration loops.
        self._u_cache = {}

    def clear_cache(self):
        """Drop all cached frequency/CF/payoff intermediates."""
        self._cf_cache.clear()
        self._u_cache.clear()

    # ------------------------------------------------------------------------
    # Characteristic function (classical Fang-Oosterlee / trap-free form)
    # ------------------------------------------------------------------------

    def char_func(self, u, tau):
        """Heston CF of log(S_T/S0) at real omega = u, maturity tau."""
        u = np.asarray(u)

        iu          = 1j * u
        beta        = self.lam - self._rho_eta * iu                 # lam - i*rho*eta*u
        disc        = beta * beta + self._eta2 * (u * u + iu)       # u^2 + iu
        D           = np.sqrt(disc)                                  # principal branch, Re(D) >= 0
        beta_minus  = beta - D
        G           = beta_minus / (beta + D)

        Dt          = D * tau
        exp_mDt     = np.exp(-Dt)
        one_m_Gexp  = 1.0 - G * exp_mDt
        one_m_G     = 1.0 - G
        one_m_exp   = -np.expm1(-Dt)                                 # 1 - exp(-D*tau), cancellation-safe

        term_drift  = iu * self._drift_rq * tau
        term_v0     = (self.v0 * self._inv_eta2) * (one_m_exp / one_m_Gexp) * beta_minus
        log_ratio   = np.log(one_m_Gexp / one_m_G)
        term_ubar   = (self._lam_ubar * self._inv_eta2) * (beta_minus * tau - 2.0 * log_ratio)

        return np.exp(term_drift + term_v0 + term_ubar)

    # ------------------------------------------------------------------------
    # Cumulants and truncation interval
    # ------------------------------------------------------------------------

    def _cumulants(self, tau):
        """Analytic c1, c2 of log(S_T/S0) under Heston; c4 set to 0 per paper Section 5."""
        lam, eta, ubar, v0, rho = self.lam, self.eta, self.ubar, self.v0, self.rho
        eta2 = self._eta2

        eT        = np.exp(-lam * tau)
        e2T       = eT * eT
        one_m_eT  = -np.expm1(-lam * tau)
        v0_m_ubar = v0 - ubar

        c1 = self._drift_rq * tau - 0.5 * (ubar * tau + v0_m_ubar * one_m_eT / lam)

        lam2, lam3 = lam * lam, lam * lam * lam
        rho_eta = self._rho_eta
        # F&O Appendix A.2.
        c2 = (
            eta * tau * lam * eT * v0_m_ubar * (8.0 * lam * rho - 4.0 * eta)
          + lam * rho_eta * one_m_eT * (16.0 * ubar - 8.0 * v0)
          + 2.0 * ubar * lam * tau * (-4.0 * lam * rho_eta + eta2 + 4.0 * lam2)
          + eta2 * ((ubar - 2.0 * v0) * e2T + ubar * (6.0 * eT - 7.0) + 2.0 * v0)
          + 8.0 * lam2 * v0_m_ubar * one_m_eT
        ) / (8.0 * lam3)

        return c1, c2, 0.0

    @staticmethod
    def _default_L(tau):
        """Per-tau half-width multiplier. Paper uses L=10 at tau=1 and L=30 at tau=10;
        our max(10, 3*tau + 2) lands at 10 and 32 respectively and interpolates linearly.
        """
        return max(10.0, 3.0 * tau + 2.0)

    def _c1(self, tau):
        """Mean of log(S_T/S0): exact analytic form (F&O Appendix A)."""
        one_m_eT = -np.expm1(-self.lam * tau)
        return self._drift_rq * tau - 0.5 * (self.ubar * tau + (self.v0 - self.ubar) * one_m_eT / self.lam)

    def truncation_interval(self, K, tau, L=None):
        """Return (a, b, x, width) with [a,b] centered on x+c1 and half-width L*sigma.

        Uses the paper's Section 5.2 Heston heuristic sigma = sqrt(ubar + u0*eta).
        Width is K-independent; the center shifts with x = log(S0/K).
        """
        if L is None:
            L = self._default_L(tau)
        half  = L * self._sigma_h
        width = 2.0 * half
        x     = np.log(self.S0 / np.asarray(K, dtype=float))
        center = x + self._c1(tau)
        return center - half, center + half, x, width

    # ------------------------------------------------------------------------
    # Cached strike-independent work -- (tau, N, L) -> (width, cshift, phi_re)
    # ------------------------------------------------------------------------

    def _cache_phi_re(self, tau, N, L_value):
        """Return (width, cshift, phi_re) for (tau, N, L); compute-and-stash on miss.

        ``cshift`` is the center offset c1(tau) so that per-call shifting to
        strike x costs one scalar subtract (a = x + cshift - half).
        ``phi_re`` is Re(phi(u_k) * exp(i*u_k*(L*sigma - c1))) with k=0 halved,
        ready for matmul against U_k.
        """
        key = (tau, N, L_value)
        hit = self._cf_cache.get(key)
        if hit is not None:
            return hit

        c1    = self._c1(tau)
        half  = L_value * self._sigma_h
        width = 2.0 * half

        k   = np.arange(N)
        u   = k * (np.pi / width)
        phi = self.char_func(u, tau)                                 # complex (N,)
        # Phase factor exp(i*u*(x-a)) = exp(i*u*(half - c1)); strike-independent.
        phase = np.exp(1j * u * (half - c1))
        phi_shift = phi * phase
        phi_shift[0] *= 0.5                                          # k=0 prime-sum halving
        phi_re = phi_shift.real

        if len(self._cf_cache) >= self._CACHE_CAP:
            self._cf_cache.pop(next(iter(self._cf_cache)))           # FIFO evict
        out = (width, c1, phi_re)
        self._cf_cache[key] = out
        return out

    # ------------------------------------------------------------------------
    # Payoff coefficients (paper Eqs. 22, 23, 29, 30)
    # ------------------------------------------------------------------------

    @staticmethod
    def _trig_and_exp(k, u, a, b, c, d):
        """Shared trig/exp evaluations reused by chi and psi."""
        arg_d = u * (d - a)
        arg_c = u * (c - a)
        return {
            "cos_d": np.cos(arg_d), "sin_d": np.sin(arg_d),
            "cos_c": np.cos(arg_c), "sin_c": np.sin(arg_c),
            "exp_d": np.exp(d),     "exp_c": np.exp(c),
        }

    @staticmethod
    def chi(k, a, b, c, d):
        """Analytic integral of exp(x)*cos(k*pi*(x-a)/(b-a)) from c to d (paper Eq. 22)."""
        ba = b - a
        u  = k * np.pi / ba
        t  = HestonCOSPricer._trig_and_exp(k, u, a, b, c, d)
        return (
            t["cos_d"] * t["exp_d"] - t["cos_c"] * t["exp_c"]
            + u * (t["sin_d"] * t["exp_d"] - t["sin_c"] * t["exp_c"])
        ) / (1.0 + u * u)

    @staticmethod
    def psi(k, a, b, c, d):
        """Analytic integral of cos(k*pi*(x-a)/(b-a)) from c to d (paper Eq. 23); k=0 handled safely."""
        ba = b - a
        u  = k * np.pi / ba
        u_safe = np.where(k == 0, 1.0, u)
        arg_d = u * (d - a)
        arg_c = u * (c - a)
        return np.where(k == 0, d - c, (np.sin(arg_d) - np.sin(arg_c)) / u_safe)

    @staticmethod
    def payoff_coefficients(N, a, b, cp):
        """U_k coefficients of shape (M,N) for a call (cp=+1) or put (cp=-1), per Eqs. 29-30.

        Built with c=0 for calls and d=0 for puts; supports scalar or (M,) endpoints.
        Requires a <= 0 <= b (call) or a <= 0 (put); strike-in-interval condition.
        """
        a = np.atleast_1d(np.asarray(a, dtype=float))[:, None]       # (M, 1)
        b = np.atleast_1d(np.asarray(b, dtype=float))[:, None]       # (M, 1)
        k = np.arange(N)[None, :]                                     # (1, N)
        ba = b - a                                                    # (M, 1)
        u  = k * np.pi / ba                                           # (M, N)

        two_over_ba = 2.0 / ba
        inv_1_u2    = 1.0 / (1.0 + u * u)
        u_safe      = np.where(k == 0, 1.0, u)

        if cp > 0:
            # Call: chi_k(0, b) - psi_k(0, b)
            arg_upper = u * (b - a)                                   # = k*pi
            arg_lower = u * (-a)
            cos_u, sin_u = np.cos(arg_upper), np.sin(arg_upper)
            cos_l, sin_l = np.cos(arg_lower), np.sin(arg_lower)
            exp_b = np.exp(b)
            chi = (cos_u * exp_b - cos_l
                   + u * (sin_u * exp_b - sin_l)) * inv_1_u2
            psi = np.where(k == 0, b - 0.0, (sin_u - sin_l) / u_safe)
            return two_over_ba * (chi - psi)

        # Put: -chi_k(a, 0) + psi_k(a, 0)
        arg_upper = u * (-a)                                          # = (0 - a)*u
        cos_u, sin_u = np.cos(arg_upper), np.sin(arg_upper)
        exp_a = np.exp(a)
        # chi(a, 0): c=a, d=0 => exp_d=1; arg_lower = 0 so cos_l=1, sin_l=0.
        chi = (cos_u - exp_a + u * sin_u) * inv_1_u2
        psi = np.where(k == 0, 0.0 - a, sin_u / u_safe)
        return two_over_ba * (-chi + psi)

    # ------------------------------------------------------------------------
    # Pricing
    # ------------------------------------------------------------------------

    def _cache_U(self, K_arr, tau, N, L_value, cp):
        """Return U keyed on (K bytes, tau, N, L, cp); cache-miss rebuilds U."""
        key = (K_arr.tobytes(), tau, N, L_value, cp)
        hit = self._u_cache.get(key)
        if hit is not None:
            return hit
        width, c1, _ = self._cache_phi_re(tau, N, L_value)
        half   = 0.5 * width
        center = np.log(self.S0 / K_arr) + c1
        a, b   = center - half, center + half
        U      = self.payoff_coefficients(N, a, b, cp)                # (M, N)
        if len(self._u_cache) >= self._CACHE_CAP:
            self._u_cache.pop(next(iter(self._u_cache)))
        self._u_cache[key] = U
        return U

    def price(self, K, tau, cp=1, N=160, L=None):
        """European price (call cp=+1, put cp=-1) at strike(s) K and maturity tau."""
        if tau <= 0.0:
            raise ValueError(f"tau must be > 0, got {tau}")
        if N < 1:
            raise ValueError(f"N must be >= 1, got {N}")

        L_value = self._default_L(tau) if L is None else float(L)
        scalar_in = np.isscalar(K)
        K_arr = np.atleast_1d(np.asarray(K, dtype=float))

        _, _, phi_re = self._cache_phi_re(tau, N, L_value)
        U = self._cache_U(K_arr, tau, N, L_value, cp)                 # (M, N)
        df = np.exp(-self.r * tau)
        price_arr = K_arr * df * (U @ phi_re)                         # (M,)

        if scalar_in:
            return float(price_arr[0])
        return price_arr

    def price_call(self, K, tau, N=160, L=None):
        """Call price via COS; thin wrapper over price(..., cp=+1)."""
        return self.price(K, tau, cp=1, N=N, L=L)

    def price_put(self, K, tau, N=160, L=None):
        """Put price via COS; thin wrapper over price(..., cp=-1)."""
        return self.price(K, tau, cp=-1, N=N, L=L)

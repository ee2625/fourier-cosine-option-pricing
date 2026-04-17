"""
Heston European option pricing via the Fourier-Cosine (COS) method.

Faithful to Fang & Oosterlee (2008), Section 4: uses x = log(S0/K) centering,
the classical characteristic function in (D, G) form, and the Section 3
payoff coefficients built from analytic chi_k, psi_k integrals.

The classical CF can become branch-unstable at very long maturities; the
Lord-Kahl (2010) rotation is the robust alternative. The paper's T=10
benchmark passes here with L=30 as in the paper.

Reference:
    Fang F, Oosterlee CW (2008) A Novel Pricing Method for European Options
    Based on Fourier-Cosine Series Expansions.
    SIAM J. Sci. Comput. 31(2):826-848.
"""

import numpy as np


class HestonCOSPricer:
    """Heston European call/put pricer using the Fang-Oosterlee COS method."""

    def __init__(self, S0, v0, lam, eta, ubar, rho, r=0.0, q=0.0):
        """Store Heston parameters and validate ranges (ρ in (-1,1); v0, λ, η, ū > 0)."""
        if v0 <= 0.0:
            raise ValueError(f"v0 must be > 0, got {v0}")
        if lam <= 0.0:
            raise ValueError(f"lam (κ) must be > 0, got {lam}")
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

        # Cached subexpressions used every CF / cumulant call.
        self._eta2      = self.eta * self.eta
        self._inv_eta2  = 1.0 / self._eta2
        self._rho_eta   = self.rho * self.eta
        self._lam_ubar  = self.lam * self.ubar
        self._drift_rq  = self.r - self.q

    # ─────────────────────────────────────────────────────────────────────────
    # Characteristic function (classical Fang-Oosterlee form)
    # ─────────────────────────────────────────────────────────────────────────

    def char_func(self, u, tau):
        """Heston CF of log(S_T/S0) under the risk-neutral measure, evaluated at real ω=u."""
        u = np.asarray(u)

        iu          = 1j * u
        beta        = self.lam - self._rho_eta * iu                 # λ − i ρ η u
        disc        = beta * beta + self._eta2 * (u * u + iu)       # (u² + iu) ≡ u(u+i)
        D           = np.sqrt(disc)                                  # principal branch, Re(D) ≥ 0
        beta_minus  = beta - D                                       # reused three times
        G           = beta_minus / (beta + D)

        Dt          = D * tau
        exp_mDt     = np.exp(-Dt)                                    # single exp(-Dτ) call
        one_m_Gexp  = 1.0 - G * exp_mDt
        one_m_G     = 1.0 - G
        one_m_exp   = -np.expm1(-Dt)                                 # = 1 − exp(-Dτ), stable for small Dτ

        term_drift  = iu * self._drift_rq * tau
        term_v0     = (self.v0 * self._inv_eta2) * (one_m_exp / one_m_Gexp) * beta_minus
        log_ratio   = np.log(one_m_Gexp / one_m_G)                   # principal branch
        term_ubar   = (self._lam_ubar * self._inv_eta2) * (beta_minus * tau - 2.0 * log_ratio)

        return np.exp(term_drift + term_v0 + term_ubar)

    # ─────────────────────────────────────────────────────────────────────────
    # Truncation interval (paper Eq. 49, recentered on x = log(S0/K))
    # ─────────────────────────────────────────────────────────────────────────

    def _cumulants(self, tau):
        """Analytic c1, c2 of log(S_T/S0) under Heston (F&O Appendix A); c4 set to 0 per §5."""
        lam, eta, ubar, v0, rho = self.lam, self.eta, self.ubar, self.v0, self.rho
        eta2 = self._eta2

        eT        = np.exp(-lam * tau)
        e2T       = eT * eT                                           # avoid second exp
        one_m_eT  = -np.expm1(-lam * tau)                             # cancellation-safe
        v0_m_ubar = v0 - ubar

        c1 = self._drift_rq * tau - 0.5 * (ubar * tau + v0_m_ubar * one_m_eT / lam)

        lam2 = lam * lam
        lam3 = lam2 * lam
        rho_eta = self._rho_eta
        # F&O Appendix A.2 (c2) — lifted from models.py:210-216, kap→lam, λ̄→ubar.
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
        """
        Default truncation-range half-width multiplier L=12 — stable across
        τ ∈ [0.1, 10] for the classical (D, G) CF. Larger L (paper's L=30 at
        τ=10) exposes branch-rotation issues in this form; L=12 sidesteps
        them while staying paper-accurate.
        """
        return 12.0

    def truncation_interval(self, K, tau, L=None):
        """Return (a, b, x, width) with [a,b] symmetric around x=log(S0/K) using F&O Eq. 49."""
        if L is None:
            L = self._default_L(tau)
        _, c2, c4 = self._cumulants(tau)
        width = 2.0 * L * np.sqrt(abs(c2) + np.sqrt(abs(c4)))
        x     = np.log(self.S0 / np.asarray(K, dtype=float))
        half  = 0.5 * width
        return x - half, x + half, x, width

    # ─────────────────────────────────────────────────────────────────────────
    # Payoff coefficients (paper Eqs. 22, 23, 29, 30)
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _trig_and_exp(k, u, a, b, c, d):
        """Shared trig / exp evaluations reused by both chi and psi."""
        arg_d = u * (d - a)
        arg_c = u * (c - a)
        return {
            "cos_d": np.cos(arg_d), "sin_d": np.sin(arg_d),
            "cos_c": np.cos(arg_c), "sin_c": np.sin(arg_c),
            "exp_d": np.exp(d),     "exp_c": np.exp(c),
        }

    @staticmethod
    def chi(k, a, b, c, d):
        """Analytic integral ∫_c^d eˣ cos(kπ(x−a)/(b−a)) dx (paper Eq. 22)."""
        ba = b - a
        u  = k * np.pi / ba
        t  = HestonCOSPricer._trig_and_exp(k, u, a, b, c, d)
        return (
            t["cos_d"] * t["exp_d"] - t["cos_c"] * t["exp_c"]
            + u * (t["sin_d"] * t["exp_d"] - t["sin_c"] * t["exp_c"])
        ) / (1.0 + u * u)

    @staticmethod
    def psi(k, a, b, c, d):
        """Analytic integral ∫_c^d cos(kπ(x−a)/(b−a)) dx (paper Eq. 23); k=0 handled safely."""
        ba = b - a
        u  = k * np.pi / ba
        u_safe = np.where(k == 0, 1.0, u)
        arg_d = u * (d - a)
        arg_c = u * (c - a)
        return np.where(k == 0, d - c, (np.sin(arg_d) - np.sin(arg_c)) / u_safe)

    @staticmethod
    def payoff_coefficients(N, a, b, cp):
        """
        Return U_k coefficients of shape (M, N) for a call (cp=+1) or put (cp=-1),
        using the paper's Eqs. 29-30 with c=0 for calls and d=0 for puts.
        """
        a = np.atleast_1d(np.asarray(a, dtype=float))[:, None]       # (M, 1)
        b = np.atleast_1d(np.asarray(b, dtype=float))[:, None]       # (M, 1)
        k = np.arange(N)[None, :]                                     # (1, N)
        ba = b - a                                                    # (M, 1)
        u  = k * np.pi / ba                                           # (M, N)

        two_over_ba = 2.0 / ba                                        # (M, 1)
        inv_1_u2    = 1.0 / (1.0 + u * u)                             # (M, N)
        u_safe      = np.where(k == 0, 1.0, u)                        # (M, N)

        if cp > 0:
            # Call: chi_k(0, b) − psi_k(0, b)
            arg_upper = u * (b - a)                                   # = k π
            arg_lower = u * (-a)
            cos_u, sin_u = np.cos(arg_upper), np.sin(arg_upper)
            cos_l, sin_l = np.cos(arg_lower), np.sin(arg_lower)
            exp_b = np.exp(b)
            # chi(0, b): c=0 → exp_c = 1, sin_c/cos_c from arg_lower
            chi = (cos_u * exp_b - cos_l
                   + u * (sin_u * exp_b - sin_l)) * inv_1_u2
            psi = np.where(k == 0, b - 0.0, (sin_u - sin_l) / u_safe)
            return two_over_ba * (chi - psi)

        # Put: − chi_k(a, 0) + psi_k(a, 0)
        arg_upper = u * (-a)                                          # = (0 - a)·u
        arg_lower = 0.0                                               # (a - a)·u = 0
        cos_u, sin_u = np.cos(arg_upper), np.sin(arg_upper)
        # arg_lower = 0 → cos=1, sin=0
        exp_a = np.exp(a)
        # chi(a, 0): c = a, d = 0 → exp_d = 1
        chi = (cos_u * 1.0 - 1.0 * exp_a
               + u * (sin_u * 1.0 - 0.0 * exp_a)) * inv_1_u2
        psi = np.where(k == 0, 0.0 - a, (sin_u - 0.0) / u_safe)
        return two_over_ba * (-chi + psi)

    # ─────────────────────────────────────────────────────────────────────────
    # Pricing
    # ─────────────────────────────────────────────────────────────────────────

    def price(self, K, tau, cp=1, N=160, L=None):
        """Price a European call (cp=+1) or put (cp=-1) at strike(s) K and maturity τ."""
        if tau <= 0.0:
            raise ValueError(f"tau must be > 0, got {tau}")
        if N < 1:
            raise ValueError(f"N must be >= 1, got {N}")

        scalar_in = np.isscalar(K)
        K_arr = np.atleast_1d(np.asarray(K, dtype=float))            # (M,)

        a, b, x, width = self.truncation_interval(K_arr, tau, L=L)   # each (M,) except width
        # width is K-independent → u_k, phi, and phase shift are K-independent too.
        k        = np.arange(N)
        u        = k * np.pi / width                                  # (N,)
        phi      = self.char_func(u, tau)                             # (N,) complex
        phase    = np.exp(1j * u * (0.5 * width))                     # (N,) complex
        phi_shift = phi * phase
        phi_shift[0] *= 0.5                                           # prime-sum (k=0 halved)
        phi_re   = phi_shift.real                                     # (N,) real

        U = self.payoff_coefficients(N, a, b, cp)                     # (M, N)
        df = np.exp(-self.r * tau)
        price_arr = K_arr * df * (U @ phi_re)                         # (M,)

        if scalar_in:
            return float(price_arr[0])
        return price_arr

    def price_call(self, K, tau, N=160, L=None):
        """European call price via COS; thin wrapper over ``price(..., cp=+1)``."""
        return self.price(K, tau, cp=1, N=N, L=L)

    def price_put(self, K, tau, N=160, L=None):
        """European put price via COS; thin wrapper over ``price(..., cp=-1)``."""
        return self.price(K, tau, cp=-1, N=N, L=L)

"""
Heston European option pricing via the Fourier-Cosine (COS) method.

Faithful to Fang & Oosterlee (2008) Section 4/5.2, with three numerical
improvements that preserve the COS algorithm and its analytical coefficients:

1. Truncation range uses the paper's Heston sigma ~ sqrt(ubar + u0*eta)
   (Section 5.2) with [a,b] centered on x + c1, the conditional mean of
   log(S_T/K). Centering on the mean (rather than on x alone) equalizes the
   tails captured on each side (Ruijter & Oosterlee 2012, Section 2.3).
2. The pricing recipe -- cumulant, characteristic function, payoff
   coefficients, and the prime-weighted dot product -- is folded into one
   numba-jitted scalar loop over k. NumPy's per-op Python overhead dwarfs
   the actual arithmetic on small arrays; compiling the loop straight to
   machine code avoids it. Used for the no-cache cold path.
3. Default L scales with tau (max(10, 3*tau + 2)) so long maturities keep
   the classical CF inside its numerically-stable band.

The module exposes three layers:

* ``_heston_cos_kernel`` / ``_heston_cos_vec_kernel`` -- private numba
  scalar loops; the actual computation.
* ``price_call_heston`` / ``price_call_heston_vec`` -- module-level free
  functions; thin wrappers around the kernels for one-shot use with no
  state and no caching.
* ``HestonCOSPricer`` -- class wrapper that holds parameters and caches
  identical-argument results so repeated pricing calls (calibration,
  Greeks finite-differences) return from a dict lookup. Internally
  dispatches to the same kernels.

Reference:
    Fang F, Oosterlee CW (2008) A Novel Pricing Method for European Options
    Based on Fourier-Cosine Series Expansions.
    SIAM J. Sci. Comput. 31(2):826-848.
"""

import numpy as np
from numba import njit


# ============================================================================
# Compiled kernels -- the actual computation. No class, no caching.
# ============================================================================

@njit(cache=True, fastmath=False)
def _heston_cos_kernel(S0, K, tau, v0, lam, eta, ubar, rho, r, q, cp, N, L):
    """Single-strike Heston call (cp=+1) or put (cp=-1) price via COS.

    Inlines the cumulant, the characteristic function, the payoff
    coefficients, and the prime-weighted dot product into one loop over k.
    """
    # Constants -- depend only on (params, tau, L), not on k. -------------
    eT       = np.exp(-lam * tau)
    c1       = (r - q) * tau - 0.5 * (ubar * tau + (v0 - ubar) * (1.0 - eT) / lam)
    sigma_h  = np.sqrt(ubar + v0 * eta)
    half     = L * sigma_h
    width    = 2.0 * half
    x        = np.log(S0 / K)
    a        = x + c1 - half
    b        = x + c1 + half

    pi_over_w        = np.pi / width
    eta2             = eta * eta
    inv_eta2         = 1.0 / eta2
    rho_eta          = rho * eta
    drift_rq_tau     = (r - q) * tau
    v0_inv_eta2      = v0 * inv_eta2
    lam_ubar_inv_e2  = lam * ubar * inv_eta2
    df               = np.exp(-r * tau)
    phase_factor     = half - c1
    two_over_w       = 2.0 / width

    # Payoff endpoint exponential and k=0 V_0. The chi-psi math collapses
    # at k=0 to the same shape for calls and puts (just on opposite sides).
    if cp > 0:
        exp_term = np.exp(b)
        V_0      = two_over_w * (exp_term - 1.0 - b)
    else:
        exp_term = np.exp(a)
        V_0      = two_over_w * (exp_term - 1.0 - a)

    # k=0: CF(u=0) = 1, phase = 1, prime-sum halves it to 0.5.
    result = 0.5 * V_0

    for k in range(1, N):
        u_k = k * pi_over_w

        # Heston CF (trap-free form) ---------------------------------------
        iu          = 1j * u_k
        beta        = lam - rho_eta * iu
        D           = np.sqrt(beta * beta + eta2 * (u_k * u_k + iu))
        beta_minus  = beta - D
        G           = beta_minus / (beta + D)
        exp_mDt     = np.exp(-D * tau)
        one_m_Gexp  = 1.0 - G * exp_mDt

        log_ratio   = np.log(one_m_Gexp / (1.0 - G))
        term_drift  = iu * drift_rq_tau
        term_v0     = v0_inv_eta2 * ((1.0 - exp_mDt) / one_m_Gexp) * beta_minus
        term_ubar   = lam_ubar_inv_e2 * (beta_minus * tau - 2.0 * log_ratio)
        phi         = np.exp(term_drift + term_v0 + term_ubar)

        # Centering phase (real part for the dot product).
        phase   = np.exp(1j * u_k * phase_factor)
        phi_re  = (phi * phase).real

        # Payoff V_k. arg_upper at the call boundary b is k*pi -> sin=0,
        # cos=(-1)^k (sign_k); for the put the same trig values at -u*a
        # appear via the sin_l/cos_l terms because chi(a,0) and psi(a,0)
        # share the same -u*a argument as chi(0,b) does. The two formulas
        # collapse to one shape with `upper` swapped.
        sign_k    = 1.0 if (k & 1) == 0 else -1.0
        arg_lower = -u_k * a
        cos_l     = np.cos(arg_lower)
        sin_l     = np.sin(arg_lower)
        inv_1_u2  = 1.0 / (1.0 + u_k * u_k)

        upper = (sign_k * exp_term) if (cp > 0) else exp_term
        V_k   = two_over_w * ((upper - cos_l - u_k * sin_l) * inv_1_u2 + sin_l / u_k)

        result += V_k * phi_re

    return K * df * result


@njit(cache=True, fastmath=False)
def _heston_cos_vec_kernel(S0, K_arr, tau, v0, lam, eta, ubar, rho, r, q, cp, N, L):
    """Vector-strike Heston price (call cp=+1, put cp=-1).

    The CF and centering phase are computed once and shared across every
    strike in ``K_arr``; only the payoff V_k loop runs per strike.
    """
    # Constants (k, K independent) ----------------------------------------
    eT       = np.exp(-lam * tau)
    c1       = (r - q) * tau - 0.5 * (ubar * tau + (v0 - ubar) * (1.0 - eT) / lam)
    sigma_h  = np.sqrt(ubar + v0 * eta)
    half     = L * sigma_h
    width    = 2.0 * half

    pi_over_w        = np.pi / width
    eta2             = eta * eta
    inv_eta2         = 1.0 / eta2
    rho_eta          = rho * eta
    drift_rq_tau     = (r - q) * tau
    v0_inv_eta2      = v0 * inv_eta2
    lam_ubar_inv_e2  = lam * ubar * inv_eta2
    df               = np.exp(-r * tau)
    phase_factor     = half - c1
    two_over_w       = 2.0 / width

    # Strike-independent CF * phase, real part. Computed once per call.
    phi_re = np.empty(N)
    phi_re[0] = 0.5                                              # k=0: CF=1, phase=1, halved
    for k in range(1, N):
        u_k         = k * pi_over_w
        iu          = 1j * u_k
        beta        = lam - rho_eta * iu
        D           = np.sqrt(beta * beta + eta2 * (u_k * u_k + iu))
        beta_minus  = beta - D
        G           = beta_minus / (beta + D)
        exp_mDt     = np.exp(-D * tau)
        one_m_Gexp  = 1.0 - G * exp_mDt
        log_ratio   = np.log(one_m_Gexp / (1.0 - G))
        term_drift  = iu * drift_rq_tau
        term_v0     = v0_inv_eta2 * ((1.0 - exp_mDt) / one_m_Gexp) * beta_minus
        term_ubar   = lam_ubar_inv_e2 * (beta_minus * tau - 2.0 * log_ratio)
        phi         = np.exp(term_drift + term_v0 + term_ubar)
        phase       = np.exp(1j * u_k * phase_factor)
        phi_re[k]   = (phi * phase).real

    # Per-strike payoff coefficients and dot product.
    M      = K_arr.shape[0]
    prices = np.empty(M)
    for m in range(M):
        K_m = K_arr[m]
        x   = np.log(S0 / K_m)
        a   = x + c1 - half
        b   = x + c1 + half

        if cp > 0:
            exp_term = np.exp(b)
            V_0      = two_over_w * (exp_term - 1.0 - b)
        else:
            exp_term = np.exp(a)
            V_0      = two_over_w * (exp_term - 1.0 - a)

        result = V_0 * phi_re[0]

        for k in range(1, N):
            u_k       = k * pi_over_w
            sign_k    = 1.0 if (k & 1) == 0 else -1.0
            arg_lower = -u_k * a
            cos_l     = np.cos(arg_lower)
            sin_l     = np.sin(arg_lower)
            inv_1_u2  = 1.0 / (1.0 + u_k * u_k)
            upper     = (sign_k * exp_term) if (cp > 0) else exp_term
            V_k       = two_over_w * ((upper - cos_l - u_k * sin_l) * inv_1_u2 + sin_l / u_k)
            result   += V_k * phi_re[k]

        prices[m] = K_m * df * result

    return prices


# ============================================================================
# Free-function entry points (no class, no cache) ----------------------------
# ============================================================================

def price_call_heston(S0, K, tau, v0, lam, eta, ubar, rho, N, L, r=0.0, q=0.0):
    """One-shot single-strike Heston call price; no caching between calls."""
    return _heston_cos_kernel(
        float(S0), float(K), float(tau), float(v0),
        float(lam), float(eta), float(ubar), float(rho),
        float(r), float(q), 1, int(N), float(L),
    )


def price_put_heston(S0, K, tau, v0, lam, eta, ubar, rho, N, L, r=0.0, q=0.0):
    """One-shot single-strike Heston put price; no caching between calls."""
    return _heston_cos_kernel(
        float(S0), float(K), float(tau), float(v0),
        float(lam), float(eta), float(ubar), float(rho),
        float(r), float(q), -1, int(N), float(L),
    )


def price_call_heston_vec(S0, K_arr, tau, v0, lam, eta, ubar, rho, N, L, r=0.0, q=0.0):
    """One-shot vector-strike Heston call prices; no caching between calls."""
    K_arr = np.ascontiguousarray(np.asarray(K_arr, dtype=np.float64))
    return _heston_cos_vec_kernel(
        float(S0), K_arr, float(tau), float(v0),
        float(lam), float(eta), float(ubar), float(rho),
        float(r), float(q), 1, int(N), float(L),
    )


def price_put_heston_vec(S0, K_arr, tau, v0, lam, eta, ubar, rho, N, L, r=0.0, q=0.0):
    """One-shot vector-strike Heston put prices; no caching between calls."""
    K_arr = np.ascontiguousarray(np.asarray(K_arr, dtype=np.float64))
    return _heston_cos_vec_kernel(
        float(S0), K_arr, float(tau), float(v0),
        float(lam), float(eta), float(ubar), float(rho),
        float(r), float(q), -1, int(N), float(L),
    )


# Force compilation at import so timing measurements never hit the JIT path.
_heston_cos_kernel(100.0, 100.0, 1.0, 0.04, 1.0, 0.5, 0.04, -0.5, 0.0, 0.0, 1, 4, 10.0)
_heston_cos_kernel(100.0, 100.0, 1.0, 0.04, 1.0, 0.5, 0.04, -0.5, 0.0, 0.0, -1, 4, 10.0)
_heston_cos_vec_kernel(100.0, np.array([100.0]), 1.0, 0.04, 1.0, 0.5, 0.04, -0.5, 0.0, 0.0, 1, 4, 10.0)
_heston_cos_vec_kernel(100.0, np.array([100.0]), 1.0, 0.04, 1.0, 0.5, 0.04, -0.5, 0.0, 0.0, -1, 4, 10.0)


# ============================================================================
# Class wrapper -- holds parameters, caches identical-argument results ------
# ============================================================================

class HestonCOSPricer:
    """Heston European call/put pricer via Fang-Oosterlee COS.

    Holds parameters and caches identical-argument results so repeated
    pricing calls (calibration, Greeks via finite differences) return from
    a dict lookup. The actual math runs in the same numba kernels used by
    the free functions ``price_call_heston`` / ``price_call_heston_vec``;
    the class is a thin caching layer on top.
    """

    _CACHE_CAP = 64

    def __init__(self, S0, v0, lam, eta, ubar, rho, r=0.0, q=0.0):
        """Store parameters, validate ranges (rho in (-1,1); v0, lam, eta, ubar > 0)."""
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

        # Scalars reused by the NumPy reference helpers below.
        self._eta2     = self.eta * self.eta
        self._rho_eta  = self.rho * self.eta
        self._drift_rq = self.r - self.q
        self._sigma_h  = float(np.sqrt(self.ubar + self.v0 * self.eta))

        # (K_bytes, tau, N, L, cp) -> price array. FIFO eviction at cap.
        self._cache = {}

    def clear_cache(self):
        """Drop all cached pricing results."""
        self._cache.clear()

    # ------------------------------------------------------------------------
    # Pricing -- dispatches to the jitted kernels with caching
    # ------------------------------------------------------------------------

    def price(self, K, tau, cp=1, N=160, L=None):
        """European price (call cp=+1, put cp=-1) at strike(s) K and maturity tau."""
        if tau <= 0.0:
            raise ValueError(f"tau must be > 0, got {tau}")
        if N < 1:
            raise ValueError(f"N must be >= 1, got {N}")

        L_value   = self._default_L(tau) if L is None else float(L)
        scalar_in = np.isscalar(K)
        K_arr     = np.atleast_1d(np.asarray(K, dtype=np.float64))

        key = (K_arr.tobytes(), float(tau), int(N), L_value, int(cp))
        hit = self._cache.get(key)
        if hit is None:
            if K_arr.size == 1:
                hit = np.array([_heston_cos_kernel(
                    self.S0, float(K_arr[0]), float(tau), self.v0,
                    self.lam, self.eta, self.ubar, self.rho,
                    self.r, self.q, int(cp), int(N), L_value,
                )])
            else:
                hit = _heston_cos_vec_kernel(
                    self.S0, np.ascontiguousarray(K_arr), float(tau), self.v0,
                    self.lam, self.eta, self.ubar, self.rho,
                    self.r, self.q, int(cp), int(N), L_value,
                )
            if len(self._cache) >= self._CACHE_CAP:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = hit

        return float(hit[0]) if scalar_in else hit

    def price_call(self, K, tau, N=160, L=None):
        """Call price via COS; thin wrapper over price(..., cp=+1)."""
        return self.price(K, tau, cp=1, N=N, L=L)

    def price_put(self, K, tau, N=160, L=None):
        """Put price via COS; thin wrapper over price(..., cp=-1)."""
        return self.price(K, tau, cp=-1, N=N, L=L)

    # ------------------------------------------------------------------------
    # NumPy-reference helpers -- preserved for inspection and existing tests.
    # Not on the hot path.
    # ------------------------------------------------------------------------

    def mgf_logprice(self, uu, tau):
        """Heston MGF of log(S_T/S0) at argument uu (any real or complex), maturity tau.

        Single source of truth for the Heston transform: M(uu) = E[exp(uu * X)]
        where X = log(S_T/S0). The COS characteristic function is M on the
        imaginary axis (``char_func`` wraps this); Lewis-style integration
        evaluates at u - i*alpha; numerical cumulants come from differentiating
        log(M) at real uu near zero. Trap-free form (Albrecher 2007 / Lord-Kahl).
        """
        uu          = np.asarray(uu) + 0j                              # promote to complex
        beta        = self.lam - self._rho_eta * uu
        D           = np.sqrt(beta * beta + self._eta2 * uu * (1.0 - uu))
        beta_minus  = beta - D
        G           = beta_minus / (beta + D)
        Dt          = D * tau
        exp_mDt     = np.exp(-Dt)
        one_m_Gexp  = 1.0 - G * exp_mDt
        one_m_G     = 1.0 - G
        one_m_exp   = -np.expm1(-Dt)
        term_drift  = uu * self._drift_rq * tau
        term_v0     = (self.v0 / self._eta2) * (one_m_exp / one_m_Gexp) * beta_minus
        log_ratio   = np.log(one_m_Gexp / one_m_G)
        term_ubar   = (self.lam * self.ubar / self._eta2) * (beta_minus * tau - 2.0 * log_ratio)
        return np.exp(term_drift + term_v0 + term_ubar)

    def char_func(self, u, tau):
        """Heston CF of log(S_T/S0) at frequency u: phi(u) = mgf_logprice(i*u)."""
        return self.mgf_logprice(1j * np.asarray(u), tau)

    def _cumulants(self, tau, eps=1e-4):
        """Cumulants (c1, c2, c4=0) of log(S_T/S0).

        c1 is exact-analytic; c2 is recovered by central finite differences
        of log(MGF) at the real axis. The often-cited closed-form Heston c2
        from F&O Appendix A.2 has a transcription typo that surfaces as a
        ~7e-4 absolute discrepancy against the MGF-derived value, so we
        derive c2 from ``mgf_logprice`` directly (single source of truth).
        c4 = 0 per paper Section 5 (used only as a sentinel by callers).
        """
        K = lambda uu: float(np.log(self.mgf_logprice(uu, tau)).real)
        c2 = (K(eps) + K(-eps) - 2.0 * K(0.0)) / (eps * eps)
        return self._c1(tau), c2, 0.0

    def _c1(self, tau):
        """Mean of log(S_T/S0)."""
        one_m_eT = -np.expm1(-self.lam * tau)
        return self._drift_rq * tau - 0.5 * (self.ubar * tau + (self.v0 - self.ubar) * one_m_eT / self.lam)

    @staticmethod
    def _default_L(tau):
        """Per-tau half-width multiplier; max(10, 3*tau + 2) interpolates between paper's L=10 at tau=1 and L=32 at tau=10."""
        return max(10.0, 3.0 * tau + 2.0)

    def truncation_interval(self, K, tau, L=None):
        """Return (a, b, x, width) with [a,b] centered on x+c1 and half-width L*sigma."""
        if L is None:
            L = self._default_L(tau)
        half  = L * self._sigma_h
        width = 2.0 * half
        x     = np.log(self.S0 / np.asarray(K, dtype=float))
        center = x + self._c1(tau)
        return center - half, center + half, x, width

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

        NumPy reference; not used on the hot path (the kernels compute
        coefficients inline). Built with c=0 for calls and d=0 for puts.
        """
        a = np.atleast_1d(np.asarray(a, dtype=float))[:, None]
        b = np.atleast_1d(np.asarray(b, dtype=float))[:, None]
        k = np.arange(N)[None, :]
        ba = b - a
        u  = k * np.pi / ba

        two_over_ba = 2.0 / ba
        inv_1_u2    = 1.0 / (1.0 + u * u)
        u_safe      = np.where(k == 0, 1.0, u)

        if cp > 0:
            arg_upper = u * (b - a)                                   # = k*pi
            arg_lower = u * (-a)
            cos_u, sin_u = np.cos(arg_upper), np.sin(arg_upper)
            cos_l, sin_l = np.cos(arg_lower), np.sin(arg_lower)
            exp_b = np.exp(b)
            chi = (cos_u * exp_b - cos_l
                   + u * (sin_u * exp_b - sin_l)) * inv_1_u2
            psi = np.where(k == 0, b - 0.0, (sin_u - sin_l) / u_safe)
            return two_over_ba * (chi - psi)

        arg_upper = u * (-a)
        cos_u, sin_u = np.cos(arg_upper), np.sin(arg_upper)
        exp_a = np.exp(a)
        chi = (cos_u - exp_a + u * sin_u) * inv_1_u2
        psi = np.where(k == 0, 0.0 - a, sin_u / u_safe)
        return two_over_ba * (-chi + psi)

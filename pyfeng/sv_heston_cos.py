"""
Heston (1993) European option pricing via the Fourier-Cosine (COS) method.

Port of ``cos_pricing.heston_cos_pricer.HestonCOSPricer`` into the PyFENG
``SvABC`` / ``HestonABC`` / ``CosABC`` class hierarchy. Faithful to
Fang & Oosterlee (2008) Sections 4 and 5.2, with the same numerical
refinements as the source pricer:

1. Truncation half-width uses the Heston-specific
   ``sigma_h = sqrt(theta + sigma*vov)`` (F&O 2008 Section 5.2), not the
   generic cumulant-based ``L*sqrt(|c2| + sqrt(|c4|))`` range. The interval
   ``[a, b]`` is centered per-strike on ``log(F/K) + c1`` (Ruijter &
   Oosterlee 2012).
2. JIT kernel folds the per-k loop (CF + payoff coefficients +
   prime-weighted dot product) into one machine-code loop. Falls back to
   a NumPy implementation if numba is not installed.
3. Default L scales with maturity: ``max(10, 3*texp + 2)``.

The MGF body is the Lord & Kahl (2010) branch-cut-safe form, identical
to ``HestonFft.mgf_logprice``, in PyFENG's log(S_T/F) convention (drift
absorbed into the forward F, not in the CF).

Caching, free-function entry points, and the ``price_call`` / ``price_put``
shortcut methods of the source pricer are deliberately omitted -- they
are not idiomatic in PyFENG. ``HestonCos(...).price(K, S, T, cp=+1)`` is
the equivalent for a call; ``cp=-1`` for a put.

References:
    Heston SL (1993) Rev. Financial Studies 6:327-343.
    Fang F, Oosterlee CW (2008) SIAM J. Sci. Comput. 31:826-848.
    Lord R, Kahl C (2010) Mathematical Finance 20:671-694.
    Ruijter MJ, Oosterlee CW (2012) SIAM J. Sci. Comput. 34:B642-B671.
"""

import numpy as np

try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        def deco(fn):
            return fn
        return deco

from . import heston
from .sv_cos import CosABC


# ============================================================================
# Compiled kernels -- log(S_T/F) world (PyFENG convention).
#
# The drift term iu*(r-q)*tau of the source pricer's log(S_T/S0) MGF is
# absorbed into the forward F = spot * exp((intr - divr) * texp); the
# kernels receive F and df only.
# ============================================================================

@njit(cache=True, fastmath=False)
def _heston_cos_kernel(F, K, tau, sigma, mr, vov, theta, rho, df, cp, N, L):
    """Single-strike Heston call (cp=+1) or put (cp=-1) price via COS."""
    eT       = np.exp(-mr * tau)
    c1       = -0.5 * (theta * tau + (sigma - theta) * (1.0 - eT) / mr)
    sigma_h  = np.sqrt(theta + sigma * vov)
    half     = L * sigma_h
    width    = 2.0 * half
    x        = np.log(F / K)
    a        = x + c1 - half
    b        = x + c1 + half

    pi_over_w        = np.pi / width
    vov2             = vov * vov
    inv_vov2         = 1.0 / vov2
    rho_vov          = rho * vov
    sigma_inv_vov2   = sigma * inv_vov2
    mr_theta_inv_vv2 = mr * theta * inv_vov2
    phase_factor     = half - c1
    two_over_w       = 2.0 / width

    if cp > 0:
        exp_term = np.exp(b)
        V_0      = two_over_w * (exp_term - 1.0 - b)
    else:
        exp_term = np.exp(a)
        V_0      = two_over_w * (exp_term - 1.0 - a)

    result = 0.5 * V_0

    for k in range(1, N):
        u_k = k * pi_over_w

        iu          = 1j * u_k
        beta        = mr - rho_vov * iu
        D           = np.sqrt(beta * beta + vov2 * (u_k * u_k + iu))
        beta_minus  = beta - D
        G           = beta_minus / (beta + D)
        exp_mDt     = np.exp(-D * tau)
        one_m_Gexp  = 1.0 - G * exp_mDt
        log_ratio   = np.log(one_m_Gexp / (1.0 - G))

        # log(S_T/F) MGF -- no drift term (drift absorbed into F).
        term_v0   = sigma_inv_vov2 * ((1.0 - exp_mDt) / one_m_Gexp) * beta_minus
        term_ubar = mr_theta_inv_vv2 * (beta_minus * tau - 2.0 * log_ratio)
        phi       = np.exp(term_v0 + term_ubar)

        phase  = np.exp(1j * u_k * phase_factor)
        phi_re = (phi * phase).real

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
def _heston_cos_vec_kernel(F, K_arr, tau, sigma, mr, vov, theta, rho, df, cp, N, L):
    """Vector-strike Heston price (call cp=+1, put cp=-1).

    The CF and centering phase are computed once and shared across every
    strike in K_arr; only the per-strike payoff V_k loop runs M times.
    """
    eT       = np.exp(-mr * tau)
    c1       = -0.5 * (theta * tau + (sigma - theta) * (1.0 - eT) / mr)
    sigma_h  = np.sqrt(theta + sigma * vov)
    half     = L * sigma_h
    width    = 2.0 * half

    pi_over_w        = np.pi / width
    vov2             = vov * vov
    inv_vov2         = 1.0 / vov2
    rho_vov          = rho * vov
    sigma_inv_vov2   = sigma * inv_vov2
    mr_theta_inv_vv2 = mr * theta * inv_vov2
    phase_factor     = half - c1
    two_over_w       = 2.0 / width

    phi_re    = np.empty(N)
    phi_re[0] = 0.5
    for k in range(1, N):
        u_k         = k * pi_over_w
        iu          = 1j * u_k
        beta        = mr - rho_vov * iu
        D           = np.sqrt(beta * beta + vov2 * (u_k * u_k + iu))
        beta_minus  = beta - D
        G           = beta_minus / (beta + D)
        exp_mDt     = np.exp(-D * tau)
        one_m_Gexp  = 1.0 - G * exp_mDt
        log_ratio   = np.log(one_m_Gexp / (1.0 - G))
        term_v0     = sigma_inv_vov2 * ((1.0 - exp_mDt) / one_m_Gexp) * beta_minus
        term_ubar   = mr_theta_inv_vv2 * (beta_minus * tau - 2.0 * log_ratio)
        phi         = np.exp(term_v0 + term_ubar)
        phase       = np.exp(1j * u_k * phase_factor)
        phi_re[k]   = (phi * phase).real

    M      = K_arr.shape[0]
    prices = np.empty(M)
    for m in range(M):
        K_m = K_arr[m]
        x   = np.log(F / K_m)
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


_warmed_up = False


def warmup_numba():
    """Trigger JIT compilation via small dummy calls.

    No-op if numba is not installed or warmup has already run. Useful for
    benchmark scripts that want to exclude JIT cost from timings.
    """
    global _warmed_up
    if _warmed_up:
        return
    if not _HAS_NUMBA:
        _warmed_up = True
        return
    _heston_cos_kernel(100.0, 100.0, 1.0, 0.04, 1.0, 0.5, 0.04, -0.5, 1.0,  1, 4, 10.0)
    _heston_cos_kernel(100.0, 100.0, 1.0, 0.04, 1.0, 0.5, 0.04, -0.5, 1.0, -1, 4, 10.0)
    _heston_cos_vec_kernel(100.0, np.array([100.0]), 1.0, 0.04, 1.0, 0.5, 0.04, -0.5, 1.0,  1, 4, 10.0)
    _heston_cos_vec_kernel(100.0, np.array([100.0]), 1.0, 0.04, 1.0, 0.5, 0.04, -0.5, 1.0, -1, 4, 10.0)
    _warmed_up = True


# ============================================================================
# Class wrapper -- PyFENG SvABC / HestonABC / CosABC integration
# ============================================================================

class HestonCos(heston.HestonABC, CosABC):
    """
    Heston (1993) stochastic-volatility European option pricing via the
    Fourier-Cosine (COS) method, with the Heston-specific sigma_h
    truncation of F&O (2008) Section 5.2 and per-strike Ruijter &
    Oosterlee (2012) centering.

    Parameters (PyFENG ``SvABC`` convention):
        sigma  : initial variance V0 (must be > 0)
        vov    : vol-of-vol (must be > 0)
        mr     : mean-reversion speed (must be > 0)
        rho    : correlation; must be in (-1, 1)
        theta  : long-run variance (must be > 0; defaults to sigma)
        intr   : risk-free rate
        divr   : dividend yield
        is_fwd : if True, treat ``spot`` as the forward price

    Tuning attributes:
        n_cos (int)         : number of Fourier-cosine terms (default 160).
        L (float | None)    : truncation half-width multiplier.
                              ``None`` (default) -> tau-adaptive
                              ``max(10, 3*texp+2)``. Set explicitly to fix.

    The CF body is identical to ``HestonFft.mgf_logprice`` (Lord & Kahl
    2010) so the COS and FFT pricers are line-for-line cross-validatable.
    Note: CosABC's inherited ``char_func`` is named ``charfunc_logprice``.

    Caching, free-function entry points, and ``price_call`` / ``price_put``
    shortcuts of the source pricer are not provided -- use
    ``m.price(K, S, T, cp=+1)`` for calls, ``cp=-1`` for puts.

    Examples:
        >>> import numpy as np
        >>> import pyfeng as pf
        >>> m = pf.HestonCos(0.0175, vov=0.5751, mr=1.5768,
        ...                  theta=0.0398, rho=-0.5711)
        >>> m.price(np.array([90.0, 100.0, 110.0]), 100.0, 1.0)
    """

    n_cos: int = 160
    L = None  # None -> tau-adaptive default; set float to fix.

    # ------------------------------------------------------------------
    # __init__ -- validate before super, re-check after.
    # ------------------------------------------------------------------

    def __init__(self, sigma, vov=0.01, rho=0.0, mr=0.01, theta=None,
                 intr=0.0, divr=0.0, is_fwd=False):
        if sigma <= 0.0:
            raise ValueError(f"sigma (initial variance) must be > 0, got {sigma}")
        if vov <= 0.0:
            raise ValueError(f"vov (vol-of-vol) must be > 0, got {vov}")
        if mr <= 0.0:
            raise ValueError(f"mr (mean-reversion) must be > 0, got {mr}")
        if theta is not None and theta <= 0.0:
            raise ValueError(f"theta (long-run variance) must be > 0, got {theta}")
        if not (-1.0 < rho < 1.0):
            raise ValueError(f"rho must be in (-1, 1), got {rho}")
        super().__init__(sigma=sigma, vov=vov, rho=rho, mr=mr, theta=theta,
                         intr=intr, divr=divr, is_fwd=is_fwd)
        if self.theta <= 0.0:
            raise ValueError(f"theta resolved to <= 0 ({self.theta})")

    # ------------------------------------------------------------------
    # Heston-specific helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _default_L(texp):
        """Per-tau half-width multiplier; max(10, 3*texp + 2).

        Interpolates between F&O's L=10 at texp=1 and ~L=32 at texp=10,
        keeping the trap-free CF numerically stable for long maturities.
        """
        return max(10.0, 3.0 * float(texp) + 2.0)

    def _resolve_L(self, texp):
        """Resolve L: class attribute if set, otherwise tau-adaptive default."""
        return self._default_L(texp) if self.L is None else float(self.L)

    def _sigma_h(self):
        """F&O (2008) Section 5.2 Heston truncation scale: sqrt(theta + sigma*vov)."""
        return float(np.sqrt(self.theta + self.sigma * self.vov))

    def _c1_logF(self, texp):
        """Mean of log(S_T/F) under Heston (no drift term -- F absorbs it)."""
        T = float(texp)
        eT = np.exp(-self.mr * T)
        return -0.5 * (self.theta * T + (self.sigma - self.theta) * (1.0 - eT) / self.mr)

    def truncation_interval(self, strike, spot, texp, L=None):
        """COS truncation interval used by ``price()``.

        Returns:
            (a, b, x, width): per-strike arrays. ``x = log(F/K)``,
                ``width = 2*L*sigma_h``, ``[a, b] = x + c1 +/- L*sigma_h``.
                Centered per-strike on ``log(F/K) + c1`` (Ruijter &
                Oosterlee 2012).
        """
        fwd, _, _ = self._fwd_factor(spot, texp)
        L_value = self._resolve_L(texp) if L is None else float(L)
        half    = L_value * self._sigma_h()
        width   = 2.0 * half
        c1      = self._c1_logF(texp)
        x       = np.log(np.asarray(fwd, dtype=float) / np.asarray(strike, dtype=float))
        center  = x + c1
        return center - half, center + half, x, width

    def _integration_range(self, strike, spot, texp):
        """CosABC interval hook using the Heston-specific sigma_h range."""
        a, b, _, _ = self.truncation_interval(strike, spot, texp)
        return a, b

    # ------------------------------------------------------------------
    # CosABC required overrides -- MGF and cumulants
    # ------------------------------------------------------------------

    def mgf_logprice(self, uu, texp):
        """
        Heston log-price MGF -- Lord & Kahl (2010) branch-cut-safe form.

        Of log(S_T/F) (PyFENG convention; drift absorbed into F). Body
        identical to ``HestonFft.mgf_logprice`` so the two pricers share
        a single CF source of truth.
        """
        var_0 = self.sigma
        vov2  = self.vov**2

        beta = self.mr - self.vov*self.rho*uu
        dd   = np.sqrt(beta**2 + vov2*uu*(1 - uu))
        gg   = (beta - dd)/(beta + dd)
        exp_ = np.exp(-dd*texp)
        tmp1 = 1 - gg*exp_

        mgf = self.mr*self.theta*((beta - dd)*texp - 2*np.log(tmp1/(1 - gg))) \
              + var_0*(beta - dd)*(1 - exp_)/tmp1
        return np.exp(mgf/vov2)

    def _cumulants(self, texp, eps=1e-4):
        """Cumulants (c1, c2, 0, c4=0) of log(S_T/F).

        c1 from ``_c1_logF`` (analytic). c2 from central finite differences
        of log(MGF) at the real axis -- the source pricer explicitly avoids
        F&O's Appendix A.2 closed-form because of a transcription typo
        that produces a ~7e-4 discrepancy. Single source of truth: the
        MGF derives c2 directly. c4 = 0 per F&O Section 5.

        Diagnostic only -- ``price()`` uses the sigma_h truncation
        (Section 5.2), not these cumulants.
        """
        K = lambda uu: float(np.log(self.mgf_logprice(uu, texp)).real)
        c2 = (K(eps) + K(-eps) - 2.0 * K(0.0)) / (eps * eps)
        return float(self._c1_logF(texp)), float(c2), 0.0, 0.0

    # ------------------------------------------------------------------
    # Static payoff helpers (source-compatible signatures, F&O Eq. 22-23)
    # ------------------------------------------------------------------

    @staticmethod
    def chi(k, a, b, c, d):
        """Analytic integral of exp(x)*cos(k*pi*(x-a)/(b-a)) from c to d.

        F&O (2008) Eq. 22. Source-compatible static helper.
        """
        ba    = b - a
        u     = k * np.pi / ba
        arg_d = u * (d - a)
        arg_c = u * (c - a)
        cos_d = np.cos(arg_d)
        sin_d = np.sin(arg_d)
        cos_c = np.cos(arg_c)
        sin_c = np.sin(arg_c)
        exp_d = np.exp(d)
        exp_c = np.exp(c)
        return (cos_d * exp_d - cos_c * exp_c
                + u * (sin_d * exp_d - sin_c * exp_c)) / (1.0 + u * u)

    @staticmethod
    def psi(k, a, b, c, d):
        """Analytic integral of cos(k*pi*(x-a)/(b-a)) from c to d.

        F&O (2008) Eq. 23. k=0 handled safely (returns d-c).
        """
        ba     = b - a
        u      = k * np.pi / ba
        u_safe = np.where(k == 0, 1.0, u)
        arg_d  = u * (d - a)
        arg_c  = u * (c - a)
        return np.where(k == 0, d - c, (np.sin(arg_d) - np.sin(arg_c)) / u_safe)

    @staticmethod
    def payoff_coefficients(N, a, b, cp):
        """U_k coefficients of shape (M, N) for a call (cp=+1) or put (cp=-1).

        F&O (2008) Eqs. 29-30. NumPy reference; not used on the hot path
        (kernels compute coefficients inline). Built with c=0 for calls
        and d=0 for puts.
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
            arg_upper = u * (b - a)
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

    # ------------------------------------------------------------------
    # Pricing -- implementation override using the same conceptual COS
    # stages as CosABC, but fused into JIT kernels for Heston.
    # ------------------------------------------------------------------

    def price(self, strike, spot, texp, cp=1):
        """
        European call/put price via the Heston COS method.

        Vectorised over ``strike`` and ``cp``. Mixed call/put portfolios
        are handled by running both kernels and muxing on the cp sign.
        Scalar ``spot`` only (vector spot is not supported in this port).

        Args:
            strike: scalar or array of strikes.
            spot:   spot (or forward if ``is_fwd=True``) price; must be scalar.
            texp:   time to expiry (must be > 0).
            cp:     +1 call / -1 put; scalar or array of same shape as strike.

        Returns:
            Option price(s) matching the broadcast shape of (strike, cp).
        """
        if texp <= 0.0:
            raise ValueError(f"texp must be > 0, got {texp}")
        if int(self.n_cos) < 1:
            raise ValueError(f"n_cos must be >= 1, got {self.n_cos}")
        if not np.isscalar(spot) and np.asarray(spot).ndim > 0:
            raise ValueError(
                "HestonCos.price requires scalar spot; vector spot is not "
                f"supported in this port. Got shape {np.asarray(spot).shape}."
            )

        fwd, df, _ = self._fwd_factor(spot, texp)
        L_value    = self._resolve_L(texp)

        F  = float(np.asarray(fwd).reshape(-1)[0])
        df = float(np.asarray(df).reshape(-1)[0])

        scalar_in = np.isscalar(strike) and np.isscalar(cp)

        K_arr  = np.atleast_1d(np.asarray(strike, dtype=np.float64))
        cp_arr = np.broadcast_to(
            np.atleast_1d(np.asarray(cp, dtype=np.float64)), K_arr.shape
        )

        # Detect uniform cp for the fast path.
        first = float(cp_arr.flat[0])
        uniform = bool(np.all(cp_arr == first))

        if uniform:
            cp_int = 1 if first > 0 else -1
            prices = self._kernel_dispatch(F, K_arr, float(texp), df, cp_int, L_value)
        else:
            calls = self._kernel_dispatch(F, K_arr, float(texp), df,  1, L_value)
            puts  = self._kernel_dispatch(F, K_arr, float(texp), df, -1, L_value)
            prices = np.where(cp_arr > 0, calls, puts)

        if scalar_in:
            return float(prices[0])
        return prices.reshape(np.broadcast_shapes(np.shape(strike), np.shape(cp)))

    def _kernel_dispatch(self, F, K_arr, tau, df, cp_int, L_value):
        """Pick scalar vs vector kernel based on K_arr size."""
        if K_arr.size == 1:
            return np.array([_heston_cos_kernel(
                F, float(K_arr[0]), tau,
                float(self.sigma), float(self.mr), float(self.vov),
                float(self.theta), float(self.rho), df,
                cp_int, int(self.n_cos), float(L_value),
            )])
        return _heston_cos_vec_kernel(
            F, np.ascontiguousarray(K_arr), tau,
            float(self.sigma), float(self.mr), float(self.vov),
            float(self.theta), float(self.rho), df,
            cp_int, int(self.n_cos), float(L_value),
        )

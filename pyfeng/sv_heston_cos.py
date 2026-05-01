"""
Heston (1993) European option pricing via the Fourier-Cosine (COS) method.

Port of ``cos_pricing.heston_cos_pricer.HestonCOSPricer`` into the PyFENG
``HestonFft`` / ``CosABC`` class hierarchy. Faithful to Fang & Oosterlee
(2008) Sections 4 and 5.2 with the same numerical refinements as the
source pricer:

1. The truncation half-width uses the Heston-tailored proxy
   ``sigma_h = sqrt(theta + sigma*vov)`` from F&O 2008 Section 5.2,
   not the generic cumulant range ``L*sqrt(|c2| + sqrt(|c4|))`` of
   Section 5.1 (the latter is still available via
   ``CosABC._truncation_range`` for diagnostics).  The interval ``[a, b]``
   is centered per-strike on ``log(F/K) + c1`` per F&O Section 5.1.
2. The per-k loop (CF + payoff coefficients + prime-weighted dot product)
   is folded into a single numba ``@njit`` kernel.  Falls back to plain
   NumPy if numba is not installed.
3. The default truncation multiplier scales with maturity:
   ``L = max(10, 3*texp + 2)``.

The MGF (Lord & Kahl 2010 branch-cut-safe form) is inherited from
``HestonFft.mgf_logprice`` -- single CF source of truth across the COS
and FFT pricers.  PyFENG's log(S_T/F) convention applies (drift sits
inside the forward F, not in the CF).

References:
    - Heston SL (1993) Review of Financial Studies 6:327-343.
      https://doi.org/10.1093/rfs/6.2.327
    - Fang F, Oosterlee CW (2008) SIAM J. Sci. Comput. 31:826-848.
      https://doi.org/10.1137/080718061
    - Lord R, Kahl C (2010) Mathematical Finance 20:671-694.
      https://doi.org/10.1111/j.1467-9965.2010.00416.x
    - Ruijter MJ, Oosterlee CW (2012) SIAM J. Sci. Comput. 34:B642-B671.
      https://doi.org/10.1137/120862053
"""

import numpy as np

try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

    def njit(*args, **kwargs):  # noqa: ARG001
        # Pass-through decorator so kernels still work as plain Python.
        if len(args) == 1 and callable(args[0]):
            return args[0]
        def deco(fn):
            return fn
        return deco

from .sv_cos import CosABC
from .sv_fft import HestonFft


__all__ = ["HestonCos", "warmup_numba"]


# ============================================================================
# Compiled kernels -- log(S_T/F) world (PyFENG convention).
#
# The drift term iu*(r-q)*tau of the source pricer's log(S_T/S0) MGF is
# absorbed into the forward F = spot * exp((intr - divr) * texp); the
# kernels receive F and df only.
#
# These kernels INLINE two pieces of math that also live in the inherited
# class hierarchy:
#   * the Lord-Kahl (2010) CF -- mirrors HestonFft.mgf_logprice.
#   * the Heston c1 closed form -- mirrors HestonABC.avgvar_mv reduced for
#     the integrated-variance mean.
# Inlining is required: numba @njit functions cannot call instance methods
# or use Python objects.  Any change to those formulas in upstream
# HestonFft / HestonABC MUST be mirrored here.  The Python-side
# mgf_logprice (inherited) and _c1_logF (which calls avgvar_mv) are the
# reference implementations; the kernels are JIT mirrors.
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

    # k=0 payoff coefficient (vanilla call/put with one boundary at b or a).
    if cp > 0:
        exp_term = np.exp(b)
        V_0      = two_over_w * (exp_term - 1.0 - b)
    else:
        exp_term = np.exp(a)
        V_0      = two_over_w * (exp_term - 1.0 - a)

    # k=0 has the prime-sum 1/2 factor folded in.
    result = 0.5 * V_0

    for k in range(1, N):
        u_k = k * pi_over_w

        # Lord-Kahl (2010) CF -- JIT mirror of HestonFft.mgf_logprice.
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

        # Centering phase exp(-i*u*a) split: exp(i*u*(half-c1)) here, cos/sin
        # arguments below carry the strike-dependent -u*a piece.
        phase  = np.exp(1j * u_k * phase_factor)
        phi_re = (phi * phase).real

        sign_k    = 1.0 if (k & 1) == 0 else -1.0
        arg_lower = -u_k * a
        cos_l     = np.cos(arg_lower)
        sin_l     = np.sin(arg_lower)
        inv_1_u2  = 1.0 / (1.0 + u_k * u_k)

        # Vanilla call: trig at upper boundary b collapses to (-1)^k * exp(b);
        # vanilla put: trig at boundary a collapses to exp(a).
        upper = (sign_k * exp_term) if (cp > 0) else exp_term
        V_k   = two_over_w * ((upper - cos_l - u_k * sin_l) * inv_1_u2 + sin_l / u_k)

        result += V_k * phi_re

    return K * df * result


@njit(cache=True, fastmath=False)
def _heston_cos_vec_kernel(F, K_arr, tau, sigma, mr, vov, theta, rho, df, cp, N, L):
    """Vector-strike Heston price (call cp=+1, put cp=-1).

    The CF and centering phase are computed once and shared across every
    strike in ``K_arr``; only the per-strike payoff V_k loop runs M times.
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

    # Strike-independent CF * centering phase, real part.  k=0 has the
    # prime-sum 1/2 factor pre-folded.
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
    """Trigger numba JIT compilation via small dummy kernel calls.

    No-op if numba is not installed or warmup has already run.  Useful
    for benchmark scripts that want the JIT cost out of the first
    measured timing.
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
# Class wrapper -- PyFENG HestonFft / CosABC integration
# ============================================================================

class HestonCos(HestonFft, CosABC):
    """
    Heston (1993) stochastic-volatility option pricing via the COS method.

    Inherits the Lord & Kahl (2010) MGF, the parameter signature, and
    ``HestonABC`` helpers (``avgvar_mv`` etc.) from ``HestonFft``.  MRO
    would otherwise route ``price()`` through ``FftABC``; ``price`` is
    defined directly on this class (kernel-backed) and takes precedence.

    The COS truncation interval uses F&O 2008 Section 5.2's Heston-tailored
    half-width ``sigma_h = sqrt(theta + sigma*vov)``, with per-strike
    centering on ``log(F/K) + c1`` from Section 5.1 (``c1`` is the mean of
    log(S_T/F), available via ``avgvar_mv``).  The default L scales with
    maturity: ``max(10, 3*texp + 2)``.

    HestonFft does not validate parameter positivity; HestonCos does, in
    ``__init__``.  Caching, free-function entry points, and the
    ``price_call`` / ``price_put`` shortcut methods of the source pricer
    are deliberately omitted -- use ``m.price(K, S, T, cp=+1)`` for calls,
    ``cp=-1`` for puts.

    Attributes:
        n_cos (int): Number of Fourier-cosine terms (default 160; overrides
            CosABC's 128).
        L (float | None): Truncation half-width multiplier.  ``None``
            (default) means tau-adaptive ``max(10, 3*texp+2)``; set to a
            float to fix.

    Examples:
        >>> import numpy as np
        >>> import pyfeng as pf
        >>> m = pf.HestonCos(0.0175, vov=0.5751, mr=1.5768,
        ...                  theta=0.0398, rho=-0.5711)
        >>> m.price(np.array([90.0, 100.0, 110.0]), 100.0, 1.0)
    """

    n_cos: int = 160
    L = None

    def __init__(self, sigma, vov=0.01, rho=0.0, mr=0.01, theta=None,
                 intr=0.0, divr=0.0, is_fwd=False):
        """
        Args:
            sigma: initial variance V_0 (must be > 0).
            vov: vol-of-vol (must be > 0).
            rho: correlation between asset and variance, in (-1, 1).
            mr: mean-reversion speed (must be > 0).
            theta: long-run variance (must be > 0; defaults to ``sigma``).
            intr: interest rate.
            divr: dividend yield.
            is_fwd: if True, treat ``spot`` as the forward price.

        Raises:
            ValueError: on any of ``sigma <= 0``, ``vov <= 0``, ``mr <= 0``,
                ``theta <= 0``, or ``rho`` outside (-1, 1).
        """
        # Pre-validate before super() so bad inputs fail fast, before any
        # MGF-correction precomputation in the base classes runs.
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

        # SvABC sets self.theta = sigma when theta is None; re-check after super.
        if self.theta <= 0.0:
            raise ValueError(f"theta resolved to <= 0 ({self.theta})")

    # ------------------------------------------------------------------
    # Heston-specific COS helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _default_L(texp):
        """Tau-adaptive half-width multiplier ``L = max(10, 3*texp + 2)``.

        Interpolates between F&O's L=10 at texp=1 and L~32 at texp=10,
        keeping the trap-free CF numerically stable for long maturities.
        """
        return max(10.0, 3.0 * float(texp) + 2.0)

    def _resolve_L(self, texp):
        """Resolve L: ``self.L`` if set, else ``_default_L(texp)``."""
        return self._default_L(texp) if self.L is None else float(self.L)

    def _sigma_h(self):
        """F&O 2008 Section 5.2 truncation scale ``sqrt(theta + sigma*vov)``."""
        return float(np.sqrt(self.theta + self.sigma * self.vov))

    def _c1_logF(self, texp):
        """Mean of log(S_T/F) under Heston.

        By Ito's lemma log(S_T/F) = -0.5 * integrated_variance + martingale,
        so c1 = -0.5 * texp * E[avg_variance].  E[avg_variance] comes from
        ``HestonABC.avgvar_mv`` (Ball & Roma 1994 Appendix B), which is the
        canonical Heston-side source of truth -- we reuse it instead of
        re-deriving the closed form.
        """
        T = float(texp)
        return -0.5 * T * self.avgvar_mv(T)[0]

    def truncation_interval(self, strike, spot, texp, L=None):
        """COS truncation interval used by ``price()``.

        Combines two pieces from F&O 2008:

        * Half-width ``L * sigma_h`` from Section 5.2 with
          ``sigma_h = sqrt(theta + sigma*vov)``.  The paper introduces
          this Heston-tailored proxy as an alternative to the generic
          Section 5.1 cumulant range ``L*sqrt(c2 + sqrt(c4))`` because the
          Heston c2 closed form has a known transcription typo (the
          cumulant range remains available via
          ``CosABC._truncation_range`` for diagnostics).
        * Per-strike center ``log(F/K) + c1`` from Section 5.1, where c1
          is the mean of log(S_T/F).  In F&O's notation this is
          ``c1 + x0`` with ``x0 = log(S0/K)``; the log-spot/log-forward
          distinction collapses since the drift sits inside the forward.

        Ruijter & Oosterlee (2012) Section 2.3 give a useful discussion of
        why per-strike centering equalizes left/right tail capture across
        a strike vector; cited for context, not as the source of the
        formula.

        Args:
            strike: scalar or array of strikes.
            spot: spot (or forward, when ``is_fwd=True``) price; scalar.
            texp: time to expiry.
            L: optional override of the half-width multiplier; defaults to
                ``_resolve_L(texp)``.

        Returns:
            ``(a, b, x, width)``: per-strike floats/arrays.
            ``x = log(F/K)``, ``width = 2*L*sigma_h``,
            ``[a, b] = x + c1 +/- L*sigma_h``.
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
        """CosABC interval hook -- delegates to ``truncation_interval``.

        Returns ``(a, b)`` only; drops the diagnostic ``x`` and ``width``.
        """
        a, b, _, _ = self.truncation_interval(strike, spot, texp)
        return a, b

    def _smile_truncation_range(self, texp, L=None):
        """
        Strike-independent Heston interval in log(S_T/F) coordinates.

        ``truncation_interval(...)`` reports the paper's per-strike
        interval for y = log(S_T/K): ``log(F/K) + c1 +/- L*sigma_h``.
        For the reusable smile setup we work in z = log(S_T/F), so the
        equivalent interval is simply ``c1 +/- L*sigma_h`` and the strike
        only enters through the payoff boundary ``log(K/F)``.
        """
        L_value = self._resolve_L(texp) if L is None else float(L)
        half = L_value * self._sigma_h()
        center = self._c1_logF(texp)
        return center - half, center + half

    def make_smile_setup(
        self,
        spot,
        texp,
        trunc_range=None,
        eps_tol=1e-8,
        moment_order=8,
        payoff_bound=1.0,
    ):
        """
        Build reusable strike-independent Heston COS coefficients.

        If ``trunc_range`` is omitted, this uses the same Heston-tailored
        F&O half-width as ``price(...)``, translated from y = log(S_T/K)
        to the strike-independent z = log(S_T/F) variable.
        """
        if trunc_range is None:
            trunc_range = self._smile_truncation_range(texp)
        return CosABC.make_smile_setup(
            self,
            spot,
            texp,
            trunc_range=trunc_range,
            eps_tol=eps_tol,
            moment_order=moment_order,
            payoff_bound=payoff_bound,
        )

    def price_smile(
        self,
        strike,
        spot,
        texp,
        cp=1,
        trunc_range=None,
        eps_tol=1e-8,
        moment_order=8,
        payoff_bound=1.0,
    ):
        """
        European Heston prices through a reusable strike-independent setup.

        This convenience method is intentionally separate from ``price``:
        the existing Numba-backed implementation remains the default path,
        while this method exposes the reusable density coefficients needed
        for volatility-smile experiments.
        """
        setup = self.make_smile_setup(
            spot,
            texp,
            trunc_range=trunc_range,
            eps_tol=eps_tol,
            moment_order=moment_order,
            payoff_bound=payoff_bound,
        )
        return setup.price(strike, cp=cp)

    # ------------------------------------------------------------------
    # Cumulants (MGF inherited from HestonFft via MRO)
    # ------------------------------------------------------------------

    def _cumulants(self, texp, eps=1e-4):
        """Cumulants ``(c1, c2, 0, c4=0)`` of log(S_T/F).

        c1 from ``_c1_logF`` (analytic).  c2 from central finite differences
        of log(MGF) at the real axis -- the source pricer explicitly avoids
        F&O's Appendix A.2 closed form because of a transcription typo
        that produces a ~7e-4 discrepancy; we follow suit and use the
        inherited ``mgf_logprice`` as the single source of truth for c2.
        c4 = 0 per F&O Section 5.

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
        """Analytic integral of ``exp(x)*cos(k*pi*(x-a)/(b-a))`` from c to d.

        F&O 2008 Eq. (22).  Source-compatible static helper.
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
        """Analytic integral of ``cos(k*pi*(x-a)/(b-a))`` from c to d.

        F&O 2008 Eq. (23).  k=0 collapses to ``d - c`` and is handled
        separately to avoid division by zero.
        """
        ba     = b - a
        u      = k * np.pi / ba
        u_safe = np.where(k == 0, 1.0, u)
        arg_d  = u * (d - a)
        arg_c  = u * (c - a)
        return np.where(k == 0, d - c, (np.sin(arg_d) - np.sin(arg_c)) / u_safe)

    @staticmethod
    def payoff_coefficients(N, a, b, cp):
        """Vanilla call/put payoff coefficients ``U_k``, shape ``(M, N)``.

        F&O 2008 Eqs. (29)-(30).  NumPy reference; the JIT kernels compute
        the same coefficients inline.  Built with c=0 for calls and d=0
        for puts.

        Args:
            N: number of cosine terms.
            a, b: per-strike interval endpoints; arrays of shape ``(M,)``.
            cp: ``+1`` for call, ``-1`` for put.

        Returns:
            ``(M, N)`` array of payoff coefficients.
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
    # Pricing -- override of CosABC.price using the JIT kernels.
    # ------------------------------------------------------------------

    def price(self, strike, spot, texp, cp=1):
        """European call/put price via the Heston COS method.

        Vectorised over ``strike`` and ``cp``.  Mixed call/put portfolios
        run both kernels and mux on the cp sign (~2x compute on truly
        mixed inputs).  Scalar ``spot`` only -- vector spot is not
        supported in this port.

        Args:
            strike: scalar or array of strikes.
            spot: spot (or forward, when ``is_fwd=True``) price; scalar.
            texp: time to expiry; must be > 0.
            cp: ``+1`` for call, ``-1`` for put.  Scalar or array
                broadcast-compatible with ``strike``.

        Returns:
            Option price(s) matching the broadcast shape of
            ``(strike, cp)``.

        Raises:
            ValueError: if ``texp <= 0``, ``self.n_cos < 1``, or ``spot``
                is array-valued.
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

        # Uniform cp -> single kernel call. Mixed cp -> run both, mux on sign.
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
        """Dispatch to scalar or vector kernel based on ``K_arr`` size."""
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

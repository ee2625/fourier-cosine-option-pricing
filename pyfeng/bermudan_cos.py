"""
Bermudan option pricing via the Fourier-Cosine method (Fang-Oosterlee 2009).

Provides a model-agnostic ``BermudanCosMixin`` plus three concrete classes
restricted to time-homogeneous one-dimensional Lévy processes:

* ``BermudanBsmCos``   -- Black-Scholes-Merton.
* ``BermudanVgCos``    -- Variance Gamma.
* ``BermudanCgmyCos``  -- CGMY.

The Fang-Oosterlee 2009 algorithm is model-agnostic for 1D Lévy processes:
the only model-specific input is the per-timestep characteristic function
``phi(u; dt) = E[exp(i*u*log(S_{t+dt}/S_t))]``, supplied by any
``CosABC`` subclass via ``charfunc_logprice(u, dt)``.  The chi/psi payoff
coefficients and the closed-form continuation matrix M_{k,n} depend only on
the truncation interval and the vanilla payoff.

Heston is intentionally excluded.  Its 2-D state (log-spot, variance)
requires the 2-D extension of F&O 2009 Section 4 -- a separate effort that
does not fit this 1-D mixin.  Mixing ``BermudanCosMixin`` into a
``HestonCos`` subclass will silently produce incorrect prices because the
per-step CF for Heston is conditional on the variance state.

References:
    Fang F, Oosterlee CW (2009) Pricing Early-Exercise and Discrete Barrier
    Options by Fourier-Cosine Series Expansions.
    Numerische Mathematik 114:27-62.
    https://doi.org/10.1007/s00211-009-0252-4
"""

import numpy as np
from scipy.optimize import brentq

from .sv_cos import BsmCos
from .lv_cos import VarGammaCos, CgmyCos


__all__ = [
    "BermudanCosMixin",
    "BermudanBsmCos",
    "BermudanVgCos",
    "BermudanCgmyCos",
]


class BermudanCosMixin:
    """
    Mixin adding Bermudan call/put pricing on top of any 1-D Lévy
    ``CosABC`` subclass.

    Combine with a model class that provides ``charfunc_logprice``,
    ``_truncation_range``, and ``_fwd_factor`` (every ``CosABC``
    descendant does).  The mixin adds no state and no abstract methods,
    so concrete classes are typically a one-line subclass.

    Attributes:
        n_exercise (int): Number of equally-spaced exercise dates
            ``t_1, ..., t_M = T``.  Default 10.

    Notes:
        ``n_cos`` and ``L`` are inherited from the underlying
        ``CosABC`` subclass.  F&O 2009 BSM examples use ``L = 10``;
        set ``m.L = 10`` to reproduce paper numbers exactly.

        The mixin assumes time-homogeneous 1-D Lévy increments so the
        per-timestep CF reduces to ``charfunc_logprice(u, dt)``.  Do not
        mix into Heston or any other multi-state model.

    Examples:
        >>> import pyfeng as pf
        >>> m = pf.BermudanBsmCos(sigma=0.25, intr=0.10)
        >>> m.n_exercise = 50
        >>> m.price_bermudan(100.0, 100.0, 1.0, cp=-1)   # ~6.55 (American put limit)
    """

    n_exercise: int = 10

    # -- per-timestep CF on the COS grid ----------------------------------

    def _phi_dt(self, dt, u_arr):
        """CF of one log-return increment ``log(S_{t+dt}/S_t)``.

        ``charfunc_logprice(u, dt)`` returns the CF of ``log(S_dt/F_dt)``
        -- the martingale-shifted log-return.  The backward induction
        works in ``x = log(S_t/K)``, whose increment ``log(S_{t+dt}/S_t)``
        differs from the CosABC variable by a deterministic drift
        ``(intr - divr) * dt``.  Add it back via the phase factor
        ``exp(i*u*(intr-divr)*dt)``.  For BSM this reduces to
        ``exp(i*u*(r-q-0.5*sigma^2)*dt - 0.5*sigma^2*dt*u^2)``.
        """
        drift = (self.intr - self.divr) * dt
        return np.exp(1j * u_arr * drift) * self.charfunc_logprice(u_arr, dt)

    # -- chi / psi (F&O Eqs. 22-23) ---------------------------------------

    @staticmethod
    def _chi_psi(c, d, a, b, N):
        """``(chi, psi)`` arrays of length ``N`` for the analytic integrals
        of ``e^x cos(...)`` and ``cos(...)`` over ``[c, d]`` -- F&O Eqs. 22-23.
        """
        ba = b - a
        k = np.arange(N)
        u = k * np.pi / ba
        u_safe = np.where(k == 0, 1.0, u)

        sin_d = np.sin(u * (d - a))
        sin_c = np.sin(u * (c - a))
        cos_d = np.cos(u * (d - a))
        cos_c = np.cos(u * (c - a))
        exp_d = np.exp(d)
        exp_c = np.exp(c)

        psi = np.where(k == 0, d - c, (sin_d - sin_c) / u_safe)
        chi = (cos_d * exp_d - cos_c * exp_c
               + u * (sin_d * exp_d - sin_c * exp_c)) / (1.0 + u * u)
        return chi, psi

    @classmethod
    def _payoff_coeffs(cls, c, d, K, a, b, N, cp):
        """Vanilla intrinsic coefficients on ``[c, d]``.

        Put  (cp < 0):  payoff K(1 - e^x), ``G_k = (2/ba) * K * (psi - chi)``.
        Call (cp > 0):  payoff K(e^x - 1), ``G_k = (2/ba) * K * (chi - psi)``.
        """
        ba = b - a
        chi, psi = cls._chi_psi(c, d, a, b, N)
        if cp < 0:
            return (2.0 / ba) * K * (psi - chi)
        return (2.0 / ba) * K * (chi - psi)

    # -- Closed-form M_{k,n}(x*) matrix (F&O 2009 Eq. 16) ------------------

    @staticmethod
    def _M_matrix(x_star, a, b, N, cp):
        """``(N, N)`` complex matrix of continuation-region integrals.

        Continuation interval depends on ``cp``:

        * put  (cp < 0): ``[x*, b]`` -- exercise on ``[a, x*]``, continue above.
        * call (cp > 0): ``[a, x*]`` -- exercise on ``[x*, b]``, continue below.

        Uses ``2 cos(a) e^{ib} = e^{i(b+a)} + e^{i(b-a)}`` to split into
        sum/difference frequency integrals.
        """
        ba = b - a
        u = np.arange(N) * np.pi / ba
        if cp < 0:
            c = x_star - a
            d = ba                                          # b - a
        else:
            c = 0.0
            d = x_star - a

        u_sum  = u[:, None] + u[None, :]                    # (N, N)
        u_diff = u[None, :] - u[:, None]                    # (N, N)

        def I_func(omega):
            mask  = (omega == 0)
            denom = np.where(mask, 1.0, omega)
            val   = (np.exp(1j * omega * d) - np.exp(1j * omega * c)) / (1j * denom)
            return np.where(mask, d - c, val)

        return (I_func(u_sum) + I_func(u_diff)) / ba

    # -- Continuation value at a single x ---------------------------------

    @staticmethod
    def _continuation_value(x, V_k, phi, u, a, df_dt):
        """``df_dt * Re[ sum'_n V_k[n] phi[n] exp(i u_n (x - a)) ]``.

        Prime sum halves the n = 0 term.
        """
        Vw = V_k.copy()
        Vw[0] *= 0.5
        phase = np.exp(1j * u * (x - a))
        return df_dt * float(np.real(np.sum(Vw * phi * phase)))

    # -- Public API --------------------------------------------------------

    def price_bermudan(self, strike, spot, texp, cp=-1):
        """Bermudan call/put price with ``self.n_exercise`` equally-spaced
        exercise dates ``t_1, ..., t_M = texp``.

        Args:
            strike: Scalar strike.
            spot:   Scalar spot.
            texp:   Time to maturity.
            cp:     +1 call, -1 put.  Default -1 (put).

        Returns:
            Scalar price.

        Notes:
            For a call with zero dividend yield there is no early-exercise
            benefit -- the price collapses to the European call modulo COS
            truncation noise that compounds across backward-induction steps.
            Use ``divr > 0`` for a meaningful Bermudan call.
        """
        if int(self.n_exercise) < 1:
            raise ValueError(f"n_exercise must be >= 1, got {self.n_exercise}")
        if cp not in (1, -1):
            raise ValueError(f"cp must be +1 (call) or -1 (put), got {cp}")

        S = float(spot)
        K = float(strike)
        T = float(texp)
        M = int(self.n_exercise)
        N = int(self.n_cos)

        x0 = np.log(S / K)

        # CosABC._truncation_range returns [a, b] for log(S_T/F).  The
        # backward induction works in x = log(S_t/K) -- the variable in
        # which the intrinsic boundary is at x = 0.  Convert by shifting
        # the interval by log(F/K), which preserves the cumulant-prescribed
        # width while re-centering on E[log(S_T/K)] = x0 + (intr - divr)*T.
        a_F, b_F = self._truncation_range(T)
        fwd, _, _ = self._fwd_factor(spot, texp)
        log_FK = float(np.log(fwd / K))
        a = a_F + log_FK
        b = b_F + log_FK
        ba = b - a

        u = np.arange(N) * np.pi / ba

        dt    = T / M
        phi   = self._phi_dt(dt, u)
        df_dt = float(np.exp(-self.intr * dt))

        # Initialise V_k at t_M with the European intrinsic on the
        # in-the-money side of x = 0.  Put: [a, min(0, b)]; call: [max(0, a), b].
        if cp < 0:
            d_payoff = min(0.0, b)
            V_k      = self._payoff_coeffs(a, d_payoff, K, a, b, N, cp=-1)
            bnd_lo, bnd_hi = a, 0.0
            intrinsic = lambda x: K * (1.0 - np.exp(x))
        else:
            c_payoff = max(0.0, a)
            V_k      = self._payoff_coeffs(c_payoff, b, K, a, b, N, cp=+1)
            bnd_lo, bnd_hi = 0.0, b
            intrinsic = lambda x: K * (np.exp(x) - 1.0)

        # Backward induction t_{M-1} -> t_1.  Skipped entirely when M = 1.
        for _ in range(M - 1, 0, -1):
            f = lambda x: self._continuation_value(x, V_k, phi, u, a, df_dt) - intrinsic(x)
            try:
                x_star = brentq(f, bnd_lo, bnd_hi, xtol=1e-10, maxiter=200)
            except ValueError:
                # No sign change: continuation dominates -> empty exercise piece.
                x_star = bnd_lo if cp < 0 else bnd_hi

            if cp < 0:
                G_c, G_d = a, x_star            # exercise interval [a, x*]
            else:
                G_c, G_d = x_star, b            # exercise interval [x*, b]

            G = self._payoff_coeffs(G_c, G_d, K, a, b, N, cp=cp)

            M_mat         = self._M_matrix(x_star, a, b, N, cp=cp)
            V_weighted    = V_k.copy()
            V_weighted[0] *= 0.5
            C = df_dt * np.real(M_mat @ (V_weighted * phi))

            V_k = G + C

        # Final step from t_1 to t_0: option value = continuation at x_0.
        return self._continuation_value(x0, V_k, phi, u, a, df_dt)


# ─────────────────────────────────────────────────────────────────────────────
# Concrete classes -- one line each
# ─────────────────────────────────────────────────────────────────────────────


class BermudanBsmCos(BsmCos, BermudanCosMixin):
    """Bermudan call/put under Black-Scholes-Merton via COS (F&O 2009)."""


class BermudanVgCos(VarGammaCos, BermudanCosMixin):
    """Bermudan call/put under Variance Gamma via COS (F&O 2009)."""


class BermudanCgmyCos(CgmyCos, BermudanCosMixin):
    """Bermudan call/put under CGMY via COS (F&O 2009)."""

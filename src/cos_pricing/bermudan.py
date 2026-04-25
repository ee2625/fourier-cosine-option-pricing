"""
Bermudan option pricing under BSM via the Fourier-Cosine method.

Direct implementation of the Fang-Oosterlee 2009 follow-up to the 2008
paper, restricted to the BSM case (i.i.d. log-return increments) so the
recursion can run on a single state variable x = log(S/K).

Algorithm (Fang & Oosterlee 2009, Section 3, BSM specialisation):

    Truncation [a, b] from BSM cumulants over the full horizon T,
    centered on x_0 + c_1.

    Initialise at t_M = T with the European put COS coefficients:
        V_k(T) = (2/ba) * K * [psi_k(a, 0) - chi_k(a, 0)]   (paper Eq. 22-23)

    For j = M - 1, M - 2, ..., 1:

      1. Continuation value at any x:
            hat V(x) = exp(-r Δt) * Re[ sum'_n  V_n(t_{j+1}) phi(u_n) exp(i u_n (x - a)) ]
         where phi is the CF of one timestep increment and prime-sum halves
         the n = 0 term.

      2. Early-exercise boundary x* solves
            hat V(x*) = K (1 - exp(x*)),       x* in [a, 0].
         Found by Brent root-finding.

      3. New coefficients V_k(t_j) = G_k(x*) + C_k(x*), where:
           - G_k(x*) is the analytic exercise piece (chi/psi over [a, x*])
           - C_k(x*) is the continuation piece (matrix-vector product with
             a closed-form matrix M_{k,n}(x*))

      4. Final step: V(x_0, t_0) = hat V(x_0) (no exercise at inception).

Cost: O(M N^2) per option (the M_{k,n} matrix is N x N per timestep).
Validation: M = 1 reduces to the European put price (a closed-form check).

References:
    Fang F, Oosterlee CW (2009) Pricing Early-Exercise and Discrete Barrier
    Options by Fourier-Cosine Series Expansions. Numerische Mathematik
    114:27-62.
"""
import numpy as np
from scipy.optimize import brentq


class BermudanCosBSM:
    """
    Bermudan put under Black-Scholes-Merton via the COS method.

    Parameters
    ----------
    sigma : float    Constant volatility.
    intr  : float    Continuously compounded risk-free rate r.  Default 0.
    divr  : float    Continuous dividend yield q.  Default 0.

    Examples
    --------
    >>> m = BermudanCosBSM(sigma=0.25, intr=0.1)
    >>> m.price_put(S=100.0, K=100.0, T=1.0, M=10, N=128)        # Bermudan
    >>> m.price_put(S=100.0, K=100.0, T=1.0, M=1,  N=128)        # = European put
    """

    def __init__(self, sigma, intr=0.0, divr=0.0):
        self.sigma = float(sigma)
        self.intr  = float(intr)
        self.divr  = float(divr)

    # ── Truncation range ───────────────────────────────────────────────────

    def _trunc_range(self, x0, T, L=10.0):
        """[a, b] from BSM cumulants of x_T - x_0, centered on x_0 + c_1."""
        c1   = (self.intr - self.divr - 0.5 * self.sigma ** 2) * T
        c2   = self.sigma ** 2 * T
        half = L * np.sqrt(c2)
        return x0 + c1 - half, x0 + c1 + half

    # ── BSM increment CF over a single timestep dt ─────────────────────────

    def _phi_dt(self, dt):
        """CF of  log(S_{t+dt}/S_t)  under risk-neutral BSM."""
        mu     = (self.intr - self.divr - 0.5 * self.sigma ** 2) * dt
        sig2dt = self.sigma ** 2 * dt
        def cf(u):
            return np.exp(1j * u * mu - 0.5 * sig2dt * u ** 2)
        return cf

    # ── Analytic chi / psi pieces (paper Eqs. 22-23) ───────────────────────

    @staticmethod
    def _chi_psi(c, d, a, b, N):
        """Return (chi, psi) arrays of length N for the analytic integrals
        of e^x cos(...) and cos(...) over [c, d]."""
        ba    = b - a
        k     = np.arange(N)
        u     = k * np.pi / ba
        u_sf  = np.where(k == 0, 1.0, u)

        sin_d = np.sin(u * (d - a))
        sin_c = np.sin(u * (c - a))
        cos_d = np.cos(u * (d - a))
        cos_c = np.cos(u * (c - a))
        exp_d = np.exp(d)
        exp_c = np.exp(c)

        psi = np.where(k == 0, d - c, (sin_d - sin_c) / u_sf)
        chi = (cos_d * exp_d - cos_c * exp_c
               + u * (sin_d * exp_d - sin_c * exp_c)) / (1.0 + u * u)
        return chi, psi

    def _put_coeffs_partial(self, c, d, K, a, b, N):
        """G_k for a put on [c, d]: (2/ba) * K * (psi - chi)."""
        ba       = b - a
        chi, psi = self._chi_psi(c, d, a, b, N)
        return (2.0 / ba) * K * (psi - chi)

    # ── Closed-form M_{k,n}(x*) matrix (continuation-region integral) ──────

    @staticmethod
    def _M_matrix(x_star, a, b, N):
        """M[k, n] = (2/ba) * integral_{x*}^b cos(u_k (x-a)) exp(i u_n (x-a)) dx,
        a complex (N, N) matrix.

        2 cos(α) e^{iβ} = e^{i(β+α)} + e^{i(β-α)}, so
        M[k, n] = (1/ba) * [ I(u_n + u_k, x*-a, b-a) + I(u_n - u_k, x*-a, b-a) ]
        with I(ω, c, d) = (e^{iωd} - e^{iωc})/(iω) for ω != 0, and (d - c) otherwise.
        """
        ba   = b - a
        u    = np.arange(N) * np.pi / ba
        c    = x_star - a
        d    = ba                                          # b - a

        u_sum  = u[:, None] + u[None, :]                   # (N, N)
        u_diff = u[None, :] - u[:, None]                   # (N, N)

        def I_func(omega):
            mask  = (omega == 0)
            denom = np.where(mask, 1.0, omega)
            val   = (np.exp(1j * omega * d) - np.exp(1j * omega * c)) / (1j * denom)
            return np.where(mask, d - c, val)

        return (I_func(u_sum) + I_func(u_diff)) / ba

    # ── Continuation value at a single x ───────────────────────────────────

    @staticmethod
    def _hat_V(x, V_k, phi, u, a, df_dt):
        """hat V(x) = df_dt * Re[ sum'_n V_k[n] phi[n] exp(i u_n (x - a)) ]
        with the prime-sum halving the n = 0 term."""
        Vw     = V_k.copy()
        Vw[0] *= 0.5
        phase  = np.exp(1j * u * (x - a))
        return df_dt * float(np.real(np.sum(Vw * phi * phase)))

    # ── Public API ─────────────────────────────────────────────────────────

    def price_put(self, S, K, T, M=10, N=128, L=10.0):
        """Bermudan put with M equally-spaced exercise dates t_1, ..., t_M = T.

        Parameters
        ----------
        S, K, T : floats   Spot, strike, maturity.
        M : int            Number of exercise dates (M = 1 reduces to European).
        N : int            COS series cutoff.
        L : float          Truncation half-width multiplier.
        """
        if M < 1:
            raise ValueError(f"M must be >= 1, got {M}")

        x0   = float(np.log(S / K))
        a, b = self._trunc_range(x0, T, L=L)
        ba   = b - a
        u    = np.arange(N) * np.pi / ba

        dt    = T / M
        phi   = self._phi_dt(dt)(u)
        df_dt = float(np.exp(-self.intr * dt))

        # Initialise V_k at t_M = T with European put coefficients on [a, min(0, b)].
        d_payoff = min(0.0, b)
        V_k      = self._put_coeffs_partial(a, d_payoff, K, a, b, N)

        # Backward induction t_{M-1} -> t_1.  Skipped entirely when M = 1.
        for _ in range(M - 1, 0, -1):
            # 1) Find early-exercise boundary x* in [a, 0].
            f = lambda x: self._hat_V(x, V_k, phi, u, a, df_dt) - K * (1.0 - np.exp(x))
            try:
                x_star = brentq(f, a, 0.0, xtol=1e-10, maxiter=200)
            except ValueError:
                # No sign change in [a, 0]: continuation dominates everywhere
                # in the exercise region, so the option behaves like a
                # European put one step earlier.
                x_star = a

            # 2) New coefficients via G + C.
            G = self._put_coeffs_partial(a, x_star, K, a, b, N)

            M_mat       = self._M_matrix(x_star, a, b, N)
            V_weighted  = V_k.copy()
            V_weighted[0] *= 0.5                           # prime-sum halving
            C = df_dt * np.real(M_mat @ (V_weighted * phi))

            V_k = G + C

        # Final step from t_1 to t_0: option value = continuation at x_0.
        return self._hat_V(x0, V_k, phi, u, a, df_dt)

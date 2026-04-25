"""
Bermudan option pricing under BSM via the Fourier-Cosine method.

Direct implementation of the Fang-Oosterlee 2009 follow-up to the 2008
paper, restricted to the BSM case (i.i.d. log-return increments) so the
recursion can run on a single state variable x = log(S/K).

Algorithm (Fang & Oosterlee 2009, Section 3, BSM specialisation):

    Truncation [a, b] from BSM cumulants over the full horizon T,
    centered on x_0 + c_1.

    Initialise at t_M = T with the European put / call COS coefficients
    on the in-the-money side of x = 0:
        - put : V_k(T) = (2/ba) * K * [psi_k(a, 0) - chi_k(a, 0)]
        - call: V_k(T) = (2/ba) * K * [chi_k(0, b) - psi_k(0, b)]

    For j = M - 1, M - 2, ..., 1:

      1. Continuation value at any x:
            hat V(x) = exp(-r dt) * Re[ sum'_n  V_n(t_{j+1}) phi(u_n) exp(i u_n (x - a)) ]
         where phi is the CF of one timestep increment and prime-sum halves
         the n = 0 term.

      2. Early-exercise boundary x* solves
            hat V(x*) = intrinsic(x*),
         on [a, 0] for puts and [0, b] for calls. Brent root-find.

      3. New coefficients V_k(t_j) = G_k(x*) + C_k(x*), where:
           - G_k(x*) is the analytic exercise piece (chi/psi over the
             exercise interval -- [a, x*] for puts, [x*, b] for calls).
           - C_k(x*) is the continuation piece (matrix-vector product with
             a closed-form matrix M_{k,n}(x*) over the continuation
             interval -- [x*, b] for puts, [a, x*] for calls).

      4. Final step: V(x_0, t_0) = hat V(x_0) (no exercise at inception).

Cost: O(M N^2) per option (the M_{k,n} matrix is N x N per timestep).
Validation: M = 1 reduces to the European price for either cp.

References:
    Fang F, Oosterlee CW (2009) Pricing Early-Exercise and Discrete Barrier
    Options by Fourier-Cosine Series Expansions. Numerische Mathematik
    114:27-62.
"""
import numpy as np
from scipy.optimize import brentq


class BermudanCosBSM:
    """
    Bermudan call/put under Black-Scholes-Merton via the COS method.

    Parameters
    ----------
    sigma : float    Constant volatility.
    intr  : float    Continuously compounded risk-free rate r.  Default 0.
    divr  : float    Continuous dividend yield q.  Default 0.

    Examples
    --------
    >>> m = BermudanCosBSM(sigma=0.25, intr=0.1)
    >>> m.price_put(S=100.0, K=100.0, T=1.0, M=10, N=128)        # Bermudan put
    >>> m.price_put(S=100.0, K=100.0, T=1.0, M=1,  N=128)        # = European put
    >>> m.price_call(S=100.0, K=100.0, T=1.0, M=10, N=128)       # Bermudan call (early exercise only matters with q > 0)
    """

    def __init__(self, sigma, intr=0.0, divr=0.0):
        self.sigma = float(sigma)
        self.intr  = float(intr)
        self.divr  = float(divr)

    # -- Truncation range --------------------------------------------------

    def _trunc_range(self, x0, T, L=10.0):
        """[a, b] from BSM cumulants of x_T - x_0, centered on x_0 + c_1."""
        c1   = (self.intr - self.divr - 0.5 * self.sigma ** 2) * T
        c2   = self.sigma ** 2 * T
        half = L * np.sqrt(c2)
        return x0 + c1 - half, x0 + c1 + half

    # -- BSM increment CF over a single timestep dt ------------------------

    def _phi_dt(self, dt):
        """CF of  log(S_{t+dt}/S_t)  under risk-neutral BSM."""
        mu     = (self.intr - self.divr - 0.5 * self.sigma ** 2) * dt
        sig2dt = self.sigma ** 2 * dt
        def cf(u):
            return np.exp(1j * u * mu - 0.5 * sig2dt * u ** 2)
        return cf

    # -- Analytic chi / psi pieces (paper Eqs. 22-23) ----------------------

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

    def _payoff_coeffs_partial(self, c, d, K, a, b, N, cp=-1):
        """G_k for the vanilla intrinsic on [c, d].

        Put  (cp < 0): payoff K(1 - e^x), G_k = (2/ba) * K * (psi - chi).
        Call (cp > 0): payoff K(e^x - 1), G_k = (2/ba) * K * (chi - psi).
        """
        ba       = b - a
        chi, psi = self._chi_psi(c, d, a, b, N)
        if cp < 0:
            return (2.0 / ba) * K * (psi - chi)
        return (2.0 / ba) * K * (chi - psi)

    # -- Closed-form M_{k,n}(x*) matrix (continuation-region integral) -----

    @staticmethod
    def _M_matrix(x_star, a, b, N, cp=-1):
        """M[k, n] = (2/ba) * integral_continuation cos(u_k (x-a)) exp(i u_n (x-a)) dx,
        a complex (N, N) matrix.

        Continuation interval depends on cp:
        - put  (cp < 0): [x*, b] -- exercise on [a, x*], continue above
        - call (cp > 0): [a, x*] -- exercise on [x*, b], continue below

        2 cos(a) e^{ib} = e^{i(b+a)} + e^{i(b-a)}, so
        M[k, n] = (1/ba) * [ I(u_n + u_k, c, d) + I(u_n - u_k, c, d) ]
        with I(omega, c, d) = (e^{i*omega*d} - e^{i*omega*c})/(i*omega) for
        omega != 0, and (d - c) otherwise.
        """
        ba = b - a
        u  = np.arange(N) * np.pi / ba
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

    # -- Continuation value at a single x ----------------------------------

    @staticmethod
    def _hat_V(x, V_k, phi, u, a, df_dt):
        """hat V(x) = df_dt * Re[ sum'_n V_k[n] phi[n] exp(i u_n (x - a)) ]
        with the prime-sum halving the n = 0 term."""
        Vw     = V_k.copy()
        Vw[0] *= 0.5
        phase  = np.exp(1j * u * (x - a))
        return df_dt * float(np.real(np.sum(Vw * phi * phase)))

    # -- Public API --------------------------------------------------------

    def price_put(self, S, K, T, M=10, N=128, L=10.0):
        """Bermudan put with M equally-spaced exercise dates."""
        return self._price(S, K, T, M, N, L, cp=-1)

    def price_call(self, S, K, T, M=10, N=128, L=10.0):
        """Bermudan call with M equally-spaced exercise dates.

        Early exercise is only economically meaningful when ``divr > 0``;
        with ``divr == 0`` the price collapses to the European call by
        the no-early-exercise theorem (modulo COS truncation noise).
        """
        return self._price(S, K, T, M, N, L, cp=+1)

    def _price(self, S, K, T, M, N, L, cp):
        """Backward induction. Shared body for puts and calls."""
        if M < 1:
            raise ValueError(f"M must be >= 1, got {M}")
        if cp not in (1, -1):
            raise ValueError(f"cp must be +1 (call) or -1 (put), got {cp}")

        x0   = float(np.log(S / K))
        a, b = self._trunc_range(x0, T, L=L)
        ba   = b - a
        u    = np.arange(N) * np.pi / ba

        dt    = T / M
        phi   = self._phi_dt(dt)(u)
        df_dt = float(np.exp(-self.intr * dt))

        # Initialise V_k at t_M = T with the European intrinsic on the
        # in-the-money side of x = 0.  Put: [a, min(0, b)]; call: [max(0, a), b].
        if cp < 0:
            d_payoff = min(0.0, b)
            V_k      = self._payoff_coeffs_partial(a, d_payoff, K, a, b, N, cp=-1)
            bnd_lo, bnd_hi = a, 0.0
            intrinsic = lambda x: K * (1.0 - np.exp(x))
        else:
            c_payoff = max(0.0, a)
            V_k      = self._payoff_coeffs_partial(c_payoff, b, K, a, b, N, cp=+1)
            bnd_lo, bnd_hi = 0.0, b
            intrinsic = lambda x: K * (np.exp(x) - 1.0)

        # Backward induction t_{M-1} -> t_1.  Skipped entirely when M = 1.
        for _ in range(M - 1, 0, -1):
            # 1) Find early-exercise boundary x* in the in-the-money range.
            f = lambda x: self._hat_V(x, V_k, phi, u, a, df_dt) - intrinsic(x)
            try:
                x_star = brentq(f, bnd_lo, bnd_hi, xtol=1e-10, maxiter=200)
            except ValueError:
                # No sign change: continuation dominates everywhere in the
                # in-the-money region -> no early exercise this step.
                # Set x* to the boundary where the exercise piece is empty.
                x_star = bnd_lo if cp < 0 else bnd_hi

            # 2) New coefficients via G + C.
            if cp < 0:
                G_c, G_d = a, x_star            # exercise interval [a, x*]
            else:
                G_c, G_d = x_star, b            # exercise interval [x*, b]

            G = self._payoff_coeffs_partial(G_c, G_d, K, a, b, N, cp=cp)

            M_mat       = self._M_matrix(x_star, a, b, N, cp=cp)
            V_weighted  = V_k.copy()
            V_weighted[0] *= 0.5                # prime-sum halving
            C = df_dt * np.real(M_mat @ (V_weighted * phi))

            V_k = G + C

        # Final step from t_1 to t_0: option value = continuation at x_0.
        return self._hat_V(x0, V_k, phi, u, a, df_dt)

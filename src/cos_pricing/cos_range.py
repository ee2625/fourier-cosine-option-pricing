"""
Truncation-range helpers for the COS method.

The Junike-Pankrashkin range uses Markov's inequality on the centered
log-return X - E[X].  Given an even central moment and a target tolerance,
it returns a symmetric half-width around the mean.
"""

from math import comb, factorial, gamma

import numpy as np


def central_moment_from_cumulants(cumulants, order):
    """
    Convert cumulants into a central moment.

    ``cumulants[j]`` is the j-th cumulant, with index 0 unused.  The first
    cumulant is ignored because central moments are moments of X - E[X].
    """
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")
    if len(cumulants) <= order:
        raise ValueError("cumulants must contain entries through the requested order")

    kappa = np.zeros(order + 1, dtype=float)
    kappa[2:] = np.asarray(cumulants[2: order + 1], dtype=float)

    moments = np.zeros(order + 1, dtype=float)
    moments[0] = 1.0
    for n in range(1, order + 1):
        total = 0.0
        for j in range(n):
            total += comb(n - 1, j) * kappa[n - j] * moments[j]
        moments[n] = total
    return float(moments[order])


def jp_markov_half_width(
    central_moment,
    variance,
    eps_tol=1e-8,
    payoff_bound=1.0,
    moment_order=8,
):
    """
    Junike-Pankrashkin Markov half-width ``L``.

    Implements Corollary 9 of Junike and Pankrashkin (2022).  ``variance``
    is used as the variance of the matching Laplace tail proxy.  The payoff
    bound is in the same normalized units as the payoff being integrated.
    """
    if moment_order < 2 or moment_order % 2 != 0:
        raise ValueError(f"moment_order must be an even integer >= 2, got {moment_order}")
    if eps_tol <= 0.0:
        raise ValueError(f"eps_tol must be > 0, got {eps_tol}")
    if payoff_bound <= 0.0:
        raise ValueError(f"payoff_bound must be > 0, got {payoff_bound}")
    if variance <= 0.0:
        raise ValueError(f"variance must be > 0, got {variance}")
    if central_moment <= 0.0:
        raise ValueError(f"central_moment must be > 0, got {central_moment}")

    sigma_tail = float(np.sqrt(variance))
    eps = float(eps_tol)
    k_bound = float(payoff_bound)
    mu_n = float(central_moment)
    m_tail = (2.0 * k_bound * mu_n / eps) ** (1.0 / moment_order)

    sqrt2 = np.sqrt(2.0)
    prefactor = -sigma_tail / (2.0 * sqrt2)
    denom = sigma_tail**2 / m_tail**2 + 2.0 * sqrt2 * sigma_tail / m_tail + 4.0

    arg_b = sqrt2 * sigma_tail * eps**2 / (72.0 * m_tail * k_bound**2)
    arg_a = (12.0 * sqrt2 * sigma_tail * eps**2) / (
        72.0 * m_tail * k_bound**2 * np.pi**2 * denom
    )

    candidates = [m_tail]
    for arg in (arg_a, arg_b):
        if arg > 0.0:
            candidates.append(prefactor * np.log(arg))

    return float(max(c for c in candidates if np.isfinite(c)))


def jp_markov_range(
    center,
    variance,
    central_moment,
    eps_tol=1e-8,
    payoff_bound=1.0,
    moment_order=8,
):
    """Return ``(center - L, center + L)`` using the JP Markov half-width."""
    half = jp_markov_half_width(
        central_moment=central_moment,
        variance=variance,
        eps_tol=eps_tol,
        payoff_bound=payoff_bound,
        moment_order=moment_order,
    )
    return float(center - half), float(center + half)


def vg_cumulants(sigma, theta, nu, texp, order):
    """VG cumulants of log(S_T/F) through ``order``."""
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")
    sig2 = sigma**2
    arg = 1.0 - theta * nu - 0.5 * sig2 * nu
    if arg <= 0.0:
        raise ValueError("VG martingale correction is not real for these parameters")

    omega = np.log(arg) / nu
    a = theta * nu
    b = 0.5 * sig2 * nu

    p = np.zeros(order + 1, dtype=float)
    if order >= 1:
        p[1] = a
    if order >= 2:
        p[2] = b

    coeff = np.zeros(order + 1, dtype=float)
    power = np.array([1.0])
    for m in range(1, order + 1):
        power = np.convolve(power, p)[: order + 1]
        coeff[: power.size] += power / m

    cumulants = np.zeros(order + 1, dtype=float)
    cumulants[1:] = [
        factorial(j) * (texp / nu) * coeff[j]
        for j in range(1, order + 1)
    ]
    cumulants[1] += omega * texp
    return cumulants


def cgmy_cumulants(C, G, M, Y, texp, order):
    """CGMY cumulants of log(S_T/F) through ``order``."""
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")

    gamma_term = C * float(gamma(-Y))
    drift = -gamma_term * ((M - 1.0) ** Y - M**Y + (G + 1.0) ** Y - G**Y)

    cumulants = np.zeros(order + 1, dtype=float)
    falling = 1.0
    for j in range(1, order + 1):
        falling *= Y - (j - 1)
        jump = gamma_term * falling * (((-1.0) ** j) * M ** (Y - j) + G ** (Y - j))
        cumulants[j] = texp * (jump + (drift if j == 1 else 0.0))
    return cumulants

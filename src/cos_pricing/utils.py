"""
Utility functions for the COS pricing library.

Includes:
    - Analytic BSM formula (reference/validation)
    - Implied volatility via Brent's method
    - Timing / convergence benchmarks
"""

import time
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


# ─────────────────────────────────────────────────────────────────────────────
# Black-Scholes-Merton analytic formula
# ─────────────────────────────────────────────────────────────────────────────

def bsm_price(strike, spot, sigma, texp, intr=0.0, divr=0.0, cp=1):
    """
    Analytic Black-Scholes-Merton call/put price.

    Parameters
    ----------
    strike : float or array
    spot   : float
    sigma  : float  (annualised vol)
    texp   : float  (years)
    intr   : float  risk-free rate
    divr   : float  dividend yield
    cp     : +1 call / −1 put

    Returns
    -------
    float or array
    """
    fwd = spot * np.exp((intr - divr) * texp)
    df  = np.exp(-intr * texp)
    sqt = sigma * np.sqrt(texp)
    d1  = np.log(fwd / strike) / sqt + 0.5 * sqt
    d2  = d1 - sqt
    return df * cp * (fwd * norm.cdf(cp * d1) - strike * norm.cdf(cp * d2))


# ─────────────────────────────────────────────────────────────────────────────
# Implied volatility
# ─────────────────────────────────────────────────────────────────────────────

def bsm_impvol(price, strike, spot, texp, intr=0.0, divr=0.0, cp=1):
    """
    BSM implied volatility via Brent's method.

    Returns np.nan when no solution exists (e.g. price < intrinsic).
    """
    scalar = np.isscalar(price) and np.isscalar(strike)
    prices  = np.atleast_1d(np.asarray(price,  dtype=float))
    strikes = np.atleast_1d(np.asarray(strike, dtype=float))
    ivols   = np.empty(np.broadcast_shapes(prices.shape, strikes.shape))

    for idx in np.ndindex(ivols.shape):
        p = prices[idx] if prices.ndim > 0 else float(prices)
        k = strikes[idx] if strikes.ndim > 0 else float(strikes)
        try:
            ivols[idx] = brentq(
                lambda s: bsm_price(k, spot, s, texp, intr, divr, cp) - p,
                1e-8, 10.0, xtol=1e-10, maxiter=200
            )
        except ValueError:
            ivols[idx] = np.nan

    return float(ivols.flat[0]) if scalar else ivols


# ─────────────────────────────────────────────────────────────────────────────
# Convergence table
# ─────────────────────────────────────────────────────────────────────────────

def convergence_table(price_fn, ref_price, n_list=None, label="COS"):
    """
    Print a convergence table: N → price → absolute error.

    Parameters
    ----------
    price_fn : callable
        ``price_fn(n_cos) → float``
    ref_price : float
        Reference (analytic or high-N) price.
    n_list : list of int
        Values of N to test.  Default [8, 16, 32, 64, 128, 256].
    label : str

    Returns
    -------
    dict  {N: (price, abs_error)}
    """
    if n_list is None:
        n_list = [8, 16, 32, 64, 128, 256]
    results = {}
    print(f"\n{'N':>6}  {'Price':>16}  {'|Error|':>14}")
    print("-" * 42)
    for N in n_list:
        p = price_fn(N)
        e = abs(p - ref_price)
        results[N] = (p, e)
        print(f"{N:>6}  {p:>16.10f}  {e:>14.2e}")
    print(f"\n  Reference ({label}): {ref_price:.10f}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Runtime benchmark
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_runtime(price_fn, n_strikes=100, n_repeats=50, label=""):
    """
    Measure average runtime of ``price_fn`` over many strikes.

    Parameters
    ----------
    price_fn : callable
        ``price_fn(strikes: np.ndarray) → np.ndarray``
    n_strikes : int
    n_repeats : int
    label : str

    Returns
    -------
    float  mean wall-clock time in milliseconds
    """
    strikes = np.linspace(80, 120, n_strikes)
    # warm-up
    price_fn(strikes)
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        price_fn(strikes)
        times.append(time.perf_counter() - t0)
    ms = np.mean(times) * 1e3
    print(f"  {label:30s}  {ms:.4f} ms  ({n_strikes} strikes, {n_repeats} reps)")
    return ms
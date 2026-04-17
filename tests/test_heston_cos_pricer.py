"""
Tests for HestonCOSPricer — paper-faithful Heston COS method.

Run:
    PYTHONPATH=src python -m pytest tests/test_heston_cos_pricer.py -v
"""

import numpy as np
import pytest
from cos_pricing import HestonCOSPricer


PAPER_PARAMS = dict(
    S0=100.0, v0=0.0175, lam=1.5768, eta=0.5751, ubar=0.0398, rho=-0.5711,
    r=0.0, q=0.0,
)
REF_T1  = 5.785155435
REF_T10 = 22.318945791474590


# ─────────────────────────────────────────────────────────────────────────────
# Paper benchmarks
# ─────────────────────────────────────────────────────────────────────────────

def test_benchmark_T1():
    m = HestonCOSPricer(**PAPER_PARAMS)
    assert abs(m.price_call(100.0, 1.0, N=160) - REF_T1) < 1e-5


def test_benchmark_T10():
    m = HestonCOSPricer(**PAPER_PARAMS)
    assert abs(m.price_call(100.0, 10.0, N=160) - REF_T10) < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# Convergence and sensitivity
# ─────────────────────────────────────────────────────────────────────────────

def test_convergence_in_N():
    m = HestonCOSPricer(**PAPER_PARAMS)
    errs = [abs(m.price_call(100.0, 1.0, N=N) - REF_T1)
            for N in (16, 32, 64, 128, 256)]
    # Each refinement should reduce error (allow flat plateau near precision).
    for prev, curr in zip(errs, errs[1:]):
        assert curr <= prev * 1.1 + 1e-12
    assert errs[-1] < 1e-6


def test_L_sensitivity():
    """L=3 too narrow (truncation dominates); default L=12 accurate."""
    m = HestonCOSPricer(**PAPER_PARAMS)
    err_small_L = abs(m.price_call(100.0, 1.0, N=160, L=3.0) - REF_T1)
    err_good_L  = abs(m.price_call(100.0, 1.0, N=160, L=12.0) - REF_T1)
    # L=3 is truncation-dominated and should be orders of magnitude worse than L=12.
    assert err_small_L > 1e3 * err_good_L
    assert err_good_L  < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# Structural correctness
# ─────────────────────────────────────────────────────────────────────────────

def test_put_call_parity():
    """
    Parity holds up to the truncation tail of the exp(y)*f(y) integral.
    At L=12, T=1 the tail loss is ~5e-5; tighter L→larger c2 requires larger N.
    """
    m = HestonCOSPricer(**PAPER_PARAMS)
    tau = 1.0
    df  = np.exp(-m.r * tau)
    fwd = m.S0 * np.exp((m.r - m.q) * tau)
    for K in (80.0, 100.0, 120.0):
        c = m.price_call(K, tau, N=160)
        p = m.price_put (K, tau, N=160)
        assert abs((c - p) - df * (fwd - K)) < 1e-4


def test_strike_vectorized_matches_scalar():
    m = HestonCOSPricer(**PAPER_PARAMS)
    strikes = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
    vec = m.price_call(strikes, 1.0, N=160)
    scal = np.array([m.price_call(float(K), 1.0, N=160) for K in strikes])
    assert vec.shape == strikes.shape
    assert np.max(np.abs(vec - scal)) < 1e-12


def test_scalar_in_scalar_out():
    m = HestonCOSPricer(**PAPER_PARAMS)
    assert isinstance(m.price_call(100.0, 1.0), float)
    assert isinstance(m.price_put(100.0, 1.0), float)


def test_non_negative_prices():
    m = HestonCOSPricer(**PAPER_PARAMS)
    strikes = np.linspace(60, 200, 15)
    prices  = m.price_call(strikes, 1.0, N=160)
    assert np.all(prices >= -1e-10)


# ─────────────────────────────────────────────────────────────────────────────
# Static helpers
# ─────────────────────────────────────────────────────────────────────────────

def test_chi_psi_broadcasting():
    """chi/psi accept (N,) inputs and broadcast-shape (M,N) inputs equivalently."""
    k = np.arange(8)
    a, b, c, d = -2.0, 3.0, 0.0, 3.0
    chi_v = HestonCOSPricer.chi(k, a, b, c, d)
    psi_v = HestonCOSPricer.psi(k, a, b, c, d)
    assert chi_v.shape == (8,)
    assert psi_v.shape == (8,)
    # psi at k=0 should equal d - c
    assert abs(psi_v[0] - (d - c)) < 1e-14


# ─────────────────────────────────────────────────────────────────────────────
# Input validation
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("bad", [
    dict(rho=1.5),
    dict(rho=-1.0),
    dict(eta=0.0),
    dict(eta=-0.1),
    dict(lam=0.0),
    dict(ubar=-0.01),
    dict(v0=0.0),
])
def test_input_validation(bad):
    params = dict(PAPER_PARAMS)
    params.update(bad)
    with pytest.raises(ValueError):
        HestonCOSPricer(**params)


def test_invalid_tau_raises():
    m = HestonCOSPricer(**PAPER_PARAMS)
    with pytest.raises(ValueError):
        m.price_call(100.0, 0.0)
    with pytest.raises(ValueError):
        m.price_call(100.0, 1.0, N=0)

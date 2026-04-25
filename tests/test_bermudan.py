"""
Tests for the Bermudan COS pricer (Fang-Oosterlee 2009) under BSM.

Three structural checks that don't need a separate reference implementation:

1. **European limit.** M = 1 (one exercise date at maturity) must equal the
   European put price exactly, since with one exercise opportunity the
   Bermudan put degenerates to the European put.  The algorithm computes
   this through the same backward-induction code path that handles M > 1,
   so passing this catches sign errors, mis-applied prime-sums, and CF
   convention mistakes.

2. **Monotonicity in M.**  Adding exercise opportunities strictly cannot
   reduce the option's value, so prices must be non-decreasing in M.

3. **Convergence to American.**  As M -> infinity the Bermudan price
   approaches the American price.  We don't pin a particular American
   benchmark (no analytic American put under BSM), but we check that the
   sequence converges (Cauchy-style) and the limit is in a reasonable
   range for the standard test parameters.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from cos_pricing import BermudanCosBSM, bsm_price


# Standard FO2009 test parameters (Section 5.1 BSM).
SIGMA, R, Q = 0.25, 0.1, 0.0
S, K, T     = 100.0, 100.0, 1.0


@pytest.fixture
def m():
    return BermudanCosBSM(sigma=SIGMA, intr=R, divr=Q)


# ─────────────────────────────────────────────────────────────────────────────
# 1. European limit: M = 1 must equal the European put exactly.
# ─────────────────────────────────────────────────────────────────────────────

def test_M1_equals_european_put(m):
    """One exercise date at maturity = European put."""
    ber = m.price_put(S=S, K=K, T=T, M=1, N=128)
    eu  = float(bsm_price(K, S, SIGMA, T, intr=R, divr=Q, cp=-1))
    assert abs(ber - eu) < 1e-12, f"M=1 Bermudan {ber:.12f} != European {eu:.12f}"


def test_M1_equals_european_otm(m):
    """Same check at an OTM strike (S = 100, K = 90)."""
    K_otm = 90.0
    ber = m.price_put(S=100.0, K=K_otm, T=T, M=1, N=128)
    eu  = float(bsm_price(K_otm, 100.0, SIGMA, T, intr=R, divr=Q, cp=-1))
    assert abs(ber - eu) < 1e-12


def test_M1_equals_european_short_T(m):
    """Same check at a short maturity (T = 0.1)."""
    ber = m.price_put(S=S, K=K, T=0.1, M=1, N=128)
    eu  = float(bsm_price(K, S, SIGMA, 0.1, intr=R, divr=Q, cp=-1))
    assert abs(ber - eu) < 1e-12


# ─────────────────────────────────────────────────────────────────────────────
# 2. Monotonicity in M: adding exercise opportunities never reduces value.
# ─────────────────────────────────────────────────────────────────────────────

def test_monotonic_in_M(m):
    """Bermudan price must be non-decreasing in the number of exercise dates."""
    Ms     = [1, 2, 5, 10, 20, 50]
    prices = [m.price_put(S=S, K=K, T=T, M=M, N=128) for M in Ms]
    for prev, cur in zip(prices, prices[1:]):
        assert cur >= prev - 1e-10, \
            f"Monotonicity failed: prices = {prices}"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Convergence: Cauchy-style + sane limit range.
# ─────────────────────────────────────────────────────────────────────────────

def test_convergence_to_american(m):
    """Successive prices should bunch up (Cauchy-style); limit in [6.4, 6.7]
    for the standard FO2009 test parameters (American put ≈ 6.55)."""
    p10  = m.price_put(S=S, K=K, T=T, M=10,  N=128)
    p50  = m.price_put(S=S, K=K, T=T, M=50,  N=128)
    p100 = m.price_put(S=S, K=K, T=T, M=100, N=128)

    # Cauchy-style: successive differences shrink.
    assert abs(p50 - p10)  > abs(p100 - p50), "Differences not shrinking with M"
    # Sane range: the published American put under these params is ~6.55.
    assert 6.4 <= p100 <= 6.7, f"M=100 price {p100} outside expected American range"


def test_bounded_below_by_european(m):
    """Bermudan put >= European put for any M >= 1."""
    eu = float(bsm_price(K, S, SIGMA, T, intr=R, divr=Q, cp=-1))
    for M in [1, 5, 20]:
        ber = m.price_put(S=S, K=K, T=T, M=M, N=128)
        assert ber >= eu - 1e-10, f"M={M}: Bermudan {ber} < European {eu}"


def test_bounded_above_by_intrinsic_plus_european(m):
    """Bermudan put <= K (the strike, which bounds put payoff)."""
    for M in [1, 10, 50]:
        ber = m.price_put(S=S, K=K, T=T, M=M, N=128)
        assert ber <= K, f"M={M}: Bermudan {ber} exceeds strike {K}"


# ─────────────────────────────────────────────────────────────────────────────
# Convergence in N (COS series cutoff) at fixed M
# ─────────────────────────────────────────────────────────────────────────────

def test_convergence_in_N(m):
    """Doubling N at fixed M should not change the price meaningfully."""
    p_64  = m.price_put(S=S, K=K, T=T, M=10, N=64)
    p_128 = m.price_put(S=S, K=K, T=T, M=10, N=128)
    p_256 = m.price_put(S=S, K=K, T=T, M=10, N=256)
    assert abs(p_256 - p_128) < 1e-6
    assert abs(p_128 - p_64)  < 1e-3


# ─────────────────────────────────────────────────────────────────────────────
# Input validation
# ─────────────────────────────────────────────────────────────────────────────

def test_invalid_M_raises(m):
    with pytest.raises(ValueError, match="M must be"):
        m.price_put(S=S, K=K, T=T, M=0)

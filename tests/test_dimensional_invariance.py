"""
Track A — Dimensional invariance tests (Buckingham pi symmetries).

These tests are model-independent: they do not require an analytic
reference value.  They assert the symmetry groups predicted by
dimensional analysis hold for the COS pricer:

    BsmModel   — scale invariance:         C(lambda*S, lambda*K) = lambda * C(S, K)
    NormalCos  — translation invariance:   C(F + lambda, K + lambda) = C(F, K)

See: MATH5030 Lecture 3 (Cases 4 and 5), Buckingham pi theorem.
"""
import sys, os
# Support  `pytest tests/`  from repo root even without PYTHONPATH=src set.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from cos_pricing import BsmModel, NormalCos


# Fixed lambda grid from team_tasks.docx (Track A spec):
LAMBDAS = [0.1, 0.5, 2.0, 10.0, 100.0]

# Coverage: multiple strikes (ITM / ATM / OTM) and multiple maturities.
SPOT    = 100.0
STRIKES = np.array([70.0, 85.0, 100.0, 115.0, 130.0])
TEXPS   = [0.1, 1.0, 5.0]

TOL = 1e-10          # "10 decimal places" per task plan


# ─────────────────────────────────────────────────────────────────────────────
# BSM scale invariance:  C(lambda*S, lambda*K) = lambda * C(S, K)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("lam", LAMBDAS)
@pytest.mark.parametrize("cp, label", [(+1, "call"), (-1, "put")])
def test_scale_invariance_bsm(lam, cp, label):
    """
    BSM price is homogeneous of degree 1 in (S, K):
        C(lambda*S, lambda*K, sigma, T) = lambda * C(S, K, sigma, T).

    This reflects the dimensional structure: BSM has only 3 independent
    dimensionless pi-groups (K/S, sigma*sqrt(T), r*T).  Scaling S and K
    by lambda leaves K/S unchanged, so C/S is unchanged, so C scales linearly.

    Verified here for multiple (strike, maturity) pairs across each lambda.
    Tolerance: absolute error on C/S below 1e-10.
    """
    model = BsmModel(sigma=0.25, intr=0.05, divr=0.02)

    worst = 0.0
    worst_where = None
    for texp in TEXPS:
        base    = model.price(STRIKES,        SPOT,       texp, cp=cp)   # C(S, K)
        scaled  = model.price(STRIKES * lam,  SPOT * lam, texp, cp=cp)   # C(lam*S, lam*K)
        # Assert C/S is invariant: (scaled / (lam*S)) == (base / S), i.e.
        # scaled == lam * base.  Form the ratio diff for scale-free tolerance.
        err_vec = np.abs(scaled / SPOT / lam - base / SPOT)
        k = int(np.argmax(err_vec))
        if err_vec[k] > worst:
            worst       = float(err_vec[k])
            worst_where = (texp, float(STRIKES[k]), float(base[k]), float(scaled[k]))

    assert worst < TOL, (
        f"BSM scale invariance violated for lambda={lam}, {label}: "
        f"max |C(lam*S, lam*K)/(lam*S) - C(S, K)/S| = {worst:.2e} at "
        f"(T={worst_where[0]}, K={worst_where[1]}); base={worst_where[2]:.10f}, "
        f"scaled={worst_where[3]:.10f}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Bachelier translation invariance:  C(F + lambda, K + lambda) = C(F, K)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("lam", LAMBDAS + [-5.0, -30.0])   # also test negative shifts
@pytest.mark.parametrize("cp, label", [(+1, "call"), (-1, "put")])
def test_translation_invariance_normal(lam, cp, label):
    """
    Bachelier price depends on (F, K) only through the difference K - F:
        C(F + lambda, K + lambda, sigma, T) = C(F, K, sigma, T).

    This is a translation symmetry, not a scale symmetry — Bachelier's
    dimensional pi-groups are (K - F)/(sigma*sqrt(T)) and Cn/(sigma*sqrt(T)),
    both of which are invariant under a common shift of F and K.

    Because NormalCos internally uses  k* = K - F  as its only strike
    input, shifting both spot and strike by the same lambda feeds
    bit-identical arguments into the COS kernel.  So this error should
    be exactly 0, not merely below 1e-10.
    """
    model = NormalCos(sigma=25.0, intr=0.05, divr=0.02)   # absolute vol, price units

    worst = 0.0
    worst_where = None
    for texp in TEXPS:
        # Under r != q, F = S*exp((r-q)T); shifting spot by lambda shifts F by
        # lambda*exp((r-q)T), so we shift spot by lambda/exp((r-q)T) to get a
        # forward shift of exactly lambda.
        carry = np.exp((model.intr - model.divr) * texp)
        spot_shift = lam / carry                          # keeps F shift equal to lam

        base    = model.price(STRIKES,       SPOT,              texp, cp=cp)
        shifted = model.price(STRIKES + lam, SPOT + spot_shift, texp, cp=cp)
        err_vec = np.abs(shifted - base)
        k = int(np.argmax(err_vec))
        if err_vec[k] > worst:
            worst       = float(err_vec[k])
            worst_where = (texp, float(STRIKES[k]), float(base[k]), float(shifted[k]))

    assert worst < TOL, (
        f"Bachelier translation invariance violated for lambda={lam}, {label}: "
        f"max |C(F+lam, K+lam) - C(F, K)| = {worst:.2e} at "
        f"(T={worst_where[0]}, K={worst_where[1]}); base={worst_where[2]:.10f}, "
        f"shifted={worst_where[3]:.10f}"
    )
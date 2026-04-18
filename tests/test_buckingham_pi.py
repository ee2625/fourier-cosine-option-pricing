"""
Track D — Parametrised Buckingham pi test across all model classes.

Each model class in this package has a characteristic dimensional symmetry:

    BsmModel, HestonCOSPricer   →  scale invariance       C(lam*S, lam*K) = lam * C(S, K)
    NormalCos (Bachelier)       →  translation invariance C(F + lam, K + lam) = C(F, K)

This file verifies the appropriate symmetry for every model with one
parametrised test.  Adding a new model class to PRICERS below
automatically extends coverage — no new test code needed.

Works without analytic reference values, so it's a stronger cross-check
than comparison against closed-form prices.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from cos_pricing import BsmModel, NormalCos, HestonCOSPricer


# ─────────────────────────────────────────────────────────────────────────────
# Uniform pricer adapters.
# All three use r = q = 0 so F = S, which makes "shift spot by lambda" equal
# to "shift forward by lambda" — the condition under which Bachelier
# translation invariance is a pure translation in spot coordinates.
# (test_dimensional_invariance.py exercises the non-zero-rate case separately
# using a carry-corrected spot shift.)
# ─────────────────────────────────────────────────────────────────────────────

def _bsm_price(spot, strike, texp, cp):
    return BsmModel(sigma=0.25).price(strike, spot, texp, cp=cp)

def _heston_price(spot, strike, texp, cp):
    m = HestonCOSPricer(
        S0=spot, v0=0.04, lam=1.5, ubar=0.04, eta=0.5, rho=-0.5,
    )
    return m.price_call(strike, tau=texp) if cp > 0 \
           else m.price_put(strike, tau=texp)

def _normal_price(spot, strike, texp, cp):
    return NormalCos(sigma=25.0).price(strike, spot, texp, cp=cp)


# (name, pricer callable, symmetry type)
PRICERS = [
    ("BsmModel",        _bsm_price,    "scale"),
    ("HestonCOSPricer", _heston_price, "scale"),
    ("NormalCos",       _normal_price, "translation"),
]

# λ values from team_tasks.docx (Track A spec extended with negatives for
# translation — Bachelier's symmetry is bidirectional).
LAMBDAS = [0.1, 0.5, 2.0, 10.0, 100.0]

SPOT    = 100.0
STRIKES = np.array([85.0, 100.0, 115.0])
TEXPS   = [0.5, 1.0, 2.0]
TOL     = 1e-10


@pytest.mark.parametrize("name, pricer, symmetry", PRICERS,
                         ids=[p[0] for p in PRICERS])
@pytest.mark.parametrize("lam", LAMBDAS)
def test_buckingham_pi(name, pricer, symmetry, lam):
    """
    Verify the dimensional symmetry appropriate to each model.

    scale:
        C(lam*S, lam*K) / (lam*S) == C(S, K) / S
    translation:
        C(S + lam, K + lam) == C(S, K)

    Both cp values and the full (strike, texp) sweep run inside one test.
    """
    worst = 0.0
    worst_where = None

    for cp in (+1, -1):
        for texp in TEXPS:
            if symmetry == "scale":
                base   = np.array([pricer(SPOT,       K,       texp, cp) for K in STRIKES])
                other  = np.array([pricer(SPOT * lam, K * lam, texp, cp) for K in STRIKES])
                err    = np.abs(other / (SPOT * lam) - base / SPOT)

            elif symmetry == "translation":
                base    = np.array([pricer(SPOT,       K,       texp, cp) for K in STRIKES])
                other   = np.array([pricer(SPOT + lam, K + lam, texp, cp) for K in STRIKES])
                err     = np.abs(other - base)

            else:
                pytest.fail(f"Unknown symmetry type: {symmetry}")

            k = int(np.argmax(err))
            if float(err[k]) > worst:
                worst       = float(err[k])
                worst_where = (cp, texp, float(STRIKES[k]),
                               float(base[k]), float(other[k]))

    assert worst < TOL, (
        f"{name} {symmetry}-invariance violated for lambda={lam}: "
        f"max err = {worst:.2e} at cp={worst_where[0]}, T={worst_where[1]}, "
        f"K={worst_where[2]}; base={worst_where[3]:.10f}, "
        f"other={worst_where[4]:.10f}"
    )
    
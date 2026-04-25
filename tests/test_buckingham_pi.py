"""
Parametrised Buckingham pi test — verifies spatial scale invariance for
every exponential-Levy model in this package.

For each model class:
    C(lambda*S, lambda*K) / (lambda*S)  ==  C(S, K) / S   (== C/S is a pi-group)

This holds because all four models live in log-moneyness coordinates
x = log(S/K), so a common rescaling of spot and strike leaves x invariant.
The test does not need analytic reference values, so it is a stronger
cross-check than comparison against a closed form.

Adding a new exponential-Levy model class to PRICERS below automatically
extends coverage — no new test code needed.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from cos_pricing import BsmModel, HestonCOSPricer, VgModel, CgmyModel


# ─────────────────────────────────────────────────────────────────────────────
# Uniform pricer adapters. r = q = 0 keeps the forward equal to the spot, so
# scaling spot by lambda directly scales the forward by lambda.
# (test_dimensional_invariance.py exercises non-zero rates separately.)
# ─────────────────────────────────────────────────────────────────────────────

def _bsm_price(spot, strike, texp, cp):
    return BsmModel(sigma=0.25).price(strike, spot, texp, cp=cp)

def _heston_price(spot, strike, texp, cp):
    m = HestonCOSPricer(
        S0=spot, v0=0.04, lam=1.5, ubar=0.04, eta=0.5, rho=-0.5,
    )
    return m.price_call(strike, tau=texp) if cp > 0 \
           else m.price_put(strike, tau=texp)

def _vg_price(spot, strike, texp, cp):
    return VgModel(sigma=0.12, theta=-0.14, nu=0.2).price(
        strike, spot, texp, cp=cp, n_cos=256,
    )

def _cgmy_price(spot, strike, texp, cp):
    return CgmyModel(C=1.0, G=5.0, M=5.0, Y=0.5).price(
        strike, spot, texp, cp=cp, n_cos=256,
    )


PRICERS = [
    ("BsmModel",        _bsm_price),
    ("HestonCOSPricer", _heston_price),
    ("VgModel",         _vg_price),
    ("CgmyModel",       _cgmy_price),
]

LAMBDAS = [0.1, 0.5, 2.0, 10.0, 100.0]

SPOT    = 100.0
STRIKES = np.array([85.0, 100.0, 115.0])
TEXPS   = [0.5, 1.0, 2.0]
TOL     = 1e-10


@pytest.mark.parametrize("name, pricer", PRICERS,
                         ids=[p[0] for p in PRICERS])
@pytest.mark.parametrize("lam", LAMBDAS)
def test_buckingham_pi_scale(name, pricer, lam):
    """
    Spatial scale invariance:  C(lam*S, lam*K) / (lam*S) == C(S, K) / S.

    Both cp values and the full (strike, texp) sweep run inside one test.
    """
    worst = 0.0
    worst_where = None

    for cp in (+1, -1):
        for texp in TEXPS:
            base  = np.array([pricer(SPOT,       K,       texp, cp) for K in STRIKES])
            other = np.array([pricer(SPOT * lam, K * lam, texp, cp) for K in STRIKES])
            err   = np.abs(other / (SPOT * lam) - base / SPOT)
            k = int(np.argmax(err))
            if float(err[k]) > worst:
                worst       = float(err[k])
                worst_where = (cp, texp, float(STRIKES[k]),
                               float(base[k]), float(other[k]))

    assert worst < TOL, (
        f"{name} scale invariance violated for lambda={lam}: "
        f"max err = {worst:.2e} at cp={worst_where[0]}, T={worst_where[1]}, "
        f"K={worst_where[2]}; base={worst_where[3]:.10f}, "
        f"other={worst_where[4]:.10f}"
    )

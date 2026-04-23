"""
Heston temporal scale invariance — a Buckingham-pi symmetry beyond BSM.

The Heston model has 10 dimensional inputs (S0, K, r, q, T, v0, ubar,
kappa, eta, rho) and 2 fundamental units (price [$], time [T]; rho is
already dimensionless).  By Buckingham's pi theorem there are
10 - 2 = 8 independent dimensionless groups.

BSM "scale invariance" exploits one of them (K/S0 vs C/S0) — the spatial
scaling tested in test_dimensional_invariance.py.  Heston has seven more.
This file exercises the strongest of the additional symmetries — the
*temporal* one — which rotates seven parameters at once:

    (T, r, q, kappa, eta, v0, ubar) → (mu*T, r/mu, q/mu, kappa/mu,
                                       eta/mu, v0/mu, ubar/mu)
    rho, S0, K   unchanged

Proof sketch.  Substitute t = mu*tau' in the Heston SDE and rescale the
Brownian motions by sqrt(mu).  The variance and spot SDEs come out
identical in tau'-time with the *original* parameters, so the law of
log(S_T/S0) at tau' = T equals the law under the rescaled measure at
t = mu*T.  Discount: exp(-r*T) = exp(-(r/mu)(mu*T)).  Hence prices match.

Implementation note.  HestonCOSPricer's truncation half-width is
L * sqrt(ubar + v0*eta), and the default L heuristic depends on tau.
Neither piece is itself invariant under temporal rescaling, even though
their *product* would only need to be "wide enough" for both to give
the true price to ~ machine precision.  We pin both instances to the
same half-width in log-moneyness explicitly, which forces a, b,
the frequency grid, the CF samples, and the U coefficients to be
bit-identical — so the test asserts ~ machine epsilon, not just < 1e-10.
"""
import sys
import os

# Support `pytest tests/` from repo root even without PYTHONPATH=src set.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from cos_pricing import HestonCOSPricer


# Pytest parameter: temporal scale factor.  Named `mu` (not `lam`) to avoid
# shadowing the Heston mean-reversion parameter, which is conventionally
# called `lam` in this codebase (and `kappa` in the literature).
MUS = [0.1, 0.5, 2.0, 10.0, 100.0]

SPOT       = 100.0
STRIKES    = np.array([70.0, 85.0, 100.0, 115.0, 130.0])
TAU_BASE   = 1.0
N_COS      = 256

# Half-width of the truncation interval in log-moneyness units.  Picked
# generously: at the largest mu the rescaled rates are tiny, but the
# pricer's heuristic sigma_h = sqrt(ubar + v0*eta) shrinks even faster, so
# fixing the half-width (rather than L) is what guarantees identical
# truncation intervals across the rescaling.
HALF_WIDTH = 12.0

# Baseline Heston (SPX-flavoured numbers).
BASE = dict(
    S0=SPOT,
    v0=0.04, lam=1.5, eta=0.4, ubar=0.04, rho=-0.7,
    r=0.03, q=0.01,
)

# Same tolerance as test_dimensional_invariance.py.
TOL = 1e-10


def _rescale(params, mu):
    """Apply the temporal pi-symmetry: divide every rate-dimensioned input by mu."""
    out = dict(params)
    for k in ("v0", "lam", "eta", "ubar", "r", "q"):
        out[k] = params[k] / mu
    return out


def _heston_sigma_h(p):
    """Replicate HestonCOSPricer._sigma_h without poking at internals."""
    return float(np.sqrt(p["ubar"] + p["v0"] * p["eta"]))


def _price_pinned(model, params, K, tau, cp):
    """Price with L chosen so the truncation half-width = HALF_WIDTH (log-units)."""
    L = HALF_WIDTH / _heston_sigma_h(params)
    return model.price(K, tau, cp=cp, N=N_COS, L=L)


# ─────────────────────────────────────────────────────────────────────────────
# Heston temporal scale invariance:
#     C(T, r, q, kappa, eta, v0, ubar)
#         == C(mu*T, r/mu, q/mu, kappa/mu, eta/mu, v0/mu, ubar/mu)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("mu", MUS)
@pytest.mark.parametrize("cp, label", [(+1, "call"), (-1, "put")])
def test_heston_temporal_invariance(mu, cp, label):
    """
    Verify that rescaling (T, all rate-dimensioned inputs) by (mu, 1/mu)
    leaves the dimensionless Heston price unchanged.

    With the truncation half-width pinned to HALF_WIDTH on both instances,
    a, b, the frequency grid u_k = k*pi/(b-a), the characteristic-function
    samples phi(u_k, tau), and the payoff coefficients U_k all coincide
    bit-for-bit between the base and rescaled pricers.  The discount
    exp(-r*T) is invariant by construction.  Result: bit-identical prices
    up to floating-point reordering.

    Sweep covers ITM/ATM/OTM strikes for calls and puts at every mu.
    """
    base_params = BASE
    resc_params = _rescale(BASE, mu)

    base = HestonCOSPricer(**base_params)
    resc = HestonCOSPricer(**resc_params)

    p_base = _price_pinned(base, base_params, STRIKES, TAU_BASE,        cp)
    p_resc = _price_pinned(resc, resc_params, STRIKES, mu * TAU_BASE,   cp)

    err_vec = np.abs(p_resc - p_base) / SPOT       # error on dimensionless C/S0
    k       = int(np.argmax(err_vec))
    worst   = float(err_vec[k])

    assert worst < TOL, (
        f"Heston temporal pi-invariance violated for mu={mu}, {label}: "
        f"max |C_resc - C_base|/S0 = {worst:.2e} at K={STRIKES[k]:.0f}; "
        f"base={p_base[k]:.10f}, resc={p_resc[k]:.10f}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Heston spatial × temporal:  combine with the BSM-style spatial scaling
# (S0, K) → (alpha*S0, alpha*K).  Tests both pi-symmetries together.
# ─────────────────────────────────────────────────────────────────────────────

ALPHAS = [0.5, 2.0, 100.0]

@pytest.mark.parametrize("mu", [0.5, 2.0, 10.0])
@pytest.mark.parametrize("alpha", ALPHAS)
@pytest.mark.parametrize("cp, label", [(+1, "call"), (-1, "put")])
def test_heston_spatial_temporal_invariance(alpha, mu, cp, label):
    """
    Joint invariance — apply spatial scaling (alpha) AND temporal rescaling
    (mu) at the same time.  Both are pi-symmetries of the dimensionless
    price, so:

        C(alpha*S0, alpha*K, mu*T, r/mu, ..., v0/mu, ubar/mu) / (alpha*S0)
            == C(S0, K, T, r, ..., v0, ubar) / S0.

    Stronger than either symmetry alone: it asserts that two of the eight
    Heston pi-groups are simultaneously invariant under the COS pricer.
    """
    base_params = BASE
    resc_params = _rescale(BASE, mu)
    resc_params["S0"] = alpha * SPOT      # apply spatial scaling on top

    base = HestonCOSPricer(**base_params)
    resc = HestonCOSPricer(**resc_params)

    p_base = _price_pinned(base, base_params, STRIKES,         TAU_BASE,      cp)
    p_resc = _price_pinned(resc, resc_params, alpha * STRIKES, mu * TAU_BASE, cp)

    err_vec = np.abs(p_resc / (alpha * SPOT) - p_base / SPOT)
    k       = int(np.argmax(err_vec))
    worst   = float(err_vec[k])

    assert worst < TOL, (
        f"Heston spatial+temporal invariance violated for alpha={alpha}, "
        f"mu={mu}, {label}: max |C_resc/(alpha*S0) - C_base/S0| = "
        f"{worst:.2e} at K={STRIKES[k]:.0f}"
    )
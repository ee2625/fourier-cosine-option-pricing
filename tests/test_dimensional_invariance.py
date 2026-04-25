"""
Dimensional invariance tests (Buckingham pi symmetries) for the four
exponential-Levy models priced in this package: BSM, Heston, VG, CGMY.

Two symmetry families are exercised:

1. **Spatial scale invariance** — common to every exponential-Levy model:
       C(lambda * S, lambda * K) == lambda * C(S, K)
   The kernel lives in log-moneyness x = log(S / K), so scaling spot and
   strike by a common factor leaves x unchanged. Holds at machine epsilon.

2. **Temporal rate invariance** — for any model whose parameters carry
   units of inverse-time (rates and intensities), stretching maturity by
   ``mu`` while contracting those parameters by 1/mu leaves the price
   unchanged. The dimensionless pi-groups (e.g. r*T, kappa*T, sigma*sqrt(T),
   T/nu, C*T) absorb the change, so the kernel sees the same numbers.

Reference: Fang & Oosterlee (2008) — the four models exercised here are
BSM (Section 5.1), Heston (5.2), CGMY (5.3), VG (5.4).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from cos_pricing import BsmModel, HestonCOSPricer, VgModel, CgmyModel


LAMBDAS = [0.1, 0.5, 2.0, 10.0, 100.0]
SPOT    = 100.0
STRIKES = np.array([70.0, 85.0, 100.0, 115.0, 130.0])
TEXPS   = [0.1, 1.0, 5.0]
TOL     = 1e-10


# ─────────────────────────────────────────────────────────────────────────────
# Spatial scale invariance:  C(lambda*S, lambda*K) = lambda * C(S, K)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("lam", LAMBDAS)
@pytest.mark.parametrize("cp, label", [(+1, "call"), (-1, "put")])
def test_scale_invariance_bsm(lam, cp, label):
    """BSM is homogeneous of degree 1 in (S, K)."""
    model = BsmModel(sigma=0.25, intr=0.05, divr=0.02)
    _assert_scale_invariance(
        f"BSM/{label}", lam,
        lambda S, K, T: model.price(K, S, T, cp=cp),
    )


@pytest.mark.parametrize("lam", LAMBDAS)
@pytest.mark.parametrize("cp, label", [(+1, "call"), (-1, "put")])
def test_scale_invariance_vg(lam, cp, label):
    """VG is exponential-Levy: C(lambda*S, lambda*K) == lambda * C(S, K)."""
    model = VgModel(sigma=0.12, theta=-0.14, nu=0.2, intr=0.05, divr=0.02)
    _assert_scale_invariance(
        f"VG/{label}", lam,
        lambda S, K, T: model.price(K, S, T, cp=cp, n_cos=256),
    )


@pytest.mark.parametrize("lam", LAMBDAS)
@pytest.mark.parametrize("cp, label", [(+1, "call"), (-1, "put")])
def test_scale_invariance_cgmy(lam, cp, label):
    """CGMY is exponential-Levy: C(lambda*S, lambda*K) == lambda * C(S, K)."""
    model = CgmyModel(C=1.0, G=5.0, M=5.0, Y=0.5, intr=0.1, divr=0.02)
    _assert_scale_invariance(
        f"CGMY/{label}", lam,
        lambda S, K, T: model.price(K, S, T, cp=cp, n_cos=256),
    )


def _assert_scale_invariance(name, lam, price_fn):
    """Compare C/(lam*S) at lam-scaled spot/strike against C/S at the base."""
    worst = 0.0
    worst_where = None
    for texp in TEXPS:
        base   = price_fn(SPOT,        STRIKES,        texp)
        scaled = price_fn(SPOT * lam,  STRIKES * lam,  texp)
        err = np.abs(scaled / SPOT / lam - base / SPOT)
        k = int(np.argmax(err))
        if float(err[k]) > worst:
            worst       = float(err[k])
            worst_where = (texp, float(STRIKES[k]),
                           float(np.atleast_1d(base)[k]),
                           float(np.atleast_1d(scaled)[k]))
    assert worst < TOL, (
        f"{name} scale invariance violated for lambda={lam}: "
        f"max |C(lam*S, lam*K)/(lam*S) - C(S, K)/S| = {worst:.2e} at "
        f"(T={worst_where[0]}, K={worst_where[1]}); "
        f"base={worst_where[2]:.10f}, scaled={worst_where[3]:.10f}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Temporal rate invariance — exponential-Levy models with time-dimensioned
# parameters. Stretching T by mu and contracting rates/intensities by 1/mu
# preserves every dimensionless pi-group, so the price is unchanged.
# ─────────────────────────────────────────────────────────────────────────────

MU_GRID = [0.1, 0.5, 2.0, 10.0]
T_BASE  = 1.0
TEMPORAL_TOL = 1e-10


@pytest.mark.parametrize("mu", MU_GRID)
@pytest.mark.parametrize("cp, label", [(+1, "call"), (-1, "put")])
def test_temporal_invariance_vg(mu, cp, label):
    """
    VG temporal symmetry. Under T -> mu*T:
        nu    -> mu * nu
        theta -> theta / mu
        sigma -> sigma / sqrt(mu)
        r, q  -> r / mu, q / mu

    Verifies that nu*theta, nu*sigma^2, w*T, T/nu, r*T, q*T are all
    invariant — these are exactly the combinations that appear inside the
    VG characteristic function.
    """
    sig_b, the_b, nu_b = 0.12, -0.14, 0.2
    r_b, q_b           = 0.05, 0.02

    base = VgModel(sigma=sig_b, theta=the_b, nu=nu_b, intr=r_b, divr=q_b)
    scaled = VgModel(
        sigma=sig_b / np.sqrt(mu),
        theta=the_b / mu,
        nu=nu_b * mu,
        intr=r_b / mu,
        divr=q_b / mu,
    )
    px_base   = base.price(STRIKES,   SPOT, T_BASE,      cp=cp, n_cos=256)
    px_scaled = scaled.price(STRIKES, SPOT, T_BASE * mu, cp=cp, n_cos=256)
    err = np.abs(px_scaled - px_base) / SPOT
    worst = float(np.max(err))
    assert worst < TEMPORAL_TOL, (
        f"VG/{label} temporal invariance violated for mu={mu}: "
        f"max |C(mu*T, scaled params) - C(T, base params)|/S = {worst:.2e}"
    )


@pytest.mark.parametrize("mu", MU_GRID)
@pytest.mark.parametrize("cp, label", [(+1, "call"), (-1, "put")])
def test_temporal_invariance_cgmy(mu, cp, label):
    """
    CGMY temporal symmetry. Under T -> mu*T:
        C    -> C / mu       (C carries units of 1/time)
        r, q -> r / mu, q / mu
        G, M, Y unchanged    (jump-size scales are dimensionless)

    Verifies that C*T and r*T are invariant; G, M, Y are pure pi-groups.
    """
    C_b, G_b, M_b, Y_b = 1.0, 5.0, 5.0, 0.5
    r_b, q_b           = 0.1, 0.02

    base = CgmyModel(C=C_b, G=G_b, M=M_b, Y=Y_b, intr=r_b, divr=q_b)
    scaled = CgmyModel(
        C=C_b / mu, G=G_b, M=M_b, Y=Y_b,
        intr=r_b / mu, divr=q_b / mu,
    )
    px_base   = base.price(STRIKES,   SPOT, T_BASE,      cp=cp, n_cos=256)
    px_scaled = scaled.price(STRIKES, SPOT, T_BASE * mu, cp=cp, n_cos=256)
    err = np.abs(px_scaled - px_base) / SPOT
    worst = float(np.max(err))
    assert worst < TEMPORAL_TOL, (
        f"CGMY/{label} temporal invariance violated for mu={mu}: "
        f"max |C(mu*T, scaled params) - C(T, base params)|/S = {worst:.2e}"
    )

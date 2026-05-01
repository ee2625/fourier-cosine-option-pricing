"""
Fixed-seed randomized robustness checks for the production pricing APIs.

These tests complement the paper-table reproductions: instead of validating
one published parameter set, they sample ordinary market/model parameters and
verify two structural requirements that should always hold:

* prices are finite for vectorized strike inputs;
* put-call parity holds to numerical tolerance.
"""

import numpy as np
import pytest

from cos_pricing import BsmModel, CgmyModel, HestonCOSPricer, VgModel


RNG_SEED = 20240501


def _market_cases(n_cases=10):
    rng = np.random.default_rng(RNG_SEED)
    cases = []
    for _ in range(n_cases):
        spot = rng.uniform(50.0, 180.0)
        cases.append({
            "spot": spot,
            "texp": rng.uniform(0.1, 3.0),
            "intr": rng.uniform(-0.01, 0.08),
            "divr": rng.uniform(0.0, 0.05),
            "strikes": spot * np.sort(rng.uniform(0.65, 1.45, size=6)),
            "bsm_sigma": rng.uniform(0.08, 0.60),
            "heston_v0": rng.uniform(0.005, 0.12),
            "heston_lam": rng.uniform(0.5, 3.0),
            "heston_eta": rng.uniform(0.1, 0.8),
            "heston_ubar": rng.uniform(0.005, 0.12),
            "heston_rho": rng.uniform(-0.85, 0.25),
            "vg_sigma": rng.uniform(0.06, 0.35),
            "vg_theta": rng.uniform(-0.25, 0.20),
            "vg_nu": rng.uniform(0.08, 0.55),
            "cgmy_C": rng.uniform(0.1, 1.2),
            "cgmy_G": rng.uniform(3.0, 9.0),
            "cgmy_M": rng.uniform(3.0, 9.0),
            # Avoid integer singularities of Gamma(-Y), especially Y=1.
            "cgmy_Y": rng.choice([
                rng.uniform(0.25, 0.85),
                rng.uniform(1.15, 1.65),
            ]),
        })
    return cases


RANDOM_CASES = _market_cases()


def _assert_finite_and_put_call_parity(model, market, price_fn, rel_tol):
    spot = market["spot"]
    texp = market["texp"]
    strikes = market["strikes"]

    calls = np.asarray(price_fn(strikes, spot, texp, +1), dtype=float)
    puts = np.asarray(price_fn(strikes, spot, texp, -1), dtype=float)

    assert np.all(np.isfinite(calls))
    assert np.all(np.isfinite(puts))

    fwd, df = model._fwd_df(spot, texp)
    parity = df * (fwd - strikes)
    residual = np.max(np.abs((calls - puts) - parity))
    assert residual / max(1.0, spot) < rel_tol


@pytest.mark.parametrize("market", RANDOM_CASES)
def test_randomized_bsm_finite_and_put_call_parity(market):
    model = BsmModel(
        sigma=market["bsm_sigma"],
        intr=market["intr"],
        divr=market["divr"],
    )

    _assert_finite_and_put_call_parity(
        model,
        market,
        lambda strike, spot, texp, cp: model.price(
            strike, spot, texp, cp=cp, n_cos=192,
        ),
        rel_tol=1e-11,
    )


@pytest.mark.parametrize("market", RANDOM_CASES)
def test_randomized_heston_finite_and_put_call_parity(market):
    model = HestonCOSPricer(
        S0=market["spot"],
        v0=market["heston_v0"],
        lam=market["heston_lam"],
        eta=market["heston_eta"],
        ubar=market["heston_ubar"],
        rho=market["heston_rho"],
        r=market["intr"],
        q=market["divr"],
    )

    calls = np.asarray(
        model.price(market["strikes"], market["texp"], cp=+1, N=512, L=24.0),
        dtype=float,
    )
    puts = np.asarray(
        model.price(market["strikes"], market["texp"], cp=-1, N=512, L=24.0),
        dtype=float,
    )

    assert np.all(np.isfinite(calls))
    assert np.all(np.isfinite(puts))

    df = np.exp(-model.r * market["texp"])
    fwd = model.S0 * np.exp((model.r - model.q) * market["texp"])
    parity = df * (fwd - market["strikes"])
    residual = np.max(np.abs((calls - puts) - parity))
    assert residual / max(1.0, model.S0) < 1e-6


@pytest.mark.parametrize("market", RANDOM_CASES)
def test_randomized_vg_finite_and_put_call_parity(market):
    model = VgModel(
        sigma=market["vg_sigma"],
        theta=market["vg_theta"],
        nu=market["vg_nu"],
        intr=market["intr"],
        divr=market["divr"],
    )

    _assert_finite_and_put_call_parity(
        model,
        market,
        lambda strike, spot, texp, cp: model.price(
            strike, spot, texp, cp=cp, n_cos=1536,
        ),
        rel_tol=5e-8,
    )


@pytest.mark.parametrize("market", RANDOM_CASES)
def test_randomized_cgmy_finite_and_put_call_parity(market):
    model = CgmyModel(
        C=market["cgmy_C"],
        G=market["cgmy_G"],
        M=market["cgmy_M"],
        Y=market["cgmy_Y"],
        intr=market["intr"],
        divr=market["divr"],
    )

    _assert_finite_and_put_call_parity(
        model,
        market,
        lambda strike, spot, texp, cp: model.price(
            strike, spot, texp, cp=cp, n_cos=512, L=12.0,
        ),
        rel_tol=2e-6,
    )

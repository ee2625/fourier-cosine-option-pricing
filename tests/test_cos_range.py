import numpy as np

from cos_pricing import BsmModel, CgmyModel, VgModel, bsm_price, cos_price_smile
from cos_pricing.cos_range import (
    central_moment_from_cumulants,
    jp_markov_half_width,
)


def test_central_moment_from_cumulants_normal_mu8():
    variance = 0.04
    cumulants = np.zeros(9)
    cumulants[2] = variance

    got = central_moment_from_cumulants(cumulants, 8)

    assert abs(got - 105.0 * variance**4) < 1e-18


def test_jp_markov_half_width_increases_with_tighter_tolerance():
    variance = 0.04
    moment8 = 105.0 * variance**4

    loose = jp_markov_half_width(moment8, variance, eps_tol=1e-4)
    tight = jp_markov_half_width(moment8, variance, eps_tol=1e-8)

    assert tight > loose > 0.0


def test_bsm_jp_range_prices_smile():
    model = BsmModel(sigma=0.2, intr=0.05, divr=0.01)
    spot, texp = 100.0, 1.0
    strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    trunc = model.jp_trunc_range(texp, eps_tol=1e-8, moment_order=8)

    center = 0.5 * (trunc[0] + trunc[1])
    assert abs(center + 0.5 * model.sigma**2 * texp) < 1e-15

    fwd, df = model._fwd_df(spot, texp)
    got = cos_price_smile(
        model.char_func(texp),
        texp,
        strikes,
        fwd,
        df,
        cp=1,
        n_cos=512,
        trunc_range=trunc,
    )
    ref = bsm_price(strikes, spot, model.sigma, texp, model.intr, model.divr)

    assert np.max(np.abs(got - ref)) < 1e-8


def test_vg_and_cgmy_jp_ranges_are_finite_and_centered():
    vg = VgModel(sigma=0.12, theta=-0.14, nu=0.2, intr=0.1)
    vg_range = vg.jp_trunc_range(1.0, eps_tol=1e-6, moment_order=8)
    vg_fo = vg.trunc_range(1.0)
    assert np.all(np.isfinite(vg_range))
    assert vg_range[0] < vg_range[1]
    assert abs(0.5 * sum(vg_range) - 0.5 * sum(vg_fo)) < 1e-12

    cgmy = CgmyModel(C=1.0, G=5.0, M=10.0, Y=0.5)
    cgmy_range = cgmy.jp_trunc_range(1.0, eps_tol=1e-6, moment_order=8)
    assert np.all(np.isfinite(cgmy_range))
    assert cgmy_range[0] < cgmy_range[1]

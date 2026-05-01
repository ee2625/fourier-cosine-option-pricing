import numpy as np
import pytest

from cos_pricing import (
    CgmyModel,
    VgModel,
    joshi_yang_contour_bsm_vol,
    joshi_yang_real_axis_bsm_vol,
    variance_equivalent_bsm_vol,
)
from cos_pricing.cos_range import cgmy_cumulants


def test_variance_equivalent_bsm_vol():
    assert abs(variance_equivalent_bsm_vol(0.04, 1.0) - 0.2) < 1e-15
    assert abs(variance_equivalent_bsm_vol(0.02, 0.5) - 0.2) < 1e-15

    with pytest.raises(ValueError):
        variance_equivalent_bsm_vol(0.0, 1.0)
    with pytest.raises(ValueError):
        variance_equivalent_bsm_vol(0.04, 0.0)


def test_joshi_yang_vol_selectors_recover_bsm_vol():
    sigma = 0.23
    texp = 1.7
    mgf = lambda u: np.exp(-0.5 * sigma**2 * texp * np.asarray(u) * (1.0 - np.asarray(u)))

    assert abs(joshi_yang_real_axis_bsm_vol(mgf, texp) - sigma) < 1e-9
    assert abs(joshi_yang_contour_bsm_vol(mgf, texp, eta=0.5) - sigma) < 1e-12

    with pytest.raises(ValueError):
        joshi_yang_contour_bsm_vol(mgf, texp, eta=0.0)


def test_vg_control_variate_identity_shape_and_coarse_reduction():
    model = VgModel(sigma=0.12, theta=-0.14, nu=0.2, intr=0.1, divr=0.0)
    strikes = np.array([85.0, 90.0, 95.0])
    n_cos = 16

    plain = model.price(strikes, 100.0, 1.0, cp=1, n_cos=n_cos)
    adj = model.bsm_control_variate_adjustment(strikes, 100.0, 1.0, cp=1, n_cos=n_cos)
    cv = model.price_cv(strikes, 100.0, 1.0, cp=1, n_cos=n_cos)

    assert cv.shape == strikes.shape
    assert np.max(np.abs(cv - (plain + adj))) < 1e-12
    assert abs(model.equivalent_bsm_vol(1.0) - np.sqrt(model.sigma**2 + model.nu * model.theta**2)) < 1e-15

    ref = 19.099354724
    plain_err = abs(model.price(90.0, 100.0, 1.0, cp=1, n_cos=n_cos) - ref)
    cv_err = abs(model.price_cv(90.0, 100.0, 1.0, cp=1, n_cos=n_cos) - ref)
    assert cv_err < plain_err


def test_vg_joshi_yang_control_variate_methods_are_opt_in():
    model = VgModel(sigma=0.12, theta=-0.14, nu=0.2, intr=0.1, divr=0.0)
    n_cos = 16

    plain = model.price(90.0, 100.0, 1.0, cp=1, n_cos=n_cos)
    for method in ("joshi", "joshi-half"):
        vol = model.equivalent_bsm_vol(1.0, method=method)
        adj = model.bsm_control_variate_adjustment(
            90.0, 100.0, 1.0, cp=1, n_cos=n_cos, vol_method=method
        )
        cv = model.price_cv(90.0, 100.0, 1.0, cp=1, n_cos=n_cos, vol_method=method)
        assert vol > 0.0
        assert abs(cv - (plain + adj)) < 1e-12

    with pytest.raises(ValueError):
        model.equivalent_bsm_vol(1.0, method="unknown")


def test_cgmy_control_variate_identity_shape_and_coarse_reduction():
    model = CgmyModel(C=1.0, G=5.0, M=5.0, Y=0.5, intr=0.1, divr=0.0)
    strikes = np.array([90.0, 100.0, 110.0])
    n_cos = 16

    plain = model.price(strikes, 100.0, 1.0, cp=1, n_cos=n_cos)
    adj = model.bsm_control_variate_adjustment(strikes, 100.0, 1.0, cp=1, n_cos=n_cos)
    cv = model.price_cv(strikes, 100.0, 1.0, cp=1, n_cos=n_cos)

    assert cv.shape == strikes.shape
    assert np.max(np.abs(cv - (plain + adj))) < 1e-12

    c2 = cgmy_cumulants(model.C, model.G, model.M, model.Y, 1.0, order=2)[2]
    assert abs(model.equivalent_bsm_vol(1.0) - np.sqrt(c2)) < 1e-15

    ref = 19.8129488424
    plain_err = abs(model.price(100.0, 100.0, 1.0, cp=1, n_cos=n_cos) - ref)
    cv_err = abs(model.price_cv(100.0, 100.0, 1.0, cp=1, n_cos=n_cos) - ref)
    assert cv_err < plain_err


def test_cgmy_joshi_yang_control_variate_methods_are_opt_in():
    model = CgmyModel(C=1.0, G=5.0, M=5.0, Y=0.5, intr=0.1, divr=0.0)
    n_cos = 16

    plain = model.price(100.0, 100.0, 1.0, cp=1, n_cos=n_cos)
    for method in ("joshi", "joshi-half"):
        vol = model.equivalent_bsm_vol(1.0, method=method)
        adj = model.bsm_control_variate_adjustment(
            100.0, 100.0, 1.0, cp=1, n_cos=n_cos, vol_method=method
        )
        cv = model.price_cv(100.0, 100.0, 1.0, cp=1, n_cos=n_cos, vol_method=method)
        assert vol > 0.0
        assert abs(cv - (plain + adj)) < 1e-12

    with pytest.raises(ValueError):
        model.equivalent_bsm_vol(1.0, method="unknown")


def test_cgmy_extreme_y_control_variate_is_opt_in_not_default():
    model = CgmyModel(C=1.0, G=5.0, M=5.0, Y=1.98, intr=0.1, divr=0.0)
    n_cos = 64

    plain = model.price(100.0, 100.0, 1.0, cp=1, n_cos=n_cos)
    cv = model.price_cv(100.0, 100.0, 1.0, cp=1, n_cos=n_cos)

    assert np.isfinite(plain)
    assert np.isfinite(cv)
    assert plain != cv

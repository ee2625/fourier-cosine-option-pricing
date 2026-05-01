"""
Test suite for the COS pricing library.

Run with:
    pytest tests/test_cos_method.py -v
"""

import numpy as np
import pytest
from cos_pricing import (
    BsmModel,
    bsm_price,
    cos_price,
    cos_price_smile,
    make_cos_smile_setup,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def bsm_params():
    return dict(sigma=0.2, intr=0.05, divr=0.1)


# ─────────────────────────────────────────────────────────────────────────────
# BSM tests
# ─────────────────────────────────────────────────────────────────────────────

class TestBsmCos:

    def test_single_call(self, bsm_params):
        """COS call matches analytic BSM to < 1e-10."""
        m = BsmModel(**bsm_params)
        spot, texp, strike = 100.0, 1.0, 100.0
        cos_val = m.price(strike, spot, texp, cp=1)
        ref     = bsm_price(strike, spot, bsm_params['sigma'], texp,
                            bsm_params['intr'], bsm_params['divr'], cp=1)
        assert abs(cos_val - ref) < 1e-10

    def test_single_put(self, bsm_params):
        """COS put matches analytic BSM to < 1e-10."""
        m = BsmModel(**bsm_params)
        spot, texp, strike = 100.0, 1.0, 105.0
        cos_val = m.price(strike, spot, texp, cp=-1)
        ref     = bsm_price(strike, spot, bsm_params['sigma'], texp,
                            bsm_params['intr'], bsm_params['divr'], cp=-1)
        assert abs(cos_val - ref) < 1e-10

    def test_strike_array(self, bsm_params):
        """Vectorised over 5 strikes. Max error < 1e-10."""
        m = BsmModel(**bsm_params)
        strikes = np.arange(80, 121, 10, dtype=float)
        spot, texp = 100.0, 1.2
        cos_vals = m.price(strikes, spot, texp)
        refs     = bsm_price(strikes, spot, bsm_params['sigma'], texp,
                             bsm_params['intr'], bsm_params['divr'])
        assert np.max(np.abs(cos_vals - refs)) < 1e-10

    def test_put_call_parity(self, bsm_params):
        """C - P = df * (F - K) to < 1e-10."""
        m = BsmModel(**bsm_params)
        spot, texp = 100.0, 1.0
        for strike in [90.0, 100.0, 110.0]:
            c   = m.price(strike, spot, texp, cp=1)
            p   = m.price(strike, spot, texp, cp=-1)
            fwd = spot * np.exp((bsm_params['intr'] - bsm_params['divr']) * texp)
            df  = np.exp(-bsm_params['intr'] * texp)
            assert abs((c - p) - df * (fwd - strike)) < 1e-10

    def test_mixed_cp_array(self, bsm_params):
        """cp array broadcasts correctly with strike array."""
        m = BsmModel(**bsm_params)
        strikes = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
        cp      = np.array([-1,   -1,    1,     1,     1])
        spot, texp = 100.0, 1.0
        cos_vals = m.price(strikes, spot, texp, cp=cp)
        refs     = bsm_price(strikes, spot, bsm_params['sigma'], texp,
                             bsm_params['intr'], bsm_params['divr'], cp=cp)
        assert np.max(np.abs(cos_vals - refs)) < 1e-10

    def test_convergence_in_n(self, bsm_params):
        """N=64 should already reach < 1e-10; N=8 should be inaccurate."""
        m_ref = BsmModel(**bsm_params)
        ref   = bsm_price(100, 100, bsm_params['sigma'], 1.0,
                          bsm_params['intr'], bsm_params['divr'])
        m8    = BsmModel(**bsm_params)
        assert abs(m8.price(100, 100, 1.0, n_cos=8)  - ref) > 1e-3   # too few
        assert abs(m8.price(100, 100, 1.0, n_cos=64) - ref) < 1e-10  # converged

    def test_scalar_output(self, bsm_params):
        """Scalar strike -> scalar output."""
        m = BsmModel(**bsm_params)
        p = m.price(100.0, 100.0, 1.0)
        assert isinstance(p, float)

    def test_array_output_shape(self, bsm_params):
        """Array strike -> array with matching shape."""
        m = BsmModel(**bsm_params)
        K = np.linspace(90, 110, 7)
        p = m.price(K, 100.0, 1.0)
        assert p.shape == (7,)

    def test_otm_deep_call(self, bsm_params):
        """Deep OTM call (K=200): price close to 0."""
        m = BsmModel(**bsm_params)
        p = m.price(200.0, 100.0, 1.0, cp=1)
        assert p >= 0.0
        assert p < 0.01  # K=200, S=100, sigma=0.2, T=1

    def test_itm_deep_call(self, bsm_params):
        """Deep ITM call: price ~ fwd - K (discounted)."""
        m = BsmModel(**bsm_params)
        spot, strike, texp = 100.0, 50.0, 1.0
        fwd = spot * np.exp((bsm_params['intr'] - bsm_params['divr']) * texp)
        df  = np.exp(-bsm_params['intr'] * texp)
        p   = m.price(strike, spot, texp, cp=1)
        assert abs(p - df * (fwd - strike)) < 0.01


# ─────────────────────────────────────────────────────────────────────────────
# COS engine tests (model-agnostic)
# ─────────────────────────────────────────────────────────────────────────────

class TestCosEngine:

    def test_custom_char_func(self):
        """cos_price works with any char_func (BSM built manually)."""
        sigma, texp = 0.3, 1.0
        fwd, df     = 100.0, 1.0
        # BSM CF manually
        def cf(u):
            return np.exp(-0.5 * sigma**2 * texp * u * (u + 1j))
        p_cos = cos_price(cf, texp, 100.0, fwd, df, cp=1, n_cos=128,
                          trunc_range=(-5.0, 5.0))
        p_ref = bsm_price(100.0, 100.0, sigma, texp)
        assert abs(p_cos - p_ref) < 1e-6

    def test_explicit_trunc_range(self):
        """Explicit trunc_range is respected."""
        m = BsmModel(sigma=0.2)
        cf = m.char_func(1.0)
        # With a sensible range: should match
        p = cos_price(cf, 1.0, 100.0, 100.0, 1.0, cp=1,
                      n_cos=128, trunc_range=(-3.0, 3.0))
        ref = bsm_price(100.0, 100.0, 0.2, 1.0)
        assert abs(p - ref) < 1e-6

    def test_cos_price_smile_matches_cos_price(self):
        """Strike-independent smile path matches the legacy vector path."""
        m = BsmModel(sigma=0.25, intr=0.03, divr=0.01)
        spot, texp = 100.0, 0.75
        fwd, df = m._fwd_df(spot, texp)
        strikes = np.linspace(70.0, 130.0, 13)
        trunc = m.trunc_range(texp)

        legacy = cos_price(
            m.char_func(texp), texp, strikes, fwd, df,
            cp=1, n_cos=160, trunc_range=trunc,
        )
        smile = cos_price_smile(
            m.char_func(texp), texp, strikes, fwd, df,
            cp=1, n_cos=160, trunc_range=trunc,
        )

        assert np.max(np.abs(smile - legacy)) < 1e-13

    def test_cos_smile_setup_reuses_density_coefficients(self):
        """A setup calls the CF once and can price multiple strike vectors."""
        sigma, texp = 0.2, 1.0
        fwd, df = 100.0, 0.97
        calls = []

        def cf(u):
            calls.append(np.asarray(u).copy())
            return np.exp(-0.5 * sigma**2 * texp * u * (u + 1j))

        setup = make_cos_smile_setup(
            cf, texp, fwd, df, n_cos=128, trunc_range=(-4.0, 4.0),
        )
        p1 = setup.price(np.array([90.0, 100.0, 110.0]), cp=1)
        p2 = setup.price(np.array([95.0, 105.0]), cp=-1)

        assert len(calls) == 1
        assert p1.shape == (3,)
        assert p2.shape == (2,)

    def test_cos_price_smile_mixed_cp_broadcasts(self):
        """Smile setup handles mixed call/put flags."""
        m = BsmModel(sigma=0.18, intr=0.04, divr=0.0)
        spot, texp = 100.0, 1.3
        fwd, df = m._fwd_df(spot, texp)
        strikes = np.array([80.0, 95.0, 100.0, 110.0, 125.0])
        cp = np.array([1, -1, 1, -1, 1])
        trunc = m.trunc_range(texp)

        got = cos_price_smile(
            m.char_func(texp), texp, strikes, fwd, df,
            cp=cp, n_cos=192, trunc_range=trunc,
        )
        ref = bsm_price(strikes, spot, m.sigma, texp, m.intr, m.divr, cp=cp)

        assert got.shape == strikes.shape
        assert np.max(np.abs(got - ref)) < 1e-10

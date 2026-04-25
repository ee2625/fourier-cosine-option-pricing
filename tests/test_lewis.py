"""
Tests for the Lewis (2001) single-integral CF inversion pricer.

Lewis is a no-damping CF-based pricer (fixed contour shift u - i/2,
equivalent to Carr-Madan with alpha = 1/2). The integrand has analytic
decay on this contour, so Gauss-Legendre quadrature converges
geometrically in n_quad.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from cos_pricing import (
    BsmModel, HestonCOSPricer, VgModel,
    bsm_price, cos_price, lewis_price,
)


# ─────────────────────────────────────────────────────────────────────────────
# BSM analytic agreement
# ─────────────────────────────────────────────────────────────────────────────

class TestLewisBsm:
    """Lewis must reproduce analytic BSM prices to ~1e-9 at moderate n_quad."""

    SIGMA, R, Q, T, S = 0.25, 0.1, 0.0, 0.1, 100.0
    STRIKES = np.array([80.0, 100.0, 120.0])

    def setup_method(self):
        self.m   = BsmModel(sigma=self.SIGMA, intr=self.R, divr=self.Q)
        self.fwd = self.S * np.exp((self.R - self.Q) * self.T)
        self.df  = np.exp(-self.R * self.T)
        self.cf  = self.m.char_func(self.T)
        self.ana = bsm_price(self.STRIKES, self.S, self.SIGMA, self.T,
                             intr=self.R, divr=self.Q, cp=1)

    def test_call_high_n(self):
        px = lewis_price(self.cf, self.T, self.STRIKES, self.fwd, self.df,
                         cp=1, n_quad=256)
        assert np.max(np.abs(px - self.ana)) < 1e-8

    def test_put_via_parity(self):
        ana_put = bsm_price(self.STRIKES, self.S, self.SIGMA, self.T,
                            intr=self.R, divr=self.Q, cp=-1)
        px = lewis_price(self.cf, self.T, self.STRIKES, self.fwd, self.df,
                         cp=-1, n_quad=256)
        assert np.max(np.abs(px - ana_put)) < 1e-8

    def test_geometric_convergence(self):
        """Doubling n_quad should drop log10(error) by at least 4 in this regime."""
        e64  = np.max(np.abs(lewis_price(self.cf, self.T, self.STRIKES,
                                          self.fwd, self.df, n_quad=64) - self.ana))
        e128 = np.max(np.abs(lewis_price(self.cf, self.T, self.STRIKES,
                                          self.fwd, self.df, n_quad=128) - self.ana))
        e256 = np.max(np.abs(lewis_price(self.cf, self.T, self.STRIKES,
                                          self.fwd, self.df, n_quad=256) - self.ana))
        assert e128 < e64
        assert e256 < e128
        # Geometric convergence: each doubling gains many orders of magnitude.
        assert np.log10(e64) - np.log10(e128) > 3
        assert np.log10(e128) - np.log10(e256) > 3

    def test_scalar_strike_input(self):
        K       = 100.0
        px      = lewis_price(self.cf, self.T, K, self.fwd, self.df, n_quad=256)
        ana_atm = bsm_price(K, self.S, self.SIGMA, self.T,
                            intr=self.R, divr=self.Q, cp=1)
        assert isinstance(px, float)
        assert abs(px - float(ana_atm)) < 1e-8


# ─────────────────────────────────────────────────────────────────────────────
# Cross-check: Lewis must agree with COS at high N for non-BSM models
# ─────────────────────────────────────────────────────────────────────────────

class TestLewisVsCos:
    """Independent CF-based pricers must agree at high precision."""

    def test_heston_vs_cos(self):
        m   = HestonCOSPricer(S0=100.0, v0=0.04, lam=1.5, ubar=0.04,
                              eta=0.5, rho=-0.7, r=0.03, q=0.0)
        T   = 1.0
        K   = np.array([90.0, 100.0, 110.0])
        cos_px   = m.price_call(K, tau=T, N=256)
        fwd      = m.S0 * np.exp((m.r - m.q) * T)
        df       = np.exp(-m.r * T)
        # HestonCOSPricer.char_func returns CF of log(S_T/S0); lewis_price
        # expects CF of log(S_T/F), so shift by exp(-i u (r-q) T).
        drift = (m.r - m.q) * T
        cf_F  = lambda u: np.exp(-1j * u * drift) * m.char_func(u, T)
        lewis_px = lewis_price(cf_F, T, K, fwd, df, cp=1, n_quad=256)
        np.testing.assert_allclose(lewis_px, cos_px, atol=1e-6)

    def test_vg_vs_cos(self):
        m   = VgModel(sigma=0.12, theta=-0.14, nu=0.2, intr=0.1, divr=0.0)
        T   = 1.0
        K   = np.array([80.0, 100.0, 120.0])
        cos_px   = m.price(K, 100.0, T, cp=1, n_cos=2**12)
        fwd, df  = m._fwd_df(100.0, T)
        lewis_px = lewis_price(m.char_func(T), T, K, fwd, df,
                               cp=1, n_quad=256)
        np.testing.assert_allclose(lewis_px, cos_px, atol=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# Mixed cp array
# ─────────────────────────────────────────────────────────────────────────────

def test_mixed_cp_array():
    sigma, T, S = 0.25, 0.1, 100.0
    m   = BsmModel(sigma=sigma, intr=0.05, divr=0.02)
    fwd = S * np.exp((m.intr - m.divr) * T)
    df  = np.exp(-m.intr * T)
    cf  = m.char_func(T)

    K  = np.array([80.0, 100.0, 120.0])
    cp = np.array([+1, -1, +1])
    px = lewis_price(cf, T, K, fwd, df, cp=cp, n_quad=256)

    ana_call = bsm_price(K[[0, 2]], S, sigma, T, intr=m.intr, divr=m.divr, cp=1)
    ana_put  = bsm_price(K[1],      S, sigma, T, intr=m.intr, divr=m.divr, cp=-1)
    np.testing.assert_allclose([px[0], px[2]], ana_call, atol=1e-8)
    assert abs(px[1] - float(ana_put)) < 1e-8

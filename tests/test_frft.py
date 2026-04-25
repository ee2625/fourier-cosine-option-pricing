"""
Tests for the fractional-FFT (FrFT) Carr-Madan pricer.

FrFT decouples the frequency grid spacing eta from the log-strike grid
spacing lambda (plain Carr-Madan FFT enforces eta*lambda = 2*pi/N), so
both can be chosen freely.  We verify:

* FrFT reproduces plain Carr-Madan exactly when beta = 1/N (the FrFT
  reduces to the standard FFT in that limit).
* FrFT matches analytic BSM at large N.
* FrFT matches COS at high precision under Heston and VG (independent
  CF-based pricers must agree).
* Mixed cp arrays.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from cos_pricing import (
    BsmModel, HestonCOSPricer, VgModel,
    bsm_price, frft_price, carr_madan_price,
)


# ─────────────────────────────────────────────────────────────────────────────
# Reduction to plain Carr-Madan FFT
# ─────────────────────────────────────────────────────────────────────────────

def test_frft_reduces_to_carr_madan():
    """When lambda = 2*pi/(N*eta) the FrFT collapses to the standard FFT,
    so frft_price must match carr_madan_price to floating-point noise."""
    sigma, r, q, T, S = 0.25, 0.1, 0.0, 0.1, 100.0
    strikes = np.array([80.0, 100.0, 120.0])
    m   = BsmModel(sigma=sigma, intr=r, divr=q)
    fwd = S * np.exp((r - q) * T)
    df  = np.exp(-r * T)
    cf  = m.char_func(T)

    N         = 1024
    eta_grid  = 0.05
    lam_grid  = 2.0 * np.pi / (N * eta_grid)              # plain-FFT relation

    px_frft = frft_price(cf, T, strikes, fwd, df, cp=1,
                         N=N, alpha=0.75,
                         eta_grid=eta_grid, lambda_grid=lam_grid)
    px_cm   = carr_madan_price(cf, T, strikes, fwd, df, cp=1,
                               N=N, alpha=0.75, eta_grid=eta_grid)
    np.testing.assert_allclose(px_frft, px_cm, atol=1e-10)


# ─────────────────────────────────────────────────────────────────────────────
# Analytic BSM agreement at large N
# ─────────────────────────────────────────────────────────────────────────────

def test_frft_vs_analytic_bsm_call():
    sigma, r, q, T, S = 0.25, 0.1, 0.0, 0.1, 100.0
    strikes = np.array([80.0, 100.0, 120.0])
    m   = BsmModel(sigma=sigma, intr=r, divr=q)
    fwd = S * np.exp((r - q) * T)
    df  = np.exp(-r * T)

    px  = frft_price(m.char_func(T), T, strikes, fwd, df, cp=1,
                     N=4096, alpha=0.75, eta_grid=0.05, lambda_grid=0.005)
    ana = bsm_price(strikes, S, sigma, T, intr=r, divr=q, cp=1)
    assert np.max(np.abs(px - ana)) < 1e-6


def test_frft_put_via_parity():
    sigma, r, q, T, S = 0.2, 0.05, 0.02, 1.0, 100.0
    strikes = np.array([80.0, 100.0, 120.0])
    m   = BsmModel(sigma=sigma, intr=r, divr=q)
    fwd = S * np.exp((r - q) * T)
    df  = np.exp(-r * T)

    px_put  = frft_price(m.char_func(T), T, strikes, fwd, df, cp=-1,
                         N=4096, alpha=0.75, eta_grid=0.05, lambda_grid=0.005)
    ana_put = bsm_price(strikes, S, sigma, T, intr=r, divr=q, cp=-1)
    assert np.max(np.abs(px_put - ana_put)) < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# Cross-check vs COS at high precision
# ─────────────────────────────────────────────────────────────────────────────

def test_frft_vs_cos_heston():
    """Two independent CF-based pricers must agree at high precision."""
    m   = HestonCOSPricer(S0=100.0, v0=0.04, lam=1.5, ubar=0.04,
                          eta=0.5, rho=-0.7, r=0.03, q=0.0)
    T   = 1.0
    K   = np.array([90.0, 100.0, 110.0])

    cos_px = m.price_call(K, tau=T, N=256)
    fwd    = m.S0 * np.exp((m.r - m.q) * T)
    df     = np.exp(-m.r * T)
    # HestonCOSPricer.char_func returns CF of log(S_T/S0); shift to log(S_T/F).
    drift  = (m.r - m.q) * T
    cf_F   = lambda u: np.exp(-1j * u * drift) * m.char_func(u, T)
    frft_px = frft_price(cf_F, T, K, fwd, df, cp=1,
                         N=4096, alpha=0.75, eta_grid=0.05, lambda_grid=0.005)
    np.testing.assert_allclose(frft_px, cos_px, atol=1e-4)


def test_frft_vs_cos_vg():
    m   = VgModel(sigma=0.12, theta=-0.14, nu=0.2, intr=0.1, divr=0.0)
    T   = 1.0
    K   = np.array([80.0, 100.0, 120.0])
    cos_px  = m.price(K, 100.0, T, cp=1, n_cos=2**12)
    fwd, df = m._fwd_df(100.0, T)
    frft_px = frft_price(m.char_func(T), T, K, fwd, df, cp=1,
                         N=4096, alpha=0.75, eta_grid=0.05, lambda_grid=0.005)
    np.testing.assert_allclose(frft_px, cos_px, atol=1e-4)


# ─────────────────────────────────────────────────────────────────────────────
# Mixed cp array
# ─────────────────────────────────────────────────────────────────────────────

def test_frft_mixed_cp_array():
    sigma, T, S = 0.25, 0.1, 100.0
    m   = BsmModel(sigma=sigma, intr=0.05, divr=0.02)
    fwd = S * np.exp((m.intr - m.divr) * T)
    df  = np.exp(-m.intr * T)

    K  = np.array([80.0, 100.0, 120.0])
    cp = np.array([+1, -1, +1])
    px = frft_price(m.char_func(T), T, K, fwd, df, cp=cp,
                    N=4096, alpha=0.75, eta_grid=0.05, lambda_grid=0.005)

    ana_call = bsm_price(K[[0, 2]], S, sigma, T, intr=m.intr, divr=m.divr, cp=1)
    ana_put  = bsm_price(K[1],      S, sigma, T, intr=m.intr, divr=m.divr, cp=-1)
    np.testing.assert_allclose([px[0], px[2]], ana_call, atol=1e-6)
    assert abs(px[1] - float(ana_put)) < 1e-6

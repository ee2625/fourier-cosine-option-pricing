"""
Tests for the PyFENG-format FrFT pricer (``pyfeng.frft``).

Mirrors the structure of ``tests/test_pyfeng_bermudan_cos.py``. Requires:
  * ``pip install pyfeng`` (the upstream PyFENG package).
  * The repo's ``pyfeng/frft.py`` to be visible inside the installed
    ``pyfeng`` package -- handled by the sys.path / __path__ overlay in
    ``tests/conftest.py``.

Run:
    python -m pytest tests/test_pyfeng_frft.py -v
"""

import numpy as np
import pytest


pf   = pytest.importorskip("pyfeng")
frft = pytest.importorskip("pyfeng.frft")

FrftMixin    = frft.FrftMixin
BsmFrft      = frft.BsmFrft
VarGammaFrft = frft.VarGammaFrft
CgmyFrft     = frft.CgmyFrft
HestonFrft   = frft.HestonFrft


SPOT  = 100.0
TEXP  = 1.0
KS    = np.array([80.0, 90.0, 100.0, 110.0, 120.0])

BSM_PARAMS = dict(sigma=0.25, intr=0.10, divr=0.0)
VG_PARAMS  = dict(sigma=0.12, theta=-0.14, vov=0.2, intr=0.10, divr=0.0)
CGMY_PARAMS = dict(C=1.0, G=5.0, M=10.0, Y=0.5)
HES_PARAMS = dict(sigma=0.0175, vov=0.5751, mr=1.5768, theta=0.0398, rho=-0.5711)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Pyfeng exports
# ─────────────────────────────────────────────────────────────────────────────

def test_pyfeng_exports_classes():
    assert BsmFrft is frft.BsmFrft
    if not hasattr(pf, "BsmFrft"):
        pytest.skip(
            "pyfeng.BsmFrft not exported. Add this line to the installed "
            "PyFENG's __init__.py: 'from .frft import BsmFrft, VarGammaFrft, "
            "CgmyFrft, HestonFrft'"
        )
    for cls in (BsmFrft, VarGammaFrft, CgmyFrft, HestonFrft):
        assert getattr(pf, cls.__name__) is cls


# ─────────────────────────────────────────────────────────────────────────────
# 2. Cross-check port vs source (LOAD-BEARING) -- BSM at machine precision
# ─────────────────────────────────────────────────────────────────────────────

def test_bsm_cross_check_vs_source():
    """``BsmFrft.price_frft`` must match ``cos_pricing.frft_price`` to
    machine precision: same kernel, same default tuning."""
    cos_pricing = pytest.importorskip("cos_pricing")
    sigma, r, q = BSM_PARAMS["sigma"], BSM_PARAMS["intr"], BSM_PARAMS["divr"]
    fwd = SPOT * np.exp((r - q) * TEXP)
    df  = np.exp(-r * TEXP)

    def cf_bsm(u):
        return np.exp(1j * u * (-0.5 * sigma**2 * TEXP)
                      - 0.5 * sigma**2 * TEXP * u**2)

    src_call = cos_pricing.frft_price(cf_bsm, TEXP, KS, fwd, df, cp=+1)
    src_put  = cos_pricing.frft_price(cf_bsm, TEXP, KS, fwd, df, cp=-1)

    port = BsmFrft(**BSM_PARAMS)
    port_call = port.price_frft(KS, SPOT, TEXP, cp=+1)
    port_put  = port.price_frft(KS, SPOT, TEXP, cp=-1)

    assert np.max(np.abs(src_call - port_call)) < 1e-13
    assert np.max(np.abs(src_put  - port_put))  < 1e-13


# ─────────────────────────────────────────────────────────────────────────────
# 3. Agreement with Lewis (within FrFT discretisation)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("cls,params", [
    (BsmFrft,      BSM_PARAMS),
    (VarGammaFrft, VG_PARAMS),
    (CgmyFrft,     CGMY_PARAMS),
    (HestonFrft,   HES_PARAMS),
])
def test_frft_matches_lewis(cls, params):
    """FrFT and Lewis (FftABC.price) should agree to ~1e-2 with default
    tuning -- both are valid Carr-Madan integrators; the gap is grid
    resolution, not a model error."""
    m = cls(**params)
    frft_px  = m.price_frft(KS, SPOT, TEXP, cp=+1)
    lewis_px = m.price(KS, SPOT, TEXP, cp=+1)
    assert np.max(np.abs(frft_px - lewis_px)) < 5e-3


# ─────────────────────────────────────────────────────────────────────────────
# 4. Put-call parity
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("cls,params", [
    (BsmFrft,      BSM_PARAMS),
    (VarGammaFrft, VG_PARAMS),
    (CgmyFrft,     CGMY_PARAMS),
    (HestonFrft,   HES_PARAMS),
])
def test_frft_put_call_parity(cls, params):
    m = cls(**params)
    fwd, df, _ = m._fwd_factor(SPOT, TEXP)
    call = m.price_frft(KS, SPOT, TEXP, cp=+1)
    put  = m.price_frft(KS, SPOT, TEXP, cp=-1)
    parity = float(df) * (float(fwd) - KS)
    assert np.max(np.abs((call - put) - parity)) < 1e-10


# ─────────────────────────────────────────────────────────────────────────────
# 5. Scalar / vector handling
# ─────────────────────────────────────────────────────────────────────────────

def test_frft_scalar_strike_returns_scalar():
    m = BsmFrft(**BSM_PARAMS)
    px = m.price_frft(100.0, SPOT, TEXP, cp=+1)
    assert np.isscalar(px) or np.ndim(px) == 0


def test_frft_vector_strike_matches_scalar():
    m = BsmFrft(**BSM_PARAMS)
    vec  = m.price_frft(KS, SPOT, TEXP, cp=+1)
    scal = np.array([m.price_frft(float(K), SPOT, TEXP, cp=+1) for K in KS])
    assert np.max(np.abs(vec - scal)) < 1e-12


# ─────────────────────────────────────────────────────────────────────────────
# 6. Mixin composability -- ad-hoc combination matches concrete class
# ─────────────────────────────────────────────────────────────────────────────

def test_mixin_composability():
    """A user can mix ``FrftMixin`` into ``BsmFft`` ad-hoc and get identical
    prices to the pre-defined ``BsmFrft``."""
    BsmFft = pf.BsmFft

    class AdHocBsmFrft(BsmFft, FrftMixin):
        pass

    a = AdHocBsmFrft(**BSM_PARAMS)
    b = BsmFrft(**BSM_PARAMS)
    a_px = a.price_frft(KS, SPOT, TEXP, cp=+1)
    b_px = b.price_frft(KS, SPOT, TEXP, cp=+1)
    assert np.max(np.abs(a_px - b_px)) < 1e-14


# ─────────────────────────────────────────────────────────────────────────────
# 7. Tunable knobs really do tune
# ─────────────────────────────────────────────────────────────────────────────

def test_frft_lambda_grid_refines_price():
    """A finer log-strike grid (smaller ``lambda_frft``) brings FrFT closer
    to the Lewis reference."""
    m_default = BsmFrft(**BSM_PARAMS)
    m_fine    = BsmFrft(**BSM_PARAMS); m_fine.lambda_frft = 0.001

    lewis_ref = m_default.price(KS, SPOT, TEXP, cp=+1)
    err_def = float(np.max(np.abs(m_default.price_frft(KS, SPOT, TEXP, cp=+1) - lewis_ref)))
    err_fine = float(np.max(np.abs(m_fine.price_frft(KS, SPOT, TEXP, cp=+1) - lewis_ref)))
    assert err_fine < err_def, f"fine={err_fine} not better than default={err_def}"

"""
Tests for the PyFENG-format Lévy COS pricers
(``pyfeng.lv_cos.VarGammaCos`` and ``pyfeng.lv_cos.CgmyCos``).

Mirrors the structure of ``tests/test_pyfeng_heston_cos.py``. Requires:
  * ``pip install pyfeng`` (the upstream PyFENG package).
  * The repo's ``pyfeng/sv_cos.py`` and ``pyfeng/lv_cos.py`` to be
    visible inside the installed ``pyfeng`` package -- handled by the
    sys.path / __path__ overlay in ``tests/conftest.py``.

Run:
    python -m pytest tests/test_pyfeng_lv_cos.py -v
"""

import numpy as np
import pytest


pf = pytest.importorskip("pyfeng")
lv_cos = pytest.importorskip("pyfeng.lv_cos")
VarGammaCos = lv_cos.VarGammaCos
CgmyCos = lv_cos.CgmyCos


# ─────────────────────────────────────────────────────────────────────────────
# Variance Gamma -- F&O (2008) Table 7 paper benchmark
# ─────────────────────────────────────────────────────────────────────────────

VG_PARAMS = dict(sigma=0.12, theta=-0.14, vov=0.2, intr=0.1, divr=0.0)
VG_SPOT   = 100.0
VG_STRIKE = 90.0
VG_REF_T01 = 10.993703187   # texp = 0.1
VG_REF_T1  = 19.099354724   # texp = 1.0


def _vg(**overrides):
    params = dict(VG_PARAMS)
    params.update(overrides)
    return VarGammaCos(**params)


# 1. Install wiring -----------------------------------------------------------

def test_vg_install_wiring():
    assert VarGammaCos is lv_cos.VarGammaCos
    if not hasattr(pf, "VarGammaCos"):
        pytest.skip(
            "pyfeng.VarGammaCos not exported. Add this line to the installed "
            "PyFENG's __init__.py: 'from .lv_cos import VarGammaCos'"
        )
    assert pf.VarGammaCos is VarGammaCos


# 2. Paper benchmark ----------------------------------------------------------

def test_vg_paper_benchmark():
    """F&O (2008) Table 7: VG call at S=100, K=90, sigma=0.12, theta=-0.14, vov=0.2, r=0.1.

    VG converges *algebraically* (order ~3) at short maturities because the CF decays
    only as |u|^{-2T/vov}; for T=0.1, vov=0.2, that is |u|^{-1}.  The reference run in
    cos_pricing/tests uses N=2^14 to reach the F&O reference price; we match that here.
    """
    m = _vg()
    m.n_cos = 2**14
    err_t01 = abs(m.price(VG_STRIKE, VG_SPOT, 0.1) - VG_REF_T01)
    err_t1  = abs(m.price(VG_STRIKE, VG_SPOT, 1.0) - VG_REF_T1)
    assert err_t01 < 1e-4, f"texp=0.1 err={err_t01:.2e}"   # algebraic regime
    assert err_t1  < 1e-5, f"texp=1.0 err={err_t1:.2e}"


# 3. Strike vectorization -----------------------------------------------------

def test_vg_strike_vectorized_matches_scalar():
    m = _vg()
    strikes = np.array([85.0, 90.0, 95.0, 100.0, 105.0])
    vec  = m.price(strikes, VG_SPOT, 1.0)
    scal = np.array([m.price(float(K), VG_SPOT, 1.0) for K in strikes])
    assert vec.shape == strikes.shape
    assert np.max(np.abs(vec - scal)) < 1e-12


# 4. Put-call parity ----------------------------------------------------------

def test_vg_put_call_parity():
    m = _vg()
    texp = 1.0
    fwd, df, _ = m._fwd_factor(VG_SPOT, texp)
    for K in (80.0, 100.0, 120.0):
        c = m.price(K, VG_SPOT, texp, cp=+1)
        p = m.price(K, VG_SPOT, texp, cp=-1)
        assert abs((c - p) - df * (fwd - K)) < 1e-4


# 5. CF identity --------------------------------------------------------------

def test_vg_charfunc_is_mgf_on_imaginary_axis():
    m = _vg()
    u = np.linspace(0.1, 5.0, 11)
    cf  = m.charfunc_logprice(u, 1.0)
    mgf = m.mgf_logprice(1j * u, 1.0)
    assert np.max(np.abs(cf - mgf)) < 1e-13


# 6. Martingale property ------------------------------------------------------

def test_vg_martingale():
    """E[S_T/F] = 1 under risk-neutral measure → mgf_logprice(1, texp) == 1."""
    for params in [{}, dict(intr=0.05, divr=0.02)]:
        m = _vg(**params)
        for texp in (0.1, 1.0, 5.0):
            got = complex(m.mgf_logprice(1.0, texp))
            assert abs(got.imag) < 1e-12, "MGF(1) should be real"
            assert abs(got.real - 1.0) < 1e-12, f"texp={texp}: martingale violated"


# 7. Analytic cumulants vs FD reference ---------------------------------------

def test_vg_analytic_cumulants_match_FD():
    """Analytic c1/c2/c4 must agree with finite-difference of log(MGF) at zero."""
    m = _vg()
    for texp in (0.5, 1.0, 5.0):
        c1, c2, c3, c4 = m._cumulants(texp)
        # FD reference, central differences.
        eps = 1e-3
        K = lambda uu: float(np.log(m.mgf_logprice(uu, texp)).real)
        c1_num = (K(eps) - K(-eps)) / (2 * eps)
        c2_num = (K(eps) + K(-eps) - 2 * K(0.0)) / eps**2
        assert abs(c1_num - c1) < 1e-7,  f"texp={texp}: c1 {c1} vs FD {c1_num}"
        assert abs(c2_num - c2) < 1e-7,  f"texp={texp}: c2 {c2} vs FD {c2_num}"
        assert c3 == 0.0
        assert c4 > 0.0


# 8. Cross-check vs source pricer (load-bearing) ------------------------------

def test_vg_cross_check_vs_source_pricer():
    cos_pricing = pytest.importorskip("cos_pricing")
    VgModel = cos_pricing.vg_model.VgModel  # may need direct import path

    # Source uses (sigma, theta, nu) -- vov maps to nu.
    src = VgModel(sigma=VG_PARAMS["sigma"], theta=VG_PARAMS["theta"],
                  nu=VG_PARAMS["vov"], intr=VG_PARAMS["intr"], divr=VG_PARAMS["divr"])
    port = _vg()
    port.n_cos = 256
    port.L = 10.0  # match source default L=10

    strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    for texp in (0.1, 0.5, 1.0):
        src_px  = src.price(strikes, VG_SPOT, texp, cp=+1, n_cos=256)
        port_px = port.price(strikes, VG_SPOT, texp, cp=+1)
        diff = float(np.max(np.abs(src_px - port_px)))
        assert diff < 1e-8, f"texp={texp}: VG port vs source diff {diff:.2e}"


# 9. Cross-check vs upstream FFT ----------------------------------------------

def test_vg_cross_check_vs_pyfeng_fft():
    VarGammaFft = getattr(pf, "VarGammaFft", None)
    if VarGammaFft is None:
        pytest.skip("pyfeng.VarGammaFft not available in this install")

    fft = VarGammaFft(**VG_PARAMS)
    cos = _vg()
    cos.n_cos = 256

    strikes = np.array([85.0, 95.0, 105.0])
    for texp in (0.5, 1.0):
        fft_px = fft.price(strikes, VG_SPOT, texp)
        cos_px = cos.price(strikes, VG_SPOT, texp)
        diff = float(np.max(np.abs(cos_px - fft_px)))
        assert diff < 1e-5, f"texp={texp}: COS vs FFT diff {diff:.2e}"


# 10. Input validation --------------------------------------------------------

@pytest.mark.parametrize("bad", [
    dict(sigma=0.0),
    dict(sigma=-0.1),
    dict(vov=0.0),
    dict(vov=-0.1),
    # Constraint: 1 - theta*vov - 0.5*sigma^2*vov > 0.
    # With vov=10 and theta=0.5, the constraint is 1 - 5 - 0.5*0.0144*10 = -4.072 < 0.
    dict(sigma=0.12, theta=0.5, vov=10.0),
])
def test_vg_input_validation(bad):
    params = dict(VG_PARAMS)
    params.update(bad)
    with pytest.raises(ValueError):
        VarGammaCos(**params)


# ─────────────────────────────────────────────────────────────────────────────
# CGMY
# ─────────────────────────────────────────────────────────────────────────────

CGMY_PARAMS = dict(C=1.0, G=5.0, M=10.0, Y=0.5)
CGMY_SPOT = 100.0


def _cgmy(**overrides):
    params = dict(CGMY_PARAMS)
    params.update(overrides)
    return CgmyCos(**params)


# 11. Install wiring ----------------------------------------------------------

def test_cgmy_install_wiring():
    assert CgmyCos is lv_cos.CgmyCos
    if not hasattr(pf, "CgmyCos"):
        pytest.skip(
            "pyfeng.CgmyCos not exported. Add this line to the installed "
            "PyFENG's __init__.py: 'from .lv_cos import CgmyCos'"
        )
    assert pf.CgmyCos is CgmyCos


# 12. Strike vectorization ----------------------------------------------------

def test_cgmy_strike_vectorized_matches_scalar():
    m = _cgmy()
    strikes = np.array([85.0, 90.0, 95.0, 100.0, 105.0, 110.0])
    vec  = m.price(strikes, CGMY_SPOT, 1.0)
    scal = np.array([m.price(float(K), CGMY_SPOT, 1.0) for K in strikes])
    assert vec.shape == strikes.shape
    # Tolerance is 1e-10 (not 1e-12) because the batched matvec accumulates
    # additions in a different order than the per-strike loop, giving a
    # single-ulp summation difference (~5e-12 observed) that has no
    # mathematical content.
    assert np.max(np.abs(vec - scal)) < 1e-10


# 13. Put-call parity ---------------------------------------------------------

def test_cgmy_put_call_parity():
    m = _cgmy()
    texp = 1.0
    fwd, df, _ = m._fwd_factor(CGMY_SPOT, texp)
    for K in (80.0, 100.0, 120.0):
        c = m.price(K, CGMY_SPOT, texp, cp=+1)
        p = m.price(K, CGMY_SPOT, texp, cp=-1)
        assert abs((c - p) - df * (fwd - K)) < 1e-4


# 14. CF identity -------------------------------------------------------------

def test_cgmy_charfunc_is_mgf_on_imaginary_axis():
    m = _cgmy()
    u = np.linspace(0.1, 3.0, 11)
    cf  = m.charfunc_logprice(u, 1.0)
    mgf = m.mgf_logprice(1j * u, 1.0)
    assert np.max(np.abs(cf - mgf)) < 1e-13


# 15. Martingale property -----------------------------------------------------

def test_cgmy_martingale():
    """E[S_T/F] = 1 → mgf_logprice(1, texp) == 1.

    CGMY's omega correction (`_mgf1_correction`) is constructed so the MGF
    at u=1 is exactly 1 by construction; this is a strict identity test.
    """
    for params in [{}, dict(intr=0.05, divr=0.02)]:
        m = _cgmy(**params)
        for texp in (0.1, 1.0, 5.0):
            got = complex(m.mgf_logprice(1.0, texp))
            assert abs(got.imag) < 1e-12, "MGF(1) should be real"
            assert abs(got.real - 1.0) < 1e-12, f"texp={texp}: martingale violated"


# 16. Cross-check vs source pricer (load-bearing) -----------------------------

@pytest.mark.parametrize("Y", [0.5, 1.0001, 1.5, 1.8])  # avoid singular Y in {0,1,2}
def test_cgmy_cross_check_vs_source_pricer(Y):
    """Two independent COS implementations of CGMY must agree.

    The pyfeng port and the cos_pricing source compute the same algorithm
    on different code paths (CosABC inheritance vs. cos_method.cos_price);
    structural floating-point differences in chi/psi assembly and prime-sum
    handling produce ~1e-7 absolute discrepancies at high Y, where the
    truncation interval [-L*Y, L*Y] is wide and the cosine grid is coarse.
    Tolerance 1e-5 covers that regime; Y=0.5 (narrow interval) routinely
    agrees to 1e-13.
    """
    cos_pricing = pytest.importorskip("cos_pricing")
    CgmyModel = cos_pricing.cgmy_model.CgmyModel

    src  = CgmyModel(C=CGMY_PARAMS["C"], G=CGMY_PARAMS["G"],
                     M=CGMY_PARAMS["M"], Y=Y)
    port = CgmyCos(C=CGMY_PARAMS["C"], G=CGMY_PARAMS["G"],
                   M=CGMY_PARAMS["M"], Y=Y)
    port.n_cos = 512
    port.L = 10.0  # match source default

    strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    for texp in (0.5, 1.0):
        src_px  = src.price(strikes, CGMY_SPOT, texp, cp=+1, n_cos=512, L=10.0)
        port_px = port.price(strikes, CGMY_SPOT, texp, cp=+1)
        diff = float(np.max(np.abs(src_px - port_px)))
        assert diff < 1e-5, f"Y={Y}, texp={texp}: CGMY port vs source diff {diff:.2e}"


# 17. Cross-check vs upstream FFT ---------------------------------------------

def test_cgmy_cross_check_vs_pyfeng_fft():
    CgmyFft = getattr(pf, "CgmyFft", None)
    if CgmyFft is None:
        pytest.skip("pyfeng.CgmyFft not available in this install")

    fft = CgmyFft(**CGMY_PARAMS)
    cos = _cgmy()
    cos.n_cos = 256

    strikes = np.array([90.0, 100.0, 110.0])
    for texp in (0.5, 1.0):
        fft_px = fft.price(strikes, CGMY_SPOT, texp)
        cos_px = cos.price(strikes, CGMY_SPOT, texp)
        diff = float(np.max(np.abs(cos_px - fft_px)))
        assert diff < 1e-5, f"texp={texp}: COS vs FFT diff {diff:.2e}"


# 18. Truncation range Y near 1.98 --------------------------------------------

def test_cgmy_y_near_2_truncation():
    m = CgmyCos(C=1.0, G=5.0, M=10.0, Y=1.98)
    a, b = m._truncation_range(1.0)
    assert (a, b) == (-100.0, 20.0)

    # And for a "normal" Y we get the [-L*Y, L*Y] heuristic.
    m2 = CgmyCos(C=1.0, G=5.0, M=10.0, Y=0.5)
    m2.L = 10.0
    a2, b2 = m2._truncation_range(1.0)
    assert (a2, b2) == (-5.0, 5.0)


# 19. Input validation --------------------------------------------------------

@pytest.mark.parametrize("bad", [
    dict(C=0.0),
    dict(C=-0.1),
    dict(G=0.0),
    dict(G=-0.1),
    dict(M=1.0),    # must be > 1
    dict(M=0.5),
    dict(Y=2.0),
    dict(Y=2.5),
])
def test_cgmy_input_validation(bad):
    params = dict(CGMY_PARAMS)
    params.update(bad)
    with pytest.raises(ValueError):
        CgmyCos(**params)


# 20. Default L documentation behavior ----------------------------------------

def test_cgmy_default_L_matches_cosabc():
    """Default L is CosABC's 12.0; user must set m.L = 10 to match source exactly."""
    m = _cgmy()
    assert m.L == 12.0
    a, b = m._truncation_range(1.0)
    assert (a, b) == (-12.0 * m.Y, 12.0 * m.Y)

    m.L = 10.0
    a10, b10 = m._truncation_range(1.0)
    assert (a10, b10) == (-10.0 * m.Y, 10.0 * m.Y)

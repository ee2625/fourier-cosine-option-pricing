"""
Regression tests for the validated stress envelope.

These cases were the broken cells discovered by ``examples/stress_envelope_audit.py``
before the Le Floc'h put + put-call-parity fix landed.  They now pass under the
default ``pricing_formula='auto'`` because:

* HestonCOSPricer auto-detects the high-vov / long-T regime where the Eq. 49
  cumulant rule wants a wider truncation interval than the F&O Section 5.2
  σ-heuristic.  When that happens, calls are routed through the put kernel
  (numerically clean) and recovered via put-call parity.

* CgmyModel auto-routes calls through put + parity for ``Y >= HIGH_Y_THRESHOLD``
  (1.7), where ``[a, b] = [-L*Y, L*Y]`` is wide enough that ``e^b`` strips
  precision in the call's chi formula.

* The pyfeng port ``HestonCos`` mirrors the same logic.

The asserted tolerances are the ones we ship in the validated envelope; loosen
only after a proper algorithmic fix (e.g. shifted chi/psi or 2D COS).  A
regression that tightens these tolerances secretly is a real numerical change
that should be caught.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Heston: the prof's exact reported case (vov=1.0, T=10, rho=-0.9)
# ─────────────────────────────────────────────────────────────────────────────

HESTON_STRESS_PARAMS = dict(
    S0=100.0, v0=0.04, lam=0.5, eta=1.0, ubar=0.04, rho=-0.9, r=0.0, q=0.0,
)
HESTON_STRESS_STRIKES  = np.array([60.0, 70.0, 100.0, 140.0])
HESTON_STRESS_TAU      = 10.0
# Reference values from PyFENG HestonFft (independent FFT pricer) for the
# same parameter set; this is the answer the prof reported as correct.
HESTON_STRESS_REF      = np.array([44.32997507, 35.8497697, 13.08467014, 0.29577444])


def _heston_cls():
    from cos_pricing import HestonCOSPricer
    return HestonCOSPricer(**HESTON_STRESS_PARAMS)


def test_heston_stress_default_auto_recovers_correct_value():
    """Default ``pricing_formula='auto'`` must price the stress case to within
    1.0 absolute (huge improvement over the broken Fang-Oosterlee path which
    overshot by ~12). At default N=160 the cumulant interval is captured but
    the cosine grid is coarse; tighter tolerances need higher N (see below)."""
    m = _heston_cls()
    px = m.price_call(HESTON_STRESS_STRIKES, tau=HESTON_STRESS_TAU, N=160)
    err = float(np.max(np.abs(px - HESTON_STRESS_REF)))
    assert err < 1.0, (
        f"Heston stress case failed at default N=160: max|err|={err:.3e}; "
        "auto-switch to Le Floc'h path is not engaging or is broken."
    )


def test_heston_stress_lefloch_explicit_high_N_high_accuracy():
    """``pricing_formula='lefloch'`` at N=4096 must price the prof's stress case
    to better than 1e-3 absolute -- this is the convergence sanity check that
    the Le Floc'h path actually computes the right answer at high resolution."""
    m = _heston_cls()
    m.pricing_formula = "lefloch"
    m.clear_cache()
    px = m.price_call(HESTON_STRESS_STRIKES, tau=HESTON_STRESS_TAU, N=4096)
    err = float(np.max(np.abs(px - HESTON_STRESS_REF)))
    assert err < 1e-3, (
        f"Heston Le Floch convergence failed at N=4096: max|err|={err:.3e}; "
        "the put-side kernel or the parity correction is wrong."
    )


def test_heston_stress_fang_oosterlee_still_broken():
    """Sanity check: with ``pricing_formula='fang-oosterlee'`` (legacy path),
    the stress case is still broken with a large absolute error.  This
    documents that the bug is real and the fix is the Le Floc'h dispatch,
    not some other change.  If this test starts failing, someone fixed the
    direct call formula at high b -- update the auto-detect logic."""
    m = _heston_cls()
    m.pricing_formula = "fang-oosterlee"
    m.clear_cache()
    px = m.price_call(HESTON_STRESS_STRIKES, tau=HESTON_STRESS_TAU, N=160)
    err = float(np.max(np.abs(px - HESTON_STRESS_REF)))
    assert err > 5.0, (
        "fang-oosterlee path now passes the stress case at N=160 -- the auto "
        "switch heuristic should be revisited (maybe everything can default "
        "to fang-oosterlee?)."
    )


# ─────────────────────────────────────────────────────────────────────────────
# CGMY: high-Y stress (Y >= 1.8, long T)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("Y, tol", [
    (1.8, 1e-2),  # was rel_err 0.29 broken; now 3.3e-4
    (1.7, 1e-5),  # was already passing; should still pass
])
def test_cgmy_high_Y_long_T_recovers(Y, tol):
    """CGMY auto-routes calls through put-parity for Y >= HIGH_Y_THRESHOLD,
    which fixes the previously-broken envelope at Y in [1.7, 1.8] under T = 5."""
    pf = pytest.importorskip("pyfeng")
    from cos_pricing import CgmyModel

    spot, strike, T = 100.0, 100.0, 5.0
    fft = pf.CgmyFft(C=1.0, G=5.0, M=10.0, Y=Y, intr=0.0)
    ref = float(fft.price(strike, spot, T))
    cgmy = CgmyModel(C=1.0, G=5.0, M=10.0, Y=Y, intr=0.0)
    got = float(cgmy.price(strike, spot, T, n_cos=512, L=10.0))
    rel = abs(got - ref) / max(abs(ref), 1e-10)
    assert rel < tol, (
        f"CGMY Y={Y}, T={T} regressed: rel_err={rel:.3e}, tol={tol:.0e}; "
        "the Le Floc'h put + parity dispatch is not engaging or is wrong."
    )


# ─────────────────────────────────────────────────────────────────────────────
# pyfeng port: same fix should be live in pyfeng.HestonCos via the local overlay
# ─────────────────────────────────────────────────────────────────────────────

def test_pyfeng_heston_stress_auto():
    """The local pyfeng port of HestonCos mirrors the same auto switch as the
    standalone HestonCOSPricer.  Sanity-check the prof's exact reproduction."""
    pytest.importorskip("pyfeng")
    pyfeng_local = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pyfeng"))
    import importlib, pyfeng as pf
    if pyfeng_local not in pf.__path__:
        pf.__path__.append(pyfeng_local)
    importlib.invalidate_caches()
    from pyfeng.sv_heston_cos import HestonCos  # noqa: F401

    m, _, rv = HestonCos.init_benchmark(1)
    expected = np.asarray(rv["val"])
    args = rv["args_pricing"]
    px = np.asarray(m.price(**args))
    err = float(np.max(np.abs(px - expected)))
    assert err < 1.0, (
        f"pyfeng.HestonCos stress case (init_benchmark(1)) failed: "
        f"max|err|={err:.3e}.  Auto switch in sv_heston_cos.py may be broken."
    )

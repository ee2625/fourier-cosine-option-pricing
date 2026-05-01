"""
Tests for the PyFENG-format Heston COS pricer (``pyfeng.sv_heston_cos.HestonCos``).

Mirrors ``tests/test_heston_cos_pricer.py`` but exercises the PyFENG class
API. Requires:
  * ``pip install pyfeng`` (the upstream PyFENG package).
  * The repo's ``pyfeng/sv_cos.py`` and ``pyfeng/sv_heston_cos.py`` to be
    visible inside the installed ``pyfeng`` package -- handled by the
    sys.path / __path__ overlay in ``tests/conftest.py``.

If PyFENG is not installed, the suite skips cleanly via ``importorskip``.

Run:
    python -m pytest tests/test_pyfeng_heston_cos.py -v
"""

import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest


pf = pytest.importorskip("pyfeng")
sv_heston_cos = pytest.importorskip("pyfeng.sv_heston_cos")
HestonCos = sv_heston_cos.HestonCos
warmup_numba = sv_heston_cos.warmup_numba


# PyFENG-renamed F&O (2008) paper parameters.
#   v0   -> sigma
#   eta  -> vov
#   lam  -> mr
#   ubar -> theta
PF_PARAMS = dict(sigma=0.0175, vov=0.5751, mr=1.5768, theta=0.0398, rho=-0.5711)
SPOT      = 100.0
STRIKE    = 100.0
REF_T1    = 5.785155435
REF_T10   = 22.318945791474590


def _make(intr=0.0, divr=0.0, **overrides):
    params = dict(PF_PARAMS, intr=intr, divr=divr)
    params.update(overrides)
    return HestonCos(**params)


def _scalar(value):
    return float(np.asarray(value, dtype=float).reshape(-1)[0])


# ─────────────────────────────────────────────────────────────────────────────
# 1. Install / wiring smoke test (priority: catches the namespace shadow problem)
# ─────────────────────────────────────────────────────────────────────────────

def test_install_wiring():
    """Verify the install-then-overlay scheme works.

    (a) ``pyfeng.heston`` resolves -- upstream is installed and visible,
        not shadowed by the local namespace package.
    (b) ``pyfeng.sv_heston_cos.HestonCos`` is importable.
    (c) ``pf.HestonCos is HestonCos`` *if* upstream ``__init__.py``
        exports it. Skipped (with a clear message) otherwise so it can
        guide the user to the missing __init__ line.
    """
    pytest.importorskip("pyfeng.heston")  # (a)
    assert HestonCos is sv_heston_cos.HestonCos  # (b)
    if not hasattr(pf, "HestonCos"):
        pytest.skip(
            "pyfeng.HestonCos not exported. Add this line to the installed "
            "PyFENG's __init__.py: 'from .sv_heston_cos import HestonCos'"
        )
    assert pf.HestonCos is HestonCos  # (c)


# ─────────────────────────────────────────────────────────────────────────────
# 2-3. Paper benchmarks -- HARD invariant: adaptive L (m.L is None)
# ─────────────────────────────────────────────────────────────────────────────

def test_paper_benchmark_t1_uses_adaptive_L():
    """At τ=1 with the L=None sentinel (adaptive), price must hit REF_T1."""
    m = _make()
    assert m.L is None, "L sentinel must be None to exercise the adaptive default"
    err = abs(m.price(STRIKE, SPOT, 1.0) - REF_T1)
    assert err < 1e-5, f"τ=1 price err={err:.2e} exceeds 1e-5"


def test_paper_benchmark_t10_uses_adaptive_L():
    """At τ=10 with adaptive L (no manual override), price must hit REF_T10.

    This is the test that would have failed if HestonCos inherited
    CosABC.L = 12.0 instead of using the τ-adaptive default.
    """
    m = _make()
    m.n_cos = 256
    assert m.L is None, "L sentinel must be None to exercise the adaptive default"
    err = abs(m.price(STRIKE, SPOT, 10.0) - REF_T10)
    assert err < 1e-6, f"τ=10 price err={err:.2e} exceeds 1e-6"


# ─────────────────────────────────────────────────────────────────────────────
# 4-6. Convergence, sensitivity, L state management
# ─────────────────────────────────────────────────────────────────────────────

def test_convergence_in_n_cos():
    m = _make()
    errs = []
    for N in (32, 64, 128, 256):
        m.n_cos = N
        errs.append(abs(m.price(STRIKE, SPOT, 1.0) - REF_T1))
    for prev, curr in zip(errs, errs[1:]):
        assert curr <= prev * 1.1 + 1e-12
    assert errs[-1] < 1e-6


def test_L_sensitivity():
    """L=3 too narrow; L=12 accurate."""
    m = _make()
    m.n_cos = 160

    m.L = 3.0
    err_small_L = abs(m.price(STRIKE, SPOT, 1.0) - REF_T1)

    m.L = 12.0
    err_good_L = abs(m.price(STRIKE, SPOT, 1.0) - REF_T1)

    assert err_small_L > 1e3 * err_good_L
    assert err_good_L < 1e-6


def test_L_reset_to_None_resumes_adaptive():
    """After m.L is set to a number, resetting to None must restore the τ-adaptive default."""
    m = _make()
    m.L = 15.0
    assert m._resolve_L(1.0) == 15.0
    assert m._resolve_L(10.0) == 15.0

    m.L = None
    assert m._resolve_L(1.0) == 10.0           # max(10, 3*1+2) = 10
    assert m._resolve_L(10.0) == 32.0          # max(10, 3*10+2) = 32


# ─────────────────────────────────────────────────────────────────────────────
# 7-10. Structural correctness
# ─────────────────────────────────────────────────────────────────────────────

def test_put_call_parity():
    """C - P = df * (F - K), tolerated to truncation tail."""
    m = _make()
    m.n_cos = 160
    texp = 1.0
    fwd, df, _ = m._fwd_factor(SPOT, texp)
    for K in (80.0, 100.0, 120.0):
        c = m.price(K, SPOT, texp, cp=+1)
        p = m.price(K, SPOT, texp, cp=-1)
        assert abs((c - p) - df * (fwd - K)) < 1e-4


def test_strike_vectorized_matches_scalar():
    m = _make()
    m.n_cos = 160
    strikes = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
    vec  = m.price(strikes, SPOT, 1.0)
    scal = np.array([m.price(float(K), SPOT, 1.0) for K in strikes])
    assert vec.shape == strikes.shape
    assert np.max(np.abs(vec - scal)) < 1e-12


def test_price_smile_matches_price_in_log_forward_range():
    m = _make()
    m.n_cos = 160
    strikes = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
    cp = np.array([1, -1, 1, -1, 1])

    got = m.price_smile(strikes, SPOT, 1.0, cp=cp)
    ref = m.price(strikes, SPOT, 1.0, cp=cp)

    assert got.shape == strikes.shape
    assert np.max(np.abs(got - ref)) < 1e-7

    # Heston's paper interval is per-strike in y=log(S_T/K).  The reusable
    # smile setup uses the equivalent z=log(S_T/F) range.
    a_z, b_z = m._smile_truncation_range(1.0)
    a_y, b_y, x, _ = m.truncation_interval(STRIKE, SPOT, 1.0)
    assert abs((a_y - x) - a_z) < 1e-12
    assert abs((b_y - x) - b_z) < 1e-12


def test_average_variance_equivalent_vol():
    m = _make()
    texp = 1.0
    avg_var = _scalar(m.avgvar_mv(texp)[0])

    assert abs(m.avg_variance_mean(texp) - avg_var) < 1e-15
    assert abs(m.equivalent_bsm_vol(texp) - np.sqrt(avg_var)) < 1e-15


def test_black_scholes_control_variate_identity_and_shape():
    m = _make()
    m.n_cos = 32
    strikes = np.array([90.0, 100.0, 110.0])
    cp = np.array([1, -1, 1])

    plain = m.price(strikes, SPOT, 1.0, cp=cp)
    adj = m.bsm_control_variate_adjustment(strikes, SPOT, 1.0, cp=cp)
    cv = m.price_cv(strikes, SPOT, 1.0, cp=cp)

    assert cv.shape == strikes.shape
    assert np.max(np.abs(cv - (plain + adj))) < 1e-12
    a, b = m._bsm_cv_trunc_range(1.0)
    assert np.isfinite(a) and np.isfinite(b) and a < b


def test_joshi_yang_control_variate_vol_methods_are_opt_in():
    m = _make()
    m.n_cos = 8
    plain = m.price(STRIKE, SPOT, 1.0)

    for method in ("joshi", "joshi-half"):
        vol = m.equivalent_bsm_vol(1.0, method=method)
        adj = m.bsm_control_variate_adjustment(
            STRIKE, SPOT, 1.0, vol_method=method
        )
        cv = m.price_cv(STRIKE, SPOT, 1.0, vol_method=method)
        assert vol > 0.0
        assert abs(cv - (plain + adj)) < 1e-12

    with pytest.raises(ValueError):
        m.equivalent_bsm_vol(1.0, method="unknown")


def test_black_scholes_control_variate_reduces_coarse_heston_error():
    m = _make()
    m.n_cos = 8

    plain_err = abs(m.price(STRIKE, SPOT, 1.0) - REF_T1)
    cv_err = abs(m.price_cv(STRIKE, SPOT, 1.0) - REF_T1)

    assert cv_err < 0.2 * plain_err


def test_price_smile_cv_uses_same_setup_range():
    m = _make()
    m.n_cos = 32
    strikes = np.array([90.0, 100.0, 110.0])
    setup = m.make_smile_setup(SPOT, 1.0)

    got = m.price_smile_cv(strikes, SPOT, 1.0)
    expected = setup.price(strikes) + m.bsm_control_variate_adjustment(
        strikes, SPOT, 1.0, trunc_range=(setup.a, setup.b)
    )

    assert got.shape == strikes.shape
    assert np.max(np.abs(got - expected)) < 1e-12


def test_scalar_in_scalar_out():
    m = _make()
    out = m.price(STRIKE, SPOT, 1.0)
    assert isinstance(out, float)


def test_non_negative_prices():
    m = _make()
    strikes = np.linspace(60, 200, 60)
    prices = m.price(strikes, SPOT, 1.0, cp=+1)
    assert np.all(prices >= -1e-10)


# ─────────────────────────────────────────────────────────────────────────────
# 11-14. Static helpers, CF identities, martingale property
# ─────────────────────────────────────────────────────────────────────────────

def test_chi_psi_broadcasting():
    """HestonCos.chi/psi static helpers with source-compatible (k, a, b, c, d) signature."""
    k = np.arange(8)
    a, b, c, d = -2.0, 3.0, 0.0, 3.0
    chi_v = HestonCos.chi(k, a, b, c, d)
    psi_v = HestonCos.psi(k, a, b, c, d)
    assert chi_v.shape == (8,)
    assert psi_v.shape == (8,)
    # ψ at k=0 collapses to (d - c).
    assert abs(psi_v[0] - (d - c)) < 1e-14


def test_charfunc_is_mgf_on_imaginary_axis():
    """charfunc_logprice(u) must equal mgf_logprice(i*u) — sanity check on inheritance."""
    m = _make()
    u = np.linspace(0.1, 5.0, 11)
    cf  = m.charfunc_logprice(u, 1.0)
    mgf = m.mgf_logprice(1j * u, 1.0)
    assert np.max(np.abs(cf - mgf)) < 1e-14


def test_analytic_c1_matches_mgf_first_derivative():
    """_c1_logF must agree with d/du log(MGF)|_0 (FD)."""
    m = _make()
    for texp in (0.5, 1.0, 5.0, 10.0):
        c1_an = m._c1_logF(texp)
        eps = 1e-4
        K = lambda uu: float(np.log(m.mgf_logprice(uu, texp)).real)
        c1_num = (K(eps) - K(-eps)) / (2.0 * eps)
        assert abs(c1_num - c1_an) < 1e-7, f"texp={texp}: c1 mismatch {c1_num} vs {c1_an}"


def test_mgf_martingale_property():
    """E[S_T/F] = 1 under risk-neutral measure → mgf_logprice(1, texp) == 1.

    Independent of (intr, divr) since the drift sits in F, not in the CF.
    """
    for params in [{}, dict(intr=0.05, divr=0.02)]:
        m = _make(**params)
        for texp in (0.1, 1.0, 5.0):
            got = complex(m.mgf_logprice(1.0, texp))
            assert abs(got.imag) < 1e-12, "MGF(1) should be real"
            assert abs(got.real - 1.0) < 1e-12, f"texp={texp}: martingale violated ({got.real})"


# ─────────────────────────────────────────────────────────────────────────────
# 15-17. Input validation
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("bad", [
    dict(rho=1.5),
    dict(rho=-1.0),
    dict(rho=1.0),
    dict(vov=0.0),
    dict(vov=-0.1),
    dict(mr=0.0),
    dict(mr=-0.1),
    dict(theta=-0.01),
    dict(sigma=0.0),
    dict(sigma=-0.01),
])
def test_input_validation(bad):
    params = dict(PF_PARAMS)
    params.update(bad)
    with pytest.raises(ValueError):
        HestonCos(**params)


def test_theta_none_defaults_to_sigma():
    """theta=None must be accepted (SvABC sets self.theta = sigma)."""
    m = HestonCos(sigma=0.0175, vov=0.5, mr=1.5, rho=-0.5)
    assert m.theta == 0.0175


def test_invalid_texp_raises():
    m = _make()
    with pytest.raises(ValueError):
        m.price(STRIKE, SPOT, 0.0)
    with pytest.raises(ValueError):
        m.price(STRIKE, SPOT, -1.0)
    m.n_cos = 0
    with pytest.raises(ValueError):
        m.price(STRIKE, SPOT, 1.0)


def test_vector_spot_rejected():
    """Vector spot is not supported in this port; reject with a clear error."""
    m = _make()
    with pytest.raises(ValueError, match="scalar spot"):
        m.price(STRIKE, np.array([100.0, 105.0]), 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# 18-19. Cross-checks against validated reference and upstream FFT pricer
# ─────────────────────────────────────────────────────────────────────────────

def test_cross_check_vs_reference_pricer():
    """Match prices against src/cos_pricing/heston_cos_pricer.py (validated reference).

    This is the most load-bearing test: it directly verifies the PyFENG
    port did not drift numerically from the source.
    """
    cos_pricing = pytest.importorskip("cos_pricing")
    HestonCOSPricer = cos_pricing.HestonCOSPricer

    ref = HestonCOSPricer(
        S0=SPOT, v0=PF_PARAMS["sigma"], lam=PF_PARAMS["mr"],
        eta=PF_PARAMS["vov"], ubar=PF_PARAMS["theta"], rho=PF_PARAMS["rho"],
        r=0.0, q=0.0,
    )
    pf_pricer = _make()
    pf_pricer.n_cos = 256
    pf_pricer.L = None  # adaptive

    strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    for texp in (0.5, 1.0, 5.0):
        L_val = pf_pricer._resolve_L(texp)
        ref_px = ref.price(strikes, texp, cp=+1, N=256, L=L_val)
        pf_px  = pf_pricer.price(strikes, SPOT, texp, cp=+1)
        diff = float(np.max(np.abs(pf_px - ref_px)))
        assert diff < 1e-7, f"texp={texp}: PyFENG vs reference drift {diff:.2e}"


def test_cross_check_vs_heston_fft():
    """Cross-check vs pyfeng.HestonFft (same MGF, different transform). Skip if absent."""
    HestonFft = getattr(pf, "HestonFft", None)
    if HestonFft is None:
        pytest.skip("pyfeng.HestonFft not available in this install")

    fft = HestonFft(**PF_PARAMS)
    cos = _make()
    cos.n_cos = 256

    strikes = np.array([90.0, 100.0, 110.0])
    for texp in (0.5, 1.0):
        fft_px = fft.price(strikes, SPOT, texp)
        cos_px = cos.price(strikes, SPOT, texp)
        diff = float(np.max(np.abs(cos_px - fft_px)))
        assert diff < 1e-5, f"texp={texp}: COS vs FFT drift {diff:.2e}"


# ─────────────────────────────────────────────────────────────────────────────
# 20-22. Mixed cp, truncation interval, payoff coefficients
# ─────────────────────────────────────────────────────────────────────────────

def test_mixed_cp():
    """Vector cp (mixed call/put) must match per-element scalar calls."""
    m = _make()
    m.n_cos = 160
    strikes = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
    cp      = np.array([+1, -1, +1, -1, +1])
    mixed = m.price(strikes, SPOT, 1.0, cp=cp)
    scal  = np.array([
        m.price(float(K), SPOT, 1.0, cp=int(c))
        for K, c in zip(strikes, cp)
    ])
    assert mixed.shape == strikes.shape
    assert np.max(np.abs(mixed - scal)) < 1e-12


def test_truncation_interval_per_strike():
    """truncation_interval returns 4-tuple; midpoint is log(F/K)+c1 per strike."""
    m = _make()
    strikes = np.array([80.0, 100.0, 120.0])
    a, b, x, width = m.truncation_interval(strikes, SPOT, 1.0)
    assert a.shape == strikes.shape == b.shape == x.shape
    assert np.isscalar(width) or width.shape == ()
    fwd, _, _ = m._fwd_factor(SPOT, 1.0)
    expected_x = np.log(float(fwd) / strikes)
    assert np.max(np.abs(x - expected_x)) < 1e-14
    midpoint = 0.5 * (a + b)
    expected_center = expected_x + m._c1_logF(1.0)
    assert np.max(np.abs(midpoint - expected_center)) < 1e-12


def test_payoff_coefficients_static_helper():
    """payoff_coefficients returns U_k matrix of expected shape and finite values."""
    a = np.array([-1.0, -1.5])
    b = np.array([+1.0, +1.5])
    N = 16

    Uc = HestonCos.payoff_coefficients(N, a, b, cp=+1)
    Up = HestonCos.payoff_coefficients(N, a, b, cp=-1)

    assert Uc.shape == (2, N)
    assert Up.shape == (2, N)
    assert np.all(np.isfinite(Uc))
    assert np.all(np.isfinite(Up))
    # Sanity: call U_0 is positive (in-the-money intrinsic accumulation),
    # put  U_0 is positive too. Both are the average payoff over [a,b].
    assert np.all(Uc[:, 0] > 0)
    assert np.all(Up[:, 0] > 0)


# ─────────────────────────────────────────────────────────────────────────────
# 23-24. Numba warmup and NumPy fallback
# ─────────────────────────────────────────────────────────────────────────────

def test_warmup_numba_idempotent():
    """warmup_numba can be called repeatedly without error."""
    warmup_numba()
    warmup_numba()  # second call: no-op


def test_numpy_fallback_runs(tmp_path):
    """Module imports and prices correctly with NUMBA_DISABLE_JIT=1.

    Run as a subprocess so the env var change isolates from the parent
    test process (numba caches the JIT decision per-process).
    """
    snippet = textwrap.dedent("""
        import os, sys
        os.environ['NUMBA_DISABLE_JIT'] = '1'
        # Replicate the conftest sys.path fix so 'pyfeng' resolves correctly.
        repo = os.environ['REPO']
        sys.path[:] = [p for p in sys.path if os.path.abspath(p) != repo]
        try:
            import pyfeng
        except ImportError:
            print('SKIP_NO_PYFENG'); raise SystemExit(0)
        pyfeng_local = os.path.join(repo, 'pyfeng')
        if pyfeng_local not in pyfeng.__path__:
            pyfeng.__path__.append(pyfeng_local)

        from pyfeng.sv_heston_cos import HestonCos
        m = HestonCos(0.0175, vov=0.5751, mr=1.5768, theta=0.0398, rho=-0.5711)
        px = m.price(100.0, 100.0, 1.0)
        assert abs(px - 5.785155435) < 1e-5, px
        print('OK', px)
    """)
    script = tmp_path / "fallback_check.py"
    script.write_text(snippet)
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, NUMBA_DISABLE_JIT="1", REPO=repo)
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True, text=True, env=env, timeout=120,
    )
    if "SKIP_NO_PYFENG" in result.stdout:
        pytest.skip("pyfeng not installed in subprocess env")
    assert result.returncode == 0, (
        f"Subprocess failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert result.stdout.startswith("OK"), f"unexpected stdout: {result.stdout!r}"

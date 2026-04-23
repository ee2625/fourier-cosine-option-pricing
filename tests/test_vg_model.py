"""
Tests for the Variance Gamma model: CF properties, COS pricing (Table 7
reproduction), Carr-Madan FFT, and density recovery.

Reference:
    Fang F, Oosterlee CW (2008) SIAM J. Sci. Comput. 31(2):826-848, Table 7.

Run:
    pytest tests/test_vg_model.py -v
"""
import numpy as np
import pytest
from scipy.stats import linregress
from cos_pricing import VgModel, carr_madan_price

# ── Table 7 parameters (Eq. 55) ───────────────────────────────────────────────
S0, K, R, Q      = 100.0, 90.0, 0.1, 0.0
SIGMA, THETA, NU = 0.12, -0.14, 0.2
REF_T01          = 10.993703187
REF_T1           = 19.099354724

PAPER_T01 = {64: 1.66e-3, 128: 4.35e-4, 256: 4.55e-5, 512: 1.13e-6, 1024: 2.52e-8}
PAPER_T1  = {32: 6.57e-4, 64: 2.10e-6,  96: 3.32e-8, 128: 4.19e-10, 160: 1.88e-11}


@pytest.fixture
def model():
    return VgModel(sigma=SIGMA, theta=THETA, nu=NU, intr=R, divr=Q)


# ─────────────────────────────────────────────────────────────────────────────
class TestVgModel:
    """Characteristic function and model properties."""

    def test_cf_at_zero(self, model):
        assert abs(model.char_func(1.0)(0.0) - 1.0) < 1e-15

    def test_cf_symmetry(self, model):
        # phi(-u) = conj(phi(u)) for a real-valued distribution
        u  = np.array([0.5, 1.0, 2.0, 5.0])
        cf = model.char_func(1.0)
        np.testing.assert_allclose(cf(-u), np.conj(cf(u)), rtol=1e-14)

    def test_martingale_condition(self, model):
        # E[S_T / F] = 1  <=>  phi(-i) = 1
        assert abs(model.char_func(1.0)(np.complex128(-1j)) - 1.0) < 1e-14

    def test_cumulants(self, model):
        # Analytic c1, c2 must match numerical finite-difference cumulants
        cf  = model.char_func(1.0)
        eps = 1e-4
        mgf = lambda v: cf(-1j * v).real
        lm0    = np.log(mgf(0.0))
        c1_num = (np.log(mgf(eps)) - np.log(mgf(-eps))) / (2 * eps)
        c2_num = (np.log(mgf(eps)) + np.log(mgf(-eps)) - 2 * lm0) / eps ** 2

        sig2 = model.sigma ** 2
        w    = np.log(1 - model.theta * model.nu - 0.5 * sig2 * model.nu) / model.nu
        c1_an = w + model.theta
        c2_an = sig2 + model.nu * model.theta ** 2

        assert abs(c1_num - c1_an) < 1e-6
        assert abs(c2_num - c2_an) < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
class TestVgCos:
    """COS method: Table 7 reproduction and convergence rates."""

    def test_T1_reference(self, model):
        assert abs(model.price(K, S0, 1.0, n_cos=2**14) - REF_T1) < 1e-6

    def test_T01_reference(self, model):
        assert abs(model.price(K, S0, 0.1, n_cos=2**14) - REF_T01) < 1e-4

    def test_table7_reproduction(self, model):
        # Paper reports absolute errors; verify magnitudes are within 1.5 OOM.
        for n, paper_err in PAPER_T1.items():
            err = abs(model.price(K, S0, 1.0, n_cos=n) - REF_T1)
            assert abs(np.log10(err) - np.log10(paper_err)) < 1.5, \
                f"T=1.0 N={n}: magnitude mismatch  ours={err:.2e}  paper={paper_err:.2e}"
        for n, paper_err in PAPER_T01.items():
            err = abs(model.price(K, S0, 0.1, n_cos=n) - REF_T01)
            assert abs(np.log10(err) - np.log10(paper_err)) < 2.0, \
                f"T=0.1 N={n}: magnitude mismatch  ours={err:.2e}  paper={paper_err:.2e}"

    def test_T1_exponential_convergence(self, model):
        # log10(|error|) linear in N  =>  exponential (geometric) convergence
        ns   = list(PAPER_T1.keys())
        errs = [abs(model.price(K, S0, 1.0, n_cos=n) - REF_T1) for n in ns]
        slope, _, r, _, _ = linregress(ns, np.log10(errs))
        assert slope < 0,   "Errors not shrinking with N"
        assert r ** 2 > 0.95, f"Not exponential (R²={r**2:.3f})"

    def test_T01_algebraic_convergence(self, model):
        # Use a wider N range to smooth out sign-oscillation noise in the fit.
        # log10(|error|) should be roughly linear in log10(N) (algebraic decay).
        ns   = [32, 64, 128, 256, 512, 1024, 2048]
        errs = [abs(model.price(K, S0, 0.1, n_cos=n) - REF_T01) for n in ns]
        slope, _, r, _, _ = linregress(np.log10(ns), np.log10(errs))
        assert slope < 0, f"Errors not shrinking with N (slope={slope:.2f})"
        assert r ** 2 > 0.85, f"Not algebraic convergence (R²={r**2:.3f})"

    def test_put_call_parity(self, model):
        fwd, df  = model._fwd_df(S0, 1.0)
        strikes  = np.array([80., 90., 100., 110., 120.])
        call     = model.price(strikes, S0, 1.0, cp=1,  n_cos=256)
        put      = model.price(strikes, S0, 1.0, cp=-1, n_cos=256)
        np.testing.assert_allclose(call - put, df * (fwd - strikes), atol=1e-8)

    def test_multi_strike(self, model):
        strikes = np.array([80., 90., 100., 110., 120.])
        vec     = model.price(strikes, S0, 1.0, n_cos=128)
        scalar  = np.array([model.price(k, S0, 1.0, n_cos=128) for k in strikes])
        np.testing.assert_allclose(vec, scalar, rtol=1e-13)


# ─────────────────────────────────────────────────────────────────────────────
class TestVgCarrMadan:
    """Carr-Madan FFT: agreement with COS at high N."""

    def test_vs_cos_T1(self, model):
        fwd, df = model._fwd_df(S0, 1.0)
        cm  = carr_madan_price(model.char_func(1.0), 1.0, K, fwd, df, N=2**16)
        cos = model.price(K, S0, 1.0, n_cos=2**14)
        assert abs(cm - cos) < 1e-5

    def test_vs_cos_T01(self, model):
        fwd, df = model._fwd_df(S0, 0.1)
        cm  = carr_madan_price(model.char_func(0.1), 0.1, K, fwd, df, N=2**16)
        cos = model.price(K, S0, 0.1, n_cos=2**14)
        assert abs(cm - cos) < 1e-4

    def test_multi_strike(self, model):
        strikes = np.array([80., 90., 100., 110., 120.])
        fwd, df = model._fwd_df(S0, 1.0)
        cf      = model.char_func(1.0)
        vec     = carr_madan_price(cf, 1.0, strikes, fwd, df, N=2**16)
        scalar  = np.array([carr_madan_price(cf, 1.0, k, fwd, df, N=2**16) for k in strikes])
        np.testing.assert_allclose(vec, scalar, rtol=1e-10)


# ─────────────────────────────────────────────────────────────────────────────
class TestVgDensity:
    """Density recovery via COS expansion (Eq. 11 of the paper)."""

    def _recover(self, model, texp, n_pts=2**11, n_eval=1000):
        a, b  = model.trunc_range(texp)
        ba    = b - a
        k     = np.arange(n_pts)
        u     = k * np.pi / ba
        cf    = model.char_func(texp)
        Fk    = (2.0 / ba) * (cf(u) * np.exp(-1j * u * a)).real
        Fk[0] *= 0.5
        x     = np.linspace(a, b, n_eval)
        f     = np.cos(np.outer((x - a) * np.pi / ba, k)) @ Fk
        return x, f

    def test_integrates_to_one(self, model):
        for texp in [0.1, 1.0]:
            x, f = self._recover(model, texp)
            assert abs(np.trapezoid(f, x) - 1.0) < 0.01, f"T={texp}: integral ≠ 1"

    def test_negative_skew(self, model):
        # theta < 0 => left-skewed => E[X] < 0
        for texp in [0.1, 1.0]:
            x, f = self._recover(model, texp)
            assert np.trapezoid(x * f, x) < 0, f"T={texp}: expected negative mean"

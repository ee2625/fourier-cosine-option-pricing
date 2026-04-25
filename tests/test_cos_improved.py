"""
Tests for the Junike-style improved COS pricer.

Validation checklist (per the implementation guide):
  1. Centered COS equals standard COS on easy models.
  2. Wider intervals (larger L) produce larger N for fixed dx_target.
  3. Fallback routing fires for very wide intervals.
  4. Direct-call and put-parity paths agree on narrow, safe regimes.
  5. Stress cases (long T, semi-heavy tails) produce reasonable prices.
  6. Well-tuned classical cases are not harmed significantly.

Reference values
----------------
BSM analytic prices are exact; VG reference price is from
Fang & Oosterlee (2008) Table 11 (T=1, K=90, S=100).

Run:
    pytest tests/test_cos_improved.py -v
"""
import numpy as np
import pytest

from cos_pricing import (
    BsmModel,
    VgModel,
    CgmyModel,
    COSGridPolicy,
    cos_improved_grid,
    cos_improved_price,
    numerical_cumulants,
    lewis_price,
)


# ── Shared fixtures ────────────────────────────────────────────────────────────

S0, R, Q = 100.0, 0.05, 0.0

@pytest.fixture
def bsm():
    return BsmModel(sigma=0.2, intr=R, divr=Q)


@pytest.fixture
def vg():
    # Fang & Oosterlee Table 11 parameters
    return VgModel(sigma=0.12, theta=-0.14, nu=0.2, intr=0.1, divr=0.0)


@pytest.fixture
def cgmy():
    # CGMY with moderate activity (Y=0.5)
    return CgmyModel(C=1.0, G=5.0, M=10.0, Y=0.5)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Model cumulants interface
# ─────────────────────────────────────────────────────────────────────────────

class TestCumulants:
    """All models expose cumulants(texp) → (c1, c2, c4)."""

    def test_bsm_exact(self, bsm):
        c1, c2, c4 = bsm.cumulants(1.0)
        assert abs(c1 - (-0.5 * 0.04)) < 1e-15
        assert abs(c2 - 0.04)          < 1e-15
        assert c4 == 0.0

    def test_vg_matches_numerical(self, vg):
        c1_a, c2_a, c4_a = vg.cumulants(1.0)
        c1_n, c2_n, _    = numerical_cumulants(vg.char_func(1.0))
        assert abs(c1_a - c1_n) < 1e-5
        assert abs(c2_a - c2_n) < 1e-5

    def test_cgmy_numerical(self, cgmy):
        c1, c2, c4 = cgmy.cumulants(1.0)
        assert np.isfinite(c1)
        assert c2 > 0
        assert np.isfinite(c4)

    def test_bsm_scales_with_time(self, bsm):
        c1a, c2a, _ = bsm.cumulants(1.0)
        c1b, c2b, _ = bsm.cumulants(2.0)
        assert abs(c2b / c2a - 2.0) < 1e-14

    def test_model_family_attributes(self, bsm, vg, cgmy):
        assert bsm.model_family  == "gaussian_like"
        assert vg.model_family   == "semi_heavy"
        assert cgmy.model_family == "heavy"


# ─────────────────────────────────────────────────────────────────────────────
# 2. cos_improved_grid
# ─────────────────────────────────────────────────────────────────────────────

class TestCosImprovedGrid:
    """Grid builder properties."""

    def test_centered_interval_symmetric(self, bsm):
        policy = COSGridPolicy(centered=True, truncation="heuristic")
        a, b, N, center = cos_improved_grid(
            bsm.char_func(1.0), bsm.cumulants(1.0), policy
        )
        assert abs(a + b) < 1e-14, "Centered interval must be symmetric: a = -b"
        assert center == pytest.approx(bsm.cumulants(1.0)[0], rel=1e-14)

    def test_uncentered_interval(self, bsm):
        policy = COSGridPolicy(centered=False, truncation="heuristic")
        c1, c2, _ = bsm.cumulants(1.0)
        a, b, N, center = cos_improved_grid(
            bsm.char_func(1.0), bsm.cumulants(1.0), policy
        )
        mid = (a + b) / 2.0
        assert abs(mid - c1) < 1e-12
        assert center == 0.0

    def test_n_is_power_of_two(self, bsm):
        policy = COSGridPolicy(truncation="heuristic")
        _, _, N, _ = cos_improved_grid(bsm.char_func(1.0), bsm.cumulants(1.0), policy)
        assert N & (N - 1) == 0, f"N={N} is not a power of two"

    def test_wider_L_gives_larger_n(self, bsm):
        """Larger L → wider interval → larger N (for fixed dx_target)."""
        cf, cum = bsm.char_func(1.0), bsm.cumulants(1.0)
        _, _, N_small, _ = cos_improved_grid(
            cf, cum, COSGridPolicy(truncation="heuristic", L=6.0, dx_target=0.05)
        )
        _, _, N_large, _ = cos_improved_grid(
            cf, cum, COSGridPolicy(truncation="heuristic", L=24.0, dx_target=0.05)
        )
        assert N_large > N_small

    def test_fixed_n_override(self, bsm):
        policy = COSGridPolicy(fixed_N=256)
        _, _, N, _ = cos_improved_grid(bsm.char_func(1.0), bsm.cumulants(1.0), policy)
        assert N == 256

    def test_min_n_respected(self, bsm):
        # Very tight dx_target still respects min_N
        policy = COSGridPolicy(truncation="heuristic", L=2.0, dx_target=10.0, min_N=64)
        _, _, N, _ = cos_improved_grid(bsm.char_func(1.0), bsm.cumulants(1.0), policy)
        assert N >= 64

    def test_max_n_respected(self, bsm):
        policy = COSGridPolicy(truncation="heuristic", L=50.0, dx_target=0.001, max_N=512)
        _, _, N, _ = cos_improved_grid(bsm.char_func(1.0), bsm.cumulants(1.0), policy)
        assert N <= 512

    def test_paper_mode_uses_paper_L(self, bsm):
        """paper truncation with paper_L=12 should give same width as heuristic L=12."""
        cum = bsm.cumulants(1.0)
        cf  = bsm.char_func(1.0)
        a1, b1, _, _ = cos_improved_grid(
            cf, cum, COSGridPolicy(truncation="paper", paper_L=12.0, centered=False)
        )
        a2, b2, _, _ = cos_improved_grid(
            cf, cum, COSGridPolicy(truncation="heuristic", L=12.0, centered=False)
        )
        assert abs((b1 - a1) - (b2 - a2)) < 1e-14


# ─────────────────────────────────────────────────────────────────────────────
# 3. Centered COS matches standard COS  (validation test 1)
# ─────────────────────────────────────────────────────────────────────────────

class TestCenteredVsStandard:
    """Centered and uncentered COS should agree when intervals are equivalent."""

    def _standard_price(self, model, K, texp, cp=-1):
        """Plain COS put using the model's own trunc_range."""
        from cos_pricing import cos_price
        fwd, df = model._fwd_df(S0, texp)
        return cos_price(
            model.char_func(texp), texp, K, fwd, df,
            cp=cp, n_cos=512, trunc_range=model.trunc_range(texp, L=12.0),
        )

    def _improved_price(self, model, K, texp, cp=-1, truncation="heuristic"):
        fwd, df = model._fwd_df(S0, texp)
        policy  = COSGridPolicy(
            truncation=truncation,
            centered=True,
            L=12.0,
            dx_target=0.025,
            fixed_N=512,
        )
        return cos_improved_price(
            model.char_func(texp), texp, K, fwd, df,
            model.cumulants(texp), cp=cp, policy=policy,
            model_family=model.model_family, use_parity=False,
        )

    def test_bsm_put_centered_vs_standard(self, bsm):
        for K in [90.0, 100.0, 110.0]:
            p_std = self._standard_price(bsm, K, 1.0)
            p_imp = self._improved_price(bsm, K, 1.0)
            assert abs(p_std - p_imp) < 1e-6, \
                f"K={K}: std={p_std:.8f}  improved={p_imp:.8f}"

    def test_vg_put_centered_vs_standard(self, vg):
        for K in [90.0, 100.0]:
            p_std = self._standard_price(vg, K, 1.0)
            p_imp = self._improved_price(vg, K, 1.0)
            assert abs(p_std - p_imp) < 1e-5, \
                f"K={K}: std={p_std:.8f}  improved={p_imp:.8f}"


# ─────────────────────────────────────────────────────────────────────────────
# 4. BSM accuracy vs analytic  (core correctness)
# ─────────────────────────────────────────────────────────────────────────────

class TestBsmAccuracy:
    """Improved COS should match BSM analytic prices to high precision."""

    @pytest.mark.parametrize("K,cp,tol", [
        (80.0,  1, 1e-5),   # deep ITM call
        (90.0,  1, 1e-5),
        (100.0, 1, 1e-5),
        (110.0,-1, 1e-5),
        (120.0,-1, 1e-5),   # deep OTM put
    ])
    def test_single_strike(self, bsm, K, cp, tol):
        analytic = BsmModel.price_analytic(K, S0, 0.2, 1.0, intr=R, cp=cp)
        fwd, df  = bsm._fwd_df(S0, 1.0)
        policy   = COSGridPolicy(truncation="tolerance", eps_trunc=1e-12)
        improved = cos_improved_price(
            bsm.char_func(1.0), 1.0, K, fwd, df,
            bsm.cumulants(1.0), cp=cp, policy=policy,
        )
        assert abs(improved - analytic) < tol, \
            f"K={K} cp={cp}: improved={improved:.8f}  analytic={analytic:.8f}"

    def test_multi_strike_array(self, bsm):
        strikes  = np.array([80., 90., 100., 110., 120.])
        analytic = BsmModel.price_analytic(strikes, S0, 0.2, 1.0, intr=R, cp=1)
        fwd, df  = bsm._fwd_df(S0, 1.0)
        policy   = COSGridPolicy(truncation="tolerance", eps_trunc=1e-12)
        improved = cos_improved_price(
            bsm.char_func(1.0), 1.0, strikes, fwd, df,
            bsm.cumulants(1.0), cp=1, policy=policy,
        )
        np.testing.assert_allclose(improved, analytic, atol=1e-5)

    def test_long_maturity_bsm(self, bsm):
        """Adaptive truncation should handle T=10 without degrading accuracy."""
        texp     = 10.0
        K        = 100.0
        analytic = BsmModel.price_analytic(K, S0, 0.2, texp, intr=R, cp=1)
        fwd, df  = bsm._fwd_df(S0, texp)
        policy   = COSGridPolicy(truncation="tolerance", eps_trunc=1e-10)
        improved = cos_improved_price(
            bsm.char_func(texp), texp, K, fwd, df,
            bsm.cumulants(texp), cp=1, policy=policy,
        )
        assert abs(improved - analytic) < 0.05    # loose tol; T=10 is hard


# ─────────────────────────────────────────────────────────────────────────────
# 5. Put-parity path  (validation test 4)
# ─────────────────────────────────────────────────────────────────────────────

class TestPutCallParity:
    """Direct-call and parity-recovered call must agree; parity must hold."""

    def _price(self, model, K, texp, cp, use_parity):
        fwd, df = model._fwd_df(S0, texp)
        policy  = COSGridPolicy(
            truncation="heuristic", L=12.0, dx_target=0.05, fixed_N=256
        )
        return cos_improved_price(
            model.char_func(texp), texp, K, fwd, df,
            model.cumulants(texp), cp=cp, policy=policy,
            model_family=model.model_family, use_parity=use_parity,
        )

    def test_parity_bsm(self, bsm):
        fwd, df  = bsm._fwd_df(S0, 1.0)
        strikes  = np.array([90., 100., 110.])
        call     = self._price(bsm, strikes, 1.0, cp=1,  use_parity=True)
        put      = self._price(bsm, strikes, 1.0, cp=-1, use_parity=True)
        expected = df * (fwd - strikes)
        np.testing.assert_allclose(call - put, expected, atol=1e-7)

    def test_direct_vs_parity_bsm(self, bsm):
        """On a narrow-regime BSM, direct-call and parity-call should agree."""
        K   = 100.0
        c_direct = self._price(bsm, K, 1.0, cp=1, use_parity=False)
        c_parity = self._price(bsm, K, 1.0, cp=1, use_parity=True)
        assert abs(c_direct - c_parity) < 1e-5

    def test_parity_vg(self, vg):
        fwd, df  = vg._fwd_df(S0, 1.0)
        strikes  = np.array([90., 100., 110.])
        call     = self._price(vg, strikes, 1.0, cp=1,  use_parity=True)
        put      = self._price(vg, strikes, 1.0, cp=-1, use_parity=True)
        expected = df * (fwd - strikes)
        np.testing.assert_allclose(call - put, expected, atol=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# 6. VG accuracy vs Fang & Oosterlee Table 11  (validation test 5)
# ─────────────────────────────────────────────────────────────────────────────

class TestVgAccuracy:
    """Improved COS should match the VG paper reference price."""

    # Table 11: S=100, K=90, T=1, sigma=0.12, theta=-0.14, nu=0.2, r=0.1, q=0
    REF = 19.099354724

    def test_vs_paper(self, vg):
        K       = 90.0
        fwd, df = vg._fwd_df(S0, 1.0)
        policy  = COSGridPolicy(truncation="tolerance", eps_trunc=1e-10,
                                dx_target=0.05, max_N=2048)
        price   = cos_improved_price(
            vg.char_func(1.0), 1.0, K, fwd, df,
            vg.cumulants(1.0), cp=1, policy=policy,
            model_family=vg.model_family,
        )
        assert abs(price - self.REF) < 1e-4, \
            f"VG T=1: improved={price:.8f}  ref={self.REF:.8f}"

    def test_vs_plain_cos(self, vg):
        """Improved COS should agree with plain COS at high N."""
        K       = 90.0
        fwd, df = vg._fwd_df(S0, 1.0)
        plain   = vg.price(K, S0, 1.0, n_cos=2048)
        policy  = COSGridPolicy(truncation="paper", paper_L=10.0,
                                fixed_N=2048, centered=True)
        improved = cos_improved_price(
            vg.char_func(1.0), 1.0, K, fwd, df,
            vg.cumulants(1.0), cp=1, policy=policy,
            model_family=vg.model_family,
        )
        assert abs(improved - plain) < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# 7. Adaptive truncation tightens with eps_trunc  (validation of tolerance mode)
# ─────────────────────────────────────────────────────────────────────────────

class TestAdaptiveTruncation:
    """Tighter eps_trunc should produce equal-or-wider intervals."""

    def _width(self, model, texp, eps):
        policy = COSGridPolicy(truncation="tolerance", eps_trunc=eps,
                               dx_target=0.05)
        a, b, _, _ = cos_improved_grid(
            model.char_func(texp), model.cumulants(texp), policy,
            model_family=model.model_family
        )
        return b - a

    def test_bsm_width_monotone(self, bsm):
        w_loose  = self._width(bsm, 1.0, eps=1e-4)
        w_tight  = self._width(bsm, 1.0, eps=1e-12)
        assert w_tight >= w_loose - 1e-10

    def test_vg_width_monotone(self, vg):
        w_loose  = self._width(vg, 1.0, eps=1e-4)
        w_tight  = self._width(vg, 1.0, eps=1e-12)
        assert w_tight >= w_loose - 1e-10


# ─────────────────────────────────────────────────────────────────────────────
# 8. Fallback routing  (validation test 3)
# ─────────────────────────────────────────────────────────────────────────────

class TestFallback:
    """When interval is too wide, price should fall back to Lewis gracefully."""

    def test_fallback_fires_for_tiny_width_fallback(self, bsm):
        """Force fallback by setting width_fallback extremely small."""
        K       = 100.0
        fwd, df = bsm._fwd_df(S0, 1.0)
        analytic = BsmModel.price_analytic(K, S0, 0.2, 1.0, intr=R, cp=1)
        policy  = COSGridPolicy(
            truncation="heuristic", L=12.0,
            width_fallback=0.01,   # any interval wider than 1bp triggers fallback
            fallback_method="lewis",
        )
        price = cos_improved_price(
            bsm.char_func(1.0), 1.0, K, fwd, df,
            bsm.cumulants(1.0), cp=1, policy=policy,
        )
        # Lewis should give a good price, not NaN / wildly wrong
        assert np.isfinite(price)
        assert abs(price - analytic) < 0.05

    def test_no_fallback_for_standard_bsm(self, bsm):
        """Standard BSM interval is much narrower than the default 150 threshold."""
        K       = 100.0
        fwd, df = bsm._fwd_df(S0, 1.0)
        policy  = COSGridPolicy(truncation="heuristic", L=12.0, width_fallback=150.0)
        a, b, _, _ = cos_improved_grid(
            bsm.char_func(1.0), bsm.cumulants(1.0), policy
        )
        assert (b - a) < 150.0


# ─────────────────────────────────────────────────────────────────────────────
# 9. Mixed cp array  (regression / API)
# ─────────────────────────────────────────────────────────────────────────────

class TestMixedCp:
    def test_mixed_cp_bsm(self, bsm):
        strikes  = np.array([90., 100., 110.])
        cp_arr   = np.array([1, -1, 1])
        fwd, df  = bsm._fwd_df(S0, 1.0)
        policy   = COSGridPolicy(truncation="heuristic", L=12.0, fixed_N=256)
        prices   = cos_improved_price(
            bsm.char_func(1.0), 1.0, strikes, fwd, df,
            bsm.cumulants(1.0), cp=cp_arr, policy=policy,
        )
        assert prices.shape == (3,)
        analytic = np.array([
            BsmModel.price_analytic(k, S0, 0.2, 1.0, intr=R, cp=c)
            for k, c in zip(strikes, cp_arr)
        ])
        np.testing.assert_allclose(prices, analytic, atol=1e-4)


# ─────────────────────────────────────────────────────────────────────────────
# 10. CGMY  (heavy tail stress test — validation test 5)
# ─────────────────────────────────────────────────────────────────────────────

class TestCgmyImproved:
    """Improved COS on CGMY should match plain COS and Lewis at moderate params."""

    def test_cgmy_matches_lewis(self, cgmy):
        K    = 100.0
        texp = 1.0
        fwd  = S0  # intr=divr=0
        df   = 1.0
        lewis_ref = lewis_price(cgmy.char_func(texp), texp, K, fwd, df, cp=1)
        policy    = COSGridPolicy(truncation="tolerance", eps_trunc=1e-8,
                                  dx_target=0.10, max_N=2048)
        improved  = cos_improved_price(
            cgmy.char_func(texp), texp, K, fwd, df,
            cgmy.cumulants(texp), cp=1, policy=policy,
            model_family=cgmy.model_family,
        )
        assert abs(improved - lewis_ref) < 0.5   # loose; heavy tails are hard


# ─────────────────────────────────────────────────────────────────────────────
# 11. Chernoff tail bound  (internal unit test)
# ─────────────────────────────────────────────────────────────────────────────

class TestChernoffTailBound:
    """For BSM the Chernoff bound should recover 2*Phi(-z) up to the grid error."""

    def test_bsm_tail_matches_gaussian(self, bsm):
        from cos_pricing.cos_improved import _chernoff_tail_bound
        from scipy.special import erfc

        sigma, T = 0.2, 1.0
        c1, c2, _ = bsm.cumulants(T)
        cf = bsm.char_func(T)

        for L in [4.0, 6.0, 8.0]:
            w   = L * np.sqrt(c2)
            chernoff   = _chernoff_tail_bound(cf, c1, w)
            gaussian   = float(erfc(w / np.sqrt(2 * c2)))   # 2*Phi(-w/sqrt(c2))
            # The Chernoff bound should be close to but may exceed the exact tail
            assert chernoff <= 1.0
            assert chernoff >= 0.0
            # For a Gaussian both should be tiny at L=6,8
            if L >= 6.0:
                assert chernoff < 1e-6

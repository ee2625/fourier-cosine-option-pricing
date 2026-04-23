"""
One-shot builder for notebooks/tests.ipynb.

Generates a single well-documented notebook that organises every test in
tests/ into thematic sections with narrative markdown between groups.
Delete after running; the notebook is the deliverable.
"""
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text))


# ─────────────────────────────────────────────────────────────────────────────
# 0. Front matter
# ─────────────────────────────────────────────────────────────────────────────

md(r"""# `cos_pricing` — test suite, organised

This notebook consolidates every test in `tests/` into one runnable,
narrative document. Each section corresponds to one test file and the
cells below each heading translate the pytest functions into inline
assertions — fixtures become module-level variables, `@pytest.mark.parametrize`
becomes explicit loops, `pytest.raises` becomes `try`/`except`.

Running the notebook top-to-bottom is equivalent to running
`pytest tests/ -v`: if a cell raises, that test has failed; if it finishes
silently, it has passed.

## Source map

| Notebook section | Source file | Focus |
|---|---|---|
| 1 — BSM COS method | [`test_cos_method.py`](../tests/test_cos_method.py) | Black–Scholes COS correctness & model-agnostic engine |
| 2 — Heston COS pricer | [`test_heston_cos_pricer.py`](../tests/test_heston_cos_pricer.py) | Paper benchmarks, convergence, input validation |
| 3 — Variance Gamma | [`test_vg_model.py`](../tests/test_vg_model.py) | CF properties, Table 7 reproduction, Carr–Madan, density |
| 4.1–4.2 — Dimensional invariance | [`test_dimensional_invariance.py`](../tests/test_dimensional_invariance.py) | BSM scale + Bachelier translation |
| 4.3 — Parametrised pi across models | [`test_buckingham_pi.py`](../tests/test_buckingham_pi.py) | Unified symmetry check over every pricer |
| 4.4–4.5 — Heston temporal pi | [`test_heston_temporal_invariance.py`](../tests/test_heston_temporal_invariance.py) | Seventh Heston pi-group + joint spatial×temporal |
""")

md(r"""## Setup

`cos_pricing` lives in `src/`; we put `src/` on the path so the notebook
can be run from the repo root without `pip install -e .` or
`PYTHONPATH=src`. All tests share these imports.""")

code(r"""import os, sys
# Repo root = parent of this notebooks/ directory.
ROOT = os.path.abspath(os.path.join(os.getcwd(), ".." if os.path.basename(os.getcwd()) == "notebooks" else "."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np
from scipy.stats import linregress

from cos_pricing import (
    BsmModel, NormalCos, HestonCOSPricer, VgModel,
    bsm_price, cos_price, carr_madan_price,
)

print("cos_pricing imported OK")""")


# ─────────────────────────────────────────────────────────────────────────────
# 1. BSM COS method
# ─────────────────────────────────────────────────────────────────────────────

md(r"""---
# 1. BSM COS method &nbsp; · &nbsp; `test_cos_method.py`

The BSM case has a closed form, so these tests anchor the COS machinery
against an analytic reference at < 1e-10 tolerance. We verify:

- single-strike calls and puts,
- vectorised strikes and mixed `cp` arrays,
- put-call parity,
- convergence in `n_cos`,
- input/output shape contracts,
- deep ITM / deep OTM limits,

then repeat two sanity checks directly on the model-agnostic `cos_price`
engine with a hand-built characteristic function.""")

code(r"""# Fixture -> module variable.
bsm_params = dict(sigma=0.2, intr=0.05, divr=0.1)""")

md(r"""## 1.1 &nbsp; Analytic agreement: COS vs closed-form BSM

Three baseline checks — a single ATM call, a slightly-OTM put, and a
vectorised sweep across five strikes — all demanding `|err| < 1e-10`
against `bsm_price`.""")

code(r"""m = BsmModel(**bsm_params)

# test_single_call
cos_val = m.price(100.0, 100.0, 1.0, cp=1)
ref     = bsm_price(100.0, 100.0, bsm_params["sigma"], 1.0,
                    bsm_params["intr"], bsm_params["divr"], cp=1)
assert abs(cos_val - ref) < 1e-10

# test_single_put
cos_val = m.price(105.0, 100.0, 1.0, cp=-1)
ref     = bsm_price(105.0, 100.0, bsm_params["sigma"], 1.0,
                    bsm_params["intr"], bsm_params["divr"], cp=-1)
assert abs(cos_val - ref) < 1e-10

# test_strike_array
strikes = np.arange(80, 121, 10, dtype=float)
cos_vals = m.price(strikes, 100.0, 1.2)
refs     = bsm_price(strikes, 100.0, bsm_params["sigma"], 1.2,
                     bsm_params["intr"], bsm_params["divr"])
assert np.max(np.abs(cos_vals - refs)) < 1e-10
print("BSM COS matches closed-form to < 1e-10")""")

md(r"""## 1.2 &nbsp; Put–call parity and mixed `cp` broadcasting

Parity `C − P = df · (F − K)` is a cross-check that decouples BSM
correctness from the COS pricer. The second test feeds a mixed
`cp` array (some puts, some calls) to verify broadcasting rules.""")

code(r"""m = BsmModel(**bsm_params)

# test_put_call_parity
for strike in [90.0, 100.0, 110.0]:
    c   = m.price(strike, 100.0, 1.0, cp=1)
    p   = m.price(strike, 100.0, 1.0, cp=-1)
    fwd = 100.0 * np.exp((bsm_params["intr"] - bsm_params["divr"]) * 1.0)
    df  = np.exp(-bsm_params["intr"] * 1.0)
    assert abs((c - p) - df * (fwd - strike)) < 1e-10

# test_mixed_cp_array
strikes = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
cp      = np.array([-1,   -1,    1,     1,     1])
cos_vals = m.price(strikes, 100.0, 1.0, cp=cp)
refs     = bsm_price(strikes, 100.0, bsm_params["sigma"], 1.0,
                     bsm_params["intr"], bsm_params["divr"], cp=cp)
assert np.max(np.abs(cos_vals - refs)) < 1e-10
print("put-call parity & mixed cp broadcasting OK")""")

md(r"""## 1.3 &nbsp; Convergence, shape contracts, and deep ITM/OTM limits

`n_cos=64` is enough for machine-precision on BSM; `n_cos=8` should be
visibly wrong. Scalar input → scalar output, array input → array.
Deep-OTM call ≈ 0; deep-ITM call ≈ discounted `(F − K)`.""")

code(r"""m = BsmModel(**bsm_params)

# test_convergence_in_n
ref = bsm_price(100, 100, bsm_params["sigma"], 1.0,
                bsm_params["intr"], bsm_params["divr"])
assert abs(m.price(100, 100, 1.0, n_cos=8)  - ref) > 1e-3    # too few
assert abs(m.price(100, 100, 1.0, n_cos=64) - ref) < 1e-10   # converged

# test_scalar_output
assert isinstance(m.price(100.0, 100.0, 1.0), float)

# test_array_output_shape
K = np.linspace(90, 110, 7)
assert m.price(K, 100.0, 1.0).shape == (7,)

# test_otm_deep_call  (K=200, S=100, sigma=0.2, T=1)
p = m.price(200.0, 100.0, 1.0, cp=1)
assert 0.0 <= p < 0.01

# test_itm_deep_call
fwd = 100.0 * np.exp((bsm_params["intr"] - bsm_params["divr"]) * 1.0)
df  = np.exp(-bsm_params["intr"] * 1.0)
p   = m.price(50.0, 100.0, 1.0, cp=1)
assert abs(p - df * (fwd - 50.0)) < 0.01
print("convergence, shapes, and deep-strike limits OK")""")

md(r"""## 1.4 &nbsp; Model-agnostic COS engine

`cos_price` takes *any* characteristic function, not just BSM's. We
feed it a hand-written BSM CF and an explicit `trunc_range` to verify
the engine is genuinely model-agnostic.""")

code(r"""# test_custom_char_func
sigma, texp = 0.3, 1.0
fwd, df     = 100.0, 1.0
def cf(u):
    return np.exp(-0.5 * sigma**2 * texp * u * (u + 1j))
p_cos = cos_price(cf, texp, 100.0, fwd, df, cp=1, n_cos=128,
                  trunc_range=(-5.0, 5.0))
p_ref = bsm_price(100.0, 100.0, sigma, texp)
assert abs(p_cos - p_ref) < 1e-6

# test_explicit_trunc_range
m  = BsmModel(sigma=0.2)
cf = m.char_func(1.0)
p  = cos_price(cf, 1.0, 100.0, 100.0, 1.0, cp=1,
               n_cos=128, trunc_range=(-3.0, 3.0))
ref = bsm_price(100.0, 100.0, 0.2, 1.0)
assert abs(p - ref) < 1e-6
print("cos_price engine works with custom CF and explicit truncation")""")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Heston COS pricer
# ─────────────────────────────────────────────────────────────────────────────

md(r"""---
# 2. Heston COS pricer &nbsp; · &nbsp; `test_heston_cos_pricer.py`

Heston has no closed form, so the benchmarks here come from
Fang & Oosterlee (2008) — two reference prices at T=1 and T=10 quoted
to ≥ 9 significant digits. Then convergence in `N`, sensitivity to the
truncation half-width `L`, structural sanity (put-call parity, shapes,
non-negativity), the static `chi`/`psi` helpers, and input validation.

Parameters below are the paper's canonical set (Eq. 55, `r = q = 0`).""")

code(r"""PAPER_PARAMS = dict(
    S0=100.0, v0=0.0175, lam=1.5768, eta=0.5751, ubar=0.0398, rho=-0.5711,
    r=0.0, q=0.0,
)
REF_T1  = 5.785155435
REF_T10 = 22.318945791474590""")

md(r"""## 2.1 &nbsp; Paper benchmarks

Reproduce the two reference prices to the stated precision.""")

code(r"""m = HestonCOSPricer(**PAPER_PARAMS)

# test_benchmark_T1
assert abs(m.price_call(100.0, 1.0,  N=160) - REF_T1)  < 1e-5
# test_benchmark_T10
assert abs(m.price_call(100.0, 10.0, N=160) - REF_T10) < 1e-6
print("Heston paper benchmarks T=1 and T=10 reproduced")""")

md(r"""## 2.2 &nbsp; Convergence in `N` and truncation half-width `L`

Each refinement in `N` must shrink the error (modulo a small plateau
near machine precision). With `L=3` the truncation of the integration
interval dominates; with the default `L=12` the error should be
three-plus orders of magnitude smaller and < 1e-6.""")

code(r"""m = HestonCOSPricer(**PAPER_PARAMS)

# test_convergence_in_N
errs = [abs(m.price_call(100.0, 1.0, N=N) - REF_T1)
        for N in (16, 32, 64, 128, 256)]
for prev, curr in zip(errs, errs[1:]):
    assert curr <= prev * 1.1 + 1e-12
assert errs[-1] < 1e-6

# test_L_sensitivity
err_small_L = abs(m.price_call(100.0, 1.0, N=160, L=3.0)  - REF_T1)
err_good_L  = abs(m.price_call(100.0, 1.0, N=160, L=12.0) - REF_T1)
assert err_small_L > 1e3 * err_good_L
assert err_good_L  < 1e-6
print("convergence in N and L sensitivity OK")""")

md(r"""## 2.3 &nbsp; Structural correctness

- **Put–call parity** to ≈ 1e-4 (tail of `exp(y)·f(y)` at default `L=12`).
- **Strike vectorisation** matches scalar calls bit-for-bit.
- **Scalar in → scalar out** and **prices ≥ 0** on a wide strike grid.""")

code(r"""m = HestonCOSPricer(**PAPER_PARAMS)

# test_put_call_parity
tau = 1.0
df  = np.exp(-m.r * tau)
fwd = m.S0 * np.exp((m.r - m.q) * tau)
for K in (80.0, 100.0, 120.0):
    c = m.price_call(K, tau, N=160)
    p = m.price_put (K, tau, N=160)
    assert abs((c - p) - df * (fwd - K)) < 1e-4

# test_strike_vectorized_matches_scalar
strikes = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
vec  = m.price_call(strikes, 1.0, N=160)
scal = np.array([m.price_call(float(K), 1.0, N=160) for K in strikes])
assert vec.shape == strikes.shape
assert np.max(np.abs(vec - scal)) < 1e-12

# test_scalar_in_scalar_out
assert isinstance(m.price_call(100.0, 1.0), float)
assert isinstance(m.price_put (100.0, 1.0), float)

# test_non_negative_prices
prices = m.price_call(np.linspace(60, 200, 15), 1.0, N=160)
assert np.all(prices >= -1e-10)
print("structural correctness: parity, vectorisation, shapes, positivity")""")

md(r"""## 2.4 &nbsp; Static helpers `chi` and `psi`

Shape & edge-case sanity for the two analytic integrals that sit behind
the cosine-expansion payoff coefficients.""")

code(r"""# test_chi_psi_broadcasting
k = np.arange(8)
a, b, c, d = -2.0, 3.0, 0.0, 3.0
chi_v = HestonCOSPricer.chi(k, a, b, c, d)
psi_v = HestonCOSPricer.psi(k, a, b, c, d)
assert chi_v.shape == (8,)
assert psi_v.shape == (8,)
assert abs(psi_v[0] - (d - c)) < 1e-14     # psi at k=0 == d-c
print("chi/psi shape + psi(k=0) identity OK")""")

md(r"""## 2.5 &nbsp; Input validation

Every parameter has an admissibility region; violating any of them
should raise `ValueError` at construction. Likewise, non-positive `tau`
or `N` should be rejected at pricing time.""")

code(r"""bad_configs = [
    dict(rho=1.5),   dict(rho=-1.0),
    dict(eta=0.0),   dict(eta=-0.1),
    dict(lam=0.0),   dict(ubar=-0.01),
    dict(v0=0.0),
]
for bad in bad_configs:
    params = dict(PAPER_PARAMS); params.update(bad)
    try:
        HestonCOSPricer(**params)
    except ValueError:
        continue
    raise AssertionError(f"expected ValueError for {bad}")

# test_invalid_tau_raises
m = HestonCOSPricer(**PAPER_PARAMS)
for bad_call in (
    lambda: m.price_call(100.0, 0.0),
    lambda: m.price_call(100.0, 1.0, N=0),
):
    try:
        bad_call()
    except ValueError:
        continue
    raise AssertionError("expected ValueError")
print("input validation: all bad configs rejected")""")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Variance Gamma
# ─────────────────────────────────────────────────────────────────────────────

md(r"""---
# 3. Variance Gamma &nbsp; · &nbsp; `test_vg_model.py`

VG prices are verified against Fang & Oosterlee (2008) Table 7 at two
maturities: T=1 shows geometric (exponential in `N`) convergence; T=0.1
is algebraic because the density is less smooth at short times. We
also cross-check COS against Carr–Madan FFT and recover the density
directly from the cosine coefficients.

Parameters come from Eq. 55 of the paper.""")

code(r"""# Table 7 parameters
S0, K, R, Q      = 100.0, 90.0, 0.1, 0.0
SIGMA, THETA, NU = 0.12, -0.14, 0.2
REF_T01          = 10.993703187
REF_T1           = 19.099354724

PAPER_T01 = {64: 1.66e-3, 128: 4.35e-4, 256: 4.55e-5, 512: 1.13e-6, 1024: 2.52e-8}
PAPER_T1  = {32: 6.57e-4, 64: 2.10e-6,  96: 3.32e-8, 128: 4.19e-10, 160: 1.88e-11}

vg = VgModel(sigma=SIGMA, theta=THETA, nu=NU, intr=R, divr=Q)""")

md(r"""## 3.1 &nbsp; Characteristic-function properties

Four invariants that any valid CF must satisfy:

1. `phi(0) = 1` — probability integrates to one.
2. `phi(-u) = conj(phi(u))` — real-valued log-return distribution.
3. `phi(-i) = 1` — martingale condition on `S_T / F`.
4. Analytic `c1`, `c2` match finite-difference cumulants of `log MGF`.""")

code(r"""# test_cf_at_zero
assert abs(vg.char_func(1.0)(0.0) - 1.0) < 1e-15

# test_cf_symmetry
u  = np.array([0.5, 1.0, 2.0, 5.0])
cf = vg.char_func(1.0)
np.testing.assert_allclose(cf(-u), np.conj(cf(u)), rtol=1e-14)

# test_martingale_condition
assert abs(vg.char_func(1.0)(np.complex128(-1j)) - 1.0) < 1e-14

# test_cumulants
cf  = vg.char_func(1.0)
eps = 1e-4
mgf = lambda v: cf(-1j * v).real
lm0    = np.log(mgf(0.0))
c1_num = (np.log(mgf(eps)) - np.log(mgf(-eps))) / (2 * eps)
c2_num = (np.log(mgf(eps)) + np.log(mgf(-eps)) - 2 * lm0) / eps ** 2

sig2 = vg.sigma ** 2
w    = np.log(1 - vg.theta * vg.nu - 0.5 * sig2 * vg.nu) / vg.nu
c1_an = w + vg.theta
c2_an = sig2 + vg.nu * vg.theta ** 2
assert abs(c1_num - c1_an) < 1e-6
assert abs(c2_num - c2_an) < 1e-6
print("VG CF properties: normalisation, conjugate symmetry, martingale, cumulants")""")

md(r"""## 3.2 &nbsp; COS — Table 7 reproduction and convergence orders

Paper errors are recovered within 1.5–2.0 orders of magnitude at every
`N`. At T=1 the slope of `log10(err)` vs `N` is negative with
`R² > 0.95` — geometric convergence. At T=0.1 convergence is algebraic:
`log10(err)` is linear in `log10(N)` with `R² > 0.85`.""")

code(r"""# test_T1_reference
assert abs(vg.price(K, S0, 1.0, n_cos=2**14) - REF_T1)  < 1e-6
# test_T01_reference
assert abs(vg.price(K, S0, 0.1, n_cos=2**14) - REF_T01) < 1e-4

# test_table7_reproduction
for n, paper_err in PAPER_T1.items():
    err = abs(vg.price(K, S0, 1.0, n_cos=n) - REF_T1)
    assert abs(np.log10(err) - np.log10(paper_err)) < 1.5
for n, paper_err in PAPER_T01.items():
    err = abs(vg.price(K, S0, 0.1, n_cos=n) - REF_T01)
    assert abs(np.log10(err) - np.log10(paper_err)) < 2.0

# test_T1_exponential_convergence
ns   = list(PAPER_T1.keys())
errs = [abs(vg.price(K, S0, 1.0, n_cos=n) - REF_T1) for n in ns]
slope, _, r, _, _ = linregress(ns, np.log10(errs))
assert slope < 0
assert r ** 2 > 0.95

# test_T01_algebraic_convergence
ns   = [32, 64, 128, 256, 512, 1024, 2048]
errs = [abs(vg.price(K, S0, 0.1, n_cos=n) - REF_T01) for n in ns]
slope, _, r, _, _ = linregress(np.log10(ns), np.log10(errs))
assert slope < 0
assert r ** 2 > 0.85

# test_put_call_parity + test_multi_strike
fwd, df = vg._fwd_df(S0, 1.0)
strikes = np.array([80., 90., 100., 110., 120.])
call    = vg.price(strikes, S0, 1.0, cp=1,  n_cos=256)
put     = vg.price(strikes, S0, 1.0, cp=-1, n_cos=256)
np.testing.assert_allclose(call - put, df * (fwd - strikes), atol=1e-8)
vec     = vg.price(strikes, S0, 1.0, n_cos=128)
scalar  = np.array([vg.price(k, S0, 1.0, n_cos=128) for k in strikes])
np.testing.assert_allclose(vec, scalar, rtol=1e-13)
print("VG COS: Table 7 reproduced, convergence orders confirmed")""")

md(r"""## 3.3 &nbsp; Carr–Madan FFT cross-check

An independent pricer (Carr–Madan dampened FFT) should agree with COS
at high `N` to within the quoted tolerances — same CF, different
transform.""")

code(r"""# test_vs_cos_T1
fwd, df = vg._fwd_df(S0, 1.0)
cm  = carr_madan_price(vg.char_func(1.0), 1.0, K, fwd, df, N=2**16)
cos = vg.price(K, S0, 1.0, n_cos=2**14)
assert abs(cm - cos) < 1e-5

# test_vs_cos_T01
fwd, df = vg._fwd_df(S0, 0.1)
cm  = carr_madan_price(vg.char_func(0.1), 0.1, K, fwd, df, N=2**16)
cos = vg.price(K, S0, 0.1, n_cos=2**14)
assert abs(cm - cos) < 1e-4

# test_multi_strike
strikes = np.array([80., 90., 100., 110., 120.])
fwd, df = vg._fwd_df(S0, 1.0)
cf      = vg.char_func(1.0)
vec     = carr_madan_price(cf, 1.0, strikes, fwd, df, N=2**16)
scalar  = np.array([carr_madan_price(cf, 1.0, k, fwd, df, N=2**16) for k in strikes])
np.testing.assert_allclose(vec, scalar, rtol=1e-10)
print("Carr-Madan FFT agrees with COS at high N")""")

md(r"""## 3.4 &nbsp; Density recovery

The cosine coefficients `F_k` of the CF are themselves a valid series
representation of the density on `[a, b]` (Eq. 11 of the paper). We
check that the recovered density integrates to ≈ 1 and, since
`theta < 0`, has a negative mean.""")

code(r"""def recover_density(model, texp, n_pts=2**11, n_eval=1000):
    a, b = model.trunc_range(texp)
    ba   = b - a
    k    = np.arange(n_pts)
    u    = k * np.pi / ba
    cf   = model.char_func(texp)
    Fk   = (2.0 / ba) * (cf(u) * np.exp(-1j * u * a)).real
    Fk[0] *= 0.5
    x    = np.linspace(a, b, n_eval)
    f    = np.cos(np.outer((x - a) * np.pi / ba, k)) @ Fk
    return x, f

for texp in [0.1, 1.0]:
    x, f = recover_density(vg, texp)
    assert abs(np.trapezoid(f, x) - 1.0) < 0.01        # integrates to 1
    assert np.trapezoid(x * f, x) < 0                  # theta < 0 => E[X] < 0
print("VG density: normalised and negatively-skewed, as expected")""")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Dimensional (Buckingham pi) invariance
# ─────────────────────────────────────────────────────────────────────────────

md(r"""---
# 4. Dimensional (Buckingham π) invariance

Every pricer in `cos_pricing` inherits dimensional structure from its
underlying SDE. The Buckingham π theorem tells us how many independent
dimensionless groups govern the price; each group corresponds to a
symmetry that the numerical pricer *must* preserve. Breaking one is
evidence of a bug in units, truncation, or discretisation.

| Model | Inputs | Units | π-groups | Tested symmetry |
|---|---|---|---|---|
| BSM | `(S, K, σ, T, r, q)` | `[$]` and `[T]` | 3 | Scale in `(S, K)` |
| Bachelier | `(F, K, σ_abs, T, r, q)` | `[$]` and `[T]` | 2 | Translation in `(F, K)` |
| Heston | `(S, K, T, r, q, v0, ūbar, κ, η, ρ)` | `[$]` and `[T]` (ρ dimensionless) | 8 | Scale + 6 others; temporal is tested below |

These tests are model-independent — they need no analytic reference and
catch bugs that closed-form cross-checks would miss.""")

md(r"""## 4.1 &nbsp; BSM scale invariance &nbsp; · &nbsp; `test_dimensional_invariance.py`

`C(λS, λK) = λ · C(S, K)` — scaling numeraire units leaves the
dimensionless price `C/S` unchanged. Tolerance: 1e-10 on `C/S`.""")

code(r"""LAMBDAS = [0.1, 0.5, 2.0, 10.0, 100.0]
SPOT    = 100.0
STRIKES = np.array([70.0, 85.0, 100.0, 115.0, 130.0])
TEXPS   = [0.1, 1.0, 5.0]
TOL     = 1e-10

model = BsmModel(sigma=0.25, intr=0.05, divr=0.02)

for lam in LAMBDAS:
    for cp in (+1, -1):
        worst = 0.0
        for texp in TEXPS:
            base   = model.price(STRIKES,        SPOT,       texp, cp=cp)
            scaled = model.price(STRIKES * lam,  SPOT * lam, texp, cp=cp)
            worst  = max(worst, float(np.max(np.abs(scaled / SPOT / lam - base / SPOT))))
        assert worst < TOL, f"BSM scale-invariance violated at lam={lam}, cp={cp}: {worst:.2e}"
print("BSM scale invariance: C(lam*S, lam*K) = lam * C(S, K)  -- OK across all (lam, cp, T, K)")""")

md(r"""## 4.2 &nbsp; Bachelier translation invariance

Bachelier prices depend on `(F, K)` only through the difference `K − F`;
shifting both by λ is a bit-identical operation internally (`k* = K − F`
is the sole strike input), so the error should be exactly 0, not merely
< 1e-10.

Under `r ≠ q`, the forward moves as `F = S · exp((r−q)T)`, so to
translate `F` by λ we translate the spot by `λ / exp((r−q)T)`.""")

code(r"""LAMBDAS_T = LAMBDAS + [-5.0, -30.0]   # Bachelier is bidirectional

model = NormalCos(sigma=25.0, intr=0.05, divr=0.02)

for lam in LAMBDAS_T:
    for cp in (+1, -1):
        worst = 0.0
        for texp in TEXPS:
            carry      = np.exp((model.intr - model.divr) * texp)
            spot_shift = lam / carry
            base    = model.price(STRIKES,       SPOT,              texp, cp=cp)
            shifted = model.price(STRIKES + lam, SPOT + spot_shift, texp, cp=cp)
            worst   = max(worst, float(np.max(np.abs(shifted - base))))
        assert worst < TOL, f"Bachelier translation-invariance violated at lam={lam}, cp={cp}: {worst:.2e}"
print("Bachelier translation invariance: C(F+lam, K+lam) = C(F, K)  -- OK")""")

md(r"""## 4.3 &nbsp; Parametrised π-test across all models &nbsp; · &nbsp; `test_buckingham_pi.py`

Same symmetries, unified test harness. BSM and Heston share scale
invariance; Bachelier has translation. Adding a new pricer to the
`PRICERS` list below would extend the sweep automatically.""")

code(r"""def _bsm_price(spot, strike, texp, cp):
    return BsmModel(sigma=0.25).price(strike, spot, texp, cp=cp)

def _heston_price(spot, strike, texp, cp):
    m = HestonCOSPricer(S0=spot, v0=0.04, lam=1.5, ubar=0.04, eta=0.5, rho=-0.5)
    return m.price_call(strike, tau=texp) if cp > 0 else m.price_put(strike, tau=texp)

def _normal_price(spot, strike, texp, cp):
    return NormalCos(sigma=25.0).price(strike, spot, texp, cp=cp)

PRICERS = [
    ("BsmModel",        _bsm_price,    "scale"),
    ("HestonCOSPricer", _heston_price, "scale"),
    ("NormalCos",       _normal_price, "translation"),
]

SPOT_PI    = 100.0
STRIKES_PI = np.array([85.0, 100.0, 115.0])
TEXPS_PI   = [0.5, 1.0, 2.0]
LAMBDAS_PI = [0.1, 0.5, 2.0, 10.0, 100.0]

for name, pricer, symmetry in PRICERS:
    for lam in LAMBDAS_PI:
        worst = 0.0
        for cp in (+1, -1):
            for texp in TEXPS_PI:
                if symmetry == "scale":
                    base  = np.array([pricer(SPOT_PI,       K,       texp, cp) for K in STRIKES_PI])
                    other = np.array([pricer(SPOT_PI * lam, K * lam, texp, cp) for K in STRIKES_PI])
                    err   = np.abs(other / (SPOT_PI * lam) - base / SPOT_PI)
                else:  # translation
                    base  = np.array([pricer(SPOT_PI,       K,       texp, cp) for K in STRIKES_PI])
                    other = np.array([pricer(SPOT_PI + lam, K + lam, texp, cp) for K in STRIKES_PI])
                    err   = np.abs(other - base)
                worst = max(worst, float(err.max()))
        assert worst < TOL, f"{name}/{symmetry} violated at lam={lam}: {worst:.2e}"
    print(f"  {name:<17s} {symmetry:<12s} -- OK across all lambda")""")

md(r"""## 4.4 &nbsp; Heston temporal π-invariance &nbsp; · &nbsp; `test_heston_temporal_invariance.py`

Heston has 10 dimensional inputs in 2 base units → 8 π-groups. BSM-style
scale invariance is one. The **temporal** symmetry is the strongest of
the other seven — it rotates seven parameters simultaneously:

$$
(T,\ r,\ q,\ \kappa,\ \eta,\ v_0,\ \bar u)
\ \longrightarrow\
(\mu T,\ r/\mu,\ q/\mu,\ \kappa/\mu,\ \eta/\mu,\ v_0/\mu,\ \bar u/\mu),
\qquad \rho, S_0, K\ \text{unchanged}.
$$

**Proof sketch.** Substitute `t = μτ'` in the Heston SDE and rescale the
Brownians by `√μ`. The variance and spot SDEs come out identical in
`τ'`-time with the *original* parameters. Discount: `exp(−rT) =
exp(−(r/μ)(μT))`. Hence prices match.

**Implementation note.** `HestonCOSPricer`'s heuristic half-width
`L · √(ū + v₀·η)` and the default `L(τ)` are not individually invariant
under this rescaling — only their product needs to be "wide enough". We
pin both pricers to the *same* log-moneyness half-width so `a, b`, the
frequency grid, the CF samples and the payoff `U_k` are bit-identical
between base and rescaled runs. Result: agreement at ~ machine epsilon,
not just < 1e-10.""")

code(r"""MUS        = [0.1, 0.5, 2.0, 10.0, 100.0]
STRIKES_H  = np.array([70.0, 85.0, 100.0, 115.0, 130.0])
TAU_BASE   = 1.0
N_COS      = 256
HALF_WIDTH = 12.0                      # log-moneyness truncation

BASE = dict(S0=100.0, v0=0.04, lam=1.5, eta=0.4, ubar=0.04, rho=-0.7, r=0.03, q=0.01)

def _rescale(params, mu):
    """Divide every rate-dimensioned input by mu."""
    out = dict(params)
    for k in ("v0", "lam", "eta", "ubar", "r", "q"):
        out[k] = params[k] / mu
    return out

def _sigma_h(p):
    return float(np.sqrt(p["ubar"] + p["v0"] * p["eta"]))

def _price_pinned(m, p, K, tau, cp):
    L = HALF_WIDTH / _sigma_h(p)
    return m.price(K, tau, cp=cp, N=N_COS, L=L)

for mu in MUS:
    for cp in (+1, -1):
        base_params = BASE
        resc_params = _rescale(BASE, mu)
        base = HestonCOSPricer(**base_params)
        resc = HestonCOSPricer(**resc_params)
        p_base = _price_pinned(base, base_params, STRIKES_H, TAU_BASE,      cp)
        p_resc = _price_pinned(resc, resc_params, STRIKES_H, mu * TAU_BASE, cp)
        worst  = float(np.max(np.abs(p_resc - p_base) / 100.0))
        assert worst < TOL, f"Heston temporal pi violated at mu={mu}, cp={cp}: {worst:.2e}"
print("Heston temporal pi-invariance: 7-parameter rotation leaves price unchanged")""")

md(r"""## 4.5 &nbsp; Heston spatial × temporal

Apply spatial scaling (α on `(S₀, K)`) **and** temporal rescaling (μ on
the rate-dimensioned block) simultaneously. Both are π-symmetries of
the dimensionless price, so

$$
\frac{C(\alpha S_0,\ \alpha K,\ \mu T,\ r/\mu, \dots)}{\alpha S_0}
\ =\ \frac{C(S_0,\ K,\ T,\ r, \dots)}{S_0}.
$$

Stronger than either symmetry alone: two of the eight Heston π-groups
simultaneously invariant under the COS pricer.""")

code(r"""ALPHAS = [0.5, 2.0, 100.0]

for mu in [0.5, 2.0, 10.0]:
    for alpha in ALPHAS:
        for cp in (+1, -1):
            base_params = BASE
            resc_params = _rescale(BASE, mu)
            resc_params["S0"] = alpha * 100.0    # apply spatial scale on top
            base = HestonCOSPricer(**base_params)
            resc = HestonCOSPricer(**resc_params)
            p_base = _price_pinned(base, base_params, STRIKES_H,         TAU_BASE,      cp)
            p_resc = _price_pinned(resc, resc_params, alpha * STRIKES_H, mu * TAU_BASE, cp)
            worst  = float(np.max(np.abs(p_resc / (alpha * 100.0) - p_base / 100.0)))
            assert worst < TOL, f"Heston spatial+temporal violated at alpha={alpha}, mu={mu}, cp={cp}: {worst:.2e}"
print("Heston joint spatial x temporal pi-invariance holds across all (alpha, mu, cp)")""")


# ─────────────────────────────────────────────────────────────────────────────
# Appendix
# ─────────────────────────────────────────────────────────────────────────────

md(r"""---
# Appendix — run the full suite via `pytest`

Equivalent to executing the cells above, but delegates to pytest so you
get its summary table, parametrise ids, short-traceback formatting, and
non-zero exit code on failure.""")

code(r"""import subprocess, os
repo_root = os.path.abspath(os.path.join(os.getcwd(),
    ".." if os.path.basename(os.getcwd()) == "notebooks" else "."))
env = dict(os.environ, PYTHONPATH=os.path.join(repo_root, "src"))
proc = subprocess.run(
    ["python", "-m", "pytest", "tests/", "-v", "--tb=short"],
    cwd=repo_root, env=env, capture_output=True, text=True,
)
print(proc.stdout[-4000:])      # tail, in case output is long
if proc.returncode != 0:
    print("STDERR:", proc.stderr[-2000:])""")


# ─────────────────────────────────────────────────────────────────────────────
# Write the notebook
# ─────────────────────────────────────────────────────────────────────────────

nb.cells = cells
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.13"},
}

out = Path(__file__).resolve().parent / "tests.ipynb"
with out.open("w", encoding="utf-8") as fh:
    nbf.write(nb, fh)

print(f"wrote {out}  ({len(cells)} cells)")

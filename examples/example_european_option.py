"""
example_european_option.py
==========================
End-to-end demonstration of the COS method for European options.

Covers:
    1. BSM – COS vs analytic formula (accuracy validation)
    2. BSM – convergence in N
    3. BSM – runtime benchmark
    4. Heston – COS pricing at various strikes and maturities
    5. Heston – put-call parity check
    6. Implied volatility surface from Heston COS prices

Run:
    cd fourier-cosine-option-pricing
    python examples/example_european_option.py
"""

import time
import numpy as np
from scipy.stats import norm

# --- path setup so the script works from the repo root ---
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cos_pricing import BsmModel, HestonCOSPricer, bsm_price, bsm_impvol
from cos_pricing.utils import convergence_table, benchmark_runtime

SEP = "=" * 62


# ─────────────────────────────────────────────────────────────────────────────
# 1. BSM accuracy
# ─────────────────────────────────────────────────────────────────────────────

print(SEP)
print("1. BsmModel (COS) vs analytic BSM")
print("   σ=0.20, r=0.05, q=0.10, T=1.2, spot=100")
print(SEP)

m_bsm = BsmModel(sigma=0.2, intr=0.05, divr=0.1)
strikes = np.arange(80, 121, 10, dtype=float)
spot, texp = 100.0, 1.2

cos_v = m_bsm.price(strikes, spot, texp)
ana_v = bsm_price(strikes, spot, 0.2, texp, 0.05, 0.1)

print(f"  {'Strike':>8}  {'Analytic':>14}  {'COS N=128':>14}  {'|Error|':>12}")
for K, a, c in zip(strikes, ana_v, cos_v):
    print(f"  {K:>8.0f}  {a:>14.8f}  {c:>14.8f}  {abs(a-c):>12.2e}")
print(f"\n  Max absolute error: {np.max(np.abs(cos_v - ana_v)):.2e}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. BSM convergence in N
# ─────────────────────────────────────────────────────────────────────────────

print()
print(SEP)
print("2. BsmModel – convergence as N increases")
print("   ATM call, σ=0.25, T=1.0, spot=K=100")
print(SEP)

m_conv = BsmModel(sigma=0.25)
ref    = bsm_price(100.0, 100.0, 0.25, 1.0)

convergence_table(
    price_fn  = lambda N: m_conv.price(100.0, 100.0, 1.0, n_cos=N),
    ref_price = ref,
    n_list    = [8, 16, 32, 64, 128, 256],
    label     = "Analytic BSM",
)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Runtime benchmark
# ─────────────────────────────────────────────────────────────────────────────

print()
print(SEP)
print("3. Runtime benchmark  (100 strikes, 50 repeats)")
print(SEP)

m_bench = BsmModel(sigma=0.2, intr=0.05, divr=0.1)
benchmark_runtime(
    price_fn  = lambda K: m_bench.price(K, 100.0, 1.0),
    n_strikes = 100,
    n_repeats = 50,
    label     = "BsmModel.price (N=128)",
)
benchmark_runtime(
    price_fn  = lambda K: m_bench.price(K, 100.0, 1.0, n_cos=256),
    n_strikes = 100,
    n_repeats = 50,
    label     = "BsmModel.price (N=256)",
)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Heston COS pricing
# ─────────────────────────────────────────────────────────────────────────────

print()
print(SEP)
print("4. HestonCOSPricer – standard calibrated parameters")
print("   v0=0.0398, λ=1.5768, ū=0.0398, η=0.5751, ρ=−0.5711")
print(SEP)

spot_h    = 100.0
heston_params = dict(
    v0=0.0398, lam=1.5768, ubar=0.0398,
    eta=0.5751, rho=-0.5711,
)
m_heston = HestonCOSPricer(S0=spot_h, **heston_params)
feller   = 2.0 * heston_params["lam"] * heston_params["ubar"] / heston_params["eta"]**2
print(f"  Feller ratio 2λū/η²  = {feller:.4f}")

strikes_h = np.array([80, 90, 100, 110, 120], dtype=float)

print()
print(f"  {'Strike':>8}", end="")
for T in [0.5, 1.0, 2.0, 5.0]:
    print(f"  {'T='+str(T):>14}", end="")
print()
print("  " + "-" * (8 + 4*16))

for K in strikes_h:
    print(f"  {K:>8.0f}", end="")
    for T in [0.5, 1.0, 2.0, 5.0]:
        p = m_heston.price_call(float(K), T)
        print(f"  {p:>14.6f}", end="")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 5. Heston put-call parity
# ─────────────────────────────────────────────────────────────────────────────

print()
print(SEP)
print("5. Heston put-call parity check  (r=q=0)")
print(SEP)

texp_pcp = 1.0
fwd_pcp  = spot_h  # r=q=0
df_pcp   = 1.0

print(f"  {'Strike':>8}  {'Call':>12}  {'Put':>12}  {'PCP error':>14}")
for K in [90.0, 100.0, 110.0]:
    c   = m_heston.price_call(K, texp_pcp)
    p   = m_heston.price_put (K, texp_pcp)
    err = abs((c - p) - df_pcp * (fwd_pcp - K))
    print(f"  {K:>8.0f}  {c:>12.6f}  {p:>12.6f}  {err:>14.2e}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Implied volatility smile from Heston COS prices
# ─────────────────────────────────────────────────────────────────────────────

print()
print(SEP)
print("6. Implied-volatility smile from HestonCOSPricer prices")
print("   T=1.0, spot=100")
print(SEP)

strikes_iv = np.array([75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125], dtype=float)
texp_iv    = 1.0

cos_prices = m_heston.price_call(strikes_iv, texp_iv)
iv_smile   = np.array([
    bsm_impvol(p, K, spot_h, texp_iv, cp=1)
    for p, K in zip(cos_prices, strikes_iv)
])

print(f"  {'Strike':>8}  {'COS Price':>12}  {'Impl Vol':>12}")
for K, p, iv in zip(strikes_iv, cos_prices, iv_smile):
    print(f"  {K:>8.0f}  {p:>12.6f}  {iv:>11.4%}")

print()
print("Done.")
"""
HestonCOSPricer benchmark and convergence demo.

Reproduces the Fang & Oosterlee (2008) Table 4 reference values and shows
exponential convergence in N as well as the sensitivity of the result to the
truncation-width parameter L.

Run:
    PYTHONPATH=src python examples/heston_cos_pricer.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from cos_pricing import HestonCOSPricer


SEP = "=" * 68

PARAMS = dict(
    S0=100.0, v0=0.0175, lam=1.5768, eta=0.5751, ubar=0.0398, rho=-0.5711,
    r=0.0, q=0.0,
)
REF_T1  = 5.785155435
REF_T10 = 22.318945791474590


# ─────────────────────────────────────────────────────────────────────────────
# 1. Benchmark reproduction
# ─────────────────────────────────────────────────────────────────────────────

print(SEP)
print("1. Paper benchmark reproduction")
print("   S0=100, K=100, r=q=0, lam=1.5768, eta=0.5751, ubar=0.0398,")
print("   v0=0.0175, rho=-0.5711  (F&O 2008, Table 4)")
print(SEP)

m = HestonCOSPricer(**PARAMS)

p1  = m.price_call(100.0, 1.0,  N=160)
p10 = m.price_call(100.0, 10.0, N=160)

print(f"  T= 1  COS price: {p1:.12f}  (ref {REF_T1:.12f})  err {abs(p1-REF_T1):.2e}")
print(f"  T=10  COS price: {p10:.12f}  (ref {REF_T10:.12f})  err {abs(p10-REF_T10):.2e}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Convergence in N (T=1)
# ─────────────────────────────────────────────────────────────────────────────

print()
print(SEP)
print("2. Convergence in N at T=1 (default L=12)")
print(SEP)
print(f"  {'N':>6}  {'Price':>18}  {'|Error|':>12}")
for N in (16, 32, 64, 128, 160, 256):
    p = m.price_call(100.0, 1.0, N=N)
    print(f"  {N:>6}  {p:>18.12f}  {abs(p - REF_T1):>12.2e}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Sensitivity to L (T=1, N=256)
# ─────────────────────────────────────────────────────────────────────────────

print()
print(SEP)
print("3. Sensitivity to L at T=1, N=256")
print("   Too-small L -> truncation error; too-large L -> branch-cut instability")
print(SEP)
print(f"  {'L':>6}  {'Price':>18}  {'|Error|':>12}")
for L in (3, 5, 8, 10, 12, 15, 20, 30):
    p = m.price_call(100.0, 1.0, N=256, L=float(L))
    print(f"  {L:>6}  {p:>18.12f}  {abs(p - REF_T1):>12.2e}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Strike surface (T=1)
# ─────────────────────────────────────────────────────────────────────────────

print()
print(SEP)
print("4. Call price surface, T=1, N=160")
print(SEP)
strikes = np.array([80, 90, 95, 100, 105, 110, 120], dtype=float)
prices  = m.price_call(strikes, 1.0, N=160)
puts    = m.price_put (strikes, 1.0, N=160)
print(f"  {'Strike':>8}  {'Call':>14}  {'Put':>14}  {'C - P':>14}  {'(F - K)':>10}")
for K, c, p in zip(strikes, prices, puts):
    print(f"  {K:>8.1f}  {c:>14.8f}  {p:>14.8f}  {c - p:>14.8f}  {PARAMS['S0'] - K:>10.2f}")

print()
print("Done.")

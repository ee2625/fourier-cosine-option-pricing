"""
Track B validation — NormalCos (Bachelier) COS pricer
vs the closed form on Lecture 3, slide 19.

Checks, in order:
  1. ATM collapse:  Cn(K=F) = sigma*sqrt(T) / sqrt(2*pi)
  2. Point-by-point match vs analytic across ITM / ATM / OTM, call and put
  3. Convergence to machine precision as N grows (paper-style table)
  4. Put-call parity:  C - P = df * (F - K)
  5. Translation invariance (Track A preview):  C(F+l, K+l) = C(F, K)
"""
import sys, os
sys.path.insert(0, "/home/claude")

import numpy as np
from cos_pricing import NormalCos


def check(name, cond, extra=""):
    tag = "PASS" if cond else "FAIL"
    print(f"  [{tag}] {name}{(' — ' + extra) if extra else ''}")
    return cond


# ── 1. ATM analytic collapse ────────────────────────────────────────────────
print("1. ATM collapse")
sigma, texp, spot = 0.25 * 100, 1.0, 100.0   # abs-vol = 25 (price units)
m = NormalCos(sigma=sigma)
atm = m.price(strike=spot, spot=spot, texp=texp, n_cos=128)
ref_atm = sigma * np.sqrt(texp) / np.sqrt(2 * np.pi)
check("Cn(K=F) = sigma*sqrt(T)/sqrt(2*pi)",
      abs(atm - ref_atm) < 1e-12,
      f"cos={atm:.12f}  ref={ref_atm:.12f}  |err|={abs(atm-ref_atm):.2e}")

# ── 2. Point-by-point match vs analytic, both calls and puts ───────────────
print("\n2. Analytic match across strikes (call & put)")
r = 0.05
m = NormalCos(sigma=sigma, intr=r)
strikes = np.array([70., 85., 100., 115., 130.])
for cp, label in [(+1, "call"), (-1, "put ")]:
    cos_px = m.price(strike=strikes, spot=spot, texp=texp, cp=cp, n_cos=128)
    ana_px = NormalCos.price_analytic(strikes, spot, sigma, texp, intr=r, cp=cp)
    err    = np.max(np.abs(cos_px - ana_px))
    check(f"{label}  max|cos - ana| < 1e-12",
          err < 1e-12,
          f"max err = {err:.2e}  over K={strikes.tolist()}")

# ── 3. Convergence table: error vs N ───────────────────────────────────────
print("\n3. Convergence to machine precision")
K_test = 105.0
ana = NormalCos.price_analytic(K_test, spot, sigma, texp, intr=r)
print(f"   Reference (analytic)  = {ana:.16f}")
print(f"   {'N':>4}  {'cos price':>22}  {'abs error':>12}")
for N in [8, 16, 32, 64, 128, 256]:
    p   = m.price(strike=K_test, spot=spot, texp=texp, n_cos=N)
    err = abs(p - ana)
    print(f"   {N:>4}  {p:>22.16f}  {err:>12.2e}")
# spec target: machine precision at N = 64
p64 = m.price(strike=K_test, spot=spot, texp=texp, n_cos=64)
check("machine precision at N=64", abs(p64 - ana) < 1e-13,
      f"err = {abs(p64-ana):.2e}")

# ── 4. Put-call parity:  C - P = df*(F - K)  ───────────────────────────────
print("\n4. Put-call parity")
K_arr = np.array([80., 95., 100., 110., 125.])
C = m.price(strike=K_arr, spot=spot, texp=texp, cp=+1, n_cos=128)
P = m.price(strike=K_arr, spot=spot, texp=texp, cp=-1, n_cos=128)
df  = np.exp(-r * texp)
fwd = spot * np.exp(r * texp)
lhs = C - P
rhs = df * (fwd - K_arr)
err = np.max(np.abs(lhs - rhs))
check("max|C - P - df*(F-K)| < 1e-12", err < 1e-12, f"max err = {err:.2e}")

# ── 5. Translation invariance (Track A preview) ────────────────────────────
print("\n5. Translation invariance:  C(F+l, K+l) = C(F, K)")
lambdas = [-30.0, -5.0, 0.5, 20.0, 100.0]
K_arr = np.array([80., 100., 120.])
base  = NormalCos(sigma=sigma).price(strike=K_arr, spot=spot, texp=texp, n_cos=128)
max_err = 0.0
for lam in lambdas:
    shifted = NormalCos(sigma=sigma).price(
        strike=K_arr + lam, spot=spot + lam, texp=texp, n_cos=128)
    max_err = max(max_err, float(np.max(np.abs(shifted - base))))
check("max|C(F+l, K+l) - C(F, K)| < 1e-10 across lambdas",
      max_err < 1e-10,
      f"max err over l={lambdas} is {max_err:.2e}")

print("\nDone.")
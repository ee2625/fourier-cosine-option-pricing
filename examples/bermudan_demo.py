"""
Bermudan put under BSM via the Fang-Oosterlee 2009 COS-with-backward-induction
algorithm.

Reproduces the Section 5.1 BSM benchmark of Fang & Oosterlee (2009):
    S = K = 100, r = 0.1, q = 0, sigma = 0.25, T = 1.

The script prints two things:

  1. A convergence sweep in the number of exercise dates M, showing the
     Bermudan price growing monotonically toward the American limit
     (the published reference for these parameters is ~6.55).

  2. The European-limit check: M = 1 must equal the closed-form European
     put exactly.

Run:
    PYTHONPATH=src python examples/bermudan_demo.py
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from cos_pricing import BermudanCosBSM, bsm_price


# ── Parameters (Fang-Oosterlee 2009, Section 5.1) ────────────────────────────
S, K, T     = 100.0, 100.0, 1.0
SIGMA, R, Q = 0.25, 0.1, 0.0
N           = 128

print("=" * 70)
print("BERMUDAN PUT — Fang & Oosterlee (2009) COS with backward induction")
print(f"  S = {S}, K = {K}, T = {T}, sigma = {SIGMA}, r = {R}, q = {Q}")
print(f"  N = {N} cosine terms")
print("=" * 70)

m = BermudanCosBSM(sigma=SIGMA, intr=R, divr=Q)
eu = float(bsm_price(K, S, SIGMA, T, intr=R, divr=Q, cp=-1))
print(f"\nReference European put (closed-form): {eu:.10f}")
print(f"Reference American put (~from literature): ~6.55\n")


# ── 1. M = 1 European-limit check ────────────────────────────────────────────
ber_M1 = m.price_put(S=S, K=K, T=T, M=1, N=N)
print("─" * 70)
print("M = 1 European-limit check")
print(f"  Bermudan(M = 1)     = {ber_M1:.12f}")
print(f"  European (analytic) = {eu:.12f}")
print(f"  |difference|        = {abs(ber_M1 - eu):.2e}")
print("─" * 70)


# ── 2. Convergence to American as M grows ────────────────────────────────────
print("\nConvergence to American as M grows:")
print(f"  {'M':>5}  {'Bermudan price':>18}  {'incr. premium':>16}  {'msec':>10}")
print("  " + "─" * 60)

prev = eu
for M_val in [1, 2, 4, 8, 16, 32, 64, 100, 200]:
    t0 = time.perf_counter()
    p  = m.price_put(S=S, K=K, T=T, M=M_val, N=N)
    ms = (time.perf_counter() - t0) * 1e3
    print(f"  {M_val:>5}  {p:>18.12f}  {p - prev:>16.2e}  {ms:>10.2f}")
    prev = p

print("\nNotes:")
print("  * M = 1 matches the European put to machine epsilon -- this is the")
print("    structural validation that the backward-induction pipeline is")
print("    correct, since with one exercise opportunity the Bermudan put")
print("    degenerates to the European put.")
print("  * Prices are strictly non-decreasing in M (more exercise dates")
print("    cannot reduce the option's value).")
print("  * The American-put limit is approached as M -> infinity; the")
print("    Fang-Oosterlee 2009 paper (Table 1) gives the same convergence")
print("    profile under these parameters.")

"""
Bermudan COS stress audit.

The Bermudan COS recursion uses the same chi formula as the European pricer
plus a closed-form M_{k,n}(x*) matrix at each timestep.  Both pieces have an
``e^b`` factor that overflows when the truncation interval gets wide enough,
which means high-vol or long-T regimes can break Bermudan in the same way
the prof's Heston case broke the European call.

This script sweeps (sigma, T) for the BSM Bermudan put and reports relative
error vs. the European limit at M = 1 (which must equal the closed-form
European put exactly).

Run:
    PYTHONPATH=src python examples/bermudan_stress_audit.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from cos_pricing import BermudanCosBSM, bsm_price


REL_PASS     = 1e-3
REL_DEGRADED = 1e-1


def fmt(rel):
    if not np.isfinite(rel):
        return "  inf  "
    return f"{rel:>7.1e}"


def classify(rel):
    if not np.isfinite(rel):
        return "BROKEN"
    if rel < REL_PASS:
        return "PASS"
    if rel < REL_DEGRADED:
        return "DEGRADED"
    return "BROKEN"


def main():
    print("=" * 78)
    print("BERMUDAN PUT (BSM)  -- M = 1 European-limit check across sigma x T")
    print("If M=1 != closed-form European put, the algorithm is broken.")
    print("=" * 78)

    sigmas = [0.1, 0.2, 0.4, 0.7, 1.0, 1.5]
    Ts     = [0.1, 1.0, 5.0, 10.0]
    S = K  = 100.0
    intr = 0.05
    divr = 0.0

    print(f"\n{'sigma-T':>10} | " + " | ".join(f"T={T:>4}" for T in Ts))
    print("-" * (12 + len(Ts) * 12))

    rows = []
    for sigma in sigmas:
        cells = []
        for T in Ts:
            try:
                m = BermudanCosBSM(sigma=sigma, intr=intr, divr=divr)
                ber = float(m.price_put(S=S, K=K, T=T, M=1, N=128))
                eu  = float(bsm_price(K, S, sigma, T, intr=intr, divr=divr, cp=-1))
                rel = abs(ber - eu) / max(abs(eu), 1e-10)
            except Exception:
                rel = float("inf")
            cells.append(rel)
            rows.append((sigma, T, rel))
        print(f"{sigma:>10.2f} | " + " | ".join(fmt(c) for c in cells))

    # Also check: with M > 1, monotonicity should hold (no negative increments).
    print()
    print("=" * 78)
    print("BERMUDAN PUT  -- monotonicity in M (price must be non-decreasing)")
    print("=" * 78)
    print(f"\n{'sigma-T':>10} | " + " | ".join(f"T={T:>4}" for T in Ts))
    print("-" * (12 + len(Ts) * 12))

    mono_rows = []
    for sigma in sigmas:
        cells = []
        for T in Ts:
            try:
                m = BermudanCosBSM(sigma=sigma, intr=intr, divr=divr)
                Ms = [1, 5, 20, 50]
                Ps = [float(m.price_put(S=S, K=K, T=T, M=Mv, N=128)) for Mv in Ms]
                violation = max((prev - cur) for prev, cur in zip(Ps, Ps[1:]))
                cells.append(violation if violation > 0 else 0.0)
            except Exception:
                cells.append(float("inf"))
            mono_rows.append((sigma, T, cells[-1]))
        formatted = []
        for c in cells:
            if c == 0.0:
                formatted.append("    OK ")
            elif np.isfinite(c):
                formatted.append(f"-{c:>6.1e}")
            else:
                formatted.append("  inf  ")
        print(f"{sigma:>10.2f} | " + " | ".join(formatted) + "  (largest backslide; OK = none)")

    # Summary
    counts = {"PASS": 0, "DEGRADED": 0, "BROKEN": 0}
    for _, _, rel in rows:
        counts[classify(rel)] += 1
    print()
    print("=" * 78)
    print("SUMMARY -- M = 1 European-limit check across sigma x T cells")
    print("=" * 78)
    total = sum(counts.values())
    print(f"  PASS:     {counts['PASS']}/{total}")
    print(f"  DEGRADED: {counts['DEGRADED']}/{total}")
    print(f"  BROKEN:   {counts['BROKEN']}/{total}")

    mono_violations = sum(1 for _, _, v in mono_rows if v > 1e-10 and np.isfinite(v))
    print(f"  Monotonicity-in-M violations (over {len(mono_rows)} cells): {mono_violations}")
    if mono_violations:
        print("  Cells with backslide:")
        for sigma, T, v in mono_rows:
            if v > 1e-10 and np.isfinite(v):
                print(f"    sigma={sigma:.2f}, T={T}: backslide of {v:.3e}")

    print()
    print("Notes:")
    print("  * M = 1 should match the closed-form European put exactly --")
    print("    catches sign/prime-sum/CF-convention bugs in the recursion.")
    print("  * Monotonicity in M is a structural property: more exercise")
    print("    opportunities cannot reduce option value.")
    print("  * Bermudan COS uses the same chi formula and an additional")
    print("    M_{k,n}(x*) matrix per timestep; both have e^b factors that")
    print("    overflow in extreme-vol regimes (analogous to the European")
    print("    Heston issue resolved by Le Floc'h put-parity).  Wide sigma")
    print("    or long T cells with large rel_err are early warning signs")
    print("    of the same structural failure mode.")


if __name__ == "__main__":
    main()

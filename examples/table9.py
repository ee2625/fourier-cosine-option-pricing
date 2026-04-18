"""
Reproduce Fang & Oosterlee (2008) Table 9 — CGMY, Y=1.5.
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from cos_pricing import CgmyModel
from cos_pricing.cos_method import cos_price

PARAMS = dict(C=1.0, G=5.0, M=5.0, Y=1.5, intr=0.1, divr=0.0)
K, S0, T, L = 100.0, 100.0, 1.0, 10.0
REF = 49.790905305 

ROWS = [
    (40, 1.38e+00, 0.0545),
    (45, 1.98e-02, 0.0589),
    (50, 4.52e-04, 0.0689),
    (55, 9.59e-06, 0.0690),
    (60, 1.22e-09, 0.0732),
    (65, 7.53e-10, 0.0748),
]

N_REPS = 2000

def _collect():
    # Initialize our new model
    m = CgmyModel(**PARAMS)
    fwd, df = m._fwd_df(S0, T)
    
    # Pre-compute CF and truncation range (Hoisting for speed!)
    cf = m.char_func(T)
    trunc = m.trunc_range(T, L)

    results = []
    for N, pe, pms in ROWS:
        # 1. Calculate Price and Error
        v = cos_price(cf, T, K, fwd, df, cp=1, n_cos=N, trunc_range=trunc)
        err = abs(float(v) - REF)

        # 2. Calculate Runtime ([Teammate Reference] Matches table_2.py loop)
        t0 = time.perf_counter()
        for _ in range(N_REPS):
            cos_price(cf, T, K, fwd, df, cp=1, n_cos=N, trunc_range=trunc)
        ms = (time.perf_counter() - t0) / N_REPS * 1e3

        results.append(dict(N=N, paper_err=pe, err=err, paper_ms=pms, ms=ms))
    return results

def _print_text(results):
    print(f"Fang-Oosterlee Table 8 reproducer (CGMY, Y={PARAMS['Y']})")
    print(f"Reference value: {REF}")
    print()
    header = f"{'N':>4}  {'paper err':>10}  {'our err':>10}  {'paper ms':>9}  {'our ms':>8}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['N']:>4}  {r['paper_err']:>10.2e}  {r['err']:>10.2e}  "
              f"{r['paper_ms']:>9.4f}  {r['ms']:>8.4f}")
    print("\nNote: Our CPU time should crush the paper's 2008 MATLAB benchmark.")

def _print_markdown(results):
    """[Teammate Reference] Borrowed from test4.py for easy README updates."""
    Ns = [r["N"] for r in results]
    print(f"### Table 8 reproduction — CGMY, Y={PARAMS['Y']}, T={T}, K={K}")
    print(f"Reference: {REF}")
    print()
    print("| |" + "|".join(f" N={n} " for n in Ns) + "|")
    print("|---|" + "|".join("---" for _ in Ns) + "|")
    print("| paper error    |" + "|".join(f" {r['paper_err']:.2e} " for r in results) + "|")
    print("| our error      |" + "|".join(f" {r['err']:.2e} "       for r in results) + "|")
    print("| paper ms       |" + "|".join(f" {r['paper_ms']:.4f} "  for r in results) + "|")
    print("| our ms         |" + "|".join(f" {r['ms']:.4f} "         for r in results) + "|")

def main():
    as_md = "--markdown" in sys.argv or "--md" in sys.argv
    results = _collect()
    if as_md:
        _print_markdown(results)
    else:
        _print_text(results)
    return 0

if __name__ == "__main__":
    sys.exit(main())
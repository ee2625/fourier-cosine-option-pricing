"""
Reproduces Table 7 from Fang & Oosterlee (2008):
COS method convergence for a European call under the Variance Gamma model.

"A Novel Pricing Method for European Options Based on
Fourier-Cosine Series Expansions", SIAM J. Sci. Comput. 31(2):826-848.
Section 5.4, parameters Eq. (55).

Run:
    cd fourier-cosine-option-pricing
    PYTHONPATH=src python examples/table7.py
"""
import time
import numpy as np
from scipy.stats import linregress
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from cos_pricing import VgModel

# ── Parameters (Eq. 55) ───────────────────────────────────────────────────────
S0, K, R, Q      = 100.0, 90.0, 0.1, 0.0
SIGMA, THETA, NU = 0.12, -0.14, 0.2
REF_T01          = 10.993703187
REF_T1           = 19.099354724
N_REPS           = 10_000

N_SHORT = [128, 256, 512, 1024, 2048]
N_LONG  = [30, 60, 90, 120, 150]

# Absolute errors from Table 7 for N values we compute
PAPER_T01 = {128: 4.35e-4, 256: 4.55e-5, 512: 1.13e-6, 1024: 2.52e-8}
PAPER_T1  = {}


def time_price(model, strike, spot, texp, n_cos, n_rep):
    model.price(strike, spot, texp, cp=1, n_cos=n_cos)        # warm-up
    t0 = time.perf_counter()
    for _ in range(n_rep):
        model.price(strike, spot, texp, cp=1, n_cos=n_cos)
    return (time.perf_counter() - t0) / n_rep * 1e3            # ms


def compute_rows(model, spot, strike, texp, ref, n_values, n_rep):
    return [
        (n,
         model.price(strike, spot, texp, cp=1, n_cos=n) - ref,
         time_price(model, strike, spot, texp, n, n_rep))
        for n in n_values
    ]


def print_table7(rows_short, rows_long):
    sep = "─" * 73
    print(f"\n{'COS method':^73}")
    print(sep)
    print(f"  T=0.1; Ref = {REF_T01}        │   T=1.0; Ref = {REF_T1}")
    print(sep)
    print(f"   N   │    Error     │ Time (ms)  │    N  │    Error     │ Time (ms)")
    print(f"──────┼──────────────┼────────────┼───────┼──────────────┼──────────")
    for (n0, e0, t0_), (n1, e1, t1) in zip(rows_short, rows_long):
        print(f"  {n0:4d} │ {abs(e0):.3e}    │ {t0_:8.4f}   │  {n1:4d} │ {abs(e1):.3e}    │ {t1:8.4f}")
    print(sep)
    print("  Timings differ from paper (2008 Pentium 4 / MATLAB 7.4 vs modern NumPy).")


def verify(rows, paper_errors, label):
    rows_with_ref = [(n, err, t) for n, err, t in rows if n in paper_errors]
    if not rows_with_ref:
        return
    print(f"\n{label}:")
    all_ok = True
    for n, err, _ in rows_with_ref:
        p    = paper_errors[n]
        ok   = abs(np.log10(abs(err)) - np.log10(p)) < 1.0
        print(f"  N={n:4d}:  ours={abs(err):.2e}  paper={p:.2e}  {'✓' if ok else '✗'}")
        all_ok &= ok
    print(f"  → Magnitudes: {'PASS' if all_ok else 'FAIL'}")


def main():
    model = VgModel(sigma=SIGMA, theta=THETA, nu=NU, intr=R, divr=Q)

    # High-N cross-check (N=2^14, same as paper's reference computation)
    our_T01 = model.price(K, S0, 0.1, n_cos=2**14)
    our_T1  = model.price(K, S0, 1.0, n_cos=2**14)
    print("High-N cross-check (N=2^14):")
    print(f"  T=0.1  paper={REF_T01:.9f}  ours={our_T01:.9f}  diff={our_T01-REF_T01:.2e}")
    print(f"  T=1.0  paper={REF_T1:.9f}  ours={our_T1:.9f}  diff={our_T1-REF_T1:.2e}")

    print(f"\nTiming {N_REPS:,} reps per row ...")
    rows_short = compute_rows(model, S0, K, 0.1, REF_T01, N_SHORT, N_REPS)
    rows_long  = compute_rows(model, S0, K, 1.0, REF_T1,  N_LONG,  N_REPS)

    print_table7(rows_short, rows_long)

    print("\n── Verification vs paper ──")
    verify(rows_short, PAPER_T01, "T=0.1")
    verify(rows_long,  PAPER_T1,  "T=1.0")

    print("\n── Convergence analysis ──")
    ns_s   = [r[0] for r in rows_short]
    errs_s = [abs(r[1]) for r in rows_short]
    sl_s, _, r_s, _, _ = linregress(np.log10(ns_s), np.log10(errs_s))
    print(f"T=0.1 (algebraic):   log-log slope = {sl_s:.2f}  order ≈ {-sl_s:.1f}  R²={r_s**2:.3f}")

    ns_l   = [r[0] for r in rows_long]
    errs_l = [abs(r[1]) for r in rows_long]
    sl_l, _, r_l, _, _ = linregress(ns_l, np.log10(errs_l))
    print(f"T=1.0 (exponential): semi-log slope = {sl_l:.5f}/term  R²={r_l**2:.3f}")
    print(f"                     ≈ {sl_l*32:.2f} orders of magnitude per 32 terms")


if __name__ == "__main__":
    main()

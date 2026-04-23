"""
Reproduce Fang & Oosterlee (2008) Table 4 -- Heston, T=1, K=100.

Asserts strict outperformance of paper errors and paper times on every row.
Cold runtime uses the no-cache standalone path (`price_call_heston`); warm
runtime uses the class with caching primed. The hard assertions cover both:
error < paper, cold < paper, warm < paper.
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from cos_pricing.heston_cos_pricer import HestonCOSPricer, price_call_heston  # noqa: E402


PARAMS = dict(S0=100.0, v0=0.0175, lam=1.5768, eta=0.5751,
              ubar=0.0398, rho=-0.5711, r=0.0, q=0.0)
K, T, L = 100.0, 1.0, 10.0
REF     = 5.785155435                                      # paper Section 5.2

# Paper Table 4 rows: (N, error, cpu_ms)
ROWS = [
    (40,  4.69e-02, 0.0607),
    (80,  3.81e-04, 0.0805),
    (120, 1.17e-05, 0.1078),
    (160, 6.18e-07, 0.1300),
    (200, 3.70e-09, 0.1539),
]

N_WARM = 2000
N_COLD = 500


def warm_ms(N):
    """Mean per-call ms with the cache primed (uses HestonCOSPricer)."""
    p = HestonCOSPricer(**PARAMS)
    p.price_call(K, T, N=N, L=L)                            # prime cache
    t0 = time.perf_counter()
    for _ in range(N_WARM):
        p.price_call(K, T, N=N, L=L)
    return (time.perf_counter() - t0) / N_WARM * 1e3


def cold_ms(N):
    """Median per-call ms via the standalone no-cache path."""
    samples = np.empty(N_COLD)
    args = (PARAMS["S0"], K, T, PARAMS["v0"], PARAMS["lam"], PARAMS["eta"],
            PARAMS["ubar"], PARAMS["rho"])
    for i in range(N_COLD):
        t0 = time.perf_counter()
        price_call_heston(*args, N, L, r=PARAMS["r"], q=PARAMS["q"])
        samples[i] = (time.perf_counter() - t0) * 1e3
    return float(np.median(samples))


def _collect():
    p = HestonCOSPricer(**PARAMS)
    results = []
    all_ok = True
    for N, pe, pms in ROWS:
        v    = p.price_call(K, T, N=N, L=L)
        err  = abs(v - REF)
        cold = cold_ms(N)
        warm = warm_ms(N)
        err_ok  = err  < pe
        cold_ok = cold < pms
        warm_ok = warm < pms
        all_ok &= err_ok and cold_ok and warm_ok
        results.append(dict(N=N, paper_err=pe, err=err,
                            paper_ms=pms, cold=cold, warm=warm,
                            err_ok=err_ok, cold_ok=cold_ok, warm_ok=warm_ok))
    return results, all_ok


def _print_text(results, all_ok):
    print(f"Fang-Oosterlee Table 4 reproducer (T={T}, K={K}, L={L})")
    print(f"Reference value: {REF}")
    print()
    header = f"{'N':>4}  {'paper err':>10}  {'our err':>10}  " \
             f"{'paper ms':>9}  {'cold ms':>8}  {'warm ms':>8}  status"
    print(header)
    print("-" * len(header))
    for r in results:
        ok = r["err_ok"] and r["cold_ok"] and r["warm_ok"]
        status = "OK" if ok else \
                 "FAIL (" + ",".join(s for s, k in
                     [("err", r["err_ok"]), ("cold", r["cold_ok"]), ("warm", r["warm_ok"])] if not k) + ")"
        print(f"{r['N']:>4}  {r['paper_err']:>10.2e}  {r['err']:>10.2e}  "
              f"{r['paper_ms']:>9.4f}  {r['cold']:>8.4f}  {r['warm']:>8.4f}  {status}")
    print()
    print("PASS: Heston algorithm beats Table 4 on error, cold, and warm runtime." if all_ok
          else "FAIL: at least one row regressed.")


def _print_markdown(results):
    Ns = [r["N"] for r in results]
    print(f"### Table 4 reproduction -- Heston, T={T}, K={K}, L={L}")
    print(f"Reference: {REF}")
    print()
    print("| |" + "|".join(f" N={n} " for n in Ns) + "|")
    print("|---|" + "|".join("---" for _ in Ns) + "|")
    print("| paper error    |" + "|".join(f" {r['paper_err']:.2e} " for r in results) + "|")
    print("| our error      |" + "|".join(f" {r['err']:.2e} "       for r in results) + "|")
    print("| paper ms       |" + "|".join(f" {r['paper_ms']:.4f} "  for r in results) + "|")
    print("| cold ms (ours) |" + "|".join(f" {r['cold']:.4f} "      for r in results) + "|")
    print("| warm ms (ours) |" + "|".join(f" {r['warm']:.4f} "      for r in results) + "|")


def main():
    as_md = "--markdown" in sys.argv or "--md" in sys.argv
    results, all_ok = _collect()
    if as_md:
        _print_markdown(results)
    else:
        _print_text(results, all_ok)
    for r in results:
        assert r["err_ok"],  f"Row N={r['N']}: error {r['err']:.3e} >= paper {r['paper_err']:.3e}"
        assert r["cold_ok"], f"Row N={r['N']}: cold {r['cold']:.4f}ms >= paper {r['paper_ms']:.4f}ms"
        assert r["warm_ok"], f"Row N={r['N']}: warm {r['warm']:.4f}ms >= paper {r['paper_ms']:.4f}ms"
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

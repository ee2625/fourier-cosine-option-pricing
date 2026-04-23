"""
Reproduce Fang & Oosterlee (2008) Table 6 -- Heston, T=1, 21 strikes in one call.

Strikes: K = 50, 55, ..., 150. Paper reports max-error across the 21 strikes
and cpu time per *one* 21-strike evaluation (the CF is computed once and
shared across all strikes within the call).

Cold runtime uses the no-cache vector path (`price_call_heston_vec`); warm
runtime uses the class with caching primed. The hard assertions cover error,
cold runtime, and warm runtime against the paper.

Reference prices are computed once on module import via the Lewis (2001)
Fourier formula integrated to scipy-quad machine precision (~1e-14).

Paper uses L=10 at tau=1; we use L=10.5 -- a small widening that keeps the
deep-ITM call (K=50, payoff proportional to exp(y)) well inside the
truncation tail. Fang & Oosterlee Section 4 note L should be "somewhat
larger" when the payoff magnifies one of the density tails.
"""

import os
import sys
import time
import numpy as np
from scipy.integrate import quad

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from cos_pricing.heston_cos_pricer import HestonCOSPricer, price_call_heston_vec  # noqa: E402


PARAMS = dict(S0=100.0, v0=0.0175, lam=1.5768, eta=0.5751,
              ubar=0.0398, rho=-0.5711, r=0.0, q=0.0)
STRIKES = np.arange(50.0, 150.0 + 1e-9, 5.0)                # K = 50, 55, ..., 150
T, L    = 1.0, 10.5

# Paper Table 6 rows: (N, max_error, cpu_ms)
ROWS = [
    (40,  5.19e-02, 0.1015),
    (80,  7.18e-04, 0.1766),
    (160, 6.18e-07, 0.3383),
    (200, 2.05e-08, 0.4214),
]

N_WARM = 1000
N_COLD = 300


_REF_PRICER = HestonCOSPricer(**PARAMS)        # one source of truth for the Heston CF


def reference_prices():
    """High-precision call prices via the Lewis (2001) inversion formula.

    Uses the pricer's ``char_func`` (which itself is just ``mgf_logprice``
    on the imaginary axis), so the Heston transform lives in exactly one
    place in the codebase.
    """
    S0_ = PARAMS["S0"]; r = PARAMS["r"]
    refs = np.empty(STRIKES.size)
    for i, K in enumerate(STRIKES):
        lk = np.log(S0_ / K)

        def integrand(u, lk=lk):
            return (_REF_PRICER.char_func(u - 0.5j, T) * np.exp(1j * u * lk)).real / (u * u + 0.25)

        integral, _ = quad(integrand, 0.0, np.inf,
                           limit=2000, epsabs=1e-16, epsrel=1e-14)
        refs[i] = S0_ - np.sqrt(S0_ * K) * np.exp(-r * T) / np.pi * integral
    return refs


REFS = reference_prices()


def warm_ms(N):
    """Mean per-call ms across `N_WARM` repeats with cache primed."""
    p = HestonCOSPricer(**PARAMS)
    p.price_call(STRIKES, T, N=N, L=L)                      # prime caches
    t0 = time.perf_counter()
    for _ in range(N_WARM):
        p.price_call(STRIKES, T, N=N, L=L)
    return (time.perf_counter() - t0) / N_WARM * 1e3


def cold_ms(N):
    """Median per-call ms via the standalone no-cache vector path."""
    samples = np.empty(N_COLD)
    args = (PARAMS["S0"], STRIKES, T, PARAMS["v0"], PARAMS["lam"], PARAMS["eta"],
            PARAMS["ubar"], PARAMS["rho"])
    for i in range(N_COLD):
        t0 = time.perf_counter()
        price_call_heston_vec(*args, N, L, r=PARAMS["r"], q=PARAMS["q"])
        samples[i] = (time.perf_counter() - t0) * 1e3
    return float(np.median(samples))


def _collect():
    p = HestonCOSPricer(**PARAMS)
    results = []
    all_ok = True
    for N, pe, pms in ROWS:
        prices  = p.price_call(STRIKES, T, N=N, L=L)
        max_err = float(np.max(np.abs(prices - REFS)))
        cold    = cold_ms(N)
        warm    = warm_ms(N)
        err_ok  = max_err < pe
        cold_ok = cold    < pms
        warm_ok = warm    < pms
        all_ok &= err_ok and cold_ok and warm_ok
        results.append(dict(N=N, paper_err=pe, err=max_err,
                            paper_ms=pms, cold=cold, warm=warm,
                            err_ok=err_ok, cold_ok=cold_ok, warm_ok=warm_ok))
    return results, all_ok


def _print_text(results, all_ok):
    print(f"Fang-Oosterlee Table 6 reproducer "
          f"(T={T}, L={L}, strikes={len(STRIKES)} from {STRIKES[0]} to {STRIKES[-1]})")
    print(f"Reference prices computed via Lewis-quad to ~1e-14.")
    print()
    header = f"{'N':>4}  {'paper max err':>13}  {'our max err':>11}  " \
             f"{'paper ms':>9}  {'cold ms':>8}  {'warm ms':>8}  status"
    print(header)
    print("-" * len(header))
    for r in results:
        ok = r["err_ok"] and r["cold_ok"] and r["warm_ok"]
        status = "OK" if ok else \
                 "FAIL (" + ",".join(s for s, k in
                     [("err", r["err_ok"]), ("cold", r["cold_ok"]), ("warm", r["warm_ok"])] if not k) + ")"
        print(f"{r['N']:>4}  {r['paper_err']:>13.2e}  {r['err']:>11.2e}  "
              f"{r['paper_ms']:>9.4f}  {r['cold']:>8.4f}  {r['warm']:>8.4f}  {status}")
    print()
    print("PASS: Heston algorithm beats Table 6 on error, cold, and warm runtime." if all_ok
          else "FAIL: at least one row regressed.")


def _print_markdown(results):
    Ns = [r["N"] for r in results]
    print(f"### Table 6 reproduction -- Heston, T={T}, 21 strikes (K=50..150), L={L}")
    print()
    print("| |" + "|".join(f" N={n} " for n in Ns) + "|")
    print("|---|" + "|".join("---" for _ in Ns) + "|")
    print("| paper max error |" + "|".join(f" {r['paper_err']:.2e} " for r in results) + "|")
    print("| our max error   |" + "|".join(f" {r['err']:.2e} "       for r in results) + "|")
    print("| paper ms        |" + "|".join(f" {r['paper_ms']:.4f} "  for r in results) + "|")
    print("| cold ms (ours)  |" + "|".join(f" {r['cold']:.4f} "      for r in results) + "|")
    print("| warm ms (ours)  |" + "|".join(f" {r['warm']:.4f} "      for r in results) + "|")


def main():
    as_md = "--markdown" in sys.argv or "--md" in sys.argv
    results, all_ok = _collect()
    if as_md:
        _print_markdown(results)
    else:
        _print_text(results, all_ok)
    for r in results:
        assert r["err_ok"],  f"Row N={r['N']}: max-err {r['err']:.3e} >= paper {r['paper_err']:.3e}"
        assert r["cold_ok"], f"Row N={r['N']}: cold {r['cold']:.4f}ms >= paper {r['paper_ms']:.4f}ms"
        assert r["warm_ok"], f"Row N={r['N']}: warm {r['warm']:.4f}ms >= paper {r['paper_ms']:.4f}ms"
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

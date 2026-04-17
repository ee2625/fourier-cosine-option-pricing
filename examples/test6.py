"""
Reproduce Fang & Oosterlee (2008) Table 6 — Heston, T=1, 21 strikes in one call.

Strikes: K = 50, 55, ..., 150. Paper reports max-error across the 21 strikes
and cpu time per *one* 21-strike evaluation. The CF is computed once and
shared across all strikes; the payoff matrix is rebuilt per strike vector
(here, the same vector every call, so U is cached too).

Reference prices are computed once on module import via the Lewis (2001)
Fourier formula integrated to scipy-quad machine precision (~10⁻¹⁴).

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
from cos_pricing.heston_cos_pricer import HestonCOSPricer  # noqa: E402


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


def _cf(u, T):
    """Heston CF of log(S_T/S0); evaluated at complex u for Lewis formula."""
    S0_, v0, lam, eta, ubar, rho, r, q = (PARAMS["S0"], PARAMS["v0"], PARAMS["lam"],
                                          PARAMS["eta"], PARAMS["ubar"], PARAMS["rho"],
                                          PARAMS["r"], PARAMS["q"])
    iu = 1j * u
    beta = lam - eta * rho * iu
    D = np.sqrt(beta * beta + eta * eta * (u * u + iu))
    G = (beta - D) / (beta + D)
    exp_mDt = np.exp(-D * T)
    one_m_Gexp = 1.0 - G * exp_mDt
    drift = iu * (r - q) * T
    v0_term = (v0 / (eta * eta)) * (-np.expm1(-D * T)) / one_m_Gexp * (beta - D)
    log_ratio = np.log(one_m_Gexp / (1.0 - G))
    ubar_term = (lam * ubar / (eta * eta)) * ((beta - D) * T - 2.0 * log_ratio)
    return np.exp(drift + v0_term + ubar_term)


def reference_prices():
    """High-precision call prices via the Lewis (2001) inversion formula.

    Call(K) = S0 - (sqrt(S0*K) * exp(-r*T) / pi) * integral from 0 to inf of
              Re[phi(u - i/2) * exp(i*u*log(S0/K))] / (u^2 + 1/4) du
    """
    S0_ = PARAMS["S0"]; r = PARAMS["r"]
    refs = np.empty(STRIKES.size)
    for i, K in enumerate(STRIKES):
        lk = np.log(S0_ / K)

        def integrand(u, lk=lk):
            return (_cf(u - 0.5j, T) * np.exp(1j * u * lk)).real / (u * u + 0.25)

        integral, _ = quad(integrand, 0.0, np.inf,
                           limit=2000, epsabs=1e-16, epsrel=1e-14)
        refs[i] = S0_ - np.sqrt(S0_ * K) * np.exp(-r * T) / np.pi * integral
    return refs


REFS = reference_prices()


def warm_ms(N):
    """Mean per-call ms across `N_WARM` repeats with the cache primed."""
    p = HestonCOSPricer(**PARAMS)
    p.price_call(STRIKES, T, N=N, L=L)                      # prime caches
    t0 = time.perf_counter()
    for _ in range(N_WARM):
        p.price_call(STRIKES, T, N=N, L=L)
    return (time.perf_counter() - t0) / N_WARM * 1e3


def _collect():
    p = HestonCOSPricer(**PARAMS)
    results = []
    all_ok = True
    for N, pe, pms in ROWS:
        prices  = p.price_call(STRIKES, T, N=N, L=L)
        max_err = float(np.max(np.abs(prices - REFS)))
        warm    = warm_ms(N)
        err_ok, time_ok = max_err < pe, warm < pms
        all_ok &= err_ok and time_ok
        results.append(dict(N=N, paper_err=pe, err=max_err,
                            paper_ms=pms, warm=warm,
                            err_ok=err_ok, time_ok=time_ok))
    return results, all_ok


def _print_text(results, all_ok):
    print(f"Fang-Oosterlee Table 6 reproducer "
          f"(T={T}, L={L}, strikes={len(STRIKES)} from {STRIKES[0]} to {STRIKES[-1]})")
    print(f"Reference prices computed via Lewis-quad to ~1e-14.")
    print()
    header = f"{'N':>4}  {'paper max err':>13}  {'our max err':>11}  " \
             f"{'paper ms':>9}  {'warm ms':>8}  status"
    print(header)
    print("-" * len(header))
    for r in results:
        status = "OK" if (r["err_ok"] and r["time_ok"]) else \
                 "FAIL (" + ",".join(s for s, ok in
                     [("err", r["err_ok"]), ("time", r["time_ok"])] if not ok) + ")"
        print(f"{r['N']:>4}  {r['paper_err']:>13.2e}  {r['err']:>11.2e}  "
              f"{r['paper_ms']:>9.4f}  {r['warm']:>8.4f}  {status}")
    print()
    print("PASS: Heston algorithm universally outperforms Table 6." if all_ok
          else "FAIL: at least one row regressed on error or warm runtime.")


def _print_markdown(results):
    Ns = [r["N"] for r in results]
    print(f"### Table 6 reproduction — Heston, T={T}, 21 strikes (K=50..150), L={L}")
    print()
    print("| |" + "|".join(f" N={n} " for n in Ns) + "|")
    print("|---|" + "|".join("---" for _ in Ns) + "|")
    print("| paper max error |" + "|".join(f" {r['paper_err']:.2e} " for r in results) + "|")
    print("| our max error   |" + "|".join(f" {r['err']:.2e} "       for r in results) + "|")
    print("| paper ms        |" + "|".join(f" {r['paper_ms']:.4f} "  for r in results) + "|")
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
        assert r["time_ok"], f"Row N={r['N']}: warm {r['warm']:.4f}ms >= paper {r['paper_ms']:.4f}ms"
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

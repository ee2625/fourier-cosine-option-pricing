"""
Stress-sweep audit: cos_pricing COS vs pyfeng FFT across parameter grids.

For each model (Heston, VG, CGMY), walk a grid that intentionally goes past
the paper's benchmark regime, compute COS and FFT prices for an ATM call,
report relative error, and classify each cell as PASS / DEGRADED / BROKEN.
"""
import sys, os
ROOT = os.path.abspath(".")

sys.path[:] = [p for p in sys.path if os.path.abspath(p) not in (ROOT, "")]
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np
import pyfeng as pf
import importlib
pyfeng_local = os.path.join(ROOT, "pyfeng")
if pyfeng_local not in pf.__path__:
    pf.__path__.append(pyfeng_local)
importlib.invalidate_caches()

from cos_pricing import HestonCOSPricer, VgModel, CgmyModel

REL_PASS     = 1e-3
REL_DEGRADED = 1e-1


def classify(rel_err):
    if not np.isfinite(rel_err):
        return "BROKEN"
    if rel_err < REL_PASS:
        return "PASS"
    if rel_err < REL_DEGRADED:
        return "DEGRADED"
    return "BROKEN"


def fmt(rel_err):
    if not np.isfinite(rel_err):
        return "  inf  "
    return f"{rel_err:>7.1e}"


SPOT = 100.0
ATM  = 100.0


# ── Heston grid: (vov, T) at moderate (kappa, theta, v0, rho) and stress rho
# ────────────────────────────────────────────────────────────────────────────

def heston_grid():
    print("=" * 78)
    print("HESTON  (kappa=0.5, theta=v0=0.04, rho=-0.5; ATM call at K=100)")
    print("Sweeping vov vs T:")
    print("=" * 78)

    vovs = [0.3, 0.5, 0.7, 1.0, 1.5]
    Ts   = [0.5, 1.0, 5.0, 10.0]
    kappa, theta_lr, v0, rho = 0.5, 0.04, 0.04, -0.5

    print(f"\n{'vov-T':>10} | " + " | ".join(f"T={T:>4}" for T in Ts))
    print("-" * (12 + len(Ts) * 12))

    rows = []
    for vov in vovs:
        cells = []
        for T in Ts:
            try:
                fft = pf.HestonFft(sigma=v0, vov=vov, mr=kappa, rho=rho,
                                   theta=theta_lr, intr=0.0, divr=0.0)
                ref = float(fft.price(ATM, SPOT, T))
                cos = HestonCOSPricer(S0=SPOT, v0=v0, lam=kappa, eta=vov,
                                      ubar=theta_lr, rho=rho, r=0.0, q=0.0)
                got = float(cos.price_call(ATM, tau=T, N=160))
                rel = abs(got - ref) / max(abs(ref), 1e-10)
            except Exception:
                rel = float("inf")
            cells.append(rel)
            rows.append((vov, T, "rho=-0.5", rel))
        print(f"{vov:>10.2f} | " + " | ".join(fmt(c) for c in cells))

    print(f"\n{'vov-T':>10} | " + " | ".join(f"T={T:>4}" for T in Ts) + "   (rho = -0.9)")
    print("-" * (12 + len(Ts) * 12))
    rho_stress = -0.9
    for vov in vovs:
        cells = []
        for T in Ts:
            try:
                fft = pf.HestonFft(sigma=v0, vov=vov, mr=kappa, rho=rho_stress,
                                   theta=theta_lr, intr=0.0, divr=0.0)
                ref = float(fft.price(ATM, SPOT, T))
                cos = HestonCOSPricer(S0=SPOT, v0=v0, lam=kappa, eta=vov,
                                      ubar=theta_lr, rho=rho_stress, r=0.0, q=0.0)
                got = float(cos.price_call(ATM, tau=T, N=160))
                rel = abs(got - ref) / max(abs(ref), 1e-10)
            except Exception:
                rel = float("inf")
            cells.append(rel)
            rows.append((vov, T, "rho=-0.9", rel))
        print(f"{vov:>10.2f} | " + " | ".join(fmt(c) for c in cells))

    return rows


# ── VG grid: (sigma, nu) at fixed theta, T
# ────────────────────────────────────────────────────────────────────────────

def vg_grid():
    print()
    print("=" * 78)
    print("VARIANCE GAMMA  (theta=-0.14, intr=0.0; ATM call at K=100)")
    print("Sweeping sigma vs nu at T=1:")
    print("=" * 78)

    sigmas = [0.05, 0.12, 0.25, 0.50]
    nus    = [0.05, 0.20, 0.50, 1.00, 2.00]
    theta  = -0.14

    print(f"\n{'sigma-nu':>10} | " + " | ".join(f"nu={n:>4}" for n in nus))
    print("-" * (12 + len(nus) * 12))

    rows = []
    for sigma in sigmas:
        cells = []
        for nu in nus:
            try:
                # Constraint: 1 - theta*nu - 0.5*sigma^2*nu > 0
                if 1.0 - theta * nu - 0.5 * sigma**2 * nu <= 0:
                    cells.append(float("nan"))
                    continue
                fft = pf.VarGammaFft(sigma=sigma, theta=theta, vov=nu,
                                     intr=0.0, divr=0.0)
                ref = float(fft.price(ATM, SPOT, 1.0))
                vg  = VgModel(sigma=sigma, theta=theta, nu=nu,
                              intr=0.0, divr=0.0)
                got = float(vg.price(ATM, SPOT, 1.0, n_cos=512))
                rel = abs(got - ref) / max(abs(ref), 1e-10)
            except Exception:
                rel = float("inf")
            cells.append(rel)
            rows.append((sigma, nu, "T=1", rel))
        print(f"{sigma:>10.2f} | " + " | ".join(
            "  N/A  " if (isinstance(c, float) and np.isnan(c)) else fmt(c)
            for c in cells))

    return rows


# ── CGMY grid: (Y, T) at fixed C, G, M
# ────────────────────────────────────────────────────────────────────────────

def cgmy_grid():
    print()
    print("=" * 78)
    print("CGMY  (C=1, G=5, M=10, intr=0.0; ATM call at K=100)")
    print("Sweeping Y vs T:")
    print("=" * 78)

    Ys = [0.5, 1.0001, 1.3, 1.5, 1.7, 1.8, 1.9, 1.95]  # avoid Y in {0,1,2}
    Ts = [0.5, 1.0, 2.0, 5.0]

    print(f"\n{'Y-T':>10} | " + " | ".join(f"T={T:>4}" for T in Ts))
    print("-" * (12 + len(Ts) * 12))

    rows = []
    for Y in Ys:
        cells = []
        for T in Ts:
            try:
                fft = pf.CgmyFft(C=1.0, G=5.0, M=10.0, Y=Y,
                                 intr=0.0, divr=0.0)
                ref = float(fft.price(ATM, SPOT, T))
                cgmy = CgmyModel(C=1.0, G=5.0, M=10.0, Y=Y,
                                 intr=0.0, divr=0.0)
                got = float(cgmy.price(ATM, SPOT, T, n_cos=512, L=10.0))
                rel = abs(got - ref) / max(abs(ref), 1e-10)
            except Exception:
                rel = float("inf")
            cells.append(rel)
            rows.append((Y, T, "", rel))
        print(f"{Y:>10.4g} | " + " | ".join(fmt(c) for c in cells))

    return rows


# ── Summary
# ────────────────────────────────────────────────────────────────────────────

def summarize(name, rows):
    counts = {"PASS": 0, "DEGRADED": 0, "BROKEN": 0}
    broken_cells = []
    for params in rows:
        rel = params[-1]
        if isinstance(rel, float) and np.isnan(rel):
            continue
        cls = classify(rel)
        counts[cls] += 1
        if cls == "BROKEN":
            broken_cells.append((params[:-1], rel))

    total = sum(counts.values())
    print(f"\n{name}: {counts['PASS']}/{total} PASS  "
          f"{counts['DEGRADED']}/{total} DEGRADED  "
          f"{counts['BROKEN']}/{total} BROKEN")
    if broken_cells:
        print("  Broken cells:")
        for params, rel in broken_cells:
            print(f"    {params}  rel_err={fmt(rel)}")


if __name__ == "__main__":
    heston_rows = heston_grid()
    vg_rows     = vg_grid()
    cgmy_rows   = cgmy_grid()

    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    summarize("Heston", heston_rows)
    summarize("VG    ", vg_rows)
    summarize("CGMY  ", cgmy_rows)

    print()
    print("Classification:")
    print(f"  PASS     : rel_err < {REL_PASS}")
    print(f"  DEGRADED : {REL_PASS} <= rel_err < {REL_DEGRADED}")
    print(f"  BROKEN   : rel_err >= {REL_DEGRADED} or NaN/Inf")

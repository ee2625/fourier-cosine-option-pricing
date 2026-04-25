"""
Dimensionless BSM pricing surface.

Produces one figure: the Black-Scholes-Merton dimensionless price surface
``C / S_0`` over the two pi-groups ``(K/S_0, sigma*sqrt(T))``. Scatter
points from three different ``(S_0, sigma, T)`` triples are overlaid to
show that the raw four inputs collapse onto one universal surface.

Run:
    PYTHONPATH=src python examples/dimensionless_surface.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection

from cos_pricing import BsmModel, cos_price


# BSM truncation needs to cover  |log(K/S_0)|  at every grid cell. With
# K/S_0 up to 2.0 and sigma*sqrt(T) as low as 0.05, the worst case needs
# L >= log(2) / 0.05 ~ 14.  We pick L = 20 uniformly — the COS method is
# insensitive to oversized L for Gaussian densities.
L_BSM = 20.0


# ── Dimensionless pi-group grids ──────────────────────────────────────────
#   pi_1 = K / S_0           moneyness, ranges over [0.5, 2.0]
#   pi_2 = sigma * sqrt(T)   vol-time,  ranges over [0.05, 0.8]
# Rates set to zero so there is no third pi-group to worry about (r*T = 0).
N_K, N_V = 40, 40
k_over_s = np.linspace(0.5, 2.0, N_K)
sig_sqT  = np.linspace(0.05, 0.8, N_V)
KK, VV   = np.meshgrid(k_over_s, sig_sqT, indexing="ij")

S0_REF, T_REF = 100.0, 1.0


def bsm_surface(S0, T, k_grid, sigma_grid):
    """Return C/S_0 on a meshgrid of (K/S0, sigma*sqrt(T)) using wide trunc."""
    Kgrid = k_grid * S0
    sigma = sigma_grid / np.sqrt(T)
    Z = np.empty_like(Kgrid)
    for j in range(sigma.shape[1]):
        sig = float(sigma[0, j])
        m   = BsmModel(sigma=sig)
        a, b = m.trunc_range(T, L=L_BSM)
        Z[:, j] = cos_price(m.char_func(T), T, Kgrid[:, j],
                            fwd=S0, df=1.0, cp=+1,
                            trunc_range=(a, b)) / S0
    return Z


# ── Compute the BSM surface on the canonical frame ────────────────────────
print("Computing BSM surface on a 40x40 (K/S0, sigma*sqrt(T)) grid ...")
Z_bsm = bsm_surface(S0_REF, T_REF, KK, VV)
print(f"  C/S_0 range: [{Z_bsm.min():.4f}, {Z_bsm.max():.4f}]")


# ── Collapse demo: scatter from 3 different (S0, sigma, T) triples ────────
# Each triple lives at a different sigma*sqrt(T), so the scatter points spread
# across the surface rather than stacking at one spot. Numerical equality at
# the SAME pi-point is verified separately below (to machine precision).
triples_spread = [
    ("S0=50,  T=2,    sigma=0.14",  50.0,  2.00, 0.14,  np.array([0.7, 1.0, 1.3])),
    ("S0=250, T=0.25, sigma=0.80",  250.0, 0.25, 0.80,  np.array([0.6, 0.9, 1.4])),
    ("S0=1000, T=4,   sigma=0.30", 1000.0, 4.00, 0.30,  np.array([0.8, 1.1, 1.8])),
]
triples_common = [
    (50.0,  2.00, 0.4 / np.sqrt(2.00)),
    (250.0, 0.25, 0.4 / np.sqrt(0.25)),
    (1000., 4.00, 0.4 / np.sqrt(4.00)),
]
common_moneyness = np.array([0.7, 1.0, 1.3, 1.6])

sc_x, sc_y, sc_z, sc_labels = [], [], [], []
for lbl, S0, T, sig, moneyness in triples_spread:
    m = BsmModel(sigma=sig)
    a, b = m.trunc_range(T, L=L_BSM)
    fwd  = S0
    prices = cos_price(m.char_func(T), T, moneyness * S0,
                       fwd=fwd, df=1.0, cp=+1, trunc_range=(a, b))
    sc_x.append(moneyness)
    sc_y.append(np.full_like(moneyness, sig * np.sqrt(T)))
    sc_z.append(prices / S0)
    sc_labels.append(lbl)

print("\nCollapse verification — same pi-point, different raw inputs")
print("  sigma*sqrt(T) = 0.4 for all three triples below")
hdr = f"{'K/S0':>8}" + "".join(
    f"  C/S0: S0={S0:g},T={T:g},sig={sig:.3f}" for S0, T, sig in triples_common
)
print(hdr)
collapse_vals = []
for moneyness in common_moneyness:
    row = []
    for S0, T, sig in triples_common:
        m = BsmModel(sigma=sig)
        a, b = m.trunc_range(T, L=L_BSM)
        p = float(cos_price(m.char_func(T), T, moneyness * S0,
                            fwd=S0, df=1.0, cp=+1, trunc_range=(a, b)))
        row.append(p / S0)
    collapse_vals.append(row)
    print(f"{moneyness:>8.2f}  " + "  ".join(f"{v:>26.14f}" for v in row))
spread = max(max(r) - min(r) for r in collapse_vals)
print(f"max spread across triples: {spread:.2e}   (should be ~ machine epsilon)")


# ── Plot ──────────────────────────────────────────────────────────────────
out_dir = os.path.join(os.path.dirname(__file__), "..", "docs")
os.makedirs(out_dir, exist_ok=True)

fig = plt.figure(figsize=(9, 7))
ax  = fig.add_subplot(111, projection="3d")
ax.plot_surface(KK, VV, Z_bsm, cmap="viridis", alpha=0.70,
                linewidth=0, antialiased=True)
colors  = ["#d62728", "#ff7f0e", "#1f77b4"]
markers = ["o", "s", "^"]
for i, (lbl, xs, ys, zs) in enumerate(zip(sc_labels, sc_x, sc_y, sc_z)):
    ax.scatter(xs, ys, zs, s=90, color=colors[i], edgecolor="black",
               linewidth=0.8, marker=markers[i], label=lbl, depthshade=False)
ax.set_xlabel(r"$K / S_0$")
ax.set_ylabel(r"$\sigma \sqrt{T}$")
ax.set_zlabel(r"$C / S_0$")
ax.set_title("BSM dimensionless surface — arbitrary $(S_0, \\sigma, T)$ triples land on it")
ax.legend(loc="upper left", fontsize=9, framealpha=0.92)
ax.view_init(elev=22, azim=-58)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, "fig_bsm_collapse.png"), dpi=150,
            bbox_inches="tight")
print(f"\nSaved: {os.path.normpath(os.path.join(out_dir, 'fig_bsm_collapse.png'))}")

if "--show" in sys.argv:
    plt.show()
else:
    plt.close("all")

print("\nDone.")

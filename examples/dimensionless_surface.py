"""
Dimensionless pricing surfaces — Track C of the dimensional-analysis
extension.

Produces two figures:

  1. BSM dimensionless collapse.
     The Black-Scholes-Merton price surface  C/S_0  over the two
     dimensionless pi-groups  (K/S_0, sigma*sqrt(T)).  Scatter points
     from three different (S_0, sigma, T) triples are overlaid to show
     that the raw 4 inputs collapse onto one universal surface.

  2. BSM vs Bachelier overlay.
     Both dimensionless surfaces on one axes with the Bachelier
     absolute volatility calibrated via  sigma_n = S_0 * sigma
     (the small-vol matching from MATH5030 Lecture 3, slide 22).
     The gap shrinks near ATM and for small sigma*sqrt(T) — this is
     the graphical version of the "BSM log-return ~ Bachelier
     percentage move when moves are small" argument.

Run:
    PYTHONPATH=src python examples/dimensionless_surface.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection

from cos_pricing import BsmModel, NormalCos, cos_price


# BSM truncation needs to cover  |log(K/S_0)|  at every grid cell. With
# K/S_0 up to 2.0 and sigma*sqrt(T) as low as 0.05, the worst case needs
# L >= log(2) / 0.05 ~ 14.  We pick L = 20 uniformly — the COS method is
# insensitive to oversized L for Gaussian densities.
L_BSM  = 20.0
L_NORM = 25.0     # Bachelier: need L * sigma*sqrt(T) > |K - F|/S_0 worst case


# ── Dimensionless pi-group grids (ranges from team_tasks.docx) ─────────────
#   pi_1 = K / S_0           moneyness, ranges over [0.5, 2.0]
#   pi_2 = sigma * sqrt(T)   vol-time,  ranges over [0.05, 0.8]
# Rates set to zero so there is no third pi-group to worry about (r*T = 0).
N_K, N_V = 40, 40                         # grid resolution
k_over_s  = np.linspace(0.5, 2.0, N_K)     # x-axis
sig_sqT   = np.linspace(0.05, 0.8, N_V)    # y-axis
KK, VV    = np.meshgrid(k_over_s, sig_sqT, indexing="ij")

# Canonical (S_0, T) used to realise the grid. Any choice works because of
# collapse — we verify that below in panel 1.
S0_REF, T_REF = 100.0, 1.0


def bsm_surface(S0, T, k_grid, sigma_grid):
    """Return C/S_0 on a meshgrid of (K/S0, sigma*sqrt(T)) using wide trunc."""
    Kgrid = k_grid * S0
    sigma = sigma_grid / np.sqrt(T)
    Z = np.empty_like(Kgrid)
    for j in range(sigma.shape[1]):
        sig = float(sigma[0, j])
        m   = BsmModel(sigma=sig)
        a, b = m.trunc_range(T, L=L_BSM)           # override default L
        # cos_price expects fwd and df; with r=q=0 these are trivial
        Z[:, j] = cos_price(m.char_func(T), T, Kgrid[:, j],
                            fwd=S0, df=1.0, cp=+1,
                            trunc_range=(a, b)) / S0
    return Z


def bachelier_surface(S0, T, k_grid, sigma_grid):
    """
    Return C_n/S_0 on the same meshgrid, with Bachelier absolute vol
    calibrated so sigma_n = S_0 * sigma (the small-move correspondence).

    NormalCos.trunc_range defaults to L=10, too narrow for our wide-moneyness
    corner. We widen it per-instance to L_NORM.
    """
    Kgrid = k_grid * S0
    sigma = sigma_grid / np.sqrt(T)
    Z = np.empty_like(Kgrid)
    for j in range(sigma.shape[1]):
        sig_n = float(sigma[0, j]) * S0
        m = NormalCos(sigma=sig_n)
        # One-line override: call the existing method with a different L default
        m.trunc_range = (lambda texp, _m=m, _L=L_NORM: _m.__class__.trunc_range(_m, texp, L=_L))
        Z[:, j] = m.price(Kgrid[:, j], S0, T, cp=+1) / S0
    return Z


# ── Compute the BSM surface on the canonical frame ─────────────────────────
print("Computing BSM surface on a 40x40 (K/S0, sigma*sqrt(T)) grid ...")
Z_bsm = bsm_surface(S0_REF, T_REF, KK, VV)
print(f"  C/S_0 range: [{Z_bsm.min():.4f}, {Z_bsm.max():.4f}]")


# ── Panel 1 — Collapse demo: scatter from 3 different (S0, sigma, T) triples ──
# Each triple lives at a different sigma*sqrt(T), so the scatter points spread
# across the surface rather than stacking at one spot. The visual claim is
# "arbitrary raw inputs produce points that lie on the surface" — the
# numerical equality of different triples at the SAME pi-point is verified
# separately below (to machine precision).
triples_spread = [
    # (label,                 S0,   T,    sigma,  moneyness values)
    ("S0=50,  T=2,    sigma=0.14",  50.0,  2.00, 0.14,  np.array([0.7, 1.0, 1.3])),
    ("S0=250, T=0.25, sigma=0.80",  250.0, 0.25, 0.80,  np.array([0.6, 0.9, 1.4])),
    ("S0=1000, T=4,   sigma=0.30", 1000.0, 4.00, 0.30,  np.array([0.8, 1.1, 1.8])),
]
# Pre-compute for the collapse-verification table: one common pi-point sampled
# by all three triples (sigma*sqrt(T) = 0.4, moneyness grid).
triples_common = [
    # (S0,  T,    sigma_BSM), each producing sigma*sqrt(T) = 0.4
    (50.0,  2.00, 0.4 / np.sqrt(2.00)),
    (250.0, 0.25, 0.4 / np.sqrt(0.25)),
    (1000., 4.00, 0.4 / np.sqrt(4.00)),
]
common_moneyness = np.array([0.7, 1.0, 1.3, 1.6])

# Scatter geometry for the plot
sc_x, sc_y, sc_z, sc_labels = [], [], [], []
for lbl, S0, T, sig, moneyness in triples_spread:
    # Per-triple: wide L for safety even though these triples don't hit the corner
    m = BsmModel(sigma=sig)
    a, b = m.trunc_range(T, L=L_BSM)
    fwd  = S0      # r = q = 0
    prices = cos_price(m.char_func(T), T, moneyness * S0,
                       fwd=fwd, df=1.0, cp=+1, trunc_range=(a, b))
    sc_x.append(moneyness)
    sc_y.append(np.full_like(moneyness, sig * np.sqrt(T)))
    sc_z.append(prices / S0)
    sc_labels.append(lbl)

# Numeric collapse verification at a common pi-point
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


# ── Panel 2 — Bachelier surface and gap ────────────────────────────────────
print("\nComputing Bachelier surface with sigma_n = S_0 * sigma ...")
Z_nml = bachelier_surface(S0_REF, T_REF, KK, VV)
gap   = Z_nml - Z_bsm
print(f"  max |C_n/S_0 - C/S_0| overall     : {np.max(np.abs(gap)):.4f}")
# Mask to |K/S0 - 1| < 0.1 and sigma*sqrt(T) < 0.2 → "near ATM, small vol-time"
near_atm = (np.abs(KK - 1.0) < 0.10) & (VV < 0.20)
print(f"  max |gap| near ATM & small sigT   : {np.max(np.abs(gap[near_atm])):.4f}")
far     = (np.abs(KK - 1.0) > 0.40) | (VV > 0.60)
print(f"  max |gap| far from ATM / large sigT: {np.max(np.abs(gap[far])):.4f}")


# ── Plots ──────────────────────────────────────────────────────────────────
out_dir = os.path.join(os.path.dirname(__file__), "..", "docs")
os.makedirs(out_dir, exist_ok=True)

# Figure 1: dimensionless BSM surface + collapse scatter
fig1 = plt.figure(figsize=(9, 7))
ax1  = fig1.add_subplot(111, projection="3d")
ax1.plot_surface(KK, VV, Z_bsm, cmap="viridis", alpha=0.70,
                 linewidth=0, antialiased=True)
colors  = ["#d62728", "#ff7f0e", "#1f77b4"]    # red, orange, blue
markers = ["o", "s", "^"]
for i, (lbl, xs, ys, zs) in enumerate(zip(sc_labels, sc_x, sc_y, sc_z)):
    ax1.scatter(xs, ys, zs, s=90, color=colors[i], edgecolor="black",
                linewidth=0.8, marker=markers[i], label=lbl, depthshade=False)
ax1.set_xlabel(r"$K / S_0$")
ax1.set_ylabel(r"$\sigma \sqrt{T}$")
ax1.set_zlabel(r"$C / S_0$")
ax1.set_title("BSM dimensionless surface — arbitrary $(S_0, \\sigma, T)$ triples land on it")
ax1.legend(loc="upper left", fontsize=9, framealpha=0.92)
ax1.view_init(elev=22, azim=-58)
fig1.tight_layout()
fig1.savefig(os.path.join(out_dir, "fig_bsm_collapse.png"), dpi=150,
             bbox_inches="tight")
print(f"\nSaved: {os.path.normpath(os.path.join(out_dir, 'fig_bsm_collapse.png'))}")

# Figure 2: BSM + Bachelier overlay (3D) and gap heatmap (2D)
fig2 = plt.figure(figsize=(14, 6))

ax2a = fig2.add_subplot(1, 2, 1, projection="3d")
ax2a.plot_surface(KK, VV, Z_bsm, cmap="Blues",   alpha=0.75,
                  linewidth=0, antialiased=True)
ax2a.plot_surface(KK, VV, Z_nml, cmap="Oranges", alpha=0.60,
                  linewidth=0, antialiased=True)
from matplotlib.patches import Patch
ax2a.legend(handles=[Patch(facecolor="steelblue",  label=r"BSM  $C/S_0$"),
                     Patch(facecolor="darkorange", label=r"Bachelier $C_n/S_0$, "
                                                         r"$\sigma_n = S_0 \sigma$")],
            loc="upper left", fontsize=9, framealpha=0.92)
ax2a.set_xlabel(r"$K / S_0$")
ax2a.set_ylabel(r"$\sigma \sqrt{T}$")
ax2a.set_zlabel(r"price / $S_0$")
ax2a.set_title("Overlay (3D)")
ax2a.view_init(elev=22, azim=-58)

ax2b = fig2.add_subplot(1, 2, 2)
# 2D heatmap of the absolute gap. Orient so K/S0 is the horizontal axis.
vmax = float(np.max(np.abs(gap)))
im = ax2b.pcolormesh(KK, VV, np.abs(gap),
                     cmap="magma_r", vmin=0.0, vmax=vmax, shading="auto")
cs = ax2b.contour(KK, VV, np.abs(gap),
                  levels=[1e-3, 5e-3, 1e-2, 2e-2, 5e-2],
                  colors="white", linewidths=0.9)
ax2b.clabel(cs, fontsize=8, fmt="%.0e")
ax2b.axvline(1.0, color="lime", linestyle="--", linewidth=1.0, alpha=0.8)
ax2b.set_xlabel(r"$K / S_0$")
ax2b.set_ylabel(r"$\sigma \sqrt{T}$")
ax2b.set_title(r"$\left|\,C_n/S_0 - C/S_0\,\right|$ — gap is minimal near ATM & small $\sigma\sqrt{T}$")
fig2.colorbar(im, ax=ax2b, label="absolute gap")

fig2.suptitle("BSM vs Bachelier dimensionless surfaces",
              fontsize=13, y=1.00)
fig2.tight_layout()
fig2.savefig(os.path.join(out_dir, "fig_bsm_vs_bachelier.png"), dpi=150,
             bbox_inches="tight")
print(f"Saved: {os.path.normpath(os.path.join(out_dir, 'fig_bsm_vs_bachelier.png'))}")

# If run interactively, show the figures
if "--show" in sys.argv:
    plt.show()
else:
    plt.close("all")

print("\nDone.")
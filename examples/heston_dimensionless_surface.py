"""
Heston dimensionless pricing surface — Track C extension to multi-factor models.

The Heston model has *eight* independent dimensionless pi-groups versus
BSM's three.  A natural choice (one of many — pi-bases are not unique):

    K / S0                              # moneyness
    (r - q) * T                         # carry
    kappa * T                           # mean reversion in dim'less time
    v0 * T                              # integrated initial variance
    ubar * T                            # integrated long-run variance
    eta * T                             # vol-of-vol scale
    rho                                 # correlation (already dim'less)
    eta^2 / (kappa * ubar)              # Feller ratio (alternative to eta*T)

This script realises a 2-parameter slice of the 8-D price surface.
Six pi-groups are held fixed; two — moneyness K/S0 and the BSM-equivalent
vol sqrt(v0*T) — are swept across a meshgrid.  The dimensionless price
C/S0 is plotted as a surface, then scatter from three different *raw*
sextets of (S0, T, v0, kappa, eta, ubar) is overlaid.  All three sextets
hit the same six fixed pi-points and lie on the surface to ~ machine
precision — the Heston analogue of the BSM (S0, sigma, T) collapse.

Run:
    PYTHONPATH=src python examples/heston_dimensionless_surface.py

Outputs:
    docs/fig_heston_collapse.png
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection

from cos_pricing import HestonCOSPricer


# ── Six fixed pi-groups ────────────────────────────────────────────────────
# These define which 2-D slice of the 8-D Heston price surface we are
# visualising.  Anyone reading the figure should be able to reproduce it
# by realising any (S0, T) pair satisfying these six dimensionless ratios.
PI_RHO     = -0.70    # rho  (already dimensionless)
PI_RT      =  0.0     # (r - q) * T
PI_KAPPA_T =  1.50    # kappa * T   — ~1.5 mean reversions per maturity
PI_UBAR_T  =  0.04    # ubar * T    — long-run integrated variance ~ 20%^2 * T
PI_ETA_T   =  0.40    # eta * T     — equivalent: eta/sqrt(kappa*ubar) = 1.633

# (Feller ratio at this slice: 2*kappa*ubar / eta^2 = 2*1.5*0.04/0.16 = 0.75
#  i.e. Feller is mildly violated — variance can touch zero. Mathematically
#  fine; chosen this way to keep the slice numerically interesting.)


# ── Two pi-groups varied across the meshgrid ───────────────────────────────
N_K, N_V    = 35, 35
k_over_s    = np.linspace(0.5, 2.0, N_K)            # x-axis: K/S0
sqrt_v0T    = np.linspace(0.10, 0.60, N_V)          # y-axis: BSM-equiv vol
KK, VV      = np.meshgrid(k_over_s, sqrt_v0T, indexing="ij")

# Canonical (S0, T) used to *realise* the grid as actual Heston pricers.
# Any (S0, T) works — the surface is the same — and we verify that with
# the collapse triples below.
S0_REF, T_REF = 100.0, 1.0

# Truncation half-width in log-moneyness units.  The corner cell has
# v0 = 0.36 (vol = 60%), giving stdev[log(S_T/S0)] ~ 0.6 over T = 1, so
# half = 10 covers ~16 stdevs even there.  Pinning the *half-width* (not
# the L scalar) is what guarantees identical truncation across the
# collapse triples below.
HALF_WIDTH = 10.0
N_COS      = 256


# ── Helpers ────────────────────────────────────────────────────────────────

def _make_pricer(S0, T, sqrt_v0T_val):
    """Return a HestonCOSPricer realising the six fixed pi-groups at (S0, T)."""
    v0    = (sqrt_v0T_val ** 2) / T          # back out v0 from sqrt(v0*T) and T
    kappa = PI_KAPPA_T  / T
    ubar  = PI_UBAR_T   / T
    eta   = PI_ETA_T    / T
    return HestonCOSPricer(
        S0=S0, v0=v0, lam=kappa, eta=eta, ubar=ubar, rho=PI_RHO,
        r=PI_RT / T, q=0.0,
    )


def _L_for_pricer(m):
    """Choose L such that L * sigma_h = HALF_WIDTH (log-moneyness units)."""
    sigma_h = float(np.sqrt(m.ubar + m.v0 * m.eta))
    return HALF_WIDTH / sigma_h


def heston_surface(S0, T, k_grid, sqrt_v0T_grid):
    """C/S0 on a (K/S0, sqrt(v0*T)) meshgrid, six other pi-groups fixed."""
    Kgrid = k_grid * S0
    Z     = np.empty_like(Kgrid)
    for j in range(sqrt_v0T_grid.shape[1]):
        sigT = float(sqrt_v0T_grid[0, j])
        m    = _make_pricer(S0, T, sigT)
        Z[:, j] = m.price(Kgrid[:, j], T, cp=+1, N=N_COS, L=_L_for_pricer(m)) / S0
    return Z


# ── Compute the surface ────────────────────────────────────────────────────
print("Computing Heston dimensionless surface on a "
      f"{N_K}x{N_V} (K/S0, sqrt(v0*T)) grid ...")
print(f"  fixed pi-groups: rho={PI_RHO}, (r-q)T={PI_RT}, "
      f"kappa*T={PI_KAPPA_T}, ubar*T={PI_UBAR_T}, eta*T={PI_ETA_T}")
Z_h = heston_surface(S0_REF, T_REF, KK, VV)
print(f"  C/S0 range: [{Z_h.min():.4f}, {Z_h.max():.4f}]")


# ── Scatter overlay: 3 spread triples on different parts of the surface ───
# All three realise the same six fixed pi-groups but at wildly different
# raw (S0, T, v0, kappa, eta, ubar).  They land on the surface because the
# six fixed groups + their (K/S0, sqrt(v0*T)) coordinate fully determine
# the dimensionless price.
spread_specs = [
    # (S0,    T,    sqrt(v0*T),  moneyness vector,            label)
    (   50.0, 2.00, 0.30, np.array([0.7, 1.0, 1.3]),  "S0=50,  T=2.00, v0=0.045, kappa=0.75"),
    (  250.0, 0.25, 0.55, np.array([0.6, 0.9, 1.4]),  "S0=250, T=0.25, v0=1.21,  kappa=6.00"),
    ( 1000.0, 4.00, 0.20, np.array([0.8, 1.1, 1.8]),  "S0=1000,T=4.00, v0=0.010, kappa=0.375"),
]

sc_x, sc_y, sc_z, sc_lbl = [], [], [], []
for S0, T, sigT, mn, lbl in spread_specs:
    m  = _make_pricer(S0, T, sigT)
    px = m.price(mn * S0, T, cp=+1, N=N_COS, L=_L_for_pricer(m)) / S0
    sc_x.append(mn)
    sc_y.append(np.full_like(mn, sigT))
    sc_z.append(px)
    sc_lbl.append(lbl)


# ── Numeric collapse table at one common pi-point ─────────────────────────
# Three triples all at sqrt(v0*T) = 0.4 with the same six fixed groups.
# Bit-equal C/S0 across triples is the signature of dimensional invariance.
print("\nCollapse verification — same pi-point, completely different raw inputs")
print("  sqrt(v0*T) = 0.40 for all three triples below")
print("  raw (v0, kappa, eta, ubar) span ranges of 16x across triples")

common_specs = [
    (   50.0, 2.00, 0.40, "(S0=50,  T=2.00, v0=0.080)"),
    (  250.0, 0.25, 0.40, "(S0=250, T=0.25, v0=0.640)"),
    ( 1000.0, 4.00, 0.40, "(S0=1000,T=4.00, v0=0.040)"),
]
common_moneyness = np.array([0.7, 1.0, 1.3, 1.6])

# Header
hdr = f"{'K/S0':>8}" + "".join(f"  C/S0: {lbl}" for _, _, _, lbl in common_specs)
print(hdr)

collapse = []
for mn in common_moneyness:
    row = []
    for S0, T, sigT, _ in common_specs:
        m = _make_pricer(S0, T, sigT)
        p = float(m.price(mn * S0, T, cp=+1, N=N_COS, L=_L_for_pricer(m))) / S0
        row.append(p)
    collapse.append(row)
    print(f"{mn:>8.2f}  " + "  ".join(f"{v:>26.16f}" for v in row))

spread_max = max(max(r) - min(r) for r in collapse)
print(f"\nmax spread across triples: {spread_max:.2e}   (~ machine epsilon)")


# ── Plot ───────────────────────────────────────────────────────────────────
out_dir = os.path.join(os.path.dirname(__file__), "..", "docs")
os.makedirs(out_dir, exist_ok=True)

fig = plt.figure(figsize=(10, 7.5))
ax  = fig.add_subplot(111, projection="3d")

ax.plot_surface(KK, VV, Z_h, cmap="plasma", alpha=0.70,
                linewidth=0, antialiased=True)

colors  = ["#d62728", "#ff7f0e", "#1f77b4"]    # red, orange, blue
markers = ["o", "s", "^"]
for i, (lbl, xs, ys, zs) in enumerate(zip(sc_lbl, sc_x, sc_y, sc_z)):
    ax.scatter(xs, ys, zs, s=90, color=colors[i], edgecolor="black",
               linewidth=0.8, marker=markers[i], label=lbl, depthshade=False)

ax.set_xlabel(r"$K / S_0$")
ax.set_ylabel(r"$\sqrt{v_0\,T}$")
ax.set_zlabel(r"$C / S_0$")
ax.set_title(
    "Heston dimensionless surface — different raw "
    r"$(S_0, T, v_0, \kappa, \eta, \bar v)$ collapse onto one slice"
    "\n"
    r"fixed: $\rho = -0.7,\ \kappa T = 1.5,\ \bar v T = 0.04,"
    r"\ \eta T = 0.4,\ (r-q)T = 0$",
    fontsize=10,
)
ax.legend(loc="upper left", fontsize=8.5, framealpha=0.92)
ax.view_init(elev=22, azim=-58)

fig.tight_layout()
out_path = os.path.join(out_dir, "fig_heston_collapse.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved: {os.path.normpath(out_path)}")

if "--show" in sys.argv:
    plt.show()
else:
    plt.close("all")

print("\nDone.")
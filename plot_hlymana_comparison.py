#!/usr/bin/env python
"""
Validate fixed command_hlymana vs command_hlymana_fast and measure speedup.

Focuses on:
  - Lyman-edge region (TAULYMEDGE = C0[NHI], not C0[Lyα])
  - Lyα core agreement
  - Runtime: slow vs fast cut vs fast full
"""

import contextlib
import io
import time

import matplotlib.pyplot as plt
import numpy as np

from wrplot import wrplotter

wrplot = wrplotter()

uv = wrplot.command_include("./UV_spectrum.txt")
print(f"Input spectrum: {len(uv[0])} pts,  {uv[0][0]:.1f}–{uv[0][-1]:.1f} Å")

COLDENS = [-1.0e20, -5.0e20, -1.0e21]
LABELS = [
    r"$N_\mathrm{H} = -10^{20}\,\mathrm{cm}^{-2}$",
    r"$N_\mathrm{H} = -5\times10^{20}\,\mathrm{cm}^{-2}$",
    r"$N_\mathrm{H} = -10^{21}\,\mathrm{cm}^{-2}$",
]

C0_0 = 7.056e-11
C0_NHI = 3.072e-11
XLYEDGE = 911.2670504
XLAM_NHI = 912.64482


def run_quiet(fn, data, **kwargs):
    buf = io.StringIO()
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(buf):
        out = fn(data, **kwargs)
    return out, time.perf_counter() - t0


def inp():
    return np.array([uv[0].copy(), uv[1].copy()])


# Warm up numba
_kw0 = dict(key="coldens", value=COLDENS[0], vrad=0.0, temp=-1, vturb=0)
_tiny = wrplot.command_cut(uv, (1210, 1220), axis="x")
run_quiet(
    wrplot.command_hlymana_fast,
    np.array([_tiny[0].copy(), _tiny[1].copy()]),
    **_kw0,
)

results = []
print(
    f"{'N_H':>12}  {'slow':>9}  {'fast_cut':>9}  {'fast_full':>9}  "
    f"{'s/cut':>7}  {'s/full':>7}  {'max|d| core':>12}  {'max|d| edge':>12}"
)

for N in COLDENS:
    kw = dict(key="coldens", value=N, vrad=0.0, temp=-1, vturb=0)

    slow, t_slow = run_quiet(wrplot.command_hlymana, inp(), **kw)
    fast_c, t_cut = run_quiet(wrplot.command_hlymana_fast, inp(), **kw)
    fast_f, t_full = run_quiet(
        wrplot.command_hlymana_fast, inp(), cut_lyman=False, **kw
    )

    # Lyα core window
    x_core = np.linspace(1205.0, 1225.0, 20000)
    ys = np.interp(x_core, slow[0], slow[1])
    yc = np.interp(x_core, fast_c[0], fast_c[1])
    yf = np.interp(x_core, fast_f[0], fast_f[1])
    d_core = np.max(np.abs(yc - ys))

    # Edge band
    x_edge = np.linspace(XLYEDGE + 0.02, XLAM_NHI - 0.02, 5000)
    ys_e = np.interp(x_edge, slow[0], slow[1])
    yc_e = np.interp(x_edge, fast_c[0], fast_c[1])
    yf_e = np.interp(x_edge, fast_f[0], fast_f[1])
    d_edge = np.max(np.abs(yc_e - ys_e))

    # Expected edge factor magnitude (for annotation)
    edge_fac = np.exp(N * C0_NHI)
    wrong_fac = np.exp(N * C0_0)

    print(
        f"{N:12.1e}  {t_slow:7.2f}s  {t_cut:7.3f}s  {t_full:7.3f}s  "
        f"{t_slow/t_cut:6.1f}x  {t_slow/t_full:6.1f}x  "
        f"{d_core:12.2e}  {d_edge:12.2e}"
    )
    results.append(dict(
        N=N, slow=slow, fast_c=fast_c, fast_f=fast_f,
        t_slow=t_slow, t_cut=t_cut, t_full=t_full,
        x_core=x_core, ys=ys, yc=yc, yf=yf,
        x_edge=x_edge, ys_e=ys_e, yc_e=yc_e, yf_e=yf_e,
        d_core=d_core, d_edge=d_edge,
        edge_fac=edge_fac, wrong_fac=wrong_fac,
    ))

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(14, 11))
gs = fig.add_gridspec(3, 3, height_ratios=[1.4, 1.2, 1.0], hspace=0.35, wspace=0.28)

for col, (res, lab) in enumerate(zip(results, LABELS)):
    # Row 0: full Lyman window
    ax = fig.add_subplot(gs[0, col])
    ax.plot(uv[0], uv[1], color="0.75", lw=0.4, label="original")
    ax.plot(res["slow"][0], res["slow"][1], color="C0", lw=0.8,
            label=f"slow [{res['t_slow']:.1f}s]")
    ax.plot(res["fast_c"][0], res["fast_c"][1], color="C2", lw=0.9, ls="--",
            label=f"fast cut [{res['t_cut']:.3f}s]")
    ax.set_xlim(920, 1300)
    ax.set_ylim(0, 1.35)
    ax.set_title(lab, fontsize=11)
    if col == 0:
        ax.set_ylabel("Normalised flux")
        ax.legend(loc="upper right", fontsize=7)
    ax.text(
        0.02, 0.05,
        f"speedup cut={res['t_slow']/res['t_cut']:.0f}×  "
        f"full={res['t_slow']/res['t_full']:.0f}×",
        transform=ax.transAxes, fontsize=7, family="monospace",
    )

    # Row 1: Lyα core + residual
    ax = fig.add_subplot(gs[1, col])
    ax.plot(res["x_core"], res["ys"], color="C0", lw=1.0, label="slow")
    ax.plot(res["x_core"], res["yc"], color="C2", lw=0.9, ls="--", label="fast cut")
    ax.set_xlim(1205, 1225)
    ax.set_ylim(0, 1.2)
    if col == 0:
        ax.set_ylabel("Normalised flux")
        ax.legend(loc="upper right", fontsize=7)
    ax.set_title(f"Lyα core  max|Δ|={res['d_core']:.2e}", fontsize=9)

    # Row 2: edge region on a synthetic continuum (UV_spectrum starts at 920 Å)
    ax = fig.add_subplot(gs[2, col])
    flat = np.array([
        np.linspace(911.3, 913.0, 4000),
        np.ones(4000),
    ])
    kw = dict(key="coldens", value=res["N"], vrad=0.0, temp=-1, vturb=0)
    slow_e, _ = run_quiet(wrplot.command_hlymana, flat, **kw)
    fast_e, _ = run_quiet(
        wrplot.command_hlymana_fast, flat, cut_lyman=False, **kw
    )
    xe = np.linspace(XLYEDGE + 0.02, XLAM_NHI - 0.02, 2000)
    yse = np.interp(xe, slow_e[0], slow_e[1])
    yfe = np.interp(xe, fast_e[0], fast_e[1])
    ax.plot(xe, yse, color="C0", lw=1.0, label="slow")
    ax.plot(xe, yfe, color="C2", lw=0.9, ls="--", label="fast")
    ax.axhline(res["edge_fac"], color="C3", ls=":", lw=0.9,
               label=rf"$\mathrm{{e}}^{{N C_0[\mathrm{{NHI}}]}}$")
    ax.axhline(res["wrong_fac"], color="0.5", ls=":", lw=0.8,
               label=rf"$\mathrm{{e}}^{{N C_0[\mathrm{{Ly}}\alpha]}}$ (old bug)")
    ax.set_xlim(XLYEDGE, XLAM_NHI)
    ax.set_xlabel(r"$\lambda$ / Å")
    if col == 0:
        ax.set_ylabel("Normalised flux")
        ax.legend(loc="best", fontsize=6)
    ax.set_title(
        f"Lyman edge (synthetic)  max|Δ|={np.max(np.abs(yse-yfe)):.2e}",
        fontsize=9,
    )

fig.suptitle(
    "HLYMANA fix: TAULYMEDGE = C0[NHI], Fortran-like TAUCRIT  —  "
    "slow vs fast",
    fontsize=12,
)
out = "hlymana_comparison.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nWrote {out}")
plt.close()

# Timing bar summary
fig, ax = plt.subplots(figsize=(7, 3.5))
methods = ["slow", "fast full", "fast cut"]
x = np.arange(len(COLDENS))
width = 0.25
for i, (key, label, color) in enumerate([
    ("t_slow", "slow", "0.5"),
    ("t_full", "fast full", "C3"),
    ("t_cut", "fast cut", "C2"),
]):
    vals = [r[key] for r in results]
    ax.bar(x + (i - 1) * width, vals, width, label=label, color=color, log=True)
ax.set_xticks(x)
ax.set_xticklabels([f"{N:.0e}" for N in COLDENS])
ax.set_xlabel(r"$N_\mathrm{H}$ / cm$^{-2}$")
ax.set_ylabel("Time [s] (log)")
ax.set_title("HLYMANA runtime")
ax.legend(fontsize=8)
plt.tight_layout()
out2 = "hlymana_timing.png"
plt.savefig(out2, dpi=150)
print(f"Wrote {out2}")
plt.close()

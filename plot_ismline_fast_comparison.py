#!/usr/bin/env python
"""
Benchmark and compare command_ismline vs command_ismline_fast.

Uses the OG2.plot / test2.txt setup (no convolution): two Si III λ1206
components (GAL + SMC) on the formal * IUE spectrum.
"""

import time

import matplotlib.pyplot as plt
import numpy as np

from wrplot import wrplotter

wrplot = wrplotter()

# --- OG2.plot parameters ---
EBV = 0.14
EBV_GAL = 0.06
EBV_SMC = EBV - EBV_GAL
CD_S_GAL = -(1.25 * EBV * 6.86e21) * 3.55e-5 / 1000
CD_S_SMC = -(EBV_SMC * 3.8e21 * 6) * 3.55e-5 / 1000
VRAD_GAL = -156.0
VRAD_SMC = 0.0
VTURB_GAL = 10.0
VTURB_SMC = 30.0
L0 = 1206.51
GAMM = 6.1e9
FVALUE = 1.67
ATMASS = 28.0

FORMAL = (
    "/home/daniel/POWR_MODELS/SMC/MODELS/"
    "T35_L4.5_g3.6_Mdot-8.5_v700_CN/formal.plot"
)

kw_gal = dict(
    coldens=CD_S_GAL, l0=L0, gamm=GAMM, fvalue=FVALUE, atmass=ATMASS,
    vturb=VTURB_GAL, vrad=VRAD_GAL,
)
kw_smc = dict(
    coldens=CD_S_SMC, l0=L0, gamm=GAMM, fvalue=FVALUE, atmass=ATMASS,
    vturb=VTURB_SMC, vrad=VRAD_SMC,
)


def load():
    return wrplot.command_include(FORMAL, inckey="* IUE")


def run_pair(fn, **extra):
    m = load()
    t0 = time.perf_counter()
    m = fn(m, **kw_gal, **extra)
    m = fn(m, **kw_smc, **extra)
    return m, time.perf_counter() - t0


def time_median(fn, n=10, **extra):
    times = []
    out = None
    for _ in range(n):
        out, dt = run_pair(fn, **extra)
        times.append(dt)
    return out, float(np.median(times))


# Warm up numba
_tiny = wrplot.command_cut(load(), (1200, 1210), axis="x")
wrplot.command_ismline_fast(
    np.array([_tiny[0].copy(), _tiny[1].copy()]), **kw_gal, cut_ism=True
)

ref = np.loadtxt("test2.txt")

print("Running slow command_ismline (full spectrum)...")
slow, t_slow = run_pair(wrplot.command_ismline, info=False)
print(f"  {t_slow:.2f}s  npts={len(slow[0])}")

print("Running fast cut_ism=False (full spectrum, median of 10)...")
fast_full, t_full = time_median(
    wrplot.command_ismline_fast, cut_ism=False, info=False
)
print(f"  {t_full*1000:.2f}ms  npts={len(fast_full[0])}  "
      f"speedup={t_slow/t_full:.0f}x")

print("Running fast cut_ism=True, window=30 Å (median of 10)...")
fast_cut, t_cut = time_median(
    wrplot.command_ismline_fast, cut_ism=True, window=30.0, info=False
)
print(f"  {t_cut*1000:.2f}ms  npts={len(fast_cut[0])}  "
      f"speedup={t_slow/t_cut:.0f}x")

print("Running fast cut_ism=True, window=10 Å (median of 10)...")
fast_cut10, t_cut10 = time_median(
    wrplot.command_ismline_fast, cut_ism=True, window=10.0, info=False
)
print(f"  {t_cut10*1000:.2f}ms  npts={len(fast_cut10[0])}  "
      f"speedup={t_slow/t_cut10:.0f}x")


def compare(name, model, xref, yref, lo=1170.0, hi=1250.0, direct=False):
    """Compare model to reference. If direct and npts match, align by index."""
    if direct and model[0].shape == xref.shape:
        # Same expansion → same ordering; λ may differ by ~1e-7 (float vs REAL*4)
        yp = model[1]
        x = model[0]
        yr = yref
    else:
        yp = np.interp(xref, model[0], model[1])
        x = xref
        yr = yref
    mask = (x >= lo) & (x <= hi)
    d = yp[mask] - yr[mask]
    return dict(
        name=name,
        max=np.max(np.abs(d)),
        rms=np.sqrt(np.mean(d ** 2)),
        d=d,
        x=x[mask],
        yp=yp[mask],
        yr=yr[mask],
    )


# Compare to WRPLOT test2.txt and to slow Python
st_slow = compare("slow vs WRPLOT", slow, ref[:, 0], ref[:, 1], direct=True)
st_full = compare("fast full vs WRPLOT", fast_full, ref[:, 0], ref[:, 1],
                  direct=True)
st_cut = compare("fast cut30 vs WRPLOT", fast_cut, ref[:, 0], ref[:, 1],
                 direct=(len(fast_cut[0]) == len(ref)))
st_cut10 = compare("fast cut10 vs WRPLOT", fast_cut10, ref[:, 0], ref[:, 1])
st_fs = compare("fast full vs slow", fast_full, slow[0], slow[1], direct=True)
st_cs = compare("fast cut30 vs slow", fast_cut, slow[0], slow[1],
                direct=(len(fast_cut[0]) == len(slow[0])))

print("\nAgreement in 1170–1250 Å:")
for st in (st_slow, st_full, st_cut, st_cut10, st_fs, st_cs):
    print(f"  {st['name']:28s}  max|Δ|={st['max']:.3e}  rms={st['rms']:.3e}")

print("\nTiming summary:")
print(f"  {'method':28s}  {'time':>10s}  {'speedup':>8s}")
print(f"  {'slow':28s}  {t_slow:9.2f}s  {'—':>8s}")
print(f"  {'fast full (cut_ism=False)':28s}  {t_full*1000:8.2f}ms  {t_slow/t_full:7.0f}x")
print(f"  {'fast cut window=30':28s}  {t_cut*1000:8.2f}ms  {t_slow/t_cut:7.0f}x")
print(f"  {'fast cut window=10':28s}  {t_cut10*1000:8.2f}ms  {t_slow/t_cut10:7.0f}x")

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
CLIGHT = 299792.458
lam_gal = L0 * (1.0 + VRAD_GAL / CLIGHT)
lam_smc = L0 * (1.0 + VRAD_SMC / CLIGHT)

fig, axes = plt.subplots(
    3, 2, figsize=(12, 9),
    gridspec_kw={"height_ratios": [1.4, 1.0, 1.0]},
)

# Left: fast full vs WRPLOT
ax = axes[0, 0]
ax.plot(st_full["x"], st_full["yr"], color="C0", lw=1.0, label="WRPLOT test2.txt")
ax.plot(st_full["x"], st_full["yp"], color="C3", lw=0.9, ls="--",
        label=f"fast full [{t_full*1000:.1f}ms, {t_slow/t_full:.0f}×]")
ax.axvline(lam_gal, color="0.5", ls=":", lw=0.8)
ax.axvline(lam_smc, color="0.5", ls=":", lw=0.8)
ax.set_xlim(1170, 1250)
ax.set_ylim(0, 1.4)
ax.set_ylabel("Normalised flux")
ax.set_title("cut_ism=False (full spectrum)")
ax.legend(loc="lower right", fontsize=8)
ax.text(0.02, 0.95, f"max|Δ|={st_full['max']:.2e}  rms={st_full['rms']:.2e}",
        transform=ax.transAxes, fontsize=8, family="monospace", va="top")

ax = axes[1, 0]
ax.plot(st_full["x"], st_full["d"], color="C3", lw=0.8)
ax.axhline(0, color="k", lw=0.5)
ax.set_xlim(1170, 1250)
ax.set_ylabel("Python − WRPLOT")
ax.set_title("Residual (fast full)")

ax = axes[2, 0]
zoom = (st_full["x"] >= 1204.5) & (st_full["x"] <= 1208.0)
ax.plot(st_full["x"][zoom], st_full["yr"][zoom], color="C0", lw=1.2, label="WRPLOT")
ax.plot(st_full["x"][zoom], st_full["yp"][zoom], color="C3", lw=1.0, ls="--",
        label="fast full")
ax.axvline(lam_gal, color="0.5", ls=":", lw=0.8)
ax.axvline(lam_smc, color="0.4", ls="--", lw=0.8)
ax.set_xlim(1204.5, 1208.0)
ax.set_ylim(-0.05, 1.2)
ax.set_xlabel(r"$\lambda$ / Å")
ax.set_ylabel("Normalised flux")
ax.legend(loc="upper right", fontsize=8)
ax.set_title("Zoom on line cores")

# Right: fast cut30 vs WRPLOT + timing bar
ax = axes[0, 1]
ax.plot(st_cut["x"], st_cut["yr"], color="C0", lw=1.0, label="WRPLOT test2.txt")
ax.plot(st_cut["x"], st_cut["yp"], color="C2", lw=0.9, ls="--",
        label=f"fast cut±30 [{t_cut*1000:.1f}ms, {t_slow/t_cut:.0f}×]")
ax.axvline(lam_gal, color="0.5", ls=":", lw=0.8)
ax.axvline(lam_smc, color="0.5", ls=":", lw=0.8)
ax.set_xlim(1170, 1250)
ax.set_ylim(0, 1.4)
ax.set_title("cut_ism=True, window=30 Å")
ax.legend(loc="lower right", fontsize=8)
ax.text(0.02, 0.95, f"max|Δ|={st_cut['max']:.2e}  rms={st_cut['rms']:.2e}",
        transform=ax.transAxes, fontsize=8, family="monospace", va="top")

ax = axes[1, 1]
ax.plot(st_cut["x"], st_cut["d"], color="C2", lw=0.8, label="cut±30")
ax.plot(st_cut10["x"], st_cut10["d"], color="C1", lw=0.7, alpha=0.8,
        label=f"cut±10 (max|Δ|={st_cut10['max']:.2e})")
ax.axhline(0, color="k", lw=0.5)
ax.set_xlim(1170, 1250)
ax.set_ylabel("Python − WRPLOT")
ax.set_title("Residual (windowed fast)")
ax.legend(loc="lower right", fontsize=8)

ax = axes[2, 1]
methods = ["slow", "fast full", "fast ±30", "fast ±10"]
times = [t_slow, t_full, t_cut, t_cut10]
colors = ["0.5", "C3", "C2", "C1"]
bars = ax.barh(methods[::-1], times[::-1], color=colors[::-1], log=True)
ax.set_xlabel("Time [s]  (2× ISMLINE: GAL + SMC, log scale)")
ax.set_title("Runtime")
for bar, t, s in zip(bars, times[::-1], [t_slow / t for t in times][::-1]):
    label = f"{t:.2f}s" if t >= 1 else f"{t*1000:.1f}ms"
    if s > 1.01:
        label += f"  ({s:.0f}×)"
    ax.text(bar.get_width() * 1.15, bar.get_y() + bar.get_height() / 2,
            label, va="center", fontsize=8)

fig.suptitle(
    "command_ismline_fast vs WRPLOT (test2.txt, no CONVOL)",
    fontsize=12,
)
plt.tight_layout()
out = "ismline_fast_comparison.png"
plt.savefig(out, dpi=150)
print(f"\nWrote {out}")
plt.close()

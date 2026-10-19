#!/usr/bin/env python
"""
Validate WRPLOT-faithful rotational convolution and measure speedup.

Compares command_convol (pure Python) vs command_convol_fast (numba) using
the Unsoeld kernel with fixed Δλ_max [Å].
"""

import time

import matplotlib.pyplot as plt
import numpy as np

from wrplot import wrplotter

wrplot = wrplotter()

# Synthetic spectrum: continuum + three narrow lines
x = np.linspace(4800.0, 5200.0, 25000)
y = np.ones_like(x)
for c, amp, w in [(4900.0, 0.7, 0.12), (5000.0, 0.9, 0.10), (5100.0, 0.6, 0.15)]:
    y -= amp * np.exp(-0.5 * ((x - c) / w) ** 2)
data = np.array([x, y])

D_ANG = 2.5          # WRPLOT Δλ_max [Å]
BETAS = [0.0, 0.6]

# Warm up numba
_ = wrplot.command_convol_fast(data, "rot", D_ANG, beta=0.0)


def time_median(fn, n=7):
    times = []
    out = None
    for _ in range(n):
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
    return out, float(np.median(times))


results = []
print(f"{'beta':>6}  {'slow':>10}  {'fast':>10}  {'speedup':>8}  {'max|Δ|':>10}")
for beta in BETAS:
    slow, t_slow = time_median(
        lambda b=beta: wrplot.command_convol(data, "rot", D_ANG, beta=b)
    )
    fast, t_fast = time_median(
        lambda b=beta: wrplot.command_convol_fast(
            data, "rot", D_ANG, beta=b, precision=None
        )
    )
    dmax = np.max(np.abs(slow[1] - fast[1]))
    print(
        f"{beta:6.1f}  {t_slow*1000:8.1f}ms  {t_fast*1000:8.1f}ms  "
        f"{t_slow/t_fast:7.0f}x  {dmax:10.2e}"
    )
    results.append(dict(
        beta=beta, slow=slow, fast=fast,
        t_slow=t_slow, t_fast=t_fast, dmax=dmax,
    ))

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(11, 7.5),
                         gridspec_kw={"height_ratios": [1.4, 1.0]})

for col, res in enumerate(results):
    ax = axes[0, col]
    ax.plot(data[0], data[1], color="0.7", lw=0.6, label="original")
    ax.plot(res["slow"][0], res["slow"][1], color="C0", lw=1.0,
            label=f"slow [{res['t_slow']*1000:.0f}ms]")
    ax.plot(res["fast"][0], res["fast"][1], color="C3", lw=0.9, ls="--",
            label=f"fast [{res['t_fast']*1000:.1f}ms, "
                  f"{res['t_slow']/res['t_fast']:.0f}×]")
    ax.set_xlim(4985, 5015)
    ax.set_ylim(0, 1.15)
    ax.set_title(rf"ROT Δλ={D_ANG} Å,  $\beta$={res['beta']}")
    ax.legend(loc="lower right", fontsize=8)
    if col == 0:
        ax.set_ylabel("Normalised flux")
    ax.text(
        0.02, 0.95, f"max|Δ|={res['dmax']:.2e}",
        transform=ax.transAxes, fontsize=8, family="monospace", va="top",
    )

    ax = axes[1, col]
    ax.plot(res["slow"][0], res["slow"][1] - res["fast"][1], color="C2", lw=0.8)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlim(4985, 5015)
    ax.set_xlabel(r"$\lambda$ / Å")
    if col == 0:
        ax.set_ylabel("slow − fast")
    ax.set_title("Residual")

fig.suptitle(
    "Rotational convolution: Unsoeld kernel, fixed Δλ (WRPLOT) — "
    "command_convol vs command_convol_fast",
    fontsize=11,
)
plt.tight_layout()
out = "convol_rot_comparison.png"
plt.savefig(out, dpi=150)
print(f"\nWrote {out}")
plt.close()

# Timing bar
fig, ax = plt.subplots(figsize=(6, 3.2))
labels = [rf"β={r['beta']}" for r in results]
x = np.arange(len(labels))
w = 0.35
ax.bar(x - w / 2, [r["t_slow"] * 1000 for r in results], w,
       label="slow (Python)", color="C0")
ax.bar(x + w / 2, [r["t_fast"] * 1000 for r in results], w,
       label="fast (numba)", color="C3")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Time [ms]")
ax.set_title(f"ROT convol runtime (N={len(data[0])} pts, Δλ={D_ANG} Å)")
ax.legend(fontsize=8)
for i, r in enumerate(results):
    ax.text(i, max(r["t_slow"], r["t_fast"]) * 1000 * 1.05,
            f"{r['t_slow']/r['t_fast']:.0f}×", ha="center", fontsize=9)
plt.tight_layout()
out2 = "convol_rot_timing.png"
plt.savefig(out2, dpi=150)
print(f"Wrote {out2}")
plt.close()

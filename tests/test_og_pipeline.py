#!/usr/bin/env python
"""
End-to-end validation of the OG.plot pipeline against WRPLOT.

Sequence (matches tests/test.plot plot 2 / OG.plot):
  INCLUDE data/continuum_iue.dat
  ISMLINE Si III λ1206 GAL
  ISMLINE Si III λ1206 SMC
  CONVOL GAUSS=0.5

Reference: ref/ref_ismline_convol.txt from ``./run_wrplot_refs.sh``.
"""

from __future__ import annotations

import time
import unittest

import matplotlib.pyplot as plt
import numpy as np

from helpers import (
    ATOL_FLUX,
    GAUSS_FWHM,
    WRPLOT_REF_CONVOL,
    apply_og_ismlines,
    compare_in_window,
    get_wrplotter,
    load_continuum_iue,
    load_ref,
    quiet,
    save_speedup_bar,
    savefig,
)


def run_pipeline(w, model, *, fast_ism=False, fast_convol=False, **ism_extra):
    out = apply_og_ismlines(w, model, fast=fast_ism, **ism_extra)
    # Match WRPLOT test.plot: SORT JOIN before CONVOL (no-op if already unique)
    out = quiet(w.command_sort, out)
    if fast_convol:
        out = quiet(
            w.command_convol_fast, out, "gauss", GAUSS_FWHM, precision=None
        )
    else:
        out = quiet(w.command_convol, out, "gauss", GAUSS_FWHM)
    return out


class TestOGPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.w = get_wrplotter()
        cls.model = load_continuum_iue(cls.w)
        tiny = quiet(cls.w.command_cut, cls.model, (1200, 1215), axis="x")
        apply_og_ismlines(cls.w, tiny, fast=True, cut_ism=True, window=20.0)
        cls.w.command_convol_fast(tiny, "gauss", GAUSS_FWHM)

    def test_slow_vs_wrplot(self):
        if not WRPLOT_REF_CONVOL.is_file():
            self.skipTest(
                "ref/ref_ismline_convol.txt missing — run ./run_wrplot_refs.sh"
            )
        ref = load_ref(WRPLOT_REF_CONVOL)
        out = run_pipeline(
            self.w, self.model, fast_ism=False, fast_convol=False
        )
        st = compare_in_window(out, ref[:, 0], ref[:, 1], 1170.0, 1250.0)
        self.assertLess(
            st["max"], ATOL_FLUX,
            msg=f"OG pipeline vs WRPLOT max|Δ|={st['max']:.3e} "
                f"rms={st['rms']:.3e}",
        )

    def test_fast_vs_slow(self):
        # Window for speed
        model = quiet(self.w.command_cut, self.model, (1100.0, 1350.0), axis="x")
        slow = run_pipeline(self.w, model, fast_ism=False, fast_convol=False)
        fast = run_pipeline(
            self.w, model,
            fast_ism=True, cut_ism=False, fast_convol=True,
        )
        st = compare_in_window(fast, slow[0], slow[1], 1170.0, 1250.0)
        self.assertLess(st["max"], ATOL_FLUX, msg=f"max|Δ|={st['max']:.3e}")


def make_plots():
    print("\n=== OG.plot pipeline validation ===")
    w = get_wrplotter()
    model = quiet(w.command_cut, load_continuum_iue(w), (1100.0, 1350.0), axis="x")
    print(f"Windowed input: {len(model[0])} pts, "
          f"{model[0][0]:.1f}–{model[0][-1]:.1f} Å")

    tiny = quiet(w.command_cut, model, (1200, 1215), axis="x")
    apply_og_ismlines(w, tiny, fast=True, cut_ism=True, window=20.0)
    w.command_convol_fast(tiny, "gauss", GAUSS_FWHM)

    configs = [
        ("slow+slow", dict(fast_ism=False, fast_convol=False)),
        ("fast+fast", dict(fast_ism=True, cut_ism=False, fast_convol=True)),
        ("fast_cut+fast", dict(fast_ism=True, cut_ism=True, window=30.0,
                               fast_convol=True)),
    ]

    results = {}
    for name, kw in configs:
        times = []
        out = None
        nrep = 1 if name.startswith("slow") else 5
        for _ in range(nrep):
            t0 = time.perf_counter()
            out = run_pipeline(w, model, **kw)
            times.append(time.perf_counter() - t0)
        results[name] = dict(out=out, t=float(np.median(times)))
        print(f"  {name:16s}  {results[name]['t']:.3f}s  n={len(out[0])}")

    ref = load_ref(WRPLOT_REF_CONVOL) if WRPLOT_REF_CONVOL.is_file() else None
    slow = results["slow+slow"]["out"]
    print("\nAgreement 1170–1250 Å:")
    for name, res in results.items():
        st = compare_in_window(res["out"], slow[0], slow[1], 1170, 1250)
        print(f"  {name:16s} vs slow: max|Δ|={st['max']:.3e}  rms={st['rms']:.3e}")
        if ref is not None:
            st_r = compare_in_window(res["out"], ref[:, 0], ref[:, 1], 1170, 1250)
            print(f"  {name:16s} vs WRPLOT: max|Δ|={st_r['max']:.3e}  "
                  f"rms={st_r['rms']:.3e}")

    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5),
                             gridspec_kw={"height_ratios": [1.6, 1.0]},
                             sharex=True)
    ax = axes[0]
    ax.plot(model[0], model[1], color="0.75", lw=0.35, label="input continuum")
    if ref is not None:
        ax.plot(ref[:, 0], ref[:, 1], color="k", lw=1.5, alpha=0.9,
                label="WRPLOT (test.plot → ref_ismline_convol.txt)", zorder=3)
    colors = {"slow+slow": "C0", "fast+fast": "C2", "fast_cut+fast": "C3"}
    for name, res in results.items():
        ax.plot(
            res["out"][0], res["out"][1], color=colors[name], lw=0.85,
            ls="--" if "fast" in name else "-",
            label=f"Python {name} [{res['t']:.2f}s]", zorder=2,
        )
    ax.set_xlim(1170, 1250)
    ax.set_ylim(0, 1.6)
    ax.set_ylabel("Normalised flux")
    ax.set_title(
        "OG pipeline QC: ISMLINE×2 + CONVOL GAUSS=0.5 vs WRPLOT test.plot"
    )
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1]
    if ref is not None:
        for name in ("slow+slow", "fast+fast", "fast_cut+fast"):
            st = compare_in_window(
                results[name]["out"], ref[:, 0], ref[:, 1], 1170, 1250
            )
            ax.plot(
                st["x"], st["d"], color=colors[name], lw=0.8,
                ls="--" if "fast" in name else "-",
                label=f"{name} − WRPLOT  (max={st['max']:.1e})",
            )
    ax.axhline(0, color="k", lw=0.4)
    ax.axhline(ATOL_FLUX, color="0.5", ls="--", lw=0.5, label=f"±{ATOL_FLUX}")
    ax.axhline(-ATOL_FLUX, color="0.5", ls="--", lw=0.5)
    ax.set_xlabel(r"$\lambda$ / Å")
    ax.set_ylabel("Δ flux")
    ax.legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, "og_pipeline_comparison.png")

    t_slow = results["slow+slow"]["t"]
    save_speedup_bar(
        ["slow+slow", "fast+fast", "fast_cut+fast"],
        [t_slow, t_slow, t_slow],
        [results[k]["t"] for k in ("slow+slow", "fast+fast", "fast_cut+fast")],
        title="OG.plot pipeline runtime",
        outfile="og_pipeline_speedup.png",
        ylabel="Time [s]",
    )


if __name__ == "__main__":
    make_plots()
    unittest.main(verbosity=2, exit=False)

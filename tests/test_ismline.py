#!/usr/bin/env python
"""
Validation + speedup suite for command_ismline / command_ismline_fast.

Uses OG2.plot parameters and WRPLOT reference ref/ref_ismline.txt
(produced by ``wrplot test.plot`` / ``./run_wrplot_refs.sh``).
Also sweeps several column densities / turbulent velocities on synthetic data.
"""

from __future__ import annotations

import unittest

import matplotlib.pyplot as plt
import numpy as np

from helpers import (
    ATOL_FLUX,
    WRPLOT_REF_ISMLINE,
    apply_og_ismlines,
    compare_in_window,
    flat_continuum,
    get_wrplotter,
    load_continuum_iue,
    load_ref,
    max_abs_diff,
    og_ism_kwargs_gal,
    quiet,
    save_speedup_bar,
    savefig,
    time_call,
)


class TestIsmlineAgreement(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.w = get_wrplotter()
        tiny = flat_continuum(1200.0, 1210.0, n=200)
        quiet(cls.w.command_ismline_fast, tiny, **og_ism_kwargs_gal(cut_ism=True))

    def test_slow_fast_flat_continuum_voigt(self):
        """Several VTURB / COLDENS on a flat continuum around λ1206."""
        data = flat_continuum(1190.0, 1220.0, n=5000)
        cases = [
            dict(coldens=-1e13, vturb=5.0, vrad=0.0, temp=-273.15),
            dict(coldens=-1e14, vturb=10.0, vrad=-156.0, temp=-273.15),
            dict(coldens=-5e14, vturb=30.0, vrad=0.0, temp=-273.15),
            dict(coldens=-1e14, vturb=0.0, vrad=0.0, temp=10000.0, atmass=28.0),
        ]
        for kw in cases:
            with self.subTest(**kw):
                base = dict(
                    l0=1206.51, gamm=6.1e9, fvalue=1.67, atmass=28.0
                )
                base.update(kw)
                slow = quiet(self.w.command_ismline, data, **base)
                fast = quiet(
                    self.w.command_ismline_fast, data,
                    cut_ism=False, **base,
                )
                d = max_abs_diff(slow[1], fast[1], slow[0], fast[0])
                self.assertLess(d, ATOL_FLUX, msg=f"max|Δ|={d:.3e}")

    def test_lorentz_vs_voigt_branch(self):
        """Pure Lorentz (T≤0, VTURB=0) must still agree slow↔fast."""
        data = flat_continuum(1200.0, 1215.0, n=3000)
        kw = dict(
            coldens=-1e14, l0=1206.51, gamm=6.1e9, fvalue=1.67,
            vturb=0.0, temp=-273.15, vrad=0.0,
        )
        slow = quiet(self.w.command_ismline, data, **kw)
        fast = quiet(self.w.command_ismline_fast, data, cut_ism=False, **kw)
        self.assertLess(
            max_abs_diff(slow[1], fast[1], slow[0], fast[0]), ATOL_FLUX
        )

    def test_vs_wrplot_reference(self):
        """GAL+SMC Si III vs WRPLOT ref/ref_ismline.txt (from test.plot)."""
        if not WRPLOT_REF_ISMLINE.is_file():
            self.skipTest("ref/ref_ismline.txt missing — run ./run_wrplot_refs.sh")
        ref = load_ref(WRPLOT_REF_ISMLINE)
        model = load_continuum_iue(self.w)
        slow = apply_og_ismlines(self.w, model, fast=False)
        st = compare_in_window(slow, ref[:, 0], ref[:, 1], 1170.0, 1250.0)
        self.assertLess(
            st["max"], ATOL_FLUX,
            msg=f"slow vs WRPLOT max|Δ|={st['max']:.3e} rms={st['rms']:.3e}",
        )


class TestIsmlineSpeed(unittest.TestCase):
    def test_fast_faster_on_continuum(self):
        import time

        w = get_wrplotter()
        model = quiet(w.command_cut, load_continuum_iue(w), (1150.0, 1280.0), axis="x")
        apply_og_ismlines(w, model, fast=True, cut_ism=True, window=30.0)

        t0 = time.perf_counter()
        apply_og_ismlines(w, model, fast=False)
        t_slow = time.perf_counter() - t0

        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            apply_og_ismlines(w, model, fast=True, cut_ism=False)
            times.append(time.perf_counter() - t0)
        t_fast = float(np.median(times))

        self.assertLess(t_fast, t_slow)
        self.assertGreater(t_slow / t_fast, 10.0)


def make_plots():
    print("\n=== ISMLINE validation plots ===")
    import time

    w = get_wrplotter()
    model = quiet(w.command_cut, load_continuum_iue(w), (1100.0, 1350.0), axis="x")
    print(f"Windowed input: {len(model[0])} pts")
    apply_og_ismlines(w, model, fast=True, cut_ism=True, window=30.0)

    def timed(fast, **extra):
        t0 = time.perf_counter()
        out = apply_og_ismlines(w, model, fast=fast, **extra)
        return out, time.perf_counter() - t0

    print("Running slow…")
    slow, t_slow = timed(False)
    print(f"  {t_slow:.2f}s  n={len(slow[0])}")

    print("Running fast full (median of 5)…")
    times_full = []
    fast_full = None
    for _ in range(5):
        out, dt = timed(True, cut_ism=False)
        times_full.append(dt)
        fast_full = out
    t_full = float(np.median(times_full))
    print(f"  {t_full*1e3:.2f}ms  speedup={t_slow/t_full:.0f}x")

    print("Running fast cut window=30 (median of 5)…")
    times_cut = []
    fast_cut = None
    for _ in range(5):
        out, dt = timed(True, cut_ism=True, window=30.0)
        times_cut.append(dt)
        fast_cut = out
    t_cut = float(np.median(times_cut))
    print(f"  {t_cut*1e3:.2f}ms  speedup={t_slow/t_cut:.0f}x")

    ref = load_ref(WRPLOT_REF_ISMLINE) if WRPLOT_REF_ISMLINE.is_file() else None

    st_fs = compare_in_window(fast_full, slow[0], slow[1], 1170, 1250, direct=True)
    st_cs = compare_in_window(fast_cut, slow[0], slow[1], 1170, 1250)
    print("\nAgreement 1170–1250 Å:")
    print(f"  fast full vs slow:  max|Δ|={st_fs['max']:.3e}")
    print(f"  fast cut30 vs slow: max|Δ|={st_cs['max']:.3e}")
    if ref is not None:
        st_wr = compare_in_window(slow, ref[:, 0], ref[:, 1], 1170, 1250)
        print(f"  slow vs WRPLOT:     max|Δ|={st_wr['max']:.3e}")

    # Spectrum + residual vs WRPLOT (test.plot)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5),
                             gridspec_kw={"height_ratios": [1.6, 1.0]},
                             sharex=True)
    ax = axes[0]
    ax.plot(model[0], model[1], color="0.75", lw=0.4, label="input continuum")
    if ref is not None:
        ax.plot(ref[:, 0], ref[:, 1], color="k", lw=1.5, alpha=0.9,
                label="WRPLOT (test.plot → ref_ismline.txt)", zorder=3)
    ax.plot(slow[0], slow[1], color="C0", lw=0.9,
            label=f"Python slow [{t_slow:.2f}s]", zorder=2)
    ax.plot(fast_full[0], fast_full[1], color="C2", lw=0.85, ls="--",
            label=f"Python fast full [{t_full*1e3:.1f}ms]", zorder=2)
    ax.plot(fast_cut[0], fast_cut[1], color="C3", lw=0.75, ls=":",
            label=f"Python fast cut30 [{t_cut*1e3:.1f}ms]", zorder=2)
    ax.set_xlim(1170, 1250)
    ax.set_ylim(0, 1.6)
    ax.set_ylabel("Normalised flux")
    ax.set_title("ISMLINE Si III λ1206 (GAL + SMC) — QC vs WRPLOT test.plot")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1]
    if ref is not None:
        st_wr = compare_in_window(slow, ref[:, 0], ref[:, 1], 1170, 1250)
        ax.plot(st_wr["x"], st_wr["d"], color="k", lw=0.9,
                label=f"Python slow − WRPLOT  (max={st_wr['max']:.1e})")
        st_fw = compare_in_window(fast_full, ref[:, 0], ref[:, 1], 1170, 1250)
        ax.plot(st_fw["x"], st_fw["d"], color="C2", lw=0.7, ls="--",
                label=f"Python fast − WRPLOT  (max={st_fw['max']:.1e})")
    ax.plot(st_fs["x"], st_fs["d"], color="C0", lw=0.5, alpha=0.6,
            label="fast − slow")
    ax.axhline(0, color="k", lw=0.4)
    ax.axhline(ATOL_FLUX, color="0.5", ls="--", lw=0.5, label=f"±{ATOL_FLUX}")
    ax.axhline(-ATOL_FLUX, color="0.5", ls="--", lw=0.5)
    ax.set_xlabel(r"$\lambda$ / Å")
    ax.set_ylabel("Δ flux")
    ax.legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, "ismline_comparison.png")

    save_speedup_bar(
        ["full spectrum", "cut window=30 Å"],
        [t_slow, t_slow],
        [t_full, t_cut],
        title=f"ISMLINE speedup (N≈{len(model[0])} pts windowed, GAL+SMC)",
        outfile="ismline_speedup.png",
        ylabel="Time [s]",
    )

    # Parameter sweep: coldens on synthetic continuum
    print("\nColdens sweep (synthetic)…")
    data0 = flat_continuum(1195.0, 1220.0, n=8000)
    labels, ts, tf = [], [], []
    for N in (-1e13, -5e13, -1e14, -5e14):
        kw = dict(
            coldens=N, l0=1206.51, gamm=6.1e9, fvalue=1.67,
            atmass=28.0, vturb=15.0, vrad=0.0,
        )
        quiet(w.command_ismline_fast, data0, cut_ism=False, **kw)  # warmup once
        _, t_s = time_call(w.command_ismline, data0, n=3, **kw)
        _, t_f = time_call(
            w.command_ismline_fast, data0, cut_ism=False, n=5, **kw
        )
        labels.append(f"N={N:.0e}")
        ts.append(t_s * 1000)
        tf.append(t_f * 1000)
        d = max_abs_diff(
            quiet(w.command_ismline, data0, **kw)[1],
            quiet(w.command_ismline_fast, data0, cut_ism=False, **kw)[1],
        )
        print(f"  N={N:.1e}: {t_s*1e3:.1f}ms → {t_f*1e3:.2f}ms  "
              f"({t_s/t_f:.0f}×)  max|Δ|={d:.2e}")
    save_speedup_bar(
        labels, ts, tf,
        title="ISMLINE speedup vs column density (synthetic, 8k pts)",
        outfile="ismline_speedup_coldens.png",
        ylabel="Time [ms]",
    )


if __name__ == "__main__":
    make_plots()
    unittest.main(verbosity=2, exit=False)

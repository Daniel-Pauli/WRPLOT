#!/usr/bin/env python
"""
Validation + speedup suite for command_hlymana / command_hlymana_fast.

Checks Lyman-edge τ (C0[NHI]), slow↔fast agreement, and speedup across
several column densities. Uses UV_spectrum.txt when available.
"""

from __future__ import annotations

import unittest

import matplotlib.pyplot as plt
import numpy as np

from helpers import (
    ATOL_FLUX,
    WRPLOT_REF_HLYMANA,
    WRPLOT_REF_HLYMANA_UV,
    compare_in_window,
    flat_continuum,
    get_wrplotter,
    load_continuum_lyman_flat,
    load_ref,
    load_uv,
    max_abs_diff,
    quiet,
    save_speedup_bar,
    savefig,
    time_call,
)

C0_LYA = 7.056e-11
C0_NHI = 3.072e-11
XLYEDGE = 911.2670504
XLAM_NHI = 912.64482


class TestHlymanaPhysics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.w = get_wrplotter()
        tiny = flat_continuum(1210.0, 1220.0, n=50)
        quiet(cls.w.command_hlymana_fast, tiny, key="coldens", value=-1e19,
              cut_lyman=False)

    def test_taulymedge_tracks_c0_nhi(self):
        ENH = -1.0e10
        expected = np.exp(ENH * C0_NHI)
        wrong = np.exp(ENH * C0_LYA)
        data = flat_continuum(911.4, 912.8, n=4000)
        out = quiet(self.w.command_hlymana, data, key="coldens", value=ENH)
        edge = (out[0] > XLYEDGE + 0.05) & (out[0] < XLAM_NHI - 0.05)
        red = (out[0] > XLAM_NHI + 0.02) & (out[0] < XLAM_NHI + 0.08)
        self.assertTrue(np.any(edge) and np.any(red))
        ratio = np.median(out[1][edge]) / np.median(out[1][red])
        self.assertLess(abs(ratio - expected), abs(ratio - wrong))

    def test_no_duplicate_wavelengths(self):
        data = flat_continuum(1200.0, 1225.0, n=500)
        out = quiet(self.w.command_hlymana, data, key="coldens", value=-1e21)
        self.assertTrue(np.all(np.diff(out[0]) > 0))

    def test_slow_fast_various_coldens(self):
        data = flat_continuum(1180.0, 1250.0, n=6000)
        for N in (-1e19, -1e20, -5e20, -1e21):
            with self.subTest(N=N):
                slow = quiet(self.w.command_hlymana, data,
                             key="coldens", value=N)
                fast = quiet(self.w.command_hlymana_fast, data,
                             key="coldens", value=N, cut_lyman=False)
                x = np.linspace(1205.0, 1225.0, 4000)
                d = max_abs_diff(slow[1], fast[1], slow[0], fast[0], x_common=x)
                # Adaptive TAUCRIT can leave small residual differences
                self.assertLess(d, 5e-2, msg=f"N={N}: max|Δ|={d:.3e}")

    def test_with_vturb(self):
        data = flat_continuum(1180.0, 1250.0, n=4000)
        kw = dict(key="coldens", value=-1e20, vturb=10.0, temp=10000.0)
        slow = quiet(self.w.command_hlymana, data, **kw)
        fast = quiet(self.w.command_hlymana_fast, data, cut_lyman=False, **kw)
        x = np.linspace(1205.0, 1225.0, 2000)
        d = max_abs_diff(slow[1], fast[1], slow[0], fast[0], x_common=x)
        self.assertLess(d, 5e-2)

    def test_vs_wrplot_flat_coldens(self):
        """Compare Python HLYMANA to WRPLOT refs on dense flat continuum."""
        data = load_continuum_lyman_flat(self.w)
        for N, path in WRPLOT_REF_HLYMANA.items():
            with self.subTest(N=N):
                if not path.is_file():
                    self.skipTest(f"{path.name} missing — run ./run_wrplot_refs.sh")
                ref = load_ref(path)
                out = quiet(
                    self.w.command_hlymana, data, key="coldens", value=N
                )
                st = compare_in_window(out, ref[:, 0], ref[:, 1], 920.0, 1280.0)
                self.assertLess(
                    st["max"], ATOL_FLUX,
                    msg=f"N={N}: max|Δ|={st['max']:.3e}",
                )

    def test_vs_wrplot_uv(self):
        if not WRPLOT_REF_HLYMANA_UV.is_file():
            self.skipTest("ref_hlymana_uv_1e20.txt missing — run ./run_wrplot_refs.sh")
        ref = load_ref(WRPLOT_REF_HLYMANA_UV)
        uv = load_uv(self.w)
        out = quiet(
            self.w.command_hlymana, uv, key="coldens", value=-1.0e20
        )
        st = compare_in_window(out, ref[:, 0], ref[:, 1], 920.0, 1280.0)
        self.assertLess(
            st["max"], ATOL_FLUX,
            msg=f"UV HLYMANA vs WRPLOT max|Δ|={st['max']:.3e}",
        )


class TestHlymanaSpeed(unittest.TestCase):
    def test_fast_faster(self):
        w = get_wrplotter()
        try:
            uv = load_uv(w)
        except Exception:
            uv = flat_continuum(900.0, 1300.0, n=20000)
        quiet(w.command_hlymana_fast, uv, key="coldens", value=-1e20)
        _, t_slow = time_call(
            w.command_hlymana, uv, key="coldens", value=-1e20, n=1
        )
        _, t_fast = time_call(
            w.command_hlymana_fast, uv, key="coldens", value=-1e20, n=3
        )
        self.assertLess(t_fast, t_slow)
        self.assertGreater(t_slow / t_fast, 2.0)


def make_plots():
    print("\n=== HLYMANA validation plots ===")
    w = get_wrplotter()

    # Match test.plot: dense flat continuum for N_H sweep + UV for one case
    flat = load_continuum_lyman_flat(w)
    uv = quiet(w.command_cut, load_uv(w), (900.0, 1300.0), axis="x")
    print(f"Flat continuum: {len(flat[0])} pts")
    print(f"UV continuum:   {len(uv[0])} pts")

    coldens = [-1.0e20, -5.0e20, -1.0e21]
    labels = [
        r"$N_\mathrm{H}=-10^{20}$ (flat)",
        r"$N_\mathrm{H}=-5\times10^{20}$ (flat)",
        r"$N_\mathrm{H}=-10^{21}$ (flat)",
    ]
    quiet(w.command_hlymana_fast, flat, key="coldens", value=coldens[0])

    results = []
    print(
        f"{'N_H':>12}  {'slow':>9}  {'fast_cut':>9}  {'fast_full':>9}  "
        f"{'vsWR':>10}  {'s/cut':>7}"
    )
    for N in coldens:
        kw = dict(key="coldens", value=N, vrad=0.0, temp=-1, vturb=0)
        slow, t_slow = time_call(w.command_hlymana, flat, n=1, **kw)
        fast_c, t_cut = time_call(w.command_hlymana_fast, flat, n=2, **kw)
        fast_f, t_full = time_call(
            w.command_hlymana_fast, flat, cut_lyman=False, n=2, **kw
        )
        ref_path = WRPLOT_REF_HLYMANA.get(N)
        ref = load_ref(ref_path) if ref_path and ref_path.is_file() else None
        d_wr = (
            compare_in_window(slow, ref[:, 0], ref[:, 1], 920, 1280)["max"]
            if ref is not None else np.nan
        )
        print(
            f"{N:12.1e}  {t_slow:7.2f}s  {t_cut:7.3f}s  {t_full:7.3f}s  "
            f"{d_wr:10.2e}  {t_slow/t_cut:6.1f}x"
        )
        results.append(dict(
            N=N, slow=slow, fast_c=fast_c, fast_f=fast_f, ref=ref,
            t_slow=t_slow, t_cut=t_cut, t_full=t_full, d_wr=d_wr,
        ))

    # UV case (test.plot plot 6)
    kw_uv = dict(key="coldens", value=-1.0e20, vrad=0.0, temp=-1, vturb=0)
    slow_uv, t_uv = time_call(w.command_hlymana, uv, n=1, **kw_uv)
    fast_uv, t_uvf = time_call(w.command_hlymana_fast, uv, n=2, **kw_uv)
    ref_uv = (
        load_ref(WRPLOT_REF_HLYMANA_UV)
        if WRPLOT_REF_HLYMANA_UV.is_file() else None
    )
    d_uv = (
        compare_in_window(slow_uv, ref_uv[:, 0], ref_uv[:, 1], 920, 1280)["max"]
        if ref_uv is not None else np.nan
    )
    print(f"{'UV -1e20':>12}  {t_uv:7.2f}s  {t_uvf:7.3f}s  {'—':>9}  "
          f"{d_uv:10.2e}  {t_uv/t_uvf:6.1f}x")

    # --- QC figure: 3 flat columns + residual vs WRPLOT ---
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharex="col")
    for col, (res, lab) in enumerate(zip(results, labels)):
        ax = axes[0, col]
        ax.plot(flat[0], flat[1], color="0.8", lw=0.4, label="input")
        if res["ref"] is not None:
            ax.plot(
                res["ref"][:, 0], res["ref"][:, 1],
                color="k", lw=1.3, alpha=0.9, label="WRPLOT (test.plot)",
                zorder=3,
            )
        ax.plot(res["slow"][0], res["slow"][1], color="C0", lw=0.75,
                label=f"Python slow [{res['t_slow']:.2f}s]", zorder=2)
        ax.plot(res["fast_c"][0], res["fast_c"][1], color="C2", lw=0.8, ls="--",
                label=f"Python fast [{res['t_cut']:.3f}s]", zorder=2)
        ax.set_xlim(920, 1300)
        ax.set_ylim(0, 1.2)
        ax.set_title(lab)
        ax.legend(fontsize=6.5, loc="upper right")
        if col == 0:
            ax.set_ylabel("Normalised flux")

        ax = axes[1, col]
        if res["ref"] is not None:
            st = compare_in_window(
                res["slow"], res["ref"][:, 0], res["ref"][:, 1], 920, 1280
            )
            ax.plot(st["x"], st["d"], color="k", lw=0.8,
                    label="Python − WRPLOT")
            ax.set_title(f"max|Δ| vs WRPLOT = {res['d_wr']:.1e}")
        ax.axhline(0, color="0.3", lw=0.4)
        ax.axhline(ATOL_FLUX, color="0.5", ls="--", lw=0.5)
        ax.axhline(-ATOL_FLUX, color="0.5", ls="--", lw=0.5)
        ax.set_xlabel(r"$\lambda$ / Å")
        if col == 0:
            ax.set_ylabel("Δ flux")
        ax.legend(fontsize=7)
    fig.suptitle(
        "HLYMANA QC: Python vs WRPLOT test.plot (flat continuum)",
        fontsize=12,
    )
    fig.tight_layout()
    savefig(fig, "hlymana_comparison.png")

    # UV QC panel
    fig, axes = plt.subplots(2, 1, figsize=(11, 7),
                             gridspec_kw={"height_ratios": [1.5, 1.0]},
                             sharex=True)
    ax = axes[0]
    ax.plot(uv[0], uv[1], color="0.75", lw=0.35, label="input UV")
    if ref_uv is not None:
        ax.plot(ref_uv[:, 0], ref_uv[:, 1], color="k", lw=1.3, alpha=0.9,
                label="WRPLOT (test.plot → ref_hlymana_uv_1e20.txt)", zorder=3)
    ax.plot(slow_uv[0], slow_uv[1], color="C0", lw=0.8,
            label=f"Python slow [{t_uv:.2f}s]", zorder=2)
    ax.plot(fast_uv[0], fast_uv[1], color="C2", lw=0.8, ls="--",
            label=f"Python fast [{t_uvf:.3f}s]", zorder=2)
    ax.set_xlim(920, 1300)
    ax.set_ylim(0, 1.5)
    ax.set_ylabel("Normalised flux")
    ax.set_title(r"HLYMANA on UV continuum, $N_\mathrm{H}=-10^{20}$ — QC vs WRPLOT")
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[1]
    if ref_uv is not None:
        st = compare_in_window(slow_uv, ref_uv[:, 0], ref_uv[:, 1], 920, 1280)
        ax.plot(st["x"], st["d"], color="k", lw=0.8,
                label=f"Python − WRPLOT  (max={st['max']:.1e})")
    ax.axhline(0, color="0.3", lw=0.4)
    ax.axhline(ATOL_FLUX, color="0.5", ls="--", lw=0.5, label=f"±{ATOL_FLUX}")
    ax.axhline(-ATOL_FLUX, color="0.5", ls="--", lw=0.5)
    ax.set_xlabel(r"$\lambda$ / Å")
    ax.set_ylabel("Δ flux")
    ax.legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, "hlymana_uv_comparison.png")

    save_speedup_bar(
        [f"N={r['N']:.0e}\ncut" for r in results]
        + [f"N={r['N']:.0e}\nfull" for r in results]
        + ["UV -1e20\ncut"],
        [r["t_slow"] for r in results] * 2 + [t_uv],
        [r["t_cut"] for r in results] + [r["t_full"] for r in results] + [t_uvf],
        title="HLYMANA speedup",
        outfile="hlymana_speedup.png",
        ylabel="Time [s]",
    )


if __name__ == "__main__":
    make_plots()
    unittest.main(verbosity=2, exit=False)

#!/usr/bin/env python
"""
Validation + speedup suite for command_convol_original / command_convol.

Covers Gaussian, box, and rotational kernels with several widths / β values.
Tolerance between slow and fast: ATOL_FLUX (1e-4); exact path uses ~machine eps.
"""

from __future__ import annotations

import unittest

import matplotlib.pyplot as plt
import numpy as np

from helpers import (
    ATOL_FLUX,
    WRPLOT_REF_CONVOL_BOX,
    WRPLOT_REF_CONVOL_GAUSS,
    WRPLOT_REF_CONVOL_ROT,
    compare_in_window,
    get_wrplotter,
    load_continuum_flat,
    load_ref,
    max_abs_diff,
    quiet,
    save_timing_table,
    savefig,
    synthetic_absorption,
    time_call,
)


class TestConvolAgreement(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.w = get_wrplotter()
        tiny = synthetic_absorption(n=200)
        cls.w.command_convol(tiny, "gauss", 0.5)
        cls.w.command_convol(tiny, "box", 1.0)
        cls.w.command_convol(tiny, "rot", 1.0, beta=0.0)

    def test_gauss_slow_fast_various_fwhm(self):
        data = synthetic_absorption(n=6000)
        for fwhm in (0.2, 0.5, 1.0, 2.5):
            with self.subTest(fwhm=fwhm):
                slow = self.w.command_convol_original(data, "gauss", fwhm)
                fast = self.w.command_convol(
                    data, "gauss", fwhm, precision=None
                )
                d = max_abs_diff(slow[1], fast[1], slow[0], fast[0])
                self.assertLess(d, ATOL_FLUX, msg=f"gauss FWHM={fwhm}: {d}")

    def test_box_slow_fast_various_widths(self):
        data = synthetic_absorption(n=6000)
        for width in (0.5, 1.0, 2.0, 5.0):
            with self.subTest(width=width):
                slow = self.w.command_convol_original(data, "box", width)
                fast = self.w.command_convol(
                    data, "box", width, precision=None
                )
                d = max_abs_diff(slow[1], fast[1])
                self.assertLess(d, ATOL_FLUX, msg=f"box width={width}: {d}")

    def test_rot_slow_fast_various_beta(self):
        data = synthetic_absorption(n=6000)
        d_ang = 2.5
        for beta in (0.0, 0.3, 0.6, 1.0):
            with self.subTest(beta=beta):
                slow = self.w.command_convol_original(data, "rot", d_ang, beta=beta)
                fast = self.w.command_convol(
                    data, "rot", d_ang, beta=beta, precision=None
                )
                d = max_abs_diff(slow[1], fast[1])
                self.assertLess(d, ATOL_FLUX, msg=f"rot beta={beta}: {d}")

    def test_gauss_on_irregular_grid(self):
        """After dense core clustering, CONVADD Δx weighting must still agree."""
        x = np.concatenate([
            np.linspace(4990, 4998, 500),
            np.linspace(4998, 5002, 4000),  # dense core
            np.linspace(5002, 5010, 500),
        ])
        y = 1.0 - 0.9 * np.exp(-0.5 * ((x - 5000.0) / 0.1) ** 2)
        data = np.array([x, y])
        slow = self.w.command_convol_original(data, "gauss", 0.5)
        fast = self.w.command_convol(data, "gauss", 0.5, precision=None)
        self.assertLess(max_abs_diff(slow[1], fast[1]), ATOL_FLUX)

    def test_value_zero_noop(self):
        data = synthetic_absorption(n=100)
        for shape in ("gauss", "box", "rot"):
            out = self.w.command_convol_original(data, shape, 0.0)
            np.testing.assert_array_equal(out[1], data[1])

    def test_vs_wrplot_gauss_box_rot(self):
        """Compare to WRPLOT refs on data/continuum_flat.dat (test.plot CONVOL plots)."""
        cases = [
            ("gauss", 2.0, 0.0, WRPLOT_REF_CONVOL_GAUSS),
            ("box", 2.0, 0.0, WRPLOT_REF_CONVOL_BOX),
            ("rot", 2.5, 0.6, WRPLOT_REF_CONVOL_ROT),
        ]
        data = load_continuum_flat(self.w)
        for shape, value, beta, path in cases:
            with self.subTest(shape=shape):
                if not path.is_file():
                    self.skipTest(f"{path.name} missing — run ./run_wrplot_refs.sh")
                ref = load_ref(path)
                out = quiet(
                    self.w.command_convol_original, data, shape, value, beta=beta
                )
                st = compare_in_window(
                    out, ref[:, 0], ref[:, 1], 4850.0, 5150.0
                )
                self.assertLess(
                    st["max"], ATOL_FLUX,
                    msg=f"{shape} vs WRPLOT max|Δ|={st['max']:.3e}",
                )


class TestConvolSpeed(unittest.TestCase):
    def test_fast_faster_than_slow(self):
        w = get_wrplotter()
        data = synthetic_absorption(n=8000, lo=4800, hi=5200,
                                    centers=[(4900, 0.7, 0.12),
                                             (5000, 0.9, 0.10),
                                             (5100, 0.6, 0.15)])
        w.command_convol(data, "rot", 2.5, beta=0.4)  # warmup
        _, t_slow = time_call(w.command_convol_original, data, "rot", 2.5, beta=0.4, n=3)
        _, t_fast = time_call(
            w.command_convol, data, "rot", 2.5, beta=0.4,
            precision=None, n=3,
        )
        self.assertLess(t_fast, t_slow)
        self.assertGreater(t_slow / t_fast, 5.0)


def make_plots():
    """QC plots: Python vs WRPLOT (test.plot) + slow/fast speedup."""
    print("\n=== CONVOL validation plots ===")
    w = get_wrplotter()
    data = load_continuum_flat(w)
    quiet(w.command_convol, data, "gauss", 0.5)
    quiet(w.command_convol, data, "box", 1.0)
    quiet(w.command_convol, data, "rot", 2.5, beta=0.0)

    # Cases that match tests/test.plot CONVOL plots (have WRPLOT refs)
    wrplot_cases = [
        ("gauss", 2.0, 0.0, "GAUSS FWHM=2.0", WRPLOT_REF_CONVOL_GAUSS),
        ("box", 2.0, 0.0, "BOX / KASTEN=2.0", WRPLOT_REF_CONVOL_BOX),
        ("rot", 2.5, 0.6, "ROT Δλ=2.5 β=0.6", WRPLOT_REF_CONVOL_ROT),
    ]
    # Extra slow↔fast timing cases (no WRPLOT ref required)
    timing_extra = [
        ("gauss", 0.5, 0.0, "GAUSS FWHM=0.5"),
        ("box", 1.0, 0.0, "BOX half=1.0"),
        ("rot", 2.5, 0.0, "ROT Δλ=2.5 β=0"),
    ]

    results = []
    print(f"{'case':28s}  {'slow':>10s}  {'fast':>10s}  {'vsWR':>10s}  {'s/f':>6s}")
    for shape, value, beta, label, ref_path in wrplot_cases:
        slow, t_slow = time_call(
            w.command_convol_original, data, shape, value, beta=beta, n=5
        )
        fast, t_fast = time_call(
            w.command_convol, data, shape, value, beta=beta,
            precision=None, n=5,
        )
        ref = load_ref(ref_path) if ref_path.is_file() else None
        d_wr = (
            compare_in_window(slow, ref[:, 0], ref[:, 1], 4850, 5150)["max"]
            if ref is not None else np.nan
        )
        print(
            f"{label:28s}  {t_slow*1e3:8.1f}ms  {t_fast*1e3:8.2f}ms  "
            f"{d_wr:10.2e}  {t_slow/t_fast:5.0f}x"
        )
        results.append(dict(
            label=label, shape=shape, value=value, beta=beta,
            slow=slow, fast=fast, ref=ref,
            t_slow=t_slow, t_fast=t_fast, d_wr=d_wr,
            d_sf=max_abs_diff(slow[1], fast[1]),
        ))

    # --- QC: spectrum + residual vs WRPLOT (test.plot) ---
    fig, axes = plt.subplots(
        2, 3, figsize=(13, 7.5),
        gridspec_kw={"height_ratios": [1.5, 1.0]},
        sharex="col",
    )
    for col, res in enumerate(results):
        ax = axes[0, col]
        ax.plot(data[0], data[1], color="0.75", lw=0.5, label="input")
        if res["ref"] is not None:
            ax.plot(
                res["ref"][:, 0], res["ref"][:, 1],
                color="k", lw=1.4, alpha=0.85, label="WRPLOT (test.plot)",
                zorder=3,
            )
        ax.plot(
            res["slow"][0], res["slow"][1], color="C0", lw=0.9,
            label=f"Python slow [{res['t_slow']*1e3:.0f}ms]", zorder=2,
        )
        ax.plot(
            res["fast"][0], res["fast"][1], color="C3", lw=0.8, ls="--",
            label=f"Python fast [{res['t_fast']*1e3:.1f}ms]", zorder=2,
        )
        ax.set_xlim(4985, 5015)
        ax.set_ylim(0, 1.15)
        ax.set_title(res["label"])
        ax.legend(loc="lower right", fontsize=6.5)
        if col == 0:
            ax.set_ylabel("Normalised flux")

        ax = axes[1, col]
        if res["ref"] is not None:
            st = compare_in_window(
                res["slow"], res["ref"][:, 0], res["ref"][:, 1], 4850, 5150
            )
            ax.plot(st["x"], st["d"], color="C0", lw=0.8, label="Python − WRPLOT")
            ax.set_title(f"max|Δ| vs WRPLOT = {res['d_wr']:.1e}")
        ax.axhline(0, color="k", lw=0.4)
        ax.axhline(ATOL_FLUX, color="0.5", ls="--", lw=0.5)
        ax.axhline(-ATOL_FLUX, color="0.5", ls="--", lw=0.5)
        ax.set_xlabel(r"$\lambda$ / Å")
        if col == 0:
            ax.set_ylabel("Δ flux")
        ax.legend(fontsize=7, loc="upper right")

    fig.suptitle(
        "CONVOL QC: Python vs WRPLOT test.plot (data/continuum_flat.dat)",
        fontsize=12,
    )
    fig.tight_layout()
    savefig(fig, "convol_comparison.png")

    timing_rows = [
        dict(
            case=r["label"],
            t_slow=r["t_slow"],
            t_fast=r["t_fast"],
            max_d_wrplot=r["d_wr"],
        )
        for r in results
    ]
    for shape, value, beta, label in timing_extra:
        _, t_slow = time_call(
            w.command_convol_original, data, shape, value, beta=beta, n=3
        )
        _, t_fast = time_call(
            w.command_convol, data, shape, value, beta=beta,
            precision=None, n=3,
        )
        timing_rows.append(dict(
            case=label, t_slow=t_slow, t_fast=t_fast, max_d_wrplot=None,
        ))

    save_timing_table(
        timing_rows,
        title=f"CONVOL timing (N={len(data[0])} pts, continuum_flat.dat)",
        outfile="convol_timing.txt",
        notes=[
            "slow = command_convol_original (pure-Python CONVEX/CONVADD)",
            "fast = command_convol (numba kernels)",
        ],
    )
    return results


if __name__ == "__main__":
    make_plots()
    unittest.main(verbosity=2, exit=False)

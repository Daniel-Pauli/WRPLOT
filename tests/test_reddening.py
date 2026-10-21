#!/usr/bin/env python
"""
Validation + speedup suite for command_reddening_original / command_reddening.

Compares each reddening law (and selected Rv variants) to tables produced by
``tests/test.plot`` plots 10–17 on ``data/continuum_reddening_flat.dat``, and
checks slow↔fast agreement plus timing.
"""

from __future__ import annotations

import unittest

import matplotlib.pyplot as plt
import numpy as np

from helpers import (
    ATOL_FLUX,
    REDDENING_EBV,
    WRPLOT_REF_REDDENING,
    apply_reddening_linear,
    compare_in_window,
    get_wrplotter,
    load_continuum_reddening_flat,
    load_ref,
    max_abs_diff,
    quiet,
    save_timing_table,
    savefig,
    time_call,
)

# Cases matching test.plot REDDENING plots (law, Rv, ref key, plot label)
REDDENING_CASES = [
    ("seaton", -1, "seaton", "SEATON"),
    ("cardelli", -1, "cardelli", "CARDELLI Rv=3.1"),
    ("cardelli", 5.0, "cardelli_rv5", "CARDELLI Rv=5.0"),
    ("fitzpatrick", -1, "fitzpatrick", "FITZPATRICK Rv=3.1"),
    ("fitzpatrick", 5.0, "fitzpatrick_rv5", "FITZPATRICK Rv=5.0"),
    ("lmc", -1, "lmc", "LMC"),
    ("smc", -1, "smc", "SMC"),
    ("smc_bar", -1, "smc_bar", "SMC_BAR"),
]

WIN_LO = 1100.0
WIN_HI = 90000.0


class TestReddeningVsWrplot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.w = get_wrplotter()
        cls.data = load_continuum_reddening_flat(cls.w)
        # Warmup numba
        quiet(
            apply_reddening_linear,
            cls.w, cls.data, REDDENING_EBV, law="seaton", fast=True,
        )

    def test_slow_fast_all_laws(self):
        """Fast and original paths must agree within ATOL_FLUX."""
        for law, Rv, _key, label in REDDENING_CASES:
            with self.subTest(law=law, Rv=Rv):
                slow = apply_reddening_linear(
                    self.w, self.data, REDDENING_EBV, law=law, Rv=Rv, fast=False
                )
                fast = apply_reddening_linear(
                    self.w, self.data, REDDENING_EBV, law=law, Rv=Rv, fast=True
                )
                d = max_abs_diff(slow[1], fast[1], slow[0], fast[0])
                self.assertLess(d, ATOL_FLUX, msg=f"{label} slow↔fast max|Δ|={d:.3e}")

    def test_vs_wrplot_all_laws(self):
        """Fast Python reddening must match WRPLOT refs within ATOL_FLUX."""
        for law, Rv, key, label in REDDENING_CASES:
            with self.subTest(law=law, Rv=Rv):
                path = WRPLOT_REF_REDDENING[key]
                if not path.is_file():
                    self.skipTest(f"{path.name} missing — run ./run_wrplot_refs.sh")
                ref = load_ref(path)
                out = apply_reddening_linear(
                    self.w, self.data, REDDENING_EBV, law=law, Rv=Rv, fast=True
                )
                st = compare_in_window(out, ref[:, 0], ref[:, 1], WIN_LO, WIN_HI)
                self.assertLess(
                    st["max"],
                    ATOL_FLUX,
                    msg=f"{label} vs WRPLOT max|Δ|={st['max']:.3e}",
                )

    def test_fitzpatrick_rv_changes_ir(self):
        """Custom Fitzpatrick Rv must change far-IR extinction (Moneti branch)."""
        out31 = apply_reddening_linear(
            self.w, self.data, REDDENING_EBV, law="fitzpatrick", Rv=3.1, fast=True
        )
        out50 = apply_reddening_linear(
            self.w, self.data, REDDENING_EBV, law="fitzpatrick", Rv=5.0, fast=True
        )
        m = (out31[0] > 4.0e4) & (out31[0] < 9.0e4)
        self.assertTrue(np.any(m))
        d = float(
            np.max(np.abs(out31[1][m] - np.interp(out31[0][m], out50[0], out50[1])))
        )
        self.assertGreater(d, 1e-3, msg=f"Fitzpatrick IR Δ(Rv=3.1 vs 5) too small: {d}")

    def test_laws_differ_optically(self):
        """Seaton / Fitzpatrick / SMC_BAR must differ in the optical."""
        seaton = apply_reddening_linear(
            self.w, self.data, REDDENING_EBV, law="seaton", fast=True
        )
        fitz = apply_reddening_linear(
            self.w, self.data, REDDENING_EBV, law="fitzpatrick", fast=True
        )
        smcb = apply_reddening_linear(
            self.w, self.data, REDDENING_EBV, law="smc_bar", fast=True
        )
        x = np.linspace(4000.0, 6000.0, 500)
        ys = np.interp(x, seaton[0], seaton[1])
        yf = np.interp(x, fitz[0], fitz[1])
        yb = np.interp(x, smcb[0], smcb[1])
        self.assertGreater(float(np.max(np.abs(ys - yf))), 1e-3)
        self.assertGreater(float(np.max(np.abs(ys - yb))), 1e-3)


class TestReddeningSpeed(unittest.TestCase):
    def test_fast_faster_than_slow(self):
        w = get_wrplotter()
        data = load_continuum_reddening_flat(w)
        # Warmup
        apply_reddening_linear(w, data, REDDENING_EBV, law="fitzpatrick", fast=True)
        _, t_slow = time_call(
            apply_reddening_linear, w, data, REDDENING_EBV,
            law="fitzpatrick", Rv=5.0, fast=False, n=3,
        )
        _, t_fast = time_call(
            apply_reddening_linear, w, data, REDDENING_EBV,
            law="fitzpatrick", Rv=5.0, fast=True, n=5,
        )
        self.assertLess(t_fast, t_slow)
        self.assertGreater(t_slow / t_fast, 5.0)


def make_plots():
    """QC plots: Python vs WRPLOT + slow/fast speedup."""
    print("\n=== REDDENING validation plots ===")
    w = get_wrplotter()
    data = load_continuum_reddening_flat(w)
    print(f"Input continuum: {len(data[0])} pts, "
          f"λ={data[0][0]:.0f}–{data[0][-1]:.0f} Å")

    # Warmup
    apply_reddening_linear(w, data, REDDENING_EBV, law="seaton", fast=True)

    results = []
    print(f"{'case':28s}  {'slow':>10s}  {'fast':>10s}  {'vsWR':>10s}  {'s/f':>6s}")
    for law, Rv, key, label in REDDENING_CASES:
        slow, t_slow = time_call(
            apply_reddening_linear, w, data, REDDENING_EBV,
            law=law, Rv=Rv, fast=False, n=3,
        )
        fast, t_fast = time_call(
            apply_reddening_linear, w, data, REDDENING_EBV,
            law=law, Rv=Rv, fast=True, n=5,
        )
        path = WRPLOT_REF_REDDENING[key]
        ref = load_ref(path) if path.is_file() else None
        if ref is not None:
            st = compare_in_window(fast, ref[:, 0], ref[:, 1], WIN_LO, WIN_HI)
            d_wr = st["max"]
        else:
            st, d_wr = None, np.nan
        d_sf = max_abs_diff(slow[1], fast[1], slow[0], fast[0])
        print(
            f"{label:28s}  {t_slow*1e3:8.1f}ms  {t_fast*1e3:8.2f}ms  "
            f"{d_wr:10.2e}  {t_slow/t_fast:5.0f}x"
        )
        results.append(dict(
            label=label, law=law, Rv=Rv, slow=slow, fast=fast, ref=ref, st=st,
            t_slow=t_slow, t_fast=t_fast, d_wr=d_wr, d_sf=d_sf,
        ))

    # --- Figure 1: six default-Rv laws (fast vs WRPLOT) ---
    default = [r for r in results if (r["law"], r["Rv"]) in {
        ("seaton", -1), ("cardelli", -1), ("fitzpatrick", -1),
        ("lmc", -1), ("smc", -1), ("smc_bar", -1),
    }]

    fig, axes = plt.subplots(
        4, 3, figsize=(13, 11),
        gridspec_kw={"height_ratios": [1.5, 1.0, 1.5, 1.0]},
        sharex="col",
    )
    for i, res in enumerate(default):
        row0 = 0 if i < 3 else 2
        col = i % 3
        ax = axes[row0, col]
        if res["ref"] is not None:
            ax.plot(
                res["ref"][:, 0], res["ref"][:, 1],
                color="k", lw=1.4, alpha=0.85, label="WRPLOT (test.plot)",
                zorder=3,
            )
        ax.plot(
            res["fast"][0], res["fast"][1], color="C0", lw=0.9,
            label=f"Python fast [{res['t_fast']*1e3:.1f}ms]", zorder=2,
        )
        ax.set_xscale("log")
        ax.set_xlim(1.0e3, 1.0e5)
        ax.set_ylim(0, 1.15)
        ax.set_title(res["label"])
        ax.legend(loc="lower right", fontsize=6.5)
        if col == 0:
            ax.set_ylabel("Flux (EBV=1)")

        ax = axes[row0 + 1, col]
        if res["st"] is not None:
            ax.plot(res["st"]["x"], res["st"]["d"], color="C0", lw=0.8,
                    label="Python − WRPLOT")
            ax.set_title(f"max|Δ| = {res['d_wr']:.1e}")
        ax.axhline(0, color="k", lw=0.4)
        ax.axhline(ATOL_FLUX, color="0.5", ls="--", lw=0.5)
        ax.axhline(-ATOL_FLUX, color="0.5", ls="--", lw=0.5)
        ax.set_xscale("log")
        if row0 == 2:
            ax.set_xlabel(r"$\lambda$ / Å")
        if col == 0:
            ax.set_ylabel("Δ flux")
        ax.legend(fontsize=7, loc="upper right")

    fig.suptitle(
        "REDDENING QC: Python fast vs WRPLOT test.plot "
        "(data/continuum_reddening_flat.dat, EBV=1)",
        fontsize=12,
    )
    fig.tight_layout()
    savefig(fig, "reddening_comparison.png")

    # --- Figure 2: Cardelli / Fitzpatrick Rv variants ---
    rv_pairs = [("cardelli", "CARDELLI"), ("fitzpatrick", "FITZPATRICK")]
    fig, axes = plt.subplots(
        2, 2, figsize=(11, 7.5),
        gridspec_kw={"height_ratios": [1.5, 1.0]},
        sharex="col",
    )
    for col, (law, title) in enumerate(rv_pairs):
        r31 = next(r for r in results if r["law"] == law and r["Rv"] < 0)
        r50 = next(r for r in results if r["law"] == law and r["Rv"] == 5.0)

        ax = axes[0, col]
        for res, c, ls in ((r31, "C0", "-"), (r50, "C3", "--")):
            if res["ref"] is not None:
                ax.plot(
                    res["ref"][:, 0], res["ref"][:, 1],
                    color="k", lw=1.2, alpha=0.5 if res is r50 else 0.85,
                    label="WRPLOT" if res is r31 else None, zorder=3,
                )
            ax.plot(
                res["fast"][0], res["fast"][1], color=c, lw=0.9, ls=ls,
                label=f"Python {res['label']}", zorder=2,
            )
        ax.set_xscale("log")
        ax.set_xlim(1.0e3, 1.0e5)
        ax.set_ylim(0, 1.15)
        ax.set_title(title)
        ax.legend(loc="lower right", fontsize=6.5)
        if col == 0:
            ax.set_ylabel("Flux (EBV=1)")

        ax = axes[1, col]
        for res, c in ((r31, "C0"), (r50, "C3")):
            if res["st"] is not None:
                ax.plot(
                    res["st"]["x"], res["st"]["d"], color=c, lw=0.8,
                    label=f"Δ {res['label']} (max {res['d_wr']:.1e})",
                )
        ax.axhline(0, color="k", lw=0.4)
        ax.axhline(ATOL_FLUX, color="0.5", ls="--", lw=0.5)
        ax.axhline(-ATOL_FLUX, color="0.5", ls="--", lw=0.5)
        ax.set_xscale("log")
        ax.set_xlabel(r"$\lambda$ / Å")
        if col == 0:
            ax.set_ylabel("Python − WRPLOT")
        ax.legend(fontsize=6.5, loc="upper right")

    fig.suptitle(
        "REDDENING QC: Rv variants (Cardelli / Fitzpatrick) vs WRPLOT",
        fontsize=12,
    )
    fig.tight_layout()
    savefig(fig, "reddening_rv_comparison.png")

    # --- Figure 3: slow vs fast residual for three representative laws ---
    show = [r for r in results if r["label"] in (
        "SEATON", "FITZPATRICK Rv=5.0", "SMC_BAR",
    )]
    fig, axes = plt.subplots(
        2, 3, figsize=(13, 7.0),
        gridspec_kw={"height_ratios": [1.5, 1.0]},
        sharex="col",
    )
    for col, res in enumerate(show):
        ax = axes[0, col]
        ax.plot(res["slow"][0], res["slow"][1], color="C0", lw=1.0,
                label=f"slow [{res['t_slow']*1e3:.0f}ms]")
        ax.plot(res["fast"][0], res["fast"][1], color="C3", lw=0.8, ls="--",
                label=f"fast [{res['t_fast']*1e3:.2f}ms]")
        ax.set_xscale("log")
        ax.set_xlim(1.0e3, 1.0e5)
        ax.set_ylim(0, 1.15)
        ax.set_title(res["label"])
        ax.legend(loc="lower right", fontsize=7)
        if col == 0:
            ax.set_ylabel("Flux (EBV=1)")

        ax = axes[1, col]
        yp = np.interp(res["slow"][0], res["fast"][0], res["fast"][1])
        ax.plot(res["slow"][0], res["slow"][1] - yp, color="C0", lw=0.8)
        ax.axhline(0, color="k", lw=0.4)
        ax.axhline(ATOL_FLUX, color="0.5", ls="--", lw=0.5)
        ax.axhline(-ATOL_FLUX, color="0.5", ls="--", lw=0.5)
        ax.set_xscale("log")
        ax.set_xlabel(r"$\lambda$ / Å")
        ax.set_title(f"slow−fast max|Δ|={res['d_sf']:.1e}")
        if col == 0:
            ax.set_ylabel("Δ flux")

    fig.suptitle(
        "REDDENING QC: command_reddening_original vs command_reddening",
        fontsize=12,
    )
    fig.tight_layout()
    savefig(fig, "reddening_slow_fast.png")

    save_timing_table(
        [
            dict(
                case=r["label"],
                t_slow=r["t_slow"],
                t_fast=r["t_fast"],
                max_d_wrplot=r["d_wr"],
            )
            for r in results
        ],
        title=(
            f"REDDENING timing (N_in={len(data[0])} → 1000 pts, "
            "continuum_reddening_flat.dat)"
        ),
        outfile="reddening_timing.txt",
        notes=[
            "slow = command_reddening_original (pure-Python expand + loop)",
            "fast = command_reddening (numba expand + reddening kernel)",
            "max|d| vs WRPLOT measured on the fast path",
        ],
    )
    return results


if __name__ == "__main__":
    make_plots()
    unittest.main(verbosity=2, exit=False)

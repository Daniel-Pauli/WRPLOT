#!/usr/bin/env python
"""
Tests for WRPLOT-style ``plot_idents`` (colors, multi-segment spacing, layout).

Checks that the O VI / (is) color-break gap stays a half-space (~1/3 em, matching
the ``\\,\\,`` used for each WRPLOT ``\\,``) across different y-axis ranges,
figure sizes, aspect ratios, and DPI.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from helpers import OUTPUT, REF, get_wrplotter

ID_O = REF / "id_O.dat"

# Cross-break gap for mid-string color splits (see __draw_wrplot_label).
TARGET_GAP_EM = 0.20
GAP_EM_TOL = 0.05  # allow small font/bbox variation


def _em_pixels(fig, fontsize):
    return fontsize * fig.dpi / 72.0


def _find_ovi_pair(ax):
    """Return (text_O_VI, text_is) artists for the O VI&(is) color-split label."""
    vis = [
        t for t in ax.texts
        if "VI" in t.get_text() and "(is)" not in t.get_text()
    ]
    iss = [t for t in ax.texts if t.get_text() == r"$\mathrm{(is)}$"]
    if not vis or not iss:
        return None, None
    # O VI multiplet sits near 1035 Å; pick closest VI then nearest (is) in x
    t_vi = min(vis, key=lambda t: abs(t.get_position()[0] - 1034.75))
    x_vi = t_vi.get_position()[0]
    t_is = min(iss, key=lambda t: abs(t.get_position()[0] - x_vi))
    return t_vi, t_is


def _vertical_gap_em(fig, ax, t_lo, t_hi, fontsize):
    """Gap in em along display-y between two stacked (rotation=90) labels."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bb0 = t_lo.get_window_extent(renderer=renderer)
    bb1 = t_hi.get_window_extent(renderer=renderer)
    gap_px = bb1.y0 - bb0.y1
    return gap_px / _em_pixels(fig, fontsize), gap_px, bb0, bb1


def _draw_idents(ax, *, id_y, id_len, id_fontsize, color_line_from_text=True,
                 xlim=(1000, 1600)):
    w = get_wrplotter()
    plt.sca(ax)  # plot_idents draws via plt.gca()
    ax.set_xlim(*xlim)
    ax.plot(list(xlim), [1, 1], "k-", lw=0.4, alpha=0.3)
    w.plot_idents(
        str(ID_O),
        id_y=id_y,
        id_len=id_len,
        id_fontsize=id_fontsize,
        color_line_from_text=color_line_from_text,
    )


# Figure geometries to stress pixel↔data conversion for the half-space gap
GEOMETRIES = [
    # name, figsize, dpi, ylim, id_y, id_len, fontsize
    ("default", (10, 4), 120, (0.0, 1.5), 1.05, 0.08, 9),
    ("tall", (5, 8), 120, (0.0, 1.5), 1.05, 0.08, 9),
    ("wide", (14, 3), 100, (0.0, 1.5), 1.05, 0.08, 9),
    ("small_square", (4, 4), 100, (0.0, 1.5), 1.05, 0.08, 8),
    ("large_hi_dpi", (12, 5), 200, (0.0, 1.5), 1.05, 0.08, 11),
    ("ylim_tight", (10, 4), 120, (0.9, 1.25), 1.02, 0.05, 9),
    ("ylim_wide", (10, 4), 120, (-0.5, 3.0), 1.1, 0.15, 10),
    ("ylim_flux_scale", (8, 5), 120, (0.0, 50.0), 35.0, 3.0, 9),
]


class TestIdentsParse(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.w = get_wrplotter()
        if not ID_O.is_file():
            raise unittest.SkipTest(f"missing {ID_O}")

    def test_color_split_ovi_is(self):
        segs = self.w._wrplotter__parse_wrplot_text(r"&EO\,VI&4\,(is)")
        self.assertEqual(len(segs), 2)
        self.assertIn("VI", segs[0]["text"])
        self.assertTrue(segs[1]["text"].lstrip("\\,").startswith("(is)") or
                        "(is)" in segs[1]["text"])
        self.assertEqual(segs[0]["color"], "black")
        self.assertEqual(segs[1]["color"], (0.0, 0.0, 1.0))

    def test_mathtext_thin_space_inside_segment(self):
        math = self.w._wrplotter__wrplot_segment_mathtext(r"O\,I\,(is)")
        self.assertEqual(math, r"$\mathrm{O}\,\,\mathrm{I}\,\,\mathrm{(is)}$")

    def test_greek(self):
        segs = self.w._wrplotter__parse_wrplot_text(r"&ELy#e#")
        self.assertEqual(len(segs), 1)
        math = self.w._wrplotter__wrplot_segment_mathtext(segs[0]["text"])
        self.assertEqual(math, r"$\mathrm{Ly}\epsilon$")


class TestIdentsLayout(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.w = get_wrplotter()
        if not ID_O.is_file():
            raise unittest.SkipTest(f"missing {ID_O}")

    def test_gap_matches_halfspace_across_geometries(self):
        """VI→(is) display gap ≈ 0.2 em for varied size / aspect / ylim."""
        for name, figsize, dpi, ylim, id_y, id_len, fs in GEOMETRIES:
            with self.subTest(geometry=name):
                fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
                ax.set_ylim(*ylim)
                _draw_idents(ax, id_y=id_y, id_len=id_len, id_fontsize=fs)
                t_vi, t_is = _find_ovi_pair(ax)
                self.assertIsNotNone(t_vi, msg=f"{name}: missing O VI text")
                self.assertIsNotNone(t_is, msg=f"{name}: missing (is) text")
                gap_em, gap_px, bb0, bb1 = _vertical_gap_em(
                    fig, ax, t_vi, t_is, fs,
                )
                self.assertFalse(
                    bb0.overlaps(bb1),
                    msg=f"{name}: O VI and (is) bboxes overlap",
                )
                self.assertGreater(
                    gap_px, 0.0,
                    msg=f"{name}: expected positive gap, got {gap_px}",
                )
                self.assertLess(
                    abs(gap_em - TARGET_GAP_EM), GAP_EM_TOL,
                    msg=(
                        f"{name}: gap {gap_em:.3f} em "
                        f"(target {TARGET_GAP_EM:.3f} ± {GAP_EM_TOL})"
                    ),
                )
                plt.close(fig)

    def test_mark_color_follows_first_letter(self):
        from matplotlib.colors import to_rgb

        fig, ax = plt.subplots(figsize=(8, 3), dpi=100)
        ax.set_ylim(0, 1.5)
        _draw_idents(
            ax, id_y=1.05, id_len=0.08, id_fontsize=9,
            color_line_from_text=True, xlim=(1035, 1045),
        )
        # O I (is) is fully blue (&4 from the start) → blue mark lines
        blue = to_rgb((0.0, 0.0, 1.0))
        blue_lines = [
            ln for ln in ax.lines
            if np.allclose(to_rgb(ln.get_color()), blue, atol=0.02)
        ]
        self.assertGreater(
            len(blue_lines), 0,
            msg="expected blue ident marks for &4 labels",
        )
        plt.close(fig)

    def test_no_crash_empty_range(self):
        fig, ax = plt.subplots()
        ax.set_ylim(0, 1.5)
        n_before = len(ax.texts)
        # id_O.dat spans ~900–11000 Å; pick a range with no entries
        _draw_idents(ax, id_y=1.05, id_len=0.05, id_fontsize=8, xlim=(20000, 20100))
        n_new = len(ax.texts) - n_before
        self.assertEqual(
            n_new, 0,
            msg=f"expected no ident texts outside file range, got {n_new}",
        )
        plt.close(fig)


def make_plots():
    """QC panel: idents under several y-ranges / aspect ratios / sizes."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    n = len(GEOMETRIES)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.2 * nrows), dpi=110)
    axes = np.atleast_1d(axes).ravel()
    gaps = []
    for ax, (name, figsize, dpi, ylim, id_y, id_len, fs) in zip(axes, GEOMETRIES):
        # embed each geometry in its own inset-like axes by matching ylim only;
        # figsize/dpi of the *panel* is fixed — also write per-geometry PNGs
        ax.set_ylim(*ylim)
        _draw_idents(ax, id_y=id_y, id_len=id_len, id_fontsize=fs)
        t_vi, t_is = _find_ovi_pair(ax)
        gap_note = "n/a"
        if t_vi is not None and t_is is not None:
            # measure on a dedicated figure that matches the geometry
            fig_g, ax_g = plt.subplots(figsize=figsize, dpi=dpi)
            ax_g.set_ylim(*ylim)
            _draw_idents(ax_g, id_y=id_y, id_len=id_len, id_fontsize=fs)
            t_vi_g, t_is_g = _find_ovi_pair(ax_g)
            gap_em, _, _, _ = _vertical_gap_em(fig_g, ax_g, t_vi_g, t_is_g, fs)
            gaps.append((name, gap_em))
            gap_note = f"gap={gap_em:.3f} em"
            fig_g.savefig(
                OUTPUT / f"idents_{name}.png", dpi=dpi, bbox_inches="tight",
            )
            plt.close(fig_g)
        ax.set_title(f"{name}  ylim={ylim}  {gap_note}", fontsize=9)
        ax.set_xlabel("λ")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(
        "plot_idents geometry matrix (VI→(is) gap target = "
        f"{TARGET_GAP_EM:.2f} em)",
        fontsize=11,
    )
    fig.tight_layout()
    out = OUTPUT / "idents_geometry_matrix.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)

    # smoke-style reference (legacy path used in development)
    fig_s, ax_s = plt.subplots(figsize=(10, 4), dpi=120)
    ax_s.set_ylim(0, 1.5)
    _draw_idents(ax_s, id_y=1.05, id_len=0.08, id_fontsize=9)
    fig_s.savefig(OUTPUT / "idents_smoke.png", dpi=120, bbox_inches="tight")
    plt.close(fig_s)

    print(f"  wrote {out.name}, idents_smoke.png, and {len(gaps)} geometry PNGs")
    for name, gap_em in gaps:
        status = "OK" if abs(gap_em - TARGET_GAP_EM) < GAP_EM_TOL else "FAIL"
        print(f"    {name:16s}  VI→(is) gap = {gap_em:.3f} em  [{status}]")


if __name__ == "__main__":
    unittest.main(verbosity=2)

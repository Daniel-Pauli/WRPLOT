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

from helpers import OUTPUT, REF, get_wrplotter, load_uv

ID_O = REF / "id_O.dat"

# Cross-break gap for mid-string color splits (see __ident_draw_text).
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
        segs = self.w._wrplotter__ident_parse_text(r"&EO\,VI&4\,(is)")
        self.assertEqual(len(segs), 2)
        self.assertIn("VI", segs[0]["text"])
        self.assertTrue(segs[1]["text"].lstrip("\\,").startswith("(is)") or
                        "(is)" in segs[1]["text"])
        self.assertEqual(segs[0]["color"], "black")
        self.assertEqual(segs[1]["color"], "blue")  # self.colors[4]

    def test_mathtext_thin_space_inside_segment(self):
        math = self.w._wrplotter__ident_mathtext(r"O\,I\,(is)")
        self.assertEqual(math, r"$\mathrm{O}\,\,\mathrm{I}\,\,\mathrm{(is)}$")

    def test_greek(self):
        segs = self.w._wrplotter__ident_parse_text(r"&ELy#e#")
        self.assertEqual(len(segs), 1)
        math = self.w._wrplotter__ident_mathtext(segs[0]["text"])
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


class TestIdentsAuto(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.w = get_wrplotter()
        if not ID_O.is_file():
            raise unittest.SkipTest(f"missing {ID_O}")
        cls.uv = load_uv(cls.w)

    def test_auto_requires_spectrum(self):
        fig, ax = plt.subplots()
        ax.set_xlim(1000, 1100)
        ax.set_ylim(0, 1.5)
        plt.sca(ax)
        with self.assertRaises(ValueError):
            self.w.plot_idents(str(ID_O), auto=True)
        plt.close(fig)

    def test_auto_places_marks_above_local_mean(self):
        x, y = np.asarray(self.uv[0], float), np.asarray(self.uv[1], float)
        fig, ax = plt.subplots(figsize=(8, 3), dpi=100)
        plt.sca(ax)
        ax.plot(x, y, "k-", lw=0.6)
        ax.set_xlim(1000, 1100)
        ymin = float(np.min(y[(x >= 1000) & (x <= 1100)]))
        ymax = float(np.max(y[(x >= 1000) & (x <= 1100)]))
        ax.set_ylim(ymin - 0.05, ymax + 0.35)
        offset = 0.2
        self.w.plot_idents(
            str(ID_O),
            auto=True,
            auto_offset=offset,
            id_len=0.06,
            id_fontsize=8,
            color_line_from_text=True,
        )
        # Mark stems should start above the local continuum (mean + offset),
        # or higher if neighbour-leveling raised a dip.
        m = (x >= 1032) & (x <= 1038)
        local_mean = float(np.mean(y[m]))
        stem_bottoms = []
        for ln in ax.lines[1:]:  # skip spectrum
            xd = np.asarray(ln.get_xdata(), float)
            yd = np.asarray(ln.get_ydata(), float)
            if xd.size == 2 and abs(xd[0] - xd[1]) < 1e-6:
                if abs(xd[0] - 1034.75) < 2.0:
                    stem_bottoms.append(min(yd))
        self.assertGreater(len(stem_bottoms), 0, msg="no O VI stems found")
        for y0 in stem_bottoms:
            self.assertGreaterEqual(
                y0, local_mean + 0.5 * offset,
                msg=f"stem bottom {y0} not above local mean {local_mean}",
            )
        self.assertGreater(len(ax.texts), 0)
        plt.close(fig)

    def test_auto_levels_absorption_dips_to_neighbours(self):
        """A deep trough should not pull an ident below its neighbours."""
        anchors = [1030.0, 1050.0, 1070.0]
        bases = [1.0, 0.1, 1.0]  # middle sits in the trough
        leveled = self.w._wrplotter__ident_auto_level(anchors, bases)
        self.assertAlmostEqual(leveled[0], 1.0)
        self.assertAlmostEqual(leveled[2], 1.0)
        self.assertAlmostEqual(leveled[1], 1.0)  # raised to neighbour average

    def test_auto_clears_emission_peak(self):
        """Mark bottom must sit above a sharp emission peak under the ident."""
        import tempfile
        wave = np.linspace(1000, 1100, 1001)
        flux = np.full_like(wave, 0.5)
        flux[(wave > 1048) & (wave < 1052)] = 2.0
        with tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as fh:
            fh.write("\\IDENT 1050.0 &ETest\n")
            path = fh.name
        try:
            fig, ax = plt.subplots()
            plt.sca(ax)
            ax.plot(wave, flux, "k-")
            ax.set_xlim(1000, 1100)
            ax.set_ylim(0, 3.0)
            offset = 0.2
            self.w.plot_idents(
                path, auto=True, auto_offset=offset, auto_window=0.5,
                id_len=0.05, id_fontsize=8,
            )
            stem_bottoms = []
            for ln in ax.lines[1:]:
                xd = np.asarray(ln.get_xdata(), float)
                yd = np.asarray(ln.get_ydata(), float)
                if xd.size == 2 and abs(xd[0] - xd[1]) < 1e-6 and abs(xd[0] - 1050.0) < 0.5:
                    stem_bottoms.append(min(yd))
            self.assertGreater(len(stem_bottoms), 0)
            for y0 in stem_bottoms:
                self.assertGreaterEqual(
                    y0, 2.0 + offset - 1e-6,
                    msg=f"stem {y0} collides with emission peak at 2.0",
                )
            plt.close(fig)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_auto_min_floor(self):
        """auto_min raises all marks to at least that y."""
        import tempfile
        wave = np.linspace(1000, 1100, 201)
        flux = np.full_like(wave, 0.3)
        with tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as fh:
            fh.write("\\IDENT 1050.0 &ETest\n")
            path = fh.name
        try:
            fig, ax = plt.subplots()
            plt.sca(ax)
            ax.plot(wave, flux, "k-")
            ax.set_xlim(1000, 1100)
            ax.set_ylim(0, 2.0)
            self.w.plot_idents(
                path, auto=True, auto_offset=0.05, auto_window=0.5,
                auto_min=1.1, id_len=0.05, id_fontsize=8,
            )
            stem_bottoms = []
            for ln in ax.lines[1:]:
                xd = np.asarray(ln.get_xdata(), float)
                yd = np.asarray(ln.get_ydata(), float)
                if xd.size == 2 and abs(xd[0] - xd[1]) < 1e-6:
                    stem_bottoms.append(min(yd))
            self.assertGreater(len(stem_bottoms), 0)
            for y0 in stem_bottoms:
                self.assertGreaterEqual(y0, 1.1 - 1e-6)
            plt.close(fig)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_fixed_ignores_file_idy_idleng(self):
        """fixed=True uses kwargs id_y/id_len and skips \\IDY/\\IDLENG."""
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as fh:
            fh.write("\\IDY 2.0U\n")
            fh.write("\\IDLENG 0.5U\n")
            fh.write("\\IDENT 1050.0 &ETest\n")
            path = fh.name
        try:
            fig, ax = plt.subplots()
            plt.sca(ax)
            ax.set_xlim(1000, 1100)
            ax.set_ylim(0, 3.0)
            ax.plot([1000, 1100], [1, 1], "k-", lw=0.3)
            self.w.plot_idents(
                path, fixed=True, id_y=1.05, id_len=0.08, id_fontsize=8,
            )
            stem_bottoms, stem_tops = [], []
            for ln in ax.lines[1:]:
                xd = np.asarray(ln.get_xdata(), float)
                yd = np.asarray(ln.get_ydata(), float)
                if xd.size == 2 and abs(xd[0] - xd[1]) < 1e-6 and abs(xd[0] - 1050) < 1:
                    stem_bottoms.append(min(yd))
                    stem_tops.append(max(yd))
            self.assertGreater(len(stem_bottoms), 0)
            # Lowest mark piece must start at kwargs id_y (not file \\IDY=2.0)
            self.assertAlmostEqual(min(stem_bottoms), 1.05, places=5)
            # Lowest stem top ≈ id_y + 0.4*id_len, not file \\IDLENG=0.5
            self.assertTrue(
                any(abs(yt - (1.05 + 0.08 * 0.4)) < 1e-4 for yt in stem_tops)
            )
            self.assertFalse(any(abs(yb - 2.0) < 0.05 for yb in stem_bottoms))
            plt.close(fig)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_fixed_and_auto_raises(self):
        fig, ax = plt.subplots()
        plt.sca(ax)
        ax.set_xlim(1000, 1100)
        ax.plot([1000, 1100], [1, 1], "k-")
        with self.assertRaises(ValueError):
            self.w.plot_idents(str(ID_O), fixed=True, auto=True)
        plt.close(fig)


def _plot_uv_with_auto_idents(xlim, *, auto_offset=0.2, id_len=0.05,
                              id_fontsize=8, figsize=(10, 4), dpi=120,
                              title=None):
    """Plot UV spectrum + auto-placed idents; return (fig, ax)."""
    w = get_wrplotter()
    uv = load_uv(w)
    x, y = np.asarray(uv[0], float), np.asarray(uv[1], float)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    plt.sca(ax)
    ax.plot(x, y, color="0.15", lw=0.7)
    ax.set_xlim(*xlim)
    m = (x >= xlim[0]) & (x <= xlim[1])
    ymin = float(np.nanmin(y[m]))
    ymax = float(np.nanmax(y[m]))
    pad = 0.08 * max(ymax - ymin, 0.2)
    ax.set_ylim(ymin - pad, ymax + pad + auto_offset + id_len + 0.35)
    w.plot_idents(
        str(ID_O),
        auto=True,
        auto_offset=auto_offset,
        id_len=id_len,
        id_fontsize=id_fontsize,
        color_line_from_text=True,
    )
    ax.set_xlabel("Wavelength / Å")
    ax.set_ylabel("Flux")
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig, ax


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

    # --- auto placement on UV spectrum ---
    auto_windows = [
        ("idents_auto_uv_fuse.png", (920, 1180),
         "auto idents on UV spectrum (FUSE)"),
        ("idents_auto_uv_iue.png", (1150, 1600),
         "auto idents on UV spectrum (IUE)"),
        ("idents_auto_uv_zoom_ovi.png", (1020, 1050),
         "auto idents zoom (O VI region)"),
    ]
    for fname, xlim, title in auto_windows:
        fig_a, _ax_a = _plot_uv_with_auto_idents(
            xlim, auto_offset=0.2, id_len=0.05, id_fontsize=8, title=title,
        )
        fig_a.savefig(OUTPUT / fname, dpi=130, bbox_inches="tight")
        plt.close(fig_a)
        print(f"  wrote {fname}")


if __name__ == "__main__":
    unittest.main(verbosity=2)

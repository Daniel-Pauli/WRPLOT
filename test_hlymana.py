#!/usr/bin/env python
"""
Unit tests for command_hlymana / command_hlymana_fast bugfixes.

Validates:
  1. TAULYMEDGE uses C0[-1] (Lorentz) / last-line Voigt, not C0[0]
  2. TAUCRIT inserts a single midpoint and does not duplicate wavelengths
  3. slow and fast (cut_lyman=False) agree closely on a dense grid
  4. Edge-region absorption matches the expected factor exp(ENH*C0[-1])
"""

import contextlib
import io
import unittest

import numpy as np

from wrplot import wrplotter


def _quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


class TestHlymana(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.w = wrplotter()
        # Warm up numba
        tiny = np.array([
            np.linspace(1210.0, 1220.0, 50),
            np.ones(50),
        ])
        _quiet(
            cls.w.command_hlymana_fast,
            tiny,
            key="coldens",
            value=-1e19,
            cut_lyman=False,
        )

    def _flat(self, lo, hi, n=4000):
        x = np.linspace(lo, hi, n)
        return np.array([x, np.ones_like(x)])

    def test_taulymedge_lorentz_matches_c0_nhi(self):
        """Edge-band flux ratio must track exp(ENH*C0[NHI]), not exp(ENH*C0[Lyα])."""
        C0_NHI = 3.072e-11
        C0_LYA = 7.056e-11
        # Mild column so edge factors are O(0.1–1), not underflowed to 0
        # |ENH|*C0_NHI ≈ 0.31, |ENH|*C0_Lyα ≈ 0.71
        ENH = -1.0e10
        expected = np.exp(ENH * C0_NHI)  # ≈ 0.736
        wrong = np.exp(ENH * C0_LYA)     # ≈ 0.494

        data = self._flat(911.4, 912.8, n=4000)
        out = _quiet(
            self.w.command_hlymana,
            data,
            key="coldens",
            value=ENH,
            temp=-1,
            vturb=0,
        )
        XLYEDGE = 911.2670504
        XLAM_NHI = 912.64482
        edge = (out[0] > XLYEDGE + 0.05) & (out[0] < XLAM_NHI - 0.05)
        red = (out[0] > XLAM_NHI + 0.02) & (out[0] < XLAM_NHI + 0.08)
        self.assertTrue(np.any(edge) and np.any(red))
        ratio = np.median(out[1][edge]) / np.median(out[1][red])
        self.assertLess(
            abs(ratio - expected),
            abs(ratio - wrong),
            msg=(
                f"edge/red continuum ratio {ratio:.6f} closer to old Lyα "
                f"factor {wrong:.6f} than to C0[NHI] factor {expected:.6f}"
            ),
        )

    def test_taulymedge_fast_matches_slow(self):
        ENH = -5.0e20
        data = self._flat(911.3, 913.0, n=3000)
        slow = _quiet(
            self.w.command_hlymana, data, key="coldens", value=ENH
        )
        fast = _quiet(
            self.w.command_hlymana_fast,
            data,
            key="coldens",
            value=ENH,
            cut_lyman=False,
        )
        XLYEDGE = 911.2670504
        XLAM_NHI = 912.64482
        x = np.linspace(XLYEDGE + 0.01, XLAM_NHI - 0.01, 500)
        ys = np.interp(x, slow[0], slow[1])
        yf = np.interp(x, fast[0], fast[1])
        self.assertLess(np.max(np.abs(ys - yf)), 5e-3)

    def test_taucrit_no_duplicate_wavelengths(self):
        data = self._flat(1200.0, 1225.0, n=500)
        out = _quiet(
            self.w.command_hlymana,
            data,
            key="coldens",
            value=-1.0e21,
        )
        # Strictly increasing wavelengths (no duplicates from broken TAUCRIT)
        dx = np.diff(out[0])
        self.assertTrue(np.all(dx > 0), msg="duplicate or non-monotonic λ")

    def test_slow_fast_lyman_alpha_core(self):
        ENH = -1.0e20
        data = self._flat(1180.0, 1250.0, n=8000)
        slow = _quiet(
            self.w.command_hlymana, data, key="coldens", value=ENH
        )
        fast = _quiet(
            self.w.command_hlymana_fast,
            data,
            key="coldens",
            value=ENH,
            cut_lyman=False,
        )
        x = np.linspace(1205.0, 1225.0, 4000)
        ys = np.interp(x, slow[0], slow[1])
        yf = np.interp(x, fast[0], fast[1])
        # Allow small differences from adaptive vs sequential TAUCRIT
        self.assertLess(np.max(np.abs(ys - yf)), 0.05)

    def test_c0_nhi_not_c0_alpha_in_edge_factor(self):
        """Mean optical depth in the edge band should sit nearer |ENH|*C0[NHI]
        than |ENH|*C0[Lyα] (profile τ is also present, so only a closeness test)."""
        C0_0 = 7.056e-11
        C0_NHI = 3.072e-11
        ENH = -5.0e10
        data = self._flat(911.4, 912.4, n=1500)
        out = _quiet(
            self.w.command_hlymana, data, key="coldens", value=ENH
        )
        mask = (out[0] > 911.2670504) & (out[0] < 912.64482)
        mean_t = -np.log(np.clip(np.mean(out[1][mask]), 1e-300, None))
        t_nhi = abs(ENH) * C0_NHI
        t_lya = abs(ENH) * C0_0
        self.assertLess(abs(mean_t - t_nhi), abs(mean_t - t_lya))


if __name__ == "__main__":
    unittest.main(verbosity=2)

#!/usr/bin/env python
"""
Unit tests for WRPLOT-faithful rotational convolution.

Validates Unsoeld kernel + fixed Δλ_max [Å] in command_convol /
command_convol_fast (and the rot() helper).
"""

import time
import unittest

import numpy as np

from wrplot import wrplotter, _convol_rot_python, _convol_rot_kernel


def _synthetic_line(n=8000, lo=4990.0, hi=5010.0, center=5000.0, width=0.15):
    """Narrow Gaussian absorption on a flat continuum."""
    x = np.linspace(lo, hi, n)
    y = 1.0 - 0.8 * np.exp(-0.5 * ((x - center) / width) ** 2)
    return np.array([x, y])


class TestConvolRot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.w = wrplotter()
        # Warm up numba
        tiny = _synthetic_line(n=200)
        cls.w.command_convol_fast(tiny, "rot", 1.0, beta=0.0)

    def test_rot_helper_unsoeld_beta0(self):
        x = np.linspace(-1.2, 1.2, 241)
        d = 1.0
        g = self.w.rot(x, mu=0.0, d_angstrom=d, beta=0.0)
        xa = x / d
        m = np.abs(xa) < 1.0
        expected = np.zeros_like(x)
        expected[m] = (2.0 / np.pi) * np.sqrt(1.0 - xa[m] ** 2)
        np.testing.assert_allclose(g, expected, rtol=0, atol=1e-14)
        self.assertTrue(np.all(g[~m] == 0.0))

    def test_rot_helper_unsoeld_beta(self):
        x = np.array([-0.5, 0.0, 0.5, 1.5])
        d = 1.0
        beta = 0.6
        g = self.w.rot(x, mu=0.0, d_angstrom=d, beta=beta)
        xa = np.array([-0.5, 0.0, 0.5])
        exp = (2.0 / np.pi) * np.sqrt(1.0 - xa ** 2) + beta * 0.5 * (1.0 - xa ** 2)
        np.testing.assert_allclose(g[:3], exp, rtol=0, atol=1e-14)
        self.assertEqual(g[3], 0.0)

    def test_slow_fast_agree_beta0(self):
        data = _synthetic_line()
        d_ang = 2.5
        slow = self.w.command_convol(data, "rot", d_ang, beta=0.0)
        fast = self.w.command_convol_fast(
            data, "rot", d_ang, beta=0.0, precision=None
        )
        np.testing.assert_allclose(slow[0], fast[0], rtol=0, atol=0)
        np.testing.assert_allclose(slow[1], fast[1], rtol=0, atol=1e-12)

    def test_slow_fast_agree_beta(self):
        data = _synthetic_line()
        d_ang = 3.0
        beta = 0.6
        slow = self.w.command_convol(data, "rot", d_ang, beta=beta)
        fast = self.w.command_convol_fast(
            data, "rot", d_ang, beta=beta, precision=None
        )
        np.testing.assert_allclose(slow[1], fast[1], rtol=0, atol=1e-12)

    def test_python_matches_numba_kernel(self):
        data = _synthetic_line(n=3000)
        d_ang, beta = 2.0, 0.3
        y_py = _convol_rot_python(data[0], data[1], d_ang, beta)
        y_nb = _convol_rot_kernel(data[0], data[1], d_ang, beta)
        np.testing.assert_allclose(y_py, y_nb, rtol=0, atol=1e-12)

    def test_fixed_width_not_wavelength_dependent(self):
        """WRPLOT D is fixed in Å: kernel support is ±D at every λ."""
        x = np.linspace(4000.0, 6000.0, 20001)
        y = np.ones_like(x)
        # narrow spike at two wavelengths
        y[np.argmin(np.abs(x - 4500.0))] = 0.0
        y[np.argmin(np.abs(x - 5500.0))] = 0.0
        d_ang = 5.0
        out = self.w.command_convol_fast(
            np.array([x, y]), "rot", d_ang, beta=0.0, precision=None
        )
        # Half-depth width of each dip should be ~same (fixed D), not ∝ λ
        def half_width(center):
            sl = (out[0] > center - 2 * d_ang) & (out[0] < center + 2 * d_ang)
            xx, yy = out[0][sl], out[1][sl]
            # continuum ~1, core min
            ymin = yy.min()
            half = 0.5 * (1.0 + ymin)
            below = xx[yy <= half]
            return below.max() - below.min() if below.size else 0.0

        w4500 = half_width(4500.0)
        w5500 = half_width(5500.0)
        # Old Gray/vsini API would make 5500/4500 ≈ 1.22 wider; we need ~1
        self.assertLess(abs(w5500 / w4500 - 1.0), 0.05)

    def test_value_zero_noop(self):
        data = _synthetic_line(n=100)
        out = self.w.command_convol(data, "rot", 0.0)
        np.testing.assert_array_equal(out[1], data[1])


class TestConvolRotSpeed(unittest.TestCase):
    """Speedup smoke test (not a strict bound — machines vary)."""

    def test_fast_faster_than_slow(self):
        w = wrplotter()
        # Modest grid: enough for a clear speedup without multi-minute CI
        data = _synthetic_line(n=5000)
        d_ang, beta = 3.0, 0.4
        w.command_convol_fast(data, "rot", d_ang, beta=beta)  # warmup

        t0 = time.perf_counter()
        w.command_convol(data, "rot", d_ang, beta=beta)
        t_slow = time.perf_counter() - t0

        t0 = time.perf_counter()
        w.command_convol_fast(data, "rot", d_ang, beta=beta, precision=None)
        t_fast = time.perf_counter() - t0

        self.assertLess(t_fast, t_slow)
        self.assertGreater(t_slow / t_fast, 5.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

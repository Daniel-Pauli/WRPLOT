#!/usr/bin/env python
"""
Unit tests for the normalised Voigt function voigth / _voigth_njit.

Covers several (a, v) regimes used by ISMLINE and HLYMANA, and checks
Python ↔ numba agreement.
"""

from __future__ import annotations

import unittest

import numpy as np

from helpers import get_wrplotter
from wrplot import _voigth_njit


class TestVoigth(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.w = get_wrplotter()
        # warmup numba
        _voigth_njit(0.1, 1.0)

    def test_python_matches_numba_grid(self):
        a_vals = [0.01, 0.1, 0.2, 0.5, 1.0, 1.5, 3.0]
        v_vals = [0.0, 0.5, 1.0, 1.3, 2.0, 2.4, 3.0, 5.0, 10.0]
        for a in a_vals:
            for v in v_vals:
                with self.subTest(a=a, v=v):
                    py = self.w.voigth(a, v)
                    nb = _voigth_njit(a, v)
                    self.assertTrue(
                        np.isfinite(py) and np.isfinite(nb),
                        msg=f"non-finite a={a} v={v}",
                    )
                    self.assertLess(
                        abs(py - nb), 1e-12,
                        msg=f"a={a} v={v}: py={py} nb={nb}",
                    )

    def test_positive_and_symmetric_in_v(self):
        for a in (0.05, 0.5, 2.0):
            for v in (0.0, 1.0, 5.0):
                h_pos = self.w.voigth(a, v)
                h_neg = self.w.voigth(a, -v)
                self.assertGreaterEqual(h_pos, 0.0)
                self.assertAlmostEqual(h_pos, h_neg, places=14)

    def test_gaussian_limit_small_a(self):
        """For tiny a and moderate v, H ≈ exp(-v²)/√π."""
        a = 1e-6
        for v in (0.0, 0.5, 1.0):
            h = self.w.voigth(a, v)
            expected = np.exp(-v * v) / np.sqrt(np.pi)
            # Köster approx is close but not exact — allow 5%
            self.assertLess(abs(h - expected) / expected, 0.05)


if __name__ == "__main__":
    unittest.main(verbosity=2)

#!/usr/bin/env python
"""
Light validation of arithmetic / grid helper commands used by pipelines.

Covers CUT, PLUS/MINUS/MULTIPLY/DIVIDE, SORT, INSERT on small arrays.
"""

from __future__ import annotations

import unittest

import numpy as np

from helpers import ATOL_FLUX, flat_continuum, get_wrplotter, quiet


class TestArithCommands(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.w = get_wrplotter()

    def test_cut(self):
        data = flat_continuum(1000.0, 2000.0, n=1001)
        out = self.w.command_cut(data, (1200.0, 1300.0), axis="x")
        self.assertTrue(np.all(out[0] > 1200.0) and np.all(out[0] < 1300.0))

    def test_plus_minus(self):
        data = flat_continuum(1.0, 10.0, n=50)
        p = self.w.command_plus(data, 2.0, axis="y")
        m = self.w.command_minus(p, 2.0, axis="y")
        np.testing.assert_allclose(m[1], data[1], atol=ATOL_FLUX)

    def test_multiply_divide(self):
        data = flat_continuum(1.0, 10.0, n=50)
        p = self.w.command_multiply(data, 3.5, axis="y")
        d = self.w.command_divide(p, 3.5, axis="y")
        np.testing.assert_allclose(d[1], data[1], atol=ATOL_FLUX)

    def test_sort(self):
        x = np.array([3.0, 1.0, 2.0])
        y = np.array([30.0, 10.0, 20.0])
        out = self.w.command_sort(np.array([x, y]))
        np.testing.assert_array_equal(out[0], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(out[1], [10.0, 20.0, 30.0])

    def test_insert_existing_and_new(self):
        data = flat_continuum(100.0, 110.0, n=11)
        # insert midpoint
        mid = 105.5
        out = quiet(self.w.command_insert, data, mid)
        self.assertTrue(np.any(np.isclose(out[0], mid)))
        self.assertTrue(np.all(np.diff(out[0]) >= 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)

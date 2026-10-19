#!/usr/bin/env python
"""
Run all wrplot validation suites and generate speedup / comparison plots.

Usage (from repo root or tests/):
    python tests/run_all.py
    python tests/run_all.py --unit-only
    python tests/run_all.py --plots-only
"""

from __future__ import annotations

import argparse
import os
import sys
import unittest
from pathlib import Path

# Ensure local imports and matplotlib cache work under sandbox / CI
TESTS = Path(__file__).resolve().parent
ROOT = TESTS.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TESTS))
os.environ.setdefault("MPLCONFIGDIR", str(TESTS / ".mplconfig"))
(TESTS / ".mplconfig").mkdir(parents=True, exist_ok=True)
(TESTS / "output").mkdir(parents=True, exist_ok=True)

import test_arith  # noqa: E402
import test_convol  # noqa: E402
import test_hlymana  # noqa: E402
import test_ismline  # noqa: E402
import test_og_pipeline  # noqa: E402
import test_voigt  # noqa: E402


PLOT_MODULES = [
    ("CONVOL", test_convol.make_plots),
    ("ISMLINE", test_ismline.make_plots),
    ("HLYMANA", test_hlymana.make_plots),
    ("OG pipeline", test_og_pipeline.make_plots),
]

UNIT_MODULES = [
    test_voigt,
    test_arith,
    test_convol,
    test_ismline,
    test_hlymana,
    test_og_pipeline,
]


def run_unit_tests() -> int:
    print("=" * 60, flush=True)
    print("Running unit / validation tests", flush=True)
    print("=" * 60, flush=True)
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for mod in UNIT_MODULES:
        suite.addTests(loader.loadTestsFromModule(mod))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


def run_plots() -> None:
    print("=" * 60, flush=True)
    print("Generating comparison + speedup plots → tests/output/", flush=True)
    print("=" * 60, flush=True)
    for name, fn in PLOT_MODULES:
        print(f"\n--- {name} ---", flush=True)
        fn()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--unit-only", action="store_true", help="Skip plot generation"
    )
    parser.add_argument(
        "--plots-only", action="store_true", help="Skip unit tests"
    )
    args = parser.parse_args()

    rc = 0
    if not args.plots_only:
        rc = run_unit_tests()
    if not args.unit_only:
        run_plots()
        print(f"\nPlots written to {TESTS / 'output'}")
    if rc:
        print("\nSome unit tests FAILED.", file=sys.stderr)
    else:
        print("\nAll requested steps finished successfully.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

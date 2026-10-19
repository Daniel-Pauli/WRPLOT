"""
Shared helpers for wrplot validation test suites.

All input continua and WRPLOT reference tables live under tests/:
  data/   — continua used by both WRPLOT (test.plot) and Python
  ref/    — tables produced by ``./run_wrplot_refs.sh`` (wrplot test.plot)
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# tests/ directory (this file's home)
TESTS = Path(__file__).resolve().parent
# Repo root (parent of tests/) — provides wrplot.py
ROOT = TESTS.parent
DATA = TESTS / "data"
REF = TESTS / "ref"
OUTPUT = TESTS / "output"
OUTPUT.mkdir(parents=True, exist_ok=True)

# Matplotlib cache inside the workspace (avoids ~/.cache permission issues)
os.environ.setdefault("MPLCONFIGDIR", str(TESTS / ".mplconfig"))
(TESTS / ".mplconfig").mkdir(parents=True, exist_ok=True)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wrplot import wrplotter  # noqa: E402

# Visually indistinguishable / scientifically acceptable flux difference
ATOL_FLUX = 1e-4

# ---------------------------------------------------------------------------
# Local fixtures (self-contained — no external formal.plot required)
# ---------------------------------------------------------------------------
CONTINUUM_IUE = DATA / "continuum_iue.dat"       # * IUE 900–1400 Å
CONTINUUM_FLAT = DATA / "continuum_flat.dat"     # synthetic absorption
CONTINUUM_LYMAN_FLAT = DATA / "continuum_lyman_flat.dat"  # flat 900–1300 Å
UV_SPECTRUM = DATA / "uv_spectrum.dat"          # UV continuum 900–1400 Å

WRPLOT_REF_ISMLINE = REF / "ref_ismline.txt"
WRPLOT_REF_ISMLINE_CONVOL = REF / "ref_ismline_convol.txt"
WRPLOT_REF_HLYMANA = {
    -1.0e20: REF / "ref_hlymana_1e20.txt",
    -5.0e20: REF / "ref_hlymana_5e20.txt",
    -1.0e21: REF / "ref_hlymana_1e21.txt",
}
WRPLOT_REF_HLYMANA_UV = REF / "ref_hlymana_uv_1e20.txt"
WRPLOT_REF_CONVOL_GAUSS = REF / "ref_convol_gauss.txt"
WRPLOT_REF_CONVOL_BOX = REF / "ref_convol_box.txt"
WRPLOT_REF_CONVOL_ROT = REF / "ref_convol_rot.txt"


# ---------------------------------------------------------------------------
# Si III λ1206 ISMLINE parameters (GAL + SMC components)
# ---------------------------------------------------------------------------
EBV = 0.14
EBV_GAL = 0.06
EBV_SMC = EBV - EBV_GAL
CD_S_GAL = -(1.25 * EBV * 6.86e21) * 3.55e-5 / 1000
CD_S_SMC = -(EBV_SMC * 3.8e21 * 6) * 3.55e-5 / 1000
VRAD_GAL = -156.0
VRAD_SMC = 0.0
VTURB_GAL = 10.0
VTURB_SMC = 30.0
ISM_L0 = 1206.51
ISM_GAMM = 6.1e9
ISM_F = 1.67
ISM_A = 28.0
GAUSS_FWHM = 0.5  # combined ISMLINE + CONVOL GAUSS case


def quiet(fn, *args, **kwargs):
    """Call fn with stdout suppressed (Fortran-style chatter)."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def time_call(fn, *args, n=1, **kwargs):
    """Return (result, median_seconds) over n calls."""
    times = []
    out = None
    for _ in range(n):
        t0 = time.perf_counter()
        out = quiet(fn, *args, **kwargs)
        times.append(time.perf_counter() - t0)
    return out, float(np.median(times))


def get_wrplotter():
    return wrplotter()


def load_xy(path, w=None):
    """Load a two-column continuum (WRPLOT XYTABLE / * comments OK)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Expected fixtures under tests/data/."
        )
    w = w or get_wrplotter()
    return quiet(w.command_include, str(path))


def load_continuum_iue(w=None):
    return load_xy(CONTINUUM_IUE, w)


def load_uv(w=None):
    return load_xy(UV_SPECTRUM, w)


def load_continuum_flat(w=None):
    return load_xy(CONTINUUM_FLAT, w)


def load_continuum_lyman_flat(w=None):
    return load_xy(CONTINUUM_LYMAN_FLAT, w)


def _fix_fortran_dexp(text: str) -> str:
    """Insert missing E in Fortran under/overflow dumps like ``1.23-105``."""
    import re
    return re.sub(r"([0-9]\.[0-9]+)([+-][0-9]+)", r"\1E\2", text)


def load_ref(path):
    """Load a WRPLOT WRITE table as (N, 2) array.

    WRPLOT sometimes writes underflowed values as ``1.234-105`` (missing ``E``);
    those are normalised before ``np.loadtxt``.
    """
    from io import StringIO

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing WRPLOT reference {path}.\n"
            f"Generate with:  cd {TESTS} && ./run_wrplot_refs.sh"
        )
    text = _fix_fortran_dexp(path.read_text())
    return np.loadtxt(StringIO(text))


def require_refs(*paths):
    missing = [str(p) for p in paths if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing WRPLOT reference file(s):\n  "
            + "\n  ".join(missing)
            + f"\nGenerate with:  cd {TESTS} && ./run_wrplot_refs.sh"
        )


def synthetic_absorption(n=8000, lo=4990.0, hi=5010.0, centers=None):
    """Flat continuum with one or more narrow Gaussian absorption features."""
    if centers is None:
        centers = [(5000.0, 0.8, 0.15)]
    x = np.linspace(lo, hi, n)
    y = np.ones_like(x)
    for c, amp, width in centers:
        y -= amp * np.exp(-0.5 * ((x - c) / width) ** 2)
    return np.array([x, y])


def flat_continuum(lo, hi, n=4000):
    x = np.linspace(lo, hi, n)
    return np.array([x, np.ones_like(x)])


def max_abs_diff(y_a, y_b, x_a=None, x_b=None, x_common=None):
    """
    Max |Δy|. If wavelength grids differ, interpolate onto x_common
    (or onto x_a if x_common is None).
    """
    ya = np.asarray(y_a, dtype=float)
    yb = np.asarray(y_b, dtype=float)
    if x_a is None or x_b is None:
        return float(np.max(np.abs(ya - yb)))
    xa = np.asarray(x_a, dtype=float)
    xb = np.asarray(x_b, dtype=float)
    if xa.shape == xb.shape and np.allclose(xa, xb, rtol=0, atol=1e-10):
        return float(np.max(np.abs(ya - yb)))
    xc = x_common if x_common is not None else xa
    ya_i = np.interp(xc, xa, ya)
    yb_i = np.interp(xc, xb, yb)
    return float(np.max(np.abs(ya_i - yb_i)))


def compare_in_window(model, xref, yref, lo, hi, direct=False):
    """Compare model flux to reference in [lo, hi]. Returns stats dict."""
    if direct and model[0].shape == xref.shape:
        x, yp, yr = model[0], model[1], yref
    else:
        yp = np.interp(xref, model[0], model[1])
        x, yr = xref, yref
    mask = (x >= lo) & (x <= hi)
    d = yp[mask] - yr[mask]
    return dict(
        max=float(np.max(np.abs(d))) if d.size else 0.0,
        rms=float(np.sqrt(np.mean(d ** 2))) if d.size else 0.0,
        x=x[mask],
        yp=yp[mask],
        yr=yr[mask],
        d=d,
    )


def _fmt_seconds(t):
    """Human-readable duration."""
    t = float(t)
    if t < 1e-3:
        return f"{t * 1e6:.1f} us"
    if t < 1.0:
        return f"{t * 1e3:.2f} ms"
    return f"{t:.3f} s"


def _write_table(path, title, notes, headers, rows_fmt):
    lines = [title, "=" * len(title), ""]
    if notes:
        lines.extend(notes)
        lines.append("")
    widths = [max(len(h), max((len(r[i]) for r in rows_fmt), default=0))
              for i, h in enumerate(headers)]
    def fmt(cols):
        return "  ".join(c.ljust(w) if i == 0 else c.rjust(w)
                         for i, (c, w) in enumerate(zip(cols, widths)))
    lines.append(fmt(headers))
    lines.append("  ".join("-" * w for w in widths))
    lines.extend(fmt(r) for r in rows_fmt)
    lines.append("")
    text = "\n".join(lines)
    path.write_text(text)
    print(f"  wrote {path}")
    print(text)
    return path


def save_timing_table(rows, title, outfile, notes=None):
    """
    Pairwise slow-vs-fast timing (same API, two implementations).

    Each row: case, t_slow, t_fast [s]; optional max_d_wrplot.
    speedup = t_slow / t_fast.
    """
    path = OUTPUT / outfile
    notes = list(notes or [])
    notes.append("speedup = t(slow) / t(fast)  for that case")
    fmt_rows = []
    for r in rows:
        ts, tf = float(r["t_slow"]), float(r["t_fast"])
        sp = f"{ts / tf:.1f}x" if tf > 0 else "—"
        dwr = r.get("max_d_wrplot")
        d_s = ("—" if dwr is None or (isinstance(dwr, float) and np.isnan(dwr))
               else f"{float(dwr):.2e}")
        fmt_rows.append((
            str(r["case"]),
            _fmt_seconds(ts),
            _fmt_seconds(tf),
            sp,
            d_s,
        ))
    return _write_table(
        path, title, notes,
        ("case", "slow", "fast", "speedup", "max|d| vs WRPLOT"),
        fmt_rows,
    )


def save_benchmark_table(rows, title, outfile, baseline_case=None, notes=None):
    """
    Multi-config runtimes with speedup relative to a baseline case.

    Each row: case, t [s]; optional max_d_wrplot.
    speedup = t(baseline) / t(case).  Baseline row shows 1.0x.
    """
    path = OUTPUT / outfile
    notes = list(notes or [])
    by_name = {str(r["case"]): r for r in rows}
    if baseline_case is None:
        baseline_case = str(rows[0]["case"])
    if baseline_case not in by_name:
        raise KeyError(f"baseline_case={baseline_case!r} not in rows")
    t0 = float(by_name[baseline_case]["t"])
    notes.append(f"speedup = t({baseline_case}) / t(case)")
    fmt_rows = []
    for r in rows:
        t = float(r["t"])
        sp = f"{t0 / t:.1f}x" if t > 0 else "—"
        dwr = r.get("max_d_wrplot")
        d_s = ("—" if dwr is None or (isinstance(dwr, float) and np.isnan(dwr))
               else f"{float(dwr):.2e}")
        fmt_rows.append((str(r["case"]), _fmt_seconds(t), sp, d_s))
    return _write_table(
        path, title, notes,
        ("case", "time", "speedup", "max|d| vs WRPLOT"),
        fmt_rows,
    )


def savefig(fig, name):
    path = OUTPUT / name
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path}")
    return path


def ism_kwargs_gal(**extra):
    kw = dict(
        coldens=CD_S_GAL,
        l0=ISM_L0,
        gamm=ISM_GAMM,
        fvalue=ISM_F,
        atmass=ISM_A,
        vturb=VTURB_GAL,
        vrad=VRAD_GAL,
    )
    kw.update(extra)
    return kw


def ism_kwargs_smc(**extra):
    kw = dict(
        coldens=CD_S_SMC,
        l0=ISM_L0,
        gamm=ISM_GAMM,
        fvalue=ISM_F,
        atmass=ISM_A,
        vturb=VTURB_SMC,
        vrad=VRAD_SMC,
    )
    kw.update(extra)
    return kw


def apply_ismlines_gal_smc(w, data, fast=False, **extra):
    """Apply GAL then SMC Si III λ1206 lines (test.plot ISMLINE case)."""
    fn = w.command_ismline if fast else w.command_ismline_original
    out = quiet(fn, data, **ism_kwargs_gal(**extra))
    out = quiet(fn, out, **ism_kwargs_smc(**extra))
    return out

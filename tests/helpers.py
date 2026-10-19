"""
Shared helpers for wrplot validation / speedup test suites.

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
WRPLOT_REF_CONVOL = REF / "ref_ismline_convol.txt"
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
# OG.plot / OG2.plot parameters (Si III λ1206)
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
GAUSS_FWHM = 0.5  # OG.plot COMMAND CONVOL GAUSS=0.5


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


def save_speedup_bar(labels, t_slow, t_fast, title, outfile, ylabel="Time [s]"):
    """Bar chart of slow vs fast runtimes with speedup annotations."""
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    x = np.arange(len(labels))
    w = 0.35
    ts = np.asarray(t_slow, dtype=float)
    tf = np.asarray(t_fast, dtype=float)
    ax.bar(x - w / 2, ts, w, label="slow", color="C0")
    ax.bar(x + w / 2, tf, w, label="fast", color="C3")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    for i, (a, b) in enumerate(zip(ts, tf)):
        if b > 0:
            ax.text(i, max(a, b) * 1.05, f"{a / b:.0f}×", ha="center", fontsize=9)
    fig.tight_layout()
    path = OUTPUT / outfile
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path}")
    return path


def savefig(fig, name):
    path = OUTPUT / name
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path}")
    return path


def og_ism_kwargs_gal(**extra):
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


def og_ism_kwargs_smc(**extra):
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


def apply_og_ismlines(w, data, fast=False, **extra):
    """Apply GAL then SMC Si III lines as in OG.plot / OG2.plot."""
    fn = w.command_ismline_fast if fast else w.command_ismline
    out = quiet(fn, data, **og_ism_kwargs_gal(**extra))
    out = quiet(fn, out, **og_ism_kwargs_smc(**extra))
    return out

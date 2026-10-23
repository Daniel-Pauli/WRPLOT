# wrplot.py — User Manual

This manual documents the Python module [`wrplot.py`](wrplot.py): a toolkit for
plotting and transforming astronomical spectra. The API is inspired by the
classic **WRPLOT** Fortran program, but you do **not** need prior WRPLOT
experience to use it.

> **Note (vs classic WRPLOT).** Classic WRPLOT is driven by `.plot` script files
> and `KASDEF` / `COMMAND` lines. This package is a **Python class API**: you
> call methods on a `wrplotter` instance. There is no `.plot` file interpreter.

---

## Table of contents

1. [Overview](#1-overview)
2. [Getting started](#2-getting-started)
3. [Core concepts](#3-core-concepts)
4. [Plotting](#4-plotting)
5. [Loading and saving data](#5-loading-and-saving-data)
6. [Dataset arithmetic](#6-dataset-arithmetic)
7. [Convolution](#7-convolution)
8. [Interstellar lines (`ISMLINE`)](#8-interstellar-lines-ismline)
9. [Lyman series absorption (`HLYMANA`)](#9-lyman-series-absorption-hlymana)
10. [Interstellar reddening (`EXTINCT`)](#10-interstellar-reddening-extinct)
11. [Line identifications](#11-line-identifications)
12. [Helper functions](#12-helper-functions)
13. [Fast vs original routines](#13-fast-vs-original-routines)
14. [Differences from classic WRPLOT](#14-differences-from-classic-wrplot)
15. [Testing](#15-testing)

---

## 1. Overview

`wrplotter` provides:

| Area | Main entry points |
|------|-------------------|
| Figures / axes | `figure`, `get_subplot`, `plot`, `show` |
| I/O | `command_include`, `command_write` |
| Table ops | `command_log`, `command_dex`, `command_cut`, `command_plus`, … |
| Spectra | `command_convol`, `command_ismline`, `command_hlymana`, `command_reddening` |
| Idents | `plot_idents` |

Most spectral routines return a modified **dataset** `[x, y]` (two rows /
length-2 array). Plotting draws on the current Matplotlib axes.

---

## 2. Getting started

### Install dependencies

See the [README](README.md) for NumPy, SciPy, Numba, Matplotlib, and
`svgpath2mpl`.

### Create a plotter

```python
from wrplot import wrplotter, warmup_kernels

warmup_kernels()   # optional: compile Numba kernels once
w = wrplotter()
```

### Minimal plot

```python
import numpy as np

x = np.linspace(0, 10, 101)
y = np.sin(x)

w.figure(figsize=(6, 4))
w.plot(x, y, XRANGE=(0, 10), YRANGE=(-1.2, 1.2),
       XTICKS=(2, 0.5), YTICKS=(0.5, 0.1),
       LABELS=("x", "y"), SYMBOL=5, COLOR=2, PEN=1.5)
w.show()
```

---

## 3. Core concepts

### 3.1 Datasets

A **dataset** is a length-2 sequence of arrays:

```text
dataset[0]  →  x  (often wavelength in Å)
dataset[1]  →  y  (often flux)
```

Example:

```python
dataset = np.array([wavelengths, fluxes])
```

Many `command_*` methods take a dataset and return a new or updated one. Prefer
keeping wavelengths **monotonic ascending** before spectral commands; several
routines sort automatically if needed.

### 3.2 Colors (`self.colors`)

Integer color indices `0`–`9` map to Matplotlib color names:

| Index | Name |
|------:|------|
| 0 | white |
| 1 | black |
| 2 | red |
| 3 | limegreen |
| 4 | blue |
| 5 | yellow |
| 6 | fuchsia |
| 7 | cyan |
| 8 | greenyellow |
| 9 | dodgerblue |

Used by `plot(..., COLOR=…)` and by `&0`–`&9` in identification text.

> **Note (vs classic WRPLOT).** Classic WRPLOT uses fixed RGB triples for
> colors 0–9. Here the names above are the practical equivalents in Matplotlib.

### 3.3 Plot symbols (`SYMBOL`)

`plot(..., SYMBOL=n)` accepts integer codes `0`–`33` (WRPLOT-style), or a
Matplotlib `(linestyle, marker)` tuple.

| Code | Meaning (summary) |
|-----:|-------------------|
| 0 | nothing |
| 1 | cross `+` |
| 2 | asterisk |
| 3 | triangle |
| 4 | square |
| 5 | solid polyline |
| 6 | cubic spline |
| 7 | cubic spline, dashed |
| 8 | circle |
| 9 | dashed polyline |
| 10 | dash-dotted polyline |
| 11 | inverted triangle |
| 12 | diamond |
| 13 | bomb (custom path) |
| 14 | Mercedes / three-spike |
| 15 | rhomb |
| 16 | vertical tick |
| 17 | flat rhomb |
| 18 | five-point star |
| 20 | dotted polyline |
| 21–25, 28, 31–32 | polyline + markers |
| 26 | filled plus |
| 27 | filled × |
| 29 | pentagon |
| 30 / 33 | multi-spike stars |

`SIZE > 0` sets marker size; `SIZE < 0` fills the area under a curve (or filled
markers). `HATCHED` can fill with hatch patterns (`D`/`d`, `V`/`v`, `H`/`h`,
`K`/`k`, `C`/`c`, or any Matplotlib hatch string).

### 3.4 Axes and `plt.gca()`

Plotting and `plot_idents` use Matplotlib’s **current axes** (`plt.gca()`).
After `figure` / `get_subplot` / `plot`, keep that axes active when adding
idents or overlays.

---

## 4. Plotting

### `figure(figsize=None, gridsize=None, fontsize=14, **kwargs)`

Creates a Matplotlib figure with Helvetica / TeX-oriented defaults and an
optional gridspec. Extra kwargs go to `matplotlib.pyplot.figure`.

| Argument | Meaning | Default |
|----------|---------|---------|
| `figsize` | Width, height [inches] | Matplotlib default |
| `gridsize` | `(nrows, ncols)` for subplots | `(1, 1)` |
| `fontsize` | Base font size | `14` |

```python
w.figure(figsize=(8.27, 11.69), gridsize=(2, 1), fontsize=12)  # A4-ish
```

### `get_grid()` / `get_subplot(xrange=None, yrange=None)`

Inspect or select a subplot from the gridspec. `xrange` / `yrange` are half-open
index slices on the grid (handle with care; less tested than `plot`). Both
default to `None` (single axes / full figure).

### `plot(x, y, …)`

Main plotting method. Important keyword arguments:

| Argument | Role | Default |
|----------|------|---------|
| `XRANGE`, `YRANGE` | Axis limits `(min, max)`; `None` → auto | `(None, None)` |
| `XTICKS`, `YTICKS` | `(major, minor)` spacing | `(None, None)` |
| `DARUNTER` | Tick origin `(x0, y0)` | `(0., 0.)` |
| `LABELS` | `(xlabel, ylabel)` | `("", "")` |
| `ALT` | Put labels on opposite side `(x, y)` | `(False, False)` |
| `NOLAB` | Suppress tick labels `(x, y)` | `(False, False)` |
| `SYMBOL` | Style code or `(linestyle, marker)` | `1` (cross) |
| `SIZE` | Marker size; negative → fill | `0.3` |
| `PEN` | Line width scale | `1` |
| `COLOR` | Index `0`–`9`, name, RGB(A) | `1` (black) |
| `HATCHED` | Hatch fill under curve | `None` |
| `spline_res` | Sampling for cubic-spline symbols 6/7 | `1e-3` |

Extra kwargs are forwarded to Matplotlib `plot` / `fill_between` where relevant
(`markersize`, `alpha`, `y2`, …).

Ticks are drawn inward on all four sides (twin axes), similar to a WRPLOT box.

### `show()`

Shows the current figure (`plt.show()`).

---

## 5. Loading and saving data

### `command_include(path, inckey=None, dataset=1, **kwargs)`

Load a table:

| Argument | Meaning | Default |
|----------|---------|---------|
| `path` | File to read | — (required) |
| `inckey` | If set, find this keyword then read an `N=` block | `None` (plain XY via `loadtxt`) |
| `dataset` | Which `N=` block to read when `inckey` is set | `1` |

Comment characters for the plain-table path include `#`, `*`, `N=`, `PLOT`, ….
Returns shape `(2, N)`.

```python
spec = w.command_include("spectrum.dat")
block = w.command_include("formal.plot", inckey="* IUE", dataset=1)
```

### `command_write(file, dataset, **kwargs)`

Write a dataset with `numpy.savetxt` (stores `dataset.T`). Kwargs are passed
through to `savetxt`.

---

## 6. Dataset arithmetic

Unless noted, methods return a new `[x, y]` array.

| Method | Effect | Default (`axis` / notes) |
|--------|--------|--------------------------|
| `command_log(dataset, axis=…)` | \(\log_{10}\) of both axes, or `"x"` / `"y"` only | `axis=None` (both) |
| `command_dex(dataset, axis=…)` | \(10^{x}\) (undo log) | `axis=None` (both) |
| `command_cut(dataset, cuttingrange, axis=…)` | Keep points inside a range | `axis=None` (both axes) |
| `command_plus` / `command_minus` | Add / subtract a value on an axis | `axis="x"` |
| `command_multiply` / `command_divide` | Scale an axis | `axis="x"` |
| `command_ari(dataset1, dataset2, axis=…)` | Add two spectra on a common grid | `axis="y"` |
| `command_insert(dataset, x)` | Insert one \(x\) by linear interpolation | — |
| `command_sort(dataset)` | Sort by \(x\) | — |
| `command_filter(magnitude, filter_name)` | Magnitude → \((\lambda_\mathrm{c}, F_\lambda)\) | — |

### Cut ranges

```python
# keep 1000 < λ < 2000
cut = w.command_cut(spec, (1000, 2000), axis="x")

# both axes: cuttingrange = ((xmin, xmax), (ymin, ymax))
cut2 = w.command_cut(spec, ((1000, 2000), (0, 2)))
```

### Adding spectra (`command_ari`)

Interpolates the coarser grid onto the finer one over the overlapping range,
then adds the chosen column (`axis="y"` for flux, `"x"` for the other column).

> **Note (vs classic WRPLOT).** The `"x"` / `"y"` naming for which column is
> merged may feel opposite to some WRPLOT `ARI` conventions; here `"y"` means
> “add the flux column.”

### Filters (`command_filter`)

Converts a magnitude to continuum flux at a central wavelength for named
systems, e.g. `U`, `B`, `V`, `I`, `J`, `H`, `K`, `MCPS:*`, `2MASS:*`, `Gaia:G`,
`SDSS:g`, `SMASH:*`, `UVIT:172`. Returns `(central_wavelength, flux)`.

---

## 7. Convolution

Broaden a spectrum with a Gaussian, box, or rotation kernel.

### Prefer: `command_convol`

```python
command_convol(
    dataset, shape="gaussian", value=0, beta=0,
    precision=1e-4, allow_fft=False, d_angstrom=None,
)
```

| Parameter | Meaning | Default |
|-----------|---------|---------|
| `shape` | Kernel type (see below) | `"gaussian"` |
| `value` | Width parameter (see below); ignored for rot if `d_angstrom` is set | `0` (no-op) |
| `beta` | Limb-darkening β (rotation only) | `0` |
| `precision` | Max \|Δflux\| vs original path; `None` / `≤0` → exact | `1e-4` |
| `allow_fft` | Opt-in FFT on uniform grids (gauss/box only) | `False` |
| `d_angstrom` | Fixed ROT half-width \(D\) [Å] (WRPLOT-like) | `None` (use vsini) |

| `shape` | `value` meaning |
|---------|-----------------|
| `"gauss"` / `"gaussian"` | FWHM [Å] |
| `"box"` / `"kasten"` | half-width [Å] |
| `"rot"` / `"rotation"` | **vsini [km/s]** with \(\Delta\lambda \propto \lambda\) (unless `d_angstrom` is set) |

For WRPLOT-identical rotation (fixed \(\Delta\lambda_\mathrm{max}\) in Å), pass
`d_angstrom=D` instead of using vsini:

```python
# WRPLOT-like ROT with D = 2.5 Å, β = 0.6
spec = w.command_convol(spec, shape="rot", d_angstrom=2.5, beta=0.6)

# Wavelength-scaling vsini (Python default for rot)
spec = w.command_convol(spec, shape="rot", value=150.0, beta=0.6)
```

Optional FFT path for **uniform** grids and Gaussian/box only:
`allow_fft=True` (off by default).

### Reference: `command_convol_original`

Pure-Python port of WRPLOT `CONVEX` / `CONVADD`. For `rot`, `value` is always
the fixed half-width \(D\) [Å] (not vsini). Use for validation; prefer
`command_convol` in production.

> **Note (vs classic WRPLOT).** Classic `CONVOL … ROT` uses fixed \(D\) [Å].
> Python `command_convol(..., shape="rot")` defaults to **vsini [km/s]** unless
> you set `d_angstrom`.

---

## 8. Interstellar lines (`ISMLINE`)

Apply extinction by a **single** interstellar line (Voigt or Lorentz profile).

### Prefer: `command_ismline`

```python
command_ismline(
    dataset, coldens, l0, fvalue, gamm,
    atmass=None, vrad=0.0, temp=-273.15, vturb=0.0,
    cut_ism=True, window=30.0, ism_window=None, info=False,
)
```

| Parameter | Meaning | Default |
|-----------|---------|---------|
| `coldens` | Column density [cm⁻²]; **negative** → absorption | — (required) |
| `l0` | Rest wavelength [Å] | — (required) |
| `fvalue` | Oscillator strength | — (required) |
| `gamm` | Damping constant [s⁻¹] | — (required) |
| `atmass` | Atomic mass [amu]; required if `temp > 0` | `None` |
| `vrad` | Radial velocity [km/s] | `0.0` |
| `temp` | Temperature [K]; with `vturb`, selects Voigt vs Lorentz | `-273.15` |
| `vturb` | Turbulent velocity [km/s] | `0.0` |
| `cut_ism` | Restrict work to a wavelength window | `True` |
| `window` | Half-width [Å] around the shifted line | `30.0` |
| `ism_window` | Absolute `(lo, hi)` instead of `±window` | `None` |
| `info` | Print Fortran-style status messages | `False` |

Input: linear wavelength [Å] and linear flux. The grid may be refined near the
line.

```python
out = w.command_ismline(
    spec, coldens=-5e13, l0=1206.500, fvalue=1.66, gamm=2.57e8,
    atmass=28.0855, vrad=0.0, temp=100.0, vturb=0.0,
)
```

Set `cut_ism=False` for a full-spectrum match to the slow/reference path (still
much faster).

### Reference: `command_ismline_original`

1:1 translation of Fortran `ISMLINE`. Prefer `command_ismline` for work.

---

## 9. Lyman series absorption (`HLYMANA`)

Correct a spectrum for H I Lyman lines (Lyα through higher members), including
a Lyman-edge term.

### Prefer: `command_hlymana`

```python
command_hlymana(
    dataset, key="coldens", value=-1, vrad=0, temp=-1, vturb=0,
    info=False, cut_lyman=True, lyman_window=(900.0, 1300.0),
)
```

| Parameter | Meaning | Default |
|-----------|---------|---------|
| `key` | `"coldens"` or `"ebv"` | `"coldens"` |
| `value` | Column density [cm⁻²] (typically negative for absorption), or \(E(B-V)\) if `key="ebv"` | `-1` |
| `vrad` | Radial velocity [km/s] | `0` |
| `temp` | Temperature [K]; `temp > -1` enables thermal Doppler | `-1` |
| `vturb` | Turbulent velocity [km/s]; `> 0` adds Doppler broadening | `0` |
| `info` | Print status messages | `False` |
| `cut_lyman` | Process only `lyman_window` | `True` |
| `lyman_window` | Wavelength window [Å], shifted with `vrad` | `(900.0, 1300.0)` |

If `temp ≤ -1` and `vturb ≤ 0`, Lorentz (Groenewegen & Lamers) profiles are used;
otherwise Voigt cores.

Input: wavelength [Å], **linear** flux. Wavelength axis must be monotonic
(sorted if needed).

For `key="ebv"`, \(N_\mathrm{H} \propto E(B-V)\) with the same constant as the
Fortran path.

### Reference: `command_hlymana_original`

Full-spectrum pure-Python path for validation.

---

## 10. Interstellar reddening (`EXTINCT`)

Apply extinction laws to a spectrum stored in **log** wavelength and **log** flux.

### Prefer: `command_reddening`

```python
command_reddening(dataset, ebv, law="seaton", Rv=-1, info=False)
```

| Parameter | Meaning | Default |
|-----------|---------|---------|
| `dataset` | `[log10(λ/Å), log10(flux)]` | — (required) |
| `ebv` | Colour excess \(E(B-V)\) | — (required) |
| `law` | `seaton`, `cardelli`, `fitzpatrick`, `lmc`, `smc`, `smc_bar` | `"seaton"` |
| `Rv` | Optional \(R_V\) where the law allows (Cardelli, Fitzpatrick, SMC) | `-1` (law default) |
| `info` | Kept for API compatibility | `False` |

Short inputs are expanded toward ≥1000 points (largest-gap midpoints), matching
WRPLOT behaviour.

```python
log_spec = w.command_log(spec)           # if you start from linear
red = w.command_reddening(log_spec, ebv=0.1, law="fitzpatrick", Rv=3.1)
linear = w.command_dex(red)
```

### Reference: `command_reddening_original`

Slow pure-Python path; same physics and API.

---

## 11. Line identifications

### Prefer: `plot_idents`

Draw labels from an identification file onto the **current** axes (plot the
spectrum first).

```python
plot_idents(
    file, id_xoff=0, id_y=1.05, id_len=0.05, id_fontsize=12,
    id_linesize=0.75, id_horizontal=False, color="black", bbox_alpha=0,
    color_line_from_text=False, auto=False, auto_offset=0.2,
    auto_window=None, auto_min=None, fixed=False,
)
```

| Parameter | Meaning | Default |
|-----------|---------|---------|
| `file` | Path to an idents file | — (required) |
| `id_xoff` | Global extra x-offset [data units] | `0` |
| `id_y` | Mark base height (y units); ignored if `auto=True` | `1.05` |
| `id_len` | Mark length (y units) | `0.05` |
| `id_fontsize` | Label font size | `12` |
| `id_linesize` | Mark line width | `0.75` |
| `id_horizontal` | Horizontal text if `True` (else vertical) | `False` |
| `color` | Default mark/text color | `"black"` |
| `bbox_alpha` | White text background opacity | `0` |
| `color_line_from_text` | Mark colour follows first `&n` in the label | `False` |
| `auto` | Place marks above local spectrum (needs a Line2D on the axes) | `False` |
| `auto_offset` | Clearance above spectrum when `auto=True` | `0.2` |
| `auto_window` | Half-width in \(x\) for local stats | `None` → `0.5` |
| `auto_min` | Optional floor for `id_y` when `auto=True` | `None` |
| `fixed` | Force kwargs `id_y`/`id_len`; ignore file `\IDY`/`\IDLENG` | `False` |

`auto` and `fixed` are mutually exclusive.

Marks use the same three-piece geometry as the legacy helper: lower **0.4**,
connector **0.1**, upper **0.5** of `id_len`. Labels are automatically
x-shifted to reduce overlap.

```python
w.figure(figsize=(10, 4))
w.plot(spec[0], spec[1], XRANGE=(1020, 1050), YRANGE=(0, 1.5),
       SYMBOL=5, COLOR=1, PEN=1)
w.plot_idents("id_O.dat", id_y=1.1, id_len=0.08, color_line_from_text=True)
# or:
w.plot_idents("id_O.dat", auto=True, auto_offset=0.2, auto_min=1.05)
```

### Identification file format

Lines starting with `*` are comments. Supported commands:

```text
\IDY    1.05          # base y for following marks (optional trailing U)
\IDLENG 0.08          # mark length
\IDENT  1031.9  &EO\,VI
\IDMULT 1031.9 1037.6  "&EO\,VI&4\,(is)"
```

Only idents whose wavelengths lie inside the current x-limits are drawn.

### Text notation (subset)

Prefixes and escapes in the label string are interpreted (not stripped):

| Code | Effect |
|------|--------|
| `&0`–`&9` | Text colour (`self.colors`) |
| `&T` / `&H` / `&M` | Subscript / superscript / normal |
| `&E` / `&B` | Compact / broad letter spacing |
| `&I` / `&N` / `&F` / `&f` / `&W` | Italic, reset style, bold on/off, quasi-bold |
| `&R` / `&L` / `&G` | Inclination right / left / none |
| `&PW` / `&PT` / `&PH` | Font family hints |
| `#a#`, `#b#`, … | Greek letters (e.g. `#e#` → ε) |
| `\,` | Thin space (rendered as a visible mathtext gap) |
| `\A`, `\S`, `\M`, … | Common symbols (Å, ⊙, Ṁ, …) |

Multi-segment colour breaks (e.g. `&EO\,VI&4\,(is)`) keep a small gap between
segments (~0.2 em).

> **Note (vs classic WRPLOT).** Classic WRPLOT draws idents inside the plot
> language. Here you call `plot_idents` from Python. Extra options `auto`,
> `fixed`, and `color_line_from_text` have no direct one-line WRPLOT equivalent.

### Legacy: `plot_idents_original`

Older implementation that **strips** most WRPLOT text prefixes (including
colours). Kept for comparison; prefer `plot_idents`.

---

## 12. Helper functions

| Method | Purpose | Defaults |
|--------|---------|----------|
| `gauss(x, mu=…, sigma=…)` | Normalised Gaussian samples | `mu=0`, `sigma=1` |
| `box(x, mu=…, width=…)` | Box kernel (not normalised to 1) | `mu=0`, `width=1` |
| `rot(x, mu=…, d_angstrom=…, beta=…)` | Unsoeld rotation kernel weights | `mu=0`, `d_angstrom=0`, `beta=0` |
| `voigth(a, v)` | Normalised Voigt \(H(a,v)\) (Köster / WRPLOT) | — |
| `is_float(element)` | Parse helper | — |
| `warmup_kernels()` | Module-level: compile Numba kernels once | — |

---

## 13. Fast vs original routines

Several commands exist in pairs:

| Fast (preferred) | Original (reference) |
|------------------|----------------------|
| `command_convol` | `command_convol_original` |
| `command_ismline` | `command_ismline_original` |
| `command_hlymana` | `command_hlymana_original` |
| `command_reddening` | `command_reddening_original` |
| `plot_idents` | `plot_idents_original` |

Fast paths use Numba and optional windowing (`cut_ism`, `cut_lyman`). Original
paths exist for bit-level / high-precision checks against classic WRPLOT and
against the fast path. For everyday work, use the fast names.

Call `warmup_kernels()` once per process if you care about first-call latency.

---

## 14. Differences from classic WRPLOT

Short summary of the main behavioural differences (also noted above):

1. **API, not scripts** — Python methods instead of `.plot` / `KASDEF` files.
2. **Rotation convolution** — `command_convol(..., "rot")` defaults to vsini
   [km/s]; use `d_angstrom=` for classic fixed-\(D\) ROT.
3. **Windowed ISM / Lyman** — Fast paths may restrict the wavelength range by
   default; turn windowing off for a full-spectrum match.
4. **Reddening units** — Expect **log** λ and **log** flux (as in WRPLOT
   `EXTINCT` tables).
5. **Idents** — Richer Python options (`auto`, `fixed`, `color_line_from_text`);
   text notation is preserved in `plot_idents`, not in `plot_idents_original`.
6. **Plotting backend** — Matplotlib + optional LaTeX text; not the X11 WRPLOT
   window or PostScript driver.

Physics of the ported spectral commands is intended to match WRPLOT within the
tolerances used in `tests/` (typically \(\sim 10^{-4}\) in flux vs reference
tables).

---

## 15. Testing

The [`tests/`](tests/) directory compares Python results to classic WRPLOT
reference spectra and regenerates QC plots. See [`tests/README.md`](tests/README.md).

```bash
python tests/run_all.py              # units + plots
python tests/run_all.py --unit-only
python tests/run_all.py --plots-only
```

---

*This manual documents only the public surface of `wrplot.py`. Private helpers
(`__ident_*`, Numba kernels, etc.) are implementation details and may change.*

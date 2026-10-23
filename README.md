# wrplot (Python)

Python toolkit for astronomical spectra and plots in the spirit of the classic
**WRPLOT** program (Wolf-Rainer Hamann et al.). It is a **programmatic API**
built around the `wrplotter` class in [`wrplot.py`](wrplot.py)—not an interpreter
for WRPLOT `.plot` / KASDEF script files.

Typical uses:

- style plots similarly to WRPLOT (symbols, pens, tick layout)
- load and transform wavelength–flux tables
- apply spectral operations (convolution, ISM lines, Lyman absorption, reddening)
- draw line identifications with WRPLOT-style text notation

## Documentation

Full user manual: **[MANUAL.md](MANUAL.md)**

## Requirements

- Python 3
- [NumPy](https://numpy.org/), [SciPy](https://scipy.org/), [Numba](https://numba.pydata.org/)
- [Matplotlib](https://matplotlib.org/) (loaded on first plot)
- [svgpath2mpl](https://pypi.org/project/svgpath2mpl/) (custom plot symbols)

```bash
pip install -r requirements.txt
# or: conda install numpy scipy numba matplotlib && pip install svgpath2mpl
```

Optional for `wrplotter.figure()`: a working LaTeX install (Matplotlib
`text.usetex`). The validation suite and most spectral commands do not need it.

## Setup

There is no packaged install yet. Put the repository on your Python path, or
work from the repo root:

```bash
cd /path/to/wrplot
python -c "from wrplot import wrplotter; print(wrplotter)"
```

In a notebook or script next to `wrplot.py`:

```python
from wrplot import wrplotter

w = wrplotter()
```

Optional: compile Numba kernels once at process start (avoids first-call JIT cost):

```python
from wrplot import warmup_kernels
warmup_kernels()
```

## Quick example

```python
import numpy as np
from wrplot import wrplotter

w = wrplotter()
x = np.linspace(1000, 1200, 2001)
y = np.ones_like(x)

spec = w.command_ismline(
    [x, y], coldens=-1e14, l0=1206.5, fvalue=1.66, gamm=2.57e8, atmass=28.0
)

w.figure(figsize=(8, 4))
w.plot(spec[0], spec[1], XRANGE=(1190, 1220), YRANGE=(0, 1.2),
       XTICKS=(10, 2), YTICKS=(0.2, 0.05), LABELS=(r"$\lambda$ [\AA]", "flux"),
       SYMBOL=5, COLOR=1, PEN=1.5)
w.show()
```

## Tests

Validation against classic WRPLOT references lives under [`tests/`](tests/).
See [`tests/README.md`](tests/README.md).

```bash
python tests/run_all.py          # unit tests + QC plots
python tests/run_all.py --unit-only
```

## License / provenance

Spectral physics routines follow the classic WRPLOT Fortran implementations
where noted in the manual. Plotting is Matplotlib-based with WRPLOT-inspired
defaults.

# wrplot validation suite

Self-contained tests comparing the Python `wrplot.py` implementation to classic
WRPLOT. Tolerance for flux differences vs WRPLOT: **1e-4**.

## Layout

| Path | Role |
|------|------|
| `test.plot` | WRPLOT script that writes reference spectra into `ref/` |
| `run_wrplot_refs.sh` | Runs `wrplot test.plot +` to regenerate `ref/` |
| `data/` | Input continua (IUE, UV, synthetic) used by WRPLOT and Python |
| `ref/` | WRPLOT `COMMAND WRITE` tables (ground truth) |
| `output/` | Comparison / speedup plots from the Python suite |
| `run_all.py` | Unit tests + plot generation |

## 1. Generate WRPLOT references

Requires a WRPLOT install (`PATH_WRPLOT`, default `/home/daniel/wrplot.dir`):

```bash
cd tests
./run_wrplot_refs.sh
# equivalent:  wrplot test.plot +
```

This creates nine tables in `ref/` (ISMLINE, ISMLINE+CONVOL, HLYMANA, CONVOL
gauss/box/rot).

## 2. Run Python validation

From the repo root (needs the same Python env as `wrplot.py`, e.g. anaconda):

```bash
/home/daniel/anaconda3/bin/python tests/run_all.py
/home/daniel/anaconda3/bin/python tests/run_all.py --unit-only
/home/daniel/anaconda3/bin/python tests/run_all.py --plots-only
```

Or individual modules:

```bash
cd tests
python test_ismline.py
python test_og_pipeline.py
…
```

## Notes

- No external `formal.plot` path is required; `data/continuum_iue.dat` is a
  portable 900–1400 Å extract of the `* IUE` continuum.
- Parent directory `wrplot.py` is imported automatically via `helpers.py`.

# wrplot validation suite

Self-contained tests comparing the Python `wrplot.py` implementation to classic
WRPLOT. Tolerance for flux differences vs WRPLOT: **1e-4**.

## Layout

| Path | Role |
|------|------|
| `test.plot` | WRPLOT script that writes reference spectra into `ref/` |
| `run_wrplot_refs.sh` | Runs `wrplot test.plot +` to regenerate `ref/` |
| `data/` | Input continua used by WRPLOT and Python |
| `ref/` | WRPLOT `COMMAND WRITE` tables + `id_O.dat` (idents fixture) |
| `output/` | QC spectrum plots (`.png`) and timing tables (`.txt`) |
| `run_all.py` | Unit tests + QC output generation (incl. idents figures) |
| `test_idents.py` | WRPLOT-style `plot_idents` (colors, half-space gap vs geometry) |

## 1. Generate WRPLOT references

Requires a WRPLOT install (`PATH_WRPLOT`, default `/home/daniel/wrplot.dir`):

```bash
cd tests
./run_wrplot_refs.sh
# equivalent:  wrplot test.plot +
```

This creates tables in `ref/` (ISMLINE, ISMLINE+CONVOL, HLYMANA,
CONVOL gauss/box/rot, and REDDENING for Seaton/Cardelli/Fitzpatrick/LMC/SMC/SMC_BAR).

## 2. Run Python validation

From the repo root (needs the same Python env as `wrplot.py`, e.g. anaconda):

```bash
/home/daniel/anaconda3/bin/python tests/run_all.py
/home/daniel/anaconda3/bin/python tests/run_all.py --unit-only
/home/daniel/anaconda3/bin/python tests/run_all.py --plots-only
```

## Notes

- No external `formal.plot` path is required; `data/continuum_iue.dat` is a
  portable 900–1400 Å extract of the `* IUE` continuum.
- Parent directory `wrplot.py` is imported automatically via `helpers.py`.
- REDDENING uses `data/continuum_reddening_flat.dat` (1000–1e5 Å, Y=0 ≡
  log flux of 1). Fast path is `command_reddening` (numba); slow reference is
  `command_reddening_original`. QC plots: `output/reddening_comparison.png`,
  `output/reddening_rv_comparison.png`, `output/reddening_slow_fast.png`,
  plus `output/reddening_timing.txt`.
- IDENTS (`plot_idents`) uses `ref/id_O.dat`. A full `run_all.py` or
  `run_all.py --plots-only` regenerates `output/idents_smoke.png`,
  `output/idents_geometry_matrix.png`, and `output/idents_*.png` (one per
  figure size / ylim). `--unit-only` skips those figure updates.

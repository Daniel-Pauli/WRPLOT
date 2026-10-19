# Known bugs / Fortran mismatches (todo)

Do not treat style nits as bugs. Items below are real or likely math/behaviour issues vs WRPLOT / `wrplot.f`.

Already fixed (not listed): `voigth` / `_voigth_njit` large-`a` branch; CONVADD-style Δx weighting in convol; `command_ismline` 1:1 port and new `command_ismline_fast`.

---

## High

- [x] **Wrong Lyman-edge τ (`TAULYMEDGE`) in both HLYMANA paths**
  - Where: `command_hlymana`, `command_hlymana_fast`
  - Problem: both set `TAULYMEDGE = ENH * C0[0]` (Lyα). Fortran uses `C0(NHI)` (last Lyman line) for Lorentz, or last-line Voigt centre τ for Voigt. Slow path even computes the correct value in-loop, then overwrites it.
  - Impact: flux between Lyman edge and highest line is wrong (~2.3× too strong for Lorentz).
  - Fixed: both paths now use `C0[-1]` / last-line Voigt; covered by `test_hlymana.py`.

- [x] **Broken TAUCRIT refinement in slow `command_hlymana`**
  - Where: profile loop in `command_hlymana`
  - Problem: Fortran inserts one midpoint and re-evaluates. Python inserts two midpoints, does `I = I-1`, then can append the previous wavelength again.
  - Impact: duplicate grid points / corrupted continuum near strong Lyman cores.
  - Fixed: single midpoint insert + re-eval at same index (Fortran `GOTO 16`); no duplicate λ.

---

## Medium

- [ ] **`command_hlymana_fast` TAUCRIT is still batched, not sequential Fortran**
  - Problem: uses batched adaptive midpoints with a max-iteration cap (not Fortran’s per-point `GOTO 16` walk). Inter-line midpoints are now inserted (MAXNPOINTS=0 path).
  - Impact: grids can still differ slightly; in practice slow↔fast agree to ~machine precision on tested UV grids after the other fixes.
- [x] ~~skipped inter-line midpoints in fast path~~ — fixed (insert centres + midpoints like Fortran `MAXNPOINTS=0`).

- [x] **Rotational convolution ≠ WRPLOT** (documented)
  - Where: `rot` / `_convol_rot_kernel` / `command_convol`
  - Problem: Python used Gray limb-darkening with `vsini` in km/s (width ∝ λ). Fortran uses Unsoeld with fixed Δλ in Å and different β weighting.
  - Impact: `shape="rot"` will not match WRPLOT.
  - Fixed: Unsoeld kernel + fixed `value` = Δλ_max [Å] (WRPLOT `D`); slow = pure Python, fast = numba. Covered by `test_convol_rot.py`.
  - **API change:** `command_convol(..., "rot", value)` now expects Δλ [Å], not vsini [km/s]. Convert via `Δλ = λ · v_sin_i / c`.

- [ ] **Fitzpatrick IR uses wrong `R_V`**
  - Where: `command_reddening` (~2765–2776)
  - Problem: for `law="fitzpatrick"`, IR Moneti branch always uses `RVMONETI=3.1`. Fortran uses the Fitzpatrick `R_V`.
  - Impact: λ ≳ 10⁴ Å when `Rv ≠ 3.1`.

- [ ] **HLYMANA with `VTURB>0` and default `TEMP=-1`**
  - Problem: uses `sqrt(2kT/m + v²)` with `T=-1` → negative thermal term; can NaN for very small `vturb`. ISMLINE handles this correctly (pure turbulence when `TEMP≤0`). Same as Fortran HLYMANA, but a footgun.
  - Impact: turbulence-only HLYMANA calls with tiny `vturb`.

- [ ] **`command_convol_fast(..., precision=...)` does not enforce the claimed tolerance**
  - Problem: docstring implies verified agreement within `precision`; the flag only gates FFT eligibility. No compare/retry against `command_convol`.
  - Impact: callers who trust `precision=1e-4` as a verified bound.

---

## Low

- [ ] **Misleading / copy-pasted docstrings**
  - `command_reddening` still has HLYMANA text
  - HLYMANA docs say `3.7e21` (code uses `3.8e21`) and wrong `T`/`vturb` thresholds
  - `smc_bar` warning mentions `'seaton'`

- [ ] **Gaussian convol cutoff slightly wider than WRPLOT**
  - Python: `4.5σ` ≈ 1.91×FWHM
  - Fortran: `3×0.6006×FWHM` ≈ 1.80×FWHM
  - Impact: small wing difference only.

- [ ] **τ≥10 warning in slow HLYMANA prints wrong variable**
  - Uses leftover `TAU` instead of `tt`.

- [x] **`rot()` helper can NaN outside the rotation disk**
  - Fixed with Unsoeld rewrite: returns 0 where `|xa| ≥ 1`.

- [ ] **MACRO-RT macroturbulence not ported**
  - Fortran CONVEX has it; Python only gauss/box/rot.

---

## Suggested fix order

1. Lyman-edge `TAULYMEDGE` in both HLYMANA paths
2. Slow-path TAUCRIT double-insert / `I-=1` logic
3. Align or clearly document `command_hlymana_fast` vs Fortran refinement
4. Remaining medium/low items as needed

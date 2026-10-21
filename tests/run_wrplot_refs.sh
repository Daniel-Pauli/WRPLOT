#!/usr/bin/env bash
# Generate WRPLOT reference spectra used by the Python validation suite.
#
# Usage (from anywhere):
#   ./tests/run_wrplot_refs.sh
# or from tests/:
#   ./run_wrplot_refs.sh
#
# Requires the classic WRPLOT installation (PATH_WRPLOT / wrplot alias).

set -euo pipefail

TESTS_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$TESTS_DIR"
mkdir -p ref data output

if [[ -z "${PATH_WRPLOT:-}" ]]; then
  if [[ -d /home/daniel/wrplot.dir ]]; then
    export PATH_WRPLOT=/home/daniel/wrplot.dir
  else
    echo "ERROR: PATH_WRPLOT is not set and /home/daniel/wrplot.dir not found."
    echo "Set PATH_WRPLOT to your WRPLOT installation root, then re-run."
    exit 1
  fi
fi

WRPLOT_COM="$PATH_WRPLOT/proc.dir/wrplot.com"
if [[ ! -x "$WRPLOT_COM" ]]; then
  echo "ERROR: WRPLOT launcher not found/executable: $WRPLOT_COM"
  exit 1
fi

# Aliases are off in non-interactive bash; call the launcher directly.
wrplot() { "$WRPLOT_COM" "$@"; }

echo "Working directory: $TESTS_DIR"
echo "PATH_WRPLOT=$PATH_WRPLOT"
echo "Running: wrplot test.plot +"
echo "(PostScript device; COMMAND WRITE dumps tables into ref/)"
echo

# Clean old PS scratch; keep previous refs until overwritten
rm -f wrplot.ps.* wrplot_scratch_* 2>/dev/null || true

# '+' = automatic PostScript (non-GUI). Extra newlines satisfy any
# "Enter Option" prompts between plots.
printf '\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n' | wrplot test.plot +
EXPECTED=(
  ref/ref_ismline.txt
  ref/ref_ismline_convol.txt
  ref/ref_hlymana_1e20.txt
  ref/ref_hlymana_5e20.txt
  ref/ref_hlymana_1e21.txt
  ref/ref_hlymana_uv_1e20.txt
  ref/ref_convol_gauss.txt
  ref/ref_convol_box.txt
  ref/ref_convol_rot.txt
  ref/ref_reddening_seaton.txt
  ref/ref_reddening_cardelli.txt
  ref/ref_reddening_cardelli_rv5.txt
  ref/ref_reddening_fitzpatrick.txt
  ref/ref_reddening_fitzpatrick_rv5.txt
  ref/ref_reddening_lmc.txt
  ref/ref_reddening_smc.txt
  ref/ref_reddening_smc_bar.txt
)

echo
echo "Reference files in ref/:"
ls -lh ref/ref_*.txt 2>/dev/null || true

missing=0
for f in "${EXPECTED[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "MISSING: $f"
    missing=1
  fi
done
if [[ "$missing" -ne 0 ]]; then
  echo "ERROR: not all expected reference files were produced."
  exit 1
fi

echo
echo "Done. All ${#EXPECTED[@]} WRPLOT reference tables are ready."
echo "Run Python suite with:  python run_all.py"

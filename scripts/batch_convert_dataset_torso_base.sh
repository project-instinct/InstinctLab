#!/usr/bin/env bash
# Batch-run npz_convert_pelvis_root_to_torso_root.py on:
#   - all climb_* clip folders under data/dataset_folder  (…/climb_*/*.npz)
#   - the whole lafan1_g1 tree
#
# Input:  *.retargeted.npz  (pelvis root, not already torso_base)
# Output: <stem>.torso_base.retargeted.npz in the same directory
#
# Usage:
#   ./WBCHSI/scripts/batch_convert_dataset_torso_base.sh
#   FORCE=1 ./WBCHSI/scripts/batch_convert_dataset_torso_base.sh   # overwrite existing torso_base outputs
#   PYTHON=/path/to/conda/env/bin/python ./WBCHSI/scripts/batch_convert_dataset_torso_base.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONVERTER="${SCRIPT_DIR}/npz_convert_pelvis_root_to_torso_root.py"
DATASET="${DATASET_ROOT:-${PROJECT_ROOT}/data/dataset_folder}"

if [[ ! -f "${CONVERTER}" ]]; then
  echo "error: converter not found: ${CONVERTER}" >&2
  exit 1
fi

if [[ ! -d "${DATASET}" ]]; then
  echo "error: dataset folder not found: ${DATASET}" >&2
  exit 1
fi

PY="${PYTHON:-python3}"

# Optional passthrough args for the Python script, e.g. BATCH_EXTRA_ARGS='--rot_weight 12'
# shellcheck disable=SC2206
EXTRA=( ${BATCH_EXTRA_ARGS:-} )

mapfile -d '' -t FILES < <(
  find "${DATASET}" -type f \
    \( -path "*/climb_*/*" -o -path "*/lafan1_g1/*" \) \
    -name '*.retargeted.npz' \
    ! -name '*.torso_base.retargeted.npz' \
    -print0 | sort -z
)

if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "No matching *.retargeted.npz under climb_* or lafan1_g1 (excluding *.torso_base.retargeted.npz)."
  exit 0
fi

echo "Dataset root: ${DATASET}"
echo "Files to convert: ${#FILES[@]}"
echo

n_ok=0
n_skip=0
n_fail=0

for src in "${FILES[@]}"; do
  base="$(basename "${src}")"                 # motion1.retargeted.npz
  stem="${base%.retargeted.npz}"             # motion1
  out_dir="$(dirname "${src}")"
  dst="${out_dir}/${stem}.torso_base.retargeted.npz"

  if [[ -f "${dst}" && "${FORCE:-0}" != "1" ]]; then
    echo "[skip] ${src} -> ${dst} (exists; set FORCE=1 to overwrite)"
    n_skip=$((n_skip + 1))
    continue
  fi

  echo "[run]  ${src}"
  if "${PY}" "${CONVERTER}" -i "${src}" -o "${dst}" "${EXTRA[@]}"; then
    n_ok=$((n_ok + 1))
  else
    echo "[fail] ${src}" >&2
    n_fail=$((n_fail + 1))
  fi
done

echo
echo "Done: ok=${n_ok} skip=${n_skip} fail=${n_fail}"

if [[ "${n_fail}" -gt 0 ]]; then
  exit 1
fi

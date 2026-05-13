#!/usr/bin/env bash
# 1) Recursively delete motion*.torso_base.retargeted.npz anywhere under lafan1_g1/
# 2) Regenerate walk* only from motion*.retargeted.npz via npz_convert_pelvis_root_to_torso_root.py
#
# Usage (from anywhere):
#   bash WBCHSI/scripts/batch_lafan1_walk_torso_base_npz.sh
#
# Requires: same Python env as manual conversion (torch, scipy, pytorch_kinematics).

set -euo pipefail
shopt -s nullglob

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

LAFAN_DIR="${PROJECT_ROOT}/data/dataset_folder/lafan1_g1"
CONVERTER="${SCRIPT_DIR}/npz_convert_pelvis_root_to_torso_root.py"
URDF_PELVIS="${PROJECT_ROOT}/WBCHSI/source/instinctlab/instinctlab/assets/resources/unitree_g1/omniretarget_models/g1/g1_29dof_spherehand.urdf"
URDF_TORSO="${PROJECT_ROOT}/WBCHSI/source/instinctlab/instinctlab/assets/resources/unitree_g1/urdf/g1_29dof_torsobase_popsicle_spherehand.urdf"

if [[ ! -d "${LAFAN_DIR}" ]]; then
  echo "ERROR: LAFAN directory not found: ${LAFAN_DIR}" >&2
  exit 1
fi
if [[ ! -f "${CONVERTER}" ]]; then
  echo "ERROR: converter not found: ${CONVERTER}" >&2
  exit 1
fi

deleted=0
converted=0

echo "== delete motion*.torso_base.retargeted.npz under entire ${LAFAN_DIR} (recursive) =="
mapfile -d '' stale_files < <(find "${LAFAN_DIR}" -mindepth 1 -type f -name 'motion*.torso_base.retargeted.npz' -print0 2>/dev/null)
for stale in "${stale_files[@]}"; do
  [[ -n "${stale}" ]] || continue
  echo "  delete ${stale}"
  rm -f "${stale}"
  deleted=$((deleted + 1))
done

echo "== convert walk* only =="
for sub in "${LAFAN_DIR}/walk"*/; do
  [[ -d "${sub}" ]] || continue

  echo "-- $(basename "${sub}") --"

  for inp in "${sub}"motion*.retargeted.npz; do
    [[ -f "${inp}" ]] || continue
    case "${inp}" in
      *torso_base*) continue ;;
    esac

    bn="$(basename "${inp}")"
    stem="${bn%.retargeted.npz}"
    out="${sub}${stem}.torso_base.retargeted.npz"

    echo "  convert ${bn} -> $(basename "${out}")"
    python "${CONVERTER}" \
      --input "${inp}" \
      --output "${out}" \
      --urdf_pelvis_base "${URDF_PELVIS}" \
      --urdf_torso_base "${URDF_TORSO}"
    converted=$((converted + 1))
  done
done

echo "Done. Removed ${deleted} torso_base npz, ran ${converted} conversions."

#!/usr/bin/env bash
# GeoRel-CLIP environment setup
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

echo "[setup] GeoRel-CLIP project root: ${PROJECT_ROOT}"

# Fix invalid OpenMP thread counts on some cloud platforms (e.g. OMP_NUM_THREADS=0)
fix_openmp_env() {
  local var val
  for var in OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS; do
    val="${!var:-}"
    if [[ -z "${val}" || "${val}" == "0" || ! "${val}" =~ ^[0-9]+$ ]]; then
      export "${var}=1"
      echo "[setup] ${var}=1"
    fi
  done
}
fix_openmp_env

if ! command -v python3 >/dev/null 2>&1; then
  echo "[setup] python3 not found." >&2
  exit 1
fi

python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

python3 - <<'PY'
import torch
print(f"[setup] torch={torch.__version__}, cuda={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[setup] gpu={torch.cuda.get_device_name(0)}")
PY

echo "[setup] Done."
echo "[setup] GeoRel-CLIP:"
echo "  cd /root && python autodl-tmp/main/train.py --config_path autodl-tmp/config/01_post-pre-training/GeoRel-CLIP.yaml"
echo "[setup] CLIP-Refine baseline:"
echo "  cd /root && python autodl-tmp/main/train.py --config_path autodl-tmp/config/01_post-pre-training/clip-refine.yaml"

#!/usr/bin/env bash
#SBATCH --job-name=spanbert_ft
#SBATCH --output=logs/spanbert_%j.out
#SBATCH --error=logs/spanbert_%j.err
#SBATCH --partition=students
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --mail-user=ojung@cl.uni-heidelberg.de
#SBATCH --mail-type=END,FAIL

set -euo pipefail

DATASET="${1:-}"
shift $(( $# > 0 ? 1 : 0 ))
EXTRA_ARGS=("$@")
if [[ -z "${DATASET}" ]]; then
  echo "Usage: sbatch job_spanbert_dataset.sh <ontonotes|figer|ultrafine> [extra training args]"
  exit 1
fi

cd /home/students/ojung/NERC

echo "Working dir: $(pwd)"
echo "Host: $(hostname)"
PYTHON_BIN="./.venv/bin/python"
echo "Python: ${PYTHON_BIN}"
"${PYTHON_BIN}" --version
echo "Dataset: ${DATASET}"
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  echo "Extra args: ${EXTRA_ARGS[*]}"
fi

mkdir -p logs "results/spanbert_finetuned/${DATASET}"

PREPROCESSED_DIR="datasets/spanbert_data/${DATASET}"
PREPROCESSED_TRAIN="${PREPROCESSED_DIR}/train.jsonl"

echo "=== Environment Preflight ==="
"${PYTHON_BIN}" - <<'PY'
import sys


def fail(msg):
    print(f"[PREFLIGHT][ERROR] {msg}", file=sys.stderr)
    raise SystemExit(1)


try:
    import accelerate
except Exception as e:
    fail(f"accelerate import fehlgeschlagen: {e}")

try:
    import transformers
except Exception as e:
    fail(f"transformers import fehlgeschlagen: {e}")

try:
    import torch
except Exception as e:
    fail(f"torch import fehlgeschlagen: {e}")

print(f"[PREFLIGHT] accelerate={accelerate.__version__}")
print(f"[PREFLIGHT] transformers={transformers.__version__}")
print(f"[PREFLIGHT] torch={torch.__version__}")

cuda_available = torch.cuda.is_available()
print(f"[PREFLIGHT] torch.cuda.is_available()={cuda_available}")

if not cuda_available:
    print("[PREFLIGHT][WARN] CUDA ist nicht verfuegbar. Training wird vermutlich auf CPU laufen oder spaeter fehlschlagen.")
else:
    print(f"[PREFLIGHT] cuda_device_count={torch.cuda.device_count()}")
    print(f"[PREFLIGHT] cuda_device_name={torch.cuda.get_device_name(0)}")

major, minor = (0, 0)
version_parts = accelerate.__version__.split(".")
if len(version_parts) >= 2:
    try:
        major, minor = int(version_parts[0]), int(version_parts[1])
    except ValueError:
        pass

if (major, minor) < (1, 1):
    fail(f"accelerate>=1.1.0 erforderlich, gefunden: {accelerate.__version__}")
PY

if [[ -f "${PREPROCESSED_TRAIN}" ]]; then
  echo "=== Reusing existing preprocessing for ${DATASET}: ${PREPROCESSED_TRAIN} ==="
else
  echo "=== Starting Preprocessing for ${DATASET} ==="
  "${PYTHON_BIN}" -u -m spanBERT.preprocessing_spanBERT --datasets "${DATASET}"
fi

echo "=== Starting Finetuning for ${DATASET} ==="
"${PYTHON_BIN}" -u -m spanBERT.spanbert_fine_tuning \
  --dataset "${DATASET}" \
  --output-dir "results/spanbert_finetuned/${DATASET}" \
  "${EXTRA_ARGS[@]}"

echo "Job finished"

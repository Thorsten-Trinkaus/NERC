#!/usr/bin/env bash
#SBATCH --job-name=spanbert_probe
#SBATCH --output=logs/spanbert_probe_%j.out
#SBATCH --error=logs/spanbert_probe_%j.err
#SBATCH --partition=students
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=1-00:00:00
#SBATCH --mail-user=ojung@cl.uni-heidelberg.de
#SBATCH --mail-type=END,FAIL

set -euo pipefail

DATASET="${1:-}"
SPLIT="${2:-validation}"
shift $(( $# > 0 ? 1 : 0 ))
shift $(( $# > 0 ? 1 : 0 ))
EXTRA_ARGS=("$@")
if [[ -z "${DATASET}" ]]; then
  echo "Usage: sbatch job_spanbert_probe.sh <ontonotes|figer|ultrafine> [validation|test]"
  exit 1
fi

if [[ "${SPLIT}" != "validation" && "${SPLIT}" != "test" ]]; then
  echo "[ERROR] Split muss validation oder test sein, war aber: ${SPLIT}"
  exit 1
fi

cd /home/students/ojung/NERC

PYTHON_BIN="./.venv/bin/python"
MODEL_DIR="results/spanbert_finetuned/${DATASET}/final_model"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/hf_datasets_cache}"
mkdir -p "${HF_DATASETS_CACHE}"

echo "Working dir: $(pwd)"
echo "Host: $(hostname)"
echo "Python: ${PYTHON_BIN}"
"${PYTHON_BIN}" --version
echo "Dataset: ${DATASET}"
echo "Split: ${SPLIT}"
echo "Model dir: ${MODEL_DIR}"
echo "HF_DATASETS_CACHE: ${HF_DATASETS_CACHE}"
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  echo "Extra args: ${EXTRA_ARGS[*]}"
fi

echo "=== Environment Preflight ==="
"${PYTHON_BIN}" - <<'PY'
import sys


def fail(msg):
    print(f"[PREFLIGHT][ERROR] {msg}", file=sys.stderr)
    raise SystemExit(1)


try:
    import torch
    import transformers
except Exception as e:
    fail(f"Import fehlgeschlagen: {e}")

print(f"[PREFLIGHT] transformers={transformers.__version__}")
print(f"[PREFLIGHT] torch={torch.__version__}")
print(f"[PREFLIGHT] torch.cuda.is_available()={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[PREFLIGHT] cuda_device_name={torch.cuda.get_device_name(0)}")
PY

if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "[ERROR] SpanBERT final_model fehlt unter ${MODEL_DIR}"
  exit 1
fi

echo "=== Starting SpanBERT Probing for ${DATASET} (${SPLIT}) ==="
"${PYTHON_BIN}" -u -m spanBERT.spanbert_probing --dataset "${DATASET}" --model-dir "${MODEL_DIR}" --split "${SPLIT}" "${EXTRA_ARGS[@]}"

echo "Job finished"

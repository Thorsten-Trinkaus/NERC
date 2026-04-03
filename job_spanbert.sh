#!/usr/bin/env bash
#SBATCH --job-name=spanbert_ft
#SBATCH --output=logs/spanbert_%j.out
#SBATCH --error=logs/spanbert_%j.err
#SBATCH --partition=students
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --mail-user=ojung@cl.uni-heidelberg.de
#SBATCH --mail-type=END,FAIL

set -euo pipefail

cd /home/students/ojung/NERC

# DO NOT uv sync here - keep existing environment

echo "Working dir: $(pwd)"
echo "Host: $(hostname)"
echo "Python3: $(which python3)"
python3 --version

mkdir -p logs results/spanbert_finetuned

# Preprocessing
echo "=== Starting Preprocessing ==="
uv run -- python -u -m spanBERT.preprocessing_spanBERT

# Finetuning
echo "=== Starting Finetuning ==="
uv run -- python -u -m spanBERT.spanbert_fine_tuning

# Copy results
echo "=== Copying results ==="
cp -v -r results/* results/spanbert_finetuned/ || true

echo "Job finished"

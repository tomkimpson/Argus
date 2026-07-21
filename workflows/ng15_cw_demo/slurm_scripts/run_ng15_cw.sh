#!/bin/bash
#SBATCH --job-name=ng15_cw_demo
#SBATCH --output=outputs/logfiles/ng15_cw_demo_%j.txt
#SBATCH --export=ALL
#SBATCH --partition=milan-c
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# NG15 CW injection-recovery demo (Earth-term, fixed noise).
# Run from workflows/ng15_cw_demo/. Assumes demo_data/ already prepared by:
#   python prepare_demo.py --discovery-dir /path/to/discovery/data --out-dir ./demo_data
# (the narrowband NG15 TOAs are heavy -- use the A100, not a login node).

# Run from the directory sbatch was invoked in (the workflow root). SLURM copies
# the script to node-local storage, so $0 is unreliable; use SLURM_SUBMIT_DIR.
cd "${SLURM_SUBMIT_DIR:?submit from workflows/ng15_cw_demo/}" || exit 1
mkdir -p outputs/logfiles

source ~/.bashrc
conda activate Argus

nvidia-smi -L
# Config can be passed as the first arg; defaults to the wideband demo.
CONFIG="${1:-configs/ng15_cw_injection_wb.ini}"
echo "Using config: $CONFIG"
time python -u run_analysis.py "$CONFIG"

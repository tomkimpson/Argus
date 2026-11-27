#!/bin/bash
# NS scaling study — Stage 1b (revised): Kalman likelihood cost vs N, on 1x A100.
# Direct microbenchmark of the likelihood NS evaluates, at sane params (no NS, no pathology).
# Submit:  sbatch workflows/ng15_sgwb_demo/slurm_scripts/ns_microbench_run.sh
#SBATCH --job-name=ns_ubench
#SBATCH --account=oz022
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --export=ALL
#SBATCH --chdir=/home/tkimpson/.treehouse/Argus-891104/1/Argus/workflows/ng15_sgwb_demo
#SBATCH --output=/home/tkimpson/.treehouse/Argus-891104/1/Argus/workflows/ng15_sgwb_demo/outputs/logfiles/ns_ubench_%j.out

ROOT=/home/tkimpson/.treehouse/Argus-891104/1/Argus/workflows/ng15_sgwb_demo
WORKTREE_PY=/home/tkimpson/.treehouse/Argus-891104/1/Argus/python

source ~/.bashrc
conda activate /fred/oz022/tkimpson/conda_envs/Argus
export PYTHONPATH="${WORKTREE_PY}:${PYTHONPATH}"   # editable-install fix (see blackjax_ns_run.sh)

echo "=== env check ==="; which python; python -c "import jax; print('jax', jax.__version__)"; nvidia-smi -L
echo "=== microbench N=2,4,8,16,32 ==="
python -u "${ROOT}/scripts/ns_likelihood_microbench.py" --N 2 4 8 16 32 --batch 25 --repeats 30 \
  --out "${ROOT}/outputs/scaling/likelihood_cost_vs_N.csv"
echo "=== done ==="

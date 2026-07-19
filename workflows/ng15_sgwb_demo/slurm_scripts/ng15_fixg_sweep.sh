#!/bin/bash
# NG15 SGWB fixed-gamma_a validation SWEEP on 1x A100 (T2.4 follow-up).
#
# Runs the power-law->OU recovery with log10_gamma_a FIXED at each of {-8.5, -8.0, -7.8}
# (three runs, sequentially, in one job/allocation). Purpose: show that fixing the OU
# corner (a) removes the pathological posterior (free-gamma_a gate had r_hat 1.05, ESS 51,
# 50 divergences) and (b) leaves the recovered band-referenced amplitude INVARIANT across
# the three choices -- validating gamma_a-fixing as the Stage-3 approach.
#
# Base config: configs/ng15_config_fixg.ini (log10_gamma_a_fixed=true). This script sed's the
# fixed value + a per-value output_id. Outputs -> outputs/ng15_fixg_<tag>_powerlaw/.
# Analyse afterwards on CPU:
#   for T in m85 m80 m78; do
#     JAX_PLATFORMS=cpu python scripts/compare_ou_recovery.py --mode powerlaw \
#       --run-prefix ng15_fixg_${T} --gamma-a-prior -10.5 -6.0
#   done
#
# Submit:  sbatch workflows/ng15_sgwb_demo/slurm_scripts/ng15_fixg_sweep.sh

#SBATCH --job-name=ng15_fixg
#SBATCH --account=oz022
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --export=ALL
#SBATCH --chdir=/home/tkimpson/.treehouse/Argus-891104/1/Argus/workflows/ng15_sgwb_demo
#SBATCH --output=/home/tkimpson/.treehouse/Argus-891104/1/Argus/workflows/ng15_sgwb_demo/outputs/logfiles/ng15_fixg_%j.out

# NB: no `set -e` -- sourcing ~/.bashrc / conda init returns non-zero in a non-interactive shell.

ROOT=/home/tkimpson/.treehouse/Argus-891104/1/Argus/workflows/ng15_sgwb_demo
WORKTREE_PY=/home/tkimpson/.treehouse/Argus-891104/1/Argus/python

mkdir -p "${ROOT}/outputs/derived_configs" "${ROOT}/outputs/logfiles"

source ~/.bashrc
conda activate /fred/oz022/tkimpson/conda_envs/Argus
# argus is pip-installed editable against main (no q11 fix); prepend the worktree python/.
export PYTHONPATH="${WORKTREE_PY}:${PYTHONPATH}"

echo "=== env check ==="
which python
python -c "import argus.model as m; print('argus.model from:', m.__file__)"
python -c "import inspect, argus.model as m; q11line=[l for l in inspect.getsource(m.get_Q_block).splitlines() if l.strip().startswith('q11')][0]; print('q11 fixed (uses gamma**2):', q11line.rstrip().endswith('γ**2'))"
nvidia-smi -L

for G in -8.5 -8.0 -7.8; do
  TAG="m$(echo "${G}" | sed 's/-//; s/\.//')"   # -8.5 -> m85, -8.0 -> m80, -7.8 -> m78
  OUTID="ng15_fixg_${TAG}_powerlaw"
  RUN="${ROOT}/outputs/derived_configs/run_fixg_${TAG}.ini"

  sed -e "s|^log10_gamma_a_value = .*|log10_gamma_a_value = ${G}|" \
      -e "s|^output_id = .*|output_id = ${OUTID}|" \
      "${ROOT}/configs/ng15_config_fixg.ini" > "${RUN}"

  echo "======================================================================"
  echo "=== FIXED log10_gamma_a = ${G}  (output_id=${OUTID}) ==="
  grep -E "^log10_gamma_a_fixed|^log10_gamma_a_value|^output_id|^data_path" "${RUN}"
  echo "======================================================================"
  time python -u "${ROOT}/run_analysis.py" "${RUN}"
done

echo "=== fixed-gamma_a sweep complete ==="

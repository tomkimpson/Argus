#!/bin/bash
# NS cost-scaling study — GPU runner for stages 1b / 2 / 3 (1x A100 per grid point).
#
# Two modes:
#   1. SUBMIT (run on the login node):
#          bash ns_scaling_run.sh submit 1b            # default walltime for the stage
#          bash ns_scaling_run.sh submit 2  48:00:00   # override walltime (HH:MM:SS)
#      Generates the stage's derived configs (gen_scaling_configs.py), writes a config-list
#      file, and submits a SLURM job ARRAY with one task per config (each task = one A100
#      job, so each grid point is independently time-boxed and queued).
#   2. WORKER (invoked by SLURM inside the array): reads its config from ${CONFIG_LIST} at
#      line ${SLURM_ARRAY_TASK_ID} and runs it via run_analysis.py. Not called by hand.
#
# Mirrors blackjax_ns_run.sh: same env, same critical PYTHONPATH editable-install fix,
# A100/oz022 routing. Runtime + n_steps land in each run's {output_id}_evidence.json
# (Stage-0 instrumentation); the shell `time` (total job wall) goes to the slurm .out.

#SBATCH --job-name=ns_scal
#SBATCH --account=oz022
# A100 GPU jobs on oz022 route to milan-gpu regardless of requested partition.
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --export=ALL
#SBATCH --chdir=/home/tkimpson/.treehouse/Argus-891104/1/Argus/workflows/ng15_sgwb_demo
#SBATCH --output=/home/tkimpson/.treehouse/Argus-891104/1/Argus/workflows/ng15_sgwb_demo/outputs/logfiles/ns_scal_%A_%a.out

ROOT=/home/tkimpson/.treehouse/Argus-891104/1/Argus/workflows/ng15_sgwb_demo
WORKTREE_PY=/home/tkimpson/.treehouse/Argus-891104/1/Argus/python
ENVPY=/fred/oz022/tkimpson/conda_envs/Argus

# ---------------------------------------------------------------- SUBMIT mode
if [ "$1" = "submit" ]; then
  STAGE="$2"
  if [ -z "${STAGE}" ]; then echo "usage: $0 submit <1b|2|3> [walltime]"; exit 1; fi
  # Sensible per-stage default walltimes (D=2 is fast; high-D coupled runs are long).
  case "${STAGE}" in
    1b) DEF_TIME="04:00:00" ;;
    2)  DEF_TIME="48:00:00" ;;   # escalate if a high-D point is close to the wall
    3)  DEF_TIME="24:00:00" ;;
    *)  echo "unknown stage ${STAGE}"; exit 1 ;;
  esac
  WALLTIME="${3:-${DEF_TIME}}"

  mkdir -p "${ROOT}/outputs/derived_configs" "${ROOT}/outputs/logfiles"
  LIST="${ROOT}/outputs/derived_configs/list_stage${STAGE}.txt"

  # Generate configs (pure python); stdout = config paths (one per line), stderr = summary.
  "${ENVPY}/bin/python" "${ROOT}/scripts/gen_scaling_configs.py" --stage "${STAGE}" > "${LIST}"
  N=$(wc -l < "${LIST}")
  if [ "${N}" -eq 0 ]; then echo "no configs generated"; exit 1; fi
  echo "stage ${STAGE}: ${N} configs -> ${LIST}; submitting array 1-${N}, time=${WALLTIME}"
  # Array is 1-indexed to match sed line numbers below.
  sbatch --array=1-"${N}" --time="${WALLTIME}" \
         --export=ALL,CONFIG_LIST="${LIST}" \
         "${ROOT}/slurm_scripts/ns_scaling_run.sh"
  exit $?
fi

# ---------------------------------------------------------------- WORKER mode (under SLURM)
if [ -z "${SLURM_ARRAY_TASK_ID}" ]; then
  echo "not in a SLURM array and no 'submit' arg; see header for usage."; exit 1
fi

source ~/.bashrc
conda activate "${ENVPY}"

# CRITICAL (see blackjax_ns_run.sh): argus is pip-installed EDITABLE at the main checkout,
# which lacks this worktree's engine/instrumentation. Prepend the worktree python/ so
# `import argus` resolves to the patched code.
export PYTHONPATH="${WORKTREE_PY}:${PYTHONPATH}"

CONFIG=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${CONFIG_LIST}")
echo "=== env check ==="
which python
python -c "import jax; print('jax', jax.__version__)"
python -c "import argus; print('argus', argus.__file__)"
nvidia-smi -L
echo "=== running NS scaling config: ${CONFIG} ==="
grep -E "^excluded_psrs|^num_live_points|^num_delete|^num_inner_steps|^output_id|^spin_injections_path|^noise_params_path" "${CONFIG}"
time python -u "${ROOT}/run_analysis.py" "${CONFIG}"
echo "=== done: ${CONFIG} ==="

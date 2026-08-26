#!/bin/bash
# M1 Stage A: MDC2 single-pulsar noise runs, one array task per pulsar (issue #111).
#
# Each task runs run_analysis.py on one pulsar's staged directory
# (data/mdc2_singles/<PSR>/, built by scripts/stage_mdc2.py) with the Stage A
# template config (configs/mdc2_stage_a.ini): GW fixed negligible, flat
# per-pulsar red-noise priors, EFAC/EQUAD fixed from the per-pulsar MDC2 truth
# slice. A 2-D NUTS problem per pulsar (~25 min per chain on an A100).
#
# Prerequisites (login node, Argus conda env):
#   1. python scripts/ingest_par_tim.py \
#          workflows/data/IPTA_MockDataChallenge2/dataset_2b \
#          workflows/ng15_sgwb_demo/data/mdc2_all
#   2. python workflows/ng15_sgwb_demo/scripts/stage_mdc2.py
#
# Submit:  sbatch workflows/ng15_sgwb_demo/slurm_scripts/mdc2_stage_a.sh
# After all tasks finish:  python scripts/extract_stage_a.py  (see its --help)

#SBATCH --job-name=mdc2_stage_a
#SBATCH --account=oz022
# The OzSTAR job_submit plugin canonicalizes any GPU request to milan-gpu; if
# milan-gpu is down, GPU jobs queue until it returns (milan-c has no GPU route).
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:1
# ~25 min per chain at ~2 it/s (job 14572769) x 2 sequential chains + ~10 min
# env/JIT startup; 30:00 timed out all 33 tasks at 4 chains.
#SBATCH --time=1:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --export=ALL
#SBATCH --array=0-32
#SBATCH --chdir=/home/tkimpson/.treehouse/Argus-891104/1/Argus/workflows/ng15_sgwb_demo
#SBATCH --output=/home/tkimpson/.treehouse/Argus-891104/1/Argus/workflows/ng15_sgwb_demo/outputs/logfiles/mdc2_stage_a_%A_%a.out

# NB: no `set -e` -- sourcing ~/.bashrc / conda init returns non-zero in a
# non-interactive shell, which under `set -e` aborts the job before any output.

ROOT=/home/tkimpson/.treehouse/Argus-891104/1/Argus/workflows/ng15_sgwb_demo
WORKTREE_PY=/home/tkimpson/.treehouse/Argus-891104/1/Argus/python

# Sorted directory enumeration = the canonical pulsar ordering printed by
# stage_mdc2.py and used by every downstream M1 artifact.
mapfile -t PSRS < <(ls -d "${ROOT}"/data/mdc2_singles/*/ | xargs -n1 basename | sort)
if [ "${#PSRS[@]}" -eq 0 ]; then
    echo "ERROR: no staged pulsar directories under ${ROOT}/data/mdc2_singles/" >&2
    echo "Run scripts/stage_mdc2.py first (see header)." >&2
    exit 1
fi
if [ "${SLURM_ARRAY_TASK_ID}" -ge "${#PSRS[@]}" ]; then
    echo "ERROR: array task ${SLURM_ARRAY_TASK_ID} >= ${#PSRS[@]} staged pulsars" >&2
    exit 1
fi
PSR="${PSRS[$SLURM_ARRAY_TASK_ID]}"

PSR_DIR="${ROOT}/data/mdc2_singles/${PSR}/"
PSR_NOISE="${ROOT}/data/mdc2_singles/${PSR}/psr_noise.json"
# The derived config must live OUTSIDE the per-run output dir (run_inference
# copies the config into it — shutil.SameFileError otherwise) but inside the
# workflow tree so io_manager resolves workflow_name. Paths injected here must
# be ABSOLUTE: relative paths in the derived config would resolve against
# outputs/derived_configs/, not configs/.
RUN="${ROOT}/outputs/derived_configs/stage_a_${PSR}.ini"

mkdir -p "${ROOT}/outputs/derived_configs" "${ROOT}/outputs/logfiles"

sed -e "s|^data_path = .*|data_path = ${PSR_DIR}|" \
    -e "s|^noise_params_path = .*|noise_params_path = ${PSR_NOISE}|" \
    -e "s|^output_id = .*|output_id = mdc2_stageA_${PSR}|" \
    "${ROOT}/configs/mdc2_stage_a.ini" > "${RUN}"

echo "=== Stage A task ${SLURM_ARRAY_TASK_ID}: ${PSR} ==="
echo "derived config: ${RUN}"
grep -E "^data_path|^noise_params_path|^output_id" "${RUN}"

source ~/.bashrc
conda activate /fred/oz022/tkimpson/conda_envs/Argus

# CRITICAL: argus is pip-installed EDITABLE pointing at /fred/oz022/tkimpson/Argus
# (the main checkout). run_analysis.py only sys.path.append()s the repo python/
# dir, which loses to the editable install. Prepend THIS worktree's python/ via
# PYTHONPATH so `import argus` resolves to the branch code (empirical/flat prior
# modes live only here until merged).
export PYTHONPATH="${WORKTREE_PY}:${PYTHONPATH}"

echo "=== env check ==="
which python
python -c "import jax, flax; print('jax', jax.__version__, 'flax', flax.__version__)"
python -c "import argus.prior_models as pm; print('argus.prior_models from:', pm.__file__); print('flat/empirical modes available:', hasattr(pm, 'get_empirical_noise_priors'))"
nvidia-smi -L

echo "=== running Stage A single-pulsar noise run: ${PSR} ==="
time python -u "${ROOT}/run_analysis.py" "${RUN}"

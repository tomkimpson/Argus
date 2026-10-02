#!/bin/bash
# OU-adequacy step 3, data prep (CPU): build single-pulsar NG15 wideband inputs for the
# pulsars selected by scripts/ng15_red_noise_budget.py (outputs/ng15_ou_adequacy/budget.json).
#
# Per pulsar, the same treatment the full-array (M3) run would use:
#   1. stage_symlinks.py      par/tim symlinks         -> data/ng15_single_staging/
#   2. ingest_par_tim.py      PINT ingest (ragged)     -> data/ng15_single_raw/<PSR>.feather
#   3. build_aligned_feathers 30-day inverse-variance binning, DMX_* dropped, one pulsar
#                             per call (its own epochs; the 50-epoch JOINT-alignment floor
#                             is lowered to 20) -> data/ng15_single_binned/<PSR>/
#   4. stage_mdc2.py          one input dir per pulsar + single-entry psr_noise.json from
#                             data/ng15_psr_noise_full.json -> data/ng15_singles/<PSR>/
# All outputs are gitignored data.
#
# Submit:  sbatch workflows/ng15_sgwb_demo/slurm_scripts/ng15_single_prep.sh

#SBATCH --job-name=ng15_single_prep
#SBATCH --account=oz022
#SBATCH --partition=milan
#SBATCH --time=4:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --export=ALL
#SBATCH --chdir=/fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo
#SBATCH --output=/fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo/outputs/logfiles/ng15_single_prep_%j.out

ROOT=/fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo
REPO=/fred/oz022/tkimpson/Argus
DATA="${ROOT}/data"

source ~/.bashrc
conda activate /fred/oz022/tkimpson/conda_envs/Argus
export PYTHONPATH="${REPO}/python:${PYTHONPATH:-}"
export JAX_PLATFORMS=cpu
# Strict mode only after the env is set up: ~/.bashrc and conda activate trip set -u.
set -euo pipefail

mapfile -t PSRS < <(python -c "
import json
rows = json.load(open('${ROOT}/outputs/ng15_ou_adequacy/budget.json'))
print('\n'.join(r['psr'] for r in rows if r['selected']))")
echo "=== ${#PSRS[@]} selected pulsars: ${PSRS[*]} ==="

# Re-runs skip the (slow) PINT ingest when every raw feather already exists.
if [ "$(ls "${DATA}"/ng15_single_raw/*.feather 2>/dev/null | wc -l)" -ne "${#PSRS[@]}" ]; then
    python "${ROOT}/scripts/stage_symlinks.py" --subset "${PSRS[@]}" \
        --output-dir "${DATA}/ng15_single_staging" --overwrite
    python "${REPO}/scripts/ingest_par_tim.py" "${DATA}/ng15_single_staging" \
        "${DATA}/ng15_single_raw" --timing-package pint --overwrite
fi

mkdir -p "${DATA}/ng15_single_binned_all"
for PSR in "${PSRS[@]}"; do
    echo "=== binning ${PSR} ==="
    ONE="${DATA}/ng15_single_raw_one/${PSR}"
    mkdir -p "${ONE}"
    ln -sf "${DATA}/ng15_single_raw/${PSR}.feather" "${ONE}/${PSR}.feather"
    python "${ROOT}/scripts/build_aligned_feathers.py" --data-dir "${ONE}" \
        --out-dir "${DATA}/ng15_single_binned/${PSR}" --grid intersection --min-epochs 20 --overwrite
    ln -sf "${DATA}/ng15_single_binned/${PSR}/${PSR}.feather" \
        "${DATA}/ng15_single_binned_all/${PSR}.feather"
done

python "${ROOT}/scripts/stage_mdc2.py" --feather-dir "${DATA}/ng15_single_binned_all" \
    --output-dir "${DATA}/ng15_singles" --noise-json "${DATA}/ng15_psr_noise_full.json" \
    --overwrite
echo "=== prep done ==="

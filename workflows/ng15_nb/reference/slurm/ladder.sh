#!/bin/bash
# Discovery reference ladders on NG15 narrowband (spec tasks 1.4, 1.5). One run per array index,
# one A100 each (numpyro runs the 4 chains vectorised on a single device):
#
#    0-4   pl    5-rung ladder, eps = 0, .25, .5, .75, 1   (NANOGrav hdγ/curnγ; C3 lnB)
#    5     pl13  eps = 1, common γ fixed at 13/3           (C3 amplitude A_HD,13/3)
#    6-10  ou14  5-rung ladder, OU kernel, 14 common modes (K2)
#   11-15  ou30  5-rung ladder, OU kernel, 30 common modes (mode-count sensitivity, C5 input)
#
# Code runs from a pinned worktree so edits to the main checkout cannot reach a queued job:
#   git -C /fred/oz022/tkimpson/Argus worktree add --detach /fred/oz022/tkimpson/Argus-ng15nb <commit>
#   mkdir -p workflows/ng15_nb/outputs/logfiles   # SLURM does not create it
#   sbatch --export=ALL,REPO=/fred/oz022/tkimpson/Argus-ng15nb workflows/ng15_nb/reference/slurm/ladder.sh
# Outputs land in the main checkout's workflows/ng15_nb/outputs/<output_id>_eps<NNN>/ (gitignored):
# samples.nc, summary.json, resolved_config.ini, provenance.json. Readout (Argus env, CPU):
#   python workflows/ng15_sgwb_demo/scripts/lnb_path_sampling.py \
#       --from-runs workflows/ng15_nb/outputs/ng15nb_pl_eps*/samples.nc --out .../lnb_pl.json

#SBATCH --job-name=ng15nb-ref
#SBATCH --account=oz022
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-15
#SBATCH --export=ALL
#SBATCH --chdir=/fred/oz022/tkimpson/Argus
#SBATCH --output=/fred/oz022/tkimpson/Argus/workflows/ng15_nb/outputs/logfiles/ladder_%A_%a.out

set -euo pipefail
REPO="${REPO:?set REPO to the pinned worktree}"
OUT_ROOT=/fred/oz022/tkimpson/Argus/workflows/ng15_nb/outputs
FROZEN="${REPO}/workflows/ng15_sgwb_demo/configs/frozen/evidence_procedure_v1.json"
REF="${REPO}/workflows/ng15_nb/reference"

EPS=(0.0 0.25 0.5 0.75 1.0)
i="${SLURM_ARRAY_TASK_ID:?}"
if (( i <= 4 )); then
    CONFIG=pl; ID=ng15nb_pl; EXTRA=(); e="${EPS[$i]}"
elif (( i == 5 )); then
    CONFIG=pl; ID=ng15nb_pl13; EXTRA=(--gw-gamma 4.333333333333333); e=1.0
elif (( i <= 10 )); then
    CONFIG=ou; ID=ng15nb_ou14; EXTRA=(--gw-components 14); e="${EPS[$((i - 6))]}"
else
    CONFIG=ou; ID=ng15nb_ou30; EXTRA=(--gw-components 30); e="${EPS[$((i - 11))]}"
fi

source /fred/oz022/tkimpson/miniconda3/etc/profile.d/conda.sh
conda activate discovery
export XLA_PYTHON_CLIENT_PREALLOCATE=false
echo "repo ${REPO} @ $(git -C "${REPO}" rev-parse HEAD); ${ID} eps=${e}; $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python "${REF}/run_rung.py" --config "${REF}/configs/${CONFIG}.ini" --eps "${e}" \
    --output-id "${ID}" --out-root "${OUT_ROOT}" "${EXTRA[@]}"

TAG=$(python -c "print(f'{round(${e} * 100):03d}')")
RUN="${OUT_ROOT}/${ID}_eps${TAG}"
python "${REPO}/workflows/ng15_sgwb_demo/scripts/stamp_provenance.py" \
    --output-dir "${RUN}" --config "${RUN}/resolved_config.ini" --frozen "${FROZEN}" --repo "${REPO}"

#!/bin/bash
# Readouts for the Discovery reference ladders (spec tasks 1.4-1.6), on a CPU node in the Argus
# env (the frozen estimator lives there). Reads the gitignored run directories written by
# ladder.sh and writes JSON to workflows/ng15_nb/outputs/readouts/:
#   lnb_hd_curn_{pl,ou14,ou30}.json     path-sampling lnB(HD/CURN), frozen procedure v1
#   lnb_curn_noise_{pl,ou14,ou30}.json  region Savage-Dickey lnB(CURN/noise) from eps = 0
#   n2_psd.json                         N2: common-process PSD at 1/T, 1/(5 yr); A_HD,13/3
#
#   sbatch workflows/ng15_nb/reference/slurm/readout.sh

#SBATCH --job-name=ng15nb-readout
#SBATCH --account=oz022
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --chdir=/fred/oz022/tkimpson/Argus
#SBATCH --output=/fred/oz022/tkimpson/Argus/workflows/ng15_nb/outputs/logfiles/readout_%j.out

set -euo pipefail
source /fred/oz022/tkimpson/miniconda3/etc/profile.d/conda.sh
conda activate Argus
O=workflows/ng15_nb/outputs
mkdir -p "${O}/readouts"
for id in pl ou14 ou30; do
    echo "===== lnB(HD/CURN) ${id}"
    python workflows/ng15_sgwb_demo/scripts/lnb_path_sampling.py \
        --from-runs ${O}/ng15nb_${id}_eps*/samples.nc --out "${O}/readouts/lnb_hd_curn_${id}.json"
    echo "===== lnB(CURN/noise) ${id}"
    python workflows/ng15_nb/reference/lnb_curn_vs_noise.py "${O}/ng15nb_${id}_eps000" \
        --out "${O}/readouts/lnb_curn_noise_${id}.json"
done
echo "===== N2 kernel systematic"
python workflows/ng15_nb/reference/kernel_systematic.py --out "${O}/readouts/n2_psd.json"

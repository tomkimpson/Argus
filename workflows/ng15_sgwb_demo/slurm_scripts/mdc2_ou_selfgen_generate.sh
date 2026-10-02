#!/bin/bash
# SPIKE step 1 (notes/SPIKE_noise_model_misspecification.md): generate a dataset at the 2b
# geometry drawn entirely from Argus's own generative model -- OU per-pulsar red noise at the
# 33 pivot-matched amplitudes, OU GW band-matched to 2b, MDC2 white noise, same epochs.
# The --log10-sigma-p list is in sorted-pulsar-name order (verified 2026-09-25: feather glob
# order == sorted noise-JSON keys, and the list re-derived independently to <5e-5 dex).
#
# Submit:  sbatch workflows/ng15_sgwb_demo/slurm_scripts/mdc2_ou_selfgen_generate.sh

#SBATCH --job-name=ou_selfgen_gen
#SBATCH --account=oz022
#SBATCH --partition=milan
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --export=ALL
#SBATCH --chdir=/fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo
#SBATCH --output=/fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo/outputs/logfiles/ou_selfgen_gen_%j.out

source ~/.bashrc
conda activate /fred/oz022/tkimpson/conda_envs/Argus
export PYTHONPATH="/fred/oz022/tkimpson/Argus/python:${PYTHONPATH}"
git -C /fred/oz022/tkimpson/Argus rev-parse --short HEAD

JAX_PLATFORMS=cpu python -u scripts/inject_powerlaw_gwb.py --mode ou \
  --aligned-dir data/mdc2_all \
  --noise-json ../data/IPTA_MockDataChallenge2/group1_psr_noise.json \
  --out-dir data/mdc2_ou_selfgen \
  --log10-ha -12.919767 --log10-gamma-a -9.0 \
  --red-noise --log10-gamma-p -9.0 \
  --log10-sigma-p=-14.8520,-17.2311,-17.5061,-15.3616,-15.1747,-14.3028,-18.0014,-16.8671,-18.2868,-14.5839,-16.8813,-17.2553,-15.6756,-16.8019,-17.0472,-18.1851,-18.0386,-14.9688,-13.8713,-18.1888,-18.0220,-18.3154,-14.9399,-18.2705,-17.0478,-14.0604,-17.8290,-17.2171,-17.3821,-17.8011,-15.7650,-16.7465,-18.0492 \
  --seed 0

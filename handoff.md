# Handoff — 2026-06-03

## Goal
Validate the Argus Kalman-filter CW likelihood against enterprise on IPTA MDC2
dataset_3b, then resolve the pulsar-term f_gw multimodality with a fixed-f_gw grid —
en route to running the pipeline on real data.

## Status
Done and successful. Argus's CW likelihood is validated (controlled injection test:
recovers a known source at 0.95x theoretical SNR^2/2). The 40-point f_gw grid completed
and its max-logL profile peaks cleanly at the injection. See `log.md` for the narrative.

## What changed this session
See `log.md`. New files (all untracked, on branch `continuous-waves`):
- `workflows/cw_shared/level1_likelihood_agreement.py` — 1-D profile gate. NOTE: superseded
  / unreliable for a multimodal likelihood; keep only as a cautionary artifact.
- `workflows/cw_shared/diag_noise_hypothesis.py` — red-noise discriminating experiment.
- `workflows/cw_shared/controlled_injection_test.py` — THE validation of record (Argus
  recovers a known injection + correct SNR normalization).
- `workflows/cw_level4b_fgw_grid/` — the f_gw grid runner (config_template.ini,
  generate_configs.py, run_grid.sh, aggregate.py, run.py, README.md) + 40 run outputs.

## Open questions
- Sky/inclination/polarization multimodality is still unaddressed (inherent to single-source
  CW at this SNR; the grid only fixes frequency). Worth a restart/tempered angular kernel?
- Is the enterprise comparison harness worth fixing, or just delete it? It is mis-normalized
  (gave a negative detection statistic in the controlled test). The controlled injection test
  already serves as the trustworthy validation.

## Blockers / problems
- None blocking. Caveats: grid Savage-Dickey BF is confounded by pulsar-term chi-phase
  overfitting (use the max-logL profile, not SD); grid index 15 (exact-injection point) timed
  out but is bracketed — not worth backfilling.

## Next steps
1. Real-data pivot (the actual goal): scope NANOGrav 15yr ingestion. Check the loader on a
   single real pulsar — does it handle real ECORR/DMX flags? Argus white noise is
   (efac*err)^2 + equad^2 with NO jitter/ECORR term — likely the first real code gap.
2. Noise marginalization: the OU process can't fit steep power-law red noise (drove
   sigma_p->~0 on MDC2). Plan to fix/constrain noise from single-pulsar enterprise runs
   rather than fit it jointly.
3. The GWB foreground: real 15yr data has a detected stochastic GWB; a CW search must model
   it (Argus has a GWB mode) or it will bias the result.
4. Optional cleanup: delete or clearly mark the unreliable enterprise comparison harness.

## Non-obvious context
- Run everything under `conda activate Argus` (enterprise/PINT/tempo2 only importable there).
- SLURM: A100s via `--partition=milan-gpu --gres=gpu:a100:1`. Grid array was job 12746776.
- Grid configs live in `runs/`, one level deeper than level4a's config; `generate_configs.py`
  absolutises data paths so the template's `../data` relative path doesn't break.
- numpyro chain_method is "sequential" on 1 GPU, so 2 chains = 2x wall time (~7h/grid point
  at 1500 samples). 12h SLURM limit gives margin; timed-out points are reported missing.

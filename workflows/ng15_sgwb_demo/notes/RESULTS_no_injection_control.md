# Task 1.9 — the no-injection control on MDC2 1b geometry

**Status:** ladder running (submitted 2026-09-08, job 16319044). **This section was written
before any Bayes factor was read.**

## The question

`sgwb/model-selection` asks three things of the evidence estimator. Two now pass on real MDC2
1b data under flat per-pulsar red-noise priors:

| scenario | result |
|---|---|
| Injected-signal case | `lnB = +3.043 ± 0.015`, `reliable: true` |
| Falsification by sky scramble | `lnB = -0.766 ± 0.010`, `reliable: true` |
| **Null case — no injected correlated signal** | **this ladder** |

The scramble does not close the third. A scramble mis-describes a correlation that is genuinely
present, so it lands below zero by construction, and it did. "Consistent with zero" is a
statement about data with no correlation to describe at all.

## The dataset

MDC2 group1 contains no signal-free member: `group1_gw_parameters.json` holds dataset1 and
dataset2 as GWBs and dataset3 as a continuous-wave source. So the control is synthesised at the
1b geometry by `scripts/inject_powerlaw_gwb.py`, which replaces the residual column outright
while keeping every other feather field — TOAs, errors, timing design matrix, sky positions, F0,
distance — from real 1b:

```
JAX_PLATFORMS=cpu python scripts/inject_powerlaw_gwb.py --mode powerlaw \
    --aligned-dir data/mdc2_d1_all \
    --noise-json ../data/IPTA_MockDataChallenge2/group1_psr_noise.json \
    --out-dir data/mdc2_d1_nogwb --log10-A-gw -30 --gamma 4.333333333333333 --seed 0
```

`log10_A_gw = -30` puts the correlated component at a pivot PSD of 2.8e-37 s³ against 1b's
injected 1.2e-7 — numerically nil, but reached through the same code path that produced the
tracked kernel-systematic injections, and recorded in `injection_truth.json` so the control
documents itself.

**"Matched" means white noise only, and that is not a shortcut.** The IPTA MDC2 dataset table
lists `g1.d1a(b)` as WN and `g1.d2a(b)` as WN,RN, so 1b has no injected per-pulsar red noise;
white noise from `group1_psr_noise.json` with `--red-noise` off reproduces 1b's noise model
exactly. Measured residual RMS is 0.87–1.09× the white-noise prediction
`sqrt((efac·σ)² + equad²)` across the 33 pulsars, consistent with χ² scatter at 183 epochs —
white noise and nothing else.

The ladder config is `configs/mdc2_d1_flat_rung.ini.template` with exactly two values changed
(`data_path`, `output_id`), verified by diff, so it is comparable to the `+3.043` run line for
line: same ridge basis and pivot, same flat Uniform red-noise priors with no
`empirical_priors_path`, same EFAC/EQUAD fixed from truth, same NUTS settings, same 5-rung
uniform ε grid.

## Pre-registered criterion

Recorded before the number was read, and also carried in the headers of
`configs/mdc2_d1_nogwb_rung.ini.template` and `slurm_scripts/mdc2_d1_nogwb_ladder.sh`.

The estimator's ±0.01 is its own numerical precision, not the scale on which "consistent with
zero" is judged: a no-injection control has a true `lnB` that is mildly negative from Occam, not
exactly 0.

| outcome | reading |
|---|---|
| `reliable: true` and `\|lnB\| < 1` | **PASS.** No evidence either way, decisively short of the `lnB ≥ 3` gate. Group 1's validation half is complete. |
| `lnB` near `+3` | **FAIL.** The estimator manufactures HD evidence on noise; the 3.043 does not survive. |
| `lnB` near `-0.77` | Not a failure, but a different finding: the estimator penalises *any* correlation model on noise-only data. To be reported, not filed as a pass. |

## Results

_Pending — the ladder is running. Fill in with the readout from
`outputs/lnb_path_sampling_mdc2_d1_nogwb.json`._

Early signal from the first completed rung (ε = 0.75, 1h25m, 0 divergences, r̂ = 1.00): the GW
pivot log-PSD posterior sits at median **-9.69** (sd 1.02) against the injected **-6.908** on
real 1b and the **-6.47** the flat ladder recovered there. The amplitude has collapsed well below
the injection scale, which is what a signal-free dataset should do. This is one rung and not a
Bayes factor.

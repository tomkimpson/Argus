# Task 1.9 — the no-injection control on MDC2 1b geometry

**Status:** COMPLETE — **PASS**, `lnB = -0.0131 +/- 0.0002`, `reliable: true`. The dataset,
criterion and pass band below were written and committed (d4065eb) **before any Bayes factor was
read**; the Results section was appended after.

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

## Results — PASS

    ln B(HD/CURN) = -0.0131 +/- 0.0002        reliable: true, failed_diagnostics: []

`|lnB| = 0.013` against a pre-registered pass band of `|lnB| < 1`. The estimator returns
essentially exactly zero on data with no correlated signal.

| rung | `<dlnL/deps>` | ESS |
|---|---|---|
| ε = 0.00 | −0.0150 ± 0.0005 | 4000 |
| ε = 0.25 | −0.0138 ± 0.0004 | 4000 |
| ε = 0.50 | −0.0135 ± 0.0005 | 3644 |
| ε = 0.75 | −0.0121 ± 0.0004 | 3904 |
| ε = 1.00 | −0.0117 ± 0.0005 | 3030 |

The integrand is not a set of large values cancelling: it is uniformly tiny, smooth and mildly
negative along the whole path, which is what "no correlation to find, and a small Occam penalty
for looking" should look like. Quadrature is nowhere near its tolerance — Romberg residual
2.2e-5 against a 0.1 ceiling, minimum integrand ESS 3030 against a floor of 50, endpoints
covered.

Sampling was flawless. All five rungs: r̂ = 1.010, min ess_bulk ≥ 6942, **0 divergences**,
~1h25m each, four chains agreeing to 0.1 dex. The GW pivot log-PSD sits at −9.69 flat across
the entire ε path (−9.704 at ε = 0 to −9.715 at ε = 1), i.e. turning the Hellings–Downs
correlation on from nothing to full moves the amplitude by 0.04 dex. Against the
`Normal(−9, 1.333)` prior that is a mild upper limit — posterior sd 1.04, 0.78 of prior sd,
centred slightly below the prior mean. Not a measurement, and not the prior handed back either.

## The three scenarios, together

`sgwb/model-selection` is now satisfied in full on MDC2 1b, and the three cases separate by
orders of magnitude in the integrand rather than by a threshold on the final number:

| case | integrand range | ln B |
|---|---|---|
| injected signal, true ORF | +4.695 → +1.558 | **+3.043 ± 0.015** |
| same data, sky-scrambled ORF | −0.386 → −1.130 | **−0.766 ± 0.010** |
| **no injected signal, true ORF** | **−0.015 → −0.012** | **−0.013 ± 0.0002** |

Each answer is the right *kind* of answer, not merely the right sign. Real correlation present
and correctly described → strongly positive. Real correlation present and mis-described →
negative, because a wrong pattern fits worse than none. No correlation to describe → zero. The
control's integrand is ~100× smaller in magnitude than the scramble's and ~300× smaller than the
injection's, so the estimator is not returning a number of fixed scale with a varying sign; it is
measuring how much correlation information the data actually contain.

Taken with the scramble, this closes the question the +3.043 could not answer on its own. Flat
red-noise priors do not manufacture Hellings–Downs evidence: given the same priors, the same
geometry, the same epochs and the same noise model, removing the signal removes the evidence.

## Scope

This validates the **estimator**, on 1b's geometry and noise model. It says nothing about
whether the flat-prior noise model itself survives data with per-pulsar red noise — 1b has none,
and on 2b it does not survive. See `RESULTS_2b_flat_priors.md`.

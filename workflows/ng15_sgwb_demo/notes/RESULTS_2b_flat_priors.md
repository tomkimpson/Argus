# Flat per-pulsar priors fail on MDC2 2b — the dataset that has red noise

**Date:** 2026-09-09 (updated 2026-09-10) · **Status:** COMPLETE — all five rungs run, all five
failed. No Bayes factor: a non-converged rung poisons the path integral. The ladder is being run to completion to show the failure across the ε path, but
**no Bayes factor will come out of it** — a non-converged rung poisons the path integral.

## Why this was run

Everything the project currently rests on was established on MDC2 **1b**: the amplitude
recovery, `lnB = +3.043`, the sky-scramble falsification at `-0.766`. All of it under flat
per-pulsar red-noise priors, which replaced the two-stage empirical priors after those were
measured to absorb the background.

The IPTA MDC2 dataset table settles what 1b is:

    g1.d1a(b)   15 yrs   30 days   Noise: WN       Signals: SB
    g1.d2a(b)   15 yrs   30 days   Noise: WN,RN    Signals: SB

**1b has no injected per-pulsar red noise.** Flat priors were therefore the *true* model there
— the one dataset where they cannot be wrong. 2b is the same 33 pulsars, epochs and white
noise, with per-pulsar red noise injected (`group1_psr_noise.json` carries `rn_log10_A` up to
−12.05 and `rn_spec_ind` for all 33) and a GWB at `log10_A = −14.886`. It is the only member of
group1 that exercises the red-noise truth, and so the only available test of whether the
procedure about to be frozen for real NG15 data generalises. Real NG15 data has red noise.

## The result: it does not sample

`mdc2_flat_eps000` — the CURN endpoint, the simplest model on the path — ran **13 h 28 m**
against 1.5–2.4 h for every 1b rung, and failed every convergence criterion:

| diagnostic | value | threshold |
|---|---|---|
| max r̂ over sampled sites | **1.59** | ≤ 1.01 |
| min ess_bulk over sampled sites | **7** (of 4000 draws) | — |
| divergent transitions | **372** / 4000 (9.3%) | ≤ 1% |
| chain 1 within-chain variance | **exactly 0** on every GW parameter | — |

(Two r̂ conventions appear in the artefacts and they disagree. The job log carries numpyro's
`print_summary`, which reports r̂ up to 27.5; the stored
`numpyro_diagnostics/mcmc_diagnostics.txt` and the numbers above use ArviZ's rank-normalised
split-r̂, which is less inflated by a frozen chain. The repo's own diagnostics file is ArviZ, so
ArviZ is what is quoted here. Both verdicts are the same — this run did not sample.)

Chain 1 froze at a point and stayed there for all 1000 draws. That is the stuck-chain pathology
`sgwb/array-analysis-procedure` names explicitly, and the spec is equally explicit that such a
run is marked failed with the chain identified and is **not** rescued by discarding it.

## Where the live chains went, which is the substantive part

First, a correction to how the pivot prior has been described. In ridge mode the GW pivot
log-PSD is *not* sampled under a uniform box. `parameter_sampling.py:81-85` samples
`log10_pivot_psd_prime ~ Normal(0, 1)` and maps it affinely, with `_reparam`
(`prior_models.py:47-49`) setting `mean = (min+max)/2` and `std = (max-min)/6`. So
`log10_pivot_psd_min/max = -13/-5` declares a **Normal(−9, 1.333)** prior whose ±3σ interval is
[−13, −5]. There is no hard edge to rail against, and values beyond −5 are permitted, merely
improbable a priori.

With that established, the three live chains agree with each other and concentrate at

    log10 pivot PSD = -4.80,  posterior sd 0.35   (prime = +3.15, sd 0.26)

i.e. **+3.15 prior-σ into the upper tail**, and — against the injected 2b pivot PSD of
**−6.3194** (`data/mdc2_inject_powerlaw/injection_truth.json`, same amplitude and pivot) —
**+1.5 dex above truth, confidently**. Per-pulsar `log10_σp` medians span −16.24 to −14.16
(1b flat: −16.23 to −15.31), none near a prior edge.

For contrast, the identical configuration on 1b:

| | 2b (WN+RN) | 1b (WN only) |
|---|---|---|
| ε = 0 pivot log-PSD median | **−4.80** | −6.55 |
| injected pivot log-PSD | −6.32 | −6.91 |
| offset | **+1.5 dex** | +0.36 dex |
| max r̂ (sampled sites) | **1.59** | 1.010 |
| min ess_bulk | **7** | 1251 |
| divergences | **372** | 0 |
| wall time | **13 h 28 m** | 1.5–2.4 h |

## The failure is structural, not an ε = 0 artefact

`mdc2_flat_eps025` (7 h 33 m) reproduces it exactly:

| | ε = 0 | ε = 0.25 |
|---|---|---|
| max r̂ (sampled sites) | 1.590 | 1.580 |
| min ess_bulk | 7 | 7 |
| divergences | 372 (9.3%) | **833 (20.8%)** |
| chain 1 | frozen, sd = 0, at −5.581 | frozen, sd = 0, at −6.307 |
| chains 0/2/3 pivot median | −4.771 / −4.777 / −4.775 | −4.768 / −4.768 / −4.771 |

`mdc2_flat_eps075` (10 h 56 m) then sharpened it: r̂ 1.59, ess_bulk 7, 751 divergences, chains
0/2/3 at −4.74 — and chain 1 **not frozen**, moving freely but confined to a low mode at −6.12,
near the injected −6.3194.

**So the frozen chain was a symptom, not the essence.** What is actually present is at least two
modes that do not mix: one within ~0.2 dex of truth, one ~1.5 dex above it. A chain that freezes
is just the degenerate case of a chain that cannot leave its mode. This is a better-posed problem
than "a chain got stuck", and it is what a misspecified noise model offering two competing
explanations of the same data would look like.

Two further things stand out. The three high chains land at **−4.77 in both of the first rungs**, unchanged by
turning on a quarter of the Hellings–Downs correlation — the amplitude they settle on is being
set by something other than the correlation structure. And the frozen chain sits at a
*different* place in each rung; at ε = 0.25 it sits at **−6.307**, within 0.01 dex of the
injected −6.3194.

That last detail is worth stating carefully rather than made into a story. It is consistent with
a posterior carrying at least two modes — one near truth, one ~1.5 dex high — between which the
sampler cannot move, with one chain initialised near the true mode and then unable to take a
single accepted step. It is not proof of that, and this note does not diagnose it further; the
point is that the failure is a sampling failure, not simply "the GW ate the red noise", and the
distinction matters for choosing a remedy.

## The complete ladder

| rung | r̂ | min ess_bulk | divergences | low-mode chains | high-mode chains | wall |
|---|---|---|---|---|---|---|
| ε = 0 | 1.59 | 7 | 372 | −5.58* | −4.77, −4.78, −4.77 | 13h28m |
| ε = 0.25 | 1.58 | 7 | 833 | −6.31* | −4.77, −4.77, −4.77 | 7h33m |
| ε = 0.5 | 1.59 | 7 | 938 | −6.28 | −4.76, −4.76, −4.76 | 7h17m |
| ε = 0.75 | 1.59 | 7 | 751 | −6.12 | −4.74, −4.75, −4.74 | 10h56m |
| ε = 1.0 | **2.21** | 5 | 924 | **−5.55, −6.10** | −4.73, −4.73 | 7h05m |

`*` = chain with exactly zero within-chain variance. Injected pivot log-PSD = **−6.3194**.
For comparison, every 1b rung: r̂ 1.010, ess ≥ 1251, 0 divergences, 1.5–2.4 h.

Two modes at every rung, never mixing, and the high mode is **rock-steady at −4.73 to −4.78
across the entire ε path** — wholly indifferent to the correlation structure. The low mode
tracks truth.

One suggestion, offered as such and not as a finding: at ε = 1 the split becomes 2–2 rather than
1–3, and r̂ worsens to 2.21 *because* two chains now disagree with two. It may be that the
Hellings–Downs correlation does carry information favouring the true amplitude and the sampler
simply cannot exploit it while it cannot mix. That is one rung and a change of one chain, so it
is a thing to test, not a thing to believe.

## This is the mirror image of the failure that killed the empirical priors

Under the two-stage empirical priors, each pulsar's single-pulsar fit absorbed the common
background, and the array run had nothing left for the GW to claim: `lnB = 0.053`, amplitude
posterior equal to its prior. Under flat priors on 2b the traffic runs the other way — the GW
absorbs the injected per-pulsar red noise and runs away into the upper tail of its prior.

**Neither extreme survives a dataset with real per-pulsar red noise.** The GW↔red-noise
degeneracy that 1b could not exercise, because it had no red noise, is doing the damage.

## The dataset is not the variable

The 2b feathers were, until this session, a symlink into a treehouse worktree outside `/fred`.
`dataset_2b` was re-ingested into a real directory and **all 33 feathers verified byte-identical**
to the symlink target before any A100 time was spent. The empirical-prior 2b ladder
(`outputs/mdc2_ladder_eps*`, `lnB = 0.175 ± 0.017`) converged cleanly on those same bytes. So
the comparison is controlled: same data, same geometry, same sampler settings, same ridge basis
— only the per-pulsar noise priors differ, and only the flat version fails to sample.

## What this does and does not mean

**It does not invalidate the 1b results.** 1b is white-noise-only, flat priors are the true
model there, and the injected-signal case, the sky scramble and the no-injection control all
stand on their own terms. Nothing measured on 1b is retracted by this.

**It does mean the noise-prior choice is not yet fit to be frozen.** The freeze at task 6.8
would carry this choice onto real NG15 data, which has per-pulsar red noise, and the only
evidence that flat priors work is from the one dataset where they could not fail. This is
precisely the situation the scenario added to `sgwb/array-analysis-procedure` this session was
written to prevent:

> #### Scenario: Noise-prior choice validated before it is frozen
> - **WHEN** a per-pulsar noise-prior choice is frozen as part of the production procedure
> - **THEN** it has been exercised on a dataset that contains injected per-pulsar red noise, not
>   only on one where the chosen priors are the true model

That gate now fails, on its first application.

## Open — deliberately not resolved here

The remedy is a real decision and one rung is not the evidence base for it. The obvious
candidates, none costed or tested:

- a weakly-informative per-pulsar prior between the two failed extremes — wide enough not to
  absorb the background, tight enough to stop the GW absorbing the red noise;
- the joint noise+GW reformulation deferred as issue #115, which targets exactly this degeneracy;
- accepting the degeneracy and reporting an amplitude *interval* rather than a point, if the
  sampler can be made to explore it rather than stick.

The remaining rungs will say whether the failure is uniform along the ε path or specific to the
CURN endpoint. Nothing else should be built on 2b until that lands.

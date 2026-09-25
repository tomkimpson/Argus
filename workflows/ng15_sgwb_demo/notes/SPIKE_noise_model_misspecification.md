# SPIKE — is the 2b failure noise-model misspecification, or the GW↔red-noise degeneracy?

**Status:** designed and approved 2026-09-10, **NOT YET RUN**. Everything needed to launch it is
in this file; no derivation is left to do.

## The question

Flat joint priors fail to sample on MDC2 2b — five rungs, five non-convergences, two non-mixing
modes at every one (`RESULTS_2b_flat_priors.md`). The same configuration succeeds on 1b. Two
explanations are live and they call for different remedies:

1. **Misspecification.** MDC2 injects per-pulsar red noise as a **power law**, as every standard
   PTA simulation does. Argus models it as an **OU process**, whose residual PSD bends from f⁻²
   to f⁻⁴ across the band. A power law has one constant slope, so an OU cannot be one over
   MDC2's ~2 decades **at any parameter value**. No prior width fixes that. 1b has no per-pulsar
   red noise at all, which is exactly why it works there.
2. **Degeneracy.** The GW and the per-pulsar red noise are simply covariant at this signal
   strength, flat priors make that explicit, and the posterior is genuinely bimodal regardless of
   what shape the noise has.

Evidence currently favours (1) but does not establish it. Checking 2b's truth refines the
mechanism and rules out the obvious version: only **3/33** pulsars are steeper than f⁻⁴, so "the
OU can't be steep enough" is *not* the story. Only **4/33** have red noise above their own white
floor at 1/5 yr, and all four are **shallow** (γ ≈ 2.3–2.7) but **loud** — J1939+2134 at +2.0 dex
over its white floor, J1643−1224 at +1.7. It is a *shape* mismatch, not a steepness one.

## The probe

Build one dataset at the 2b geometry in which **everything is drawn from Argus's own generative
model** — OU per-pulsar red noise, OU GW, same white noise, same epochs, same 33 pulsars — and
run a single ε = 0 rung with the identical flat-prior configuration that just failed five times.

If Argus cannot sample its own model at this red-noise strength, misspecification is exonerated
and the problem is degeneracy. If it samples cleanly, the shape mismatch is doing the damage.

## Exact recipe

**1. Generate** (CPU, minutes; from `workflows/ng15_sgwb_demo`):

```
JAX_PLATFORMS=cpu python scripts/inject_powerlaw_gwb.py --mode ou \
  --aligned-dir data/mdc2_all \
  --noise-json ../data/IPTA_MockDataChallenge2/group1_psr_noise.json \
  --out-dir data/mdc2_ou_selfgen \
  --log10-ha -12.919767 --log10-gamma-a -9.0 \
  --red-noise --log10-gamma-p -9.0 \
  --log10-sigma-p=-14.8520,-17.2311,-17.5061,-15.3616,-15.1747,-14.3028,-18.0014,-16.8671,-18.2868,-14.5839,-16.8813,-17.2553,-15.6756,-16.8019,-17.0472,-18.1851,-18.0386,-14.9688,-13.8713,-18.1888,-18.0220,-18.3154,-14.9399,-18.2705,-17.0478,-14.0604,-17.8290,-17.2171,-17.3821,-17.8011,-15.7650,-16.7465,-18.0492 \
  --seed 0
```

- **The GW parameters are 2b's, already band-matched.** `log10_ha = -12.919767`,
  `log10_gamma_a = -9.0` are exactly the values `notes/kernel_systematic_injection_pair.md`
  derived for task 5.1: they give `log10 S(1/5 yr) = -6.3194`, matching the power law at 2b's
  injected `log10_A = -14.886`, γ = 13/3 to 3e-7 dex. Do not re-derive them.
- **The 33 `--log10-sigma-p` values are derived in this file** (see below) and are in the
  **sorted-pulsar-name order** the loader globs, which is the order `_broadcast` assigns them in.
  **Verified 2026-09-10:** `sorted(glob('data/mdc2_all/*.feather'))` and
  `sorted(json.load(open(noise_json)))` give identical 33-element orderings, so the list maps
  pulsar-for-pulsar. Re-check if either the feather set or the noise JSON ever changes — a silent
  permutation here would assign every pulsar the wrong red-noise amplitude and invalidate the
  spike without any error.
- `--log10-gamma-p -9.0` puts every corner below the band (f_lo = 2.118e-09 Hz), so the injected
  red noise is f⁻⁴ in-band — the same corner convention 5.1 used for the GW, and chosen there
  because a corner near the band leaves an imprint of its own placement on a measurement whose
  whole point is spectral shape.

**2. Config and driver.** `configs/mdc2_ou_selfgen_rung.ini.template` =
`configs/mdc2_flat_rung.ini.template` with two values changed (`data_path`, `output_id`) — the
same two-value diff used for every ladder so far. Driver = `slurm_scripts/mdc2_flat_ladder.sh`
re-pointed and cut to `--array=0` (ε = 0 only). Keep the `empirical_priors_path` /
`red_noise_prior = flat` / `data_path` guards.

**3. Cost.** One rung, `gpu:4`. A healthy rung is 1.5–2.5 h (≈ 8–10 A100-h); the failing 2b rungs
took 7–13.5 h, so a long wall time is itself part of the signal.

## Pre-registered decision rule

Judge on **sampling diagnostics, not on `lnB`** — one rung yields no Bayes factor, and that is
fine, because the failure being diagnosed is a sampling failure.

| outcome | reading | what follows |
|---|---|---|
| r̂ ≤ 1.01, ess_bulk ≳ 1000, divergences ≤ 1%, no chain confined to a mode | **Misspecification confirmed.** Argus samples its own model at 2b-strength red noise; the power-law shape is what breaks it. | Run the attribution variant below, then open a change for a richer per-pulsar kernel. |
| r̂ ≳ 1.5, ess ~ 10, chains split across modes — the 2b signature | **Misspecification exonerated.** The GW↔red-noise degeneracy breaks it regardless of shape. | The remedy is sampling/parameterisation or a weakly-informative prior, not a new kernel. Issue #115 returns to the critical path. |
| something in between (converges but slowly, or one chain lags) | Partial. Record it and run the attribution variant anyway — a shape effect that is present but not decisive still matters at 68 pulsars. | |

**Attribution follow-up, only if misspecification is confirmed:** a second dataset with
**power-law per-pulsar red noise but an OU GW**, which separates the per-pulsar kernel from the
GW kernel. Note the injector cannot currently produce power-law per-pulsar red noise — `--red-noise`
is OU-only — so this variant needs a small addition to `inject_powerlaw_gwb.py`. That is the only
new code anywhere in this spike, and it is not needed for the first run.

## How the 33 amplitudes were derived

For each pulsar, the OU parameters were chosen so the injected red noise carries **the same
residual PSD at the pivot f = 1/5 yr** as its MDC2 power-law truth `(rn_log10_A, rn_spec_ind)`.

Argus's per-pulsar red noise is OU on `(dphi, df)` with `d(df) = -γ_p df dt + χ_p`,
`<χ_p²> = σ_p²`, entering the residual as `dphi/f0` (`inject_powerlaw_gwb.inject_red_noise`).
Its residual PSD is therefore

    S_OU(f) = σ_p² / ( f0² (2πf)² (γ_p² + (2πf)²) )      [s³]

the same shape as `ou_psd` with `σ_a² ↔ σ_p²/f0²`. Matching to
`S_PL(f) = A²/(12π²) (f/f_yr)^-γ f_yr^-3` at `f_p = 1/(5 yr)` gives

    σ_p = f0 · ω_p · sqrt(γ_p² + ω_p²) · sqrt(S_PL(f_p)),     ω_p = 2π f_p

**Why the pivot rather than the in-band variance.** Both were computed. Pivot matching is the
repo's established convention (5.1, `ou_psd`'s docstring: "a PTA constrains the spectrum over
about a decade, so the comparable observable between an OU and a power-law injection is the PSD
at a pivot frequency, not the spectral index"), and it avoids a trap. With the corner below the
band the OU is f⁻⁴, so its in-band variance integral is dominated by the lowest frequency;
matching *variance* to a γ ≈ 2.5 power law therefore piles nearly all the injected power at
f_lo, where the marginalised timing model absorbs it. The dataset would look loud and constrain
little. The variance-matched values are recorded below in case they are ever wanted, but the
pivot-matched list above is the one to use.

Variance-matched alternative (f_lo = 2.118e-09 to f_hi = 1.926e-07 Hz), same order:

    -14.7663,-17.3330,-17.6145,-15.5434,-15.0406,-14.5272,-18.1062,-16.9025,-18.3918,-14.7743,
    -17.1198,-17.3050,-15.7152,-16.8291,-17.2352,-18.2940,-18.1399,-14.8550,-14.0837,-18.3288,
    -18.1240,-18.4373,-14.8304,-18.3946,-17.1754,-14.2638,-17.9258,-17.2802,-17.4831,-17.9008,
    -16.0068,-16.8294,-18.1714

The four loud pulsars under pivot matching, as a sanity check on the generated data:
J1643−1224 (−13.87), J1939+2134 (−14.06), J0621+1002 (−14.30), J1012+5307 (−14.58).

## What this spike is not

It does **not** reproduce 2b. The injected red noise is f⁻⁴ where 2b's loud pulsars are f⁻²·⁵ —
deliberately, because the point is to hand Argus data from its own model at comparable red-noise
strength. So a clean result says "Argus samples its own model here", which is what discriminates
the two hypotheses; it does not say "Argus would sample 2b if only the priors were different".

It also does not choose a remedy. If misspecification is confirmed, what replaces the
single-corner OU for per-pulsar red noise is a subsystem change (cf. issue #115) and needs its
own design.

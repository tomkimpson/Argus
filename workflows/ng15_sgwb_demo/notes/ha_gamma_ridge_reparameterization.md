# The h_a–γ_a ridge and the ridge GW reparameterization

**Context:** M1 MDC2 dress rehearsal (issue #111), branch `m1-mdc2-dress-rehearsal`.
**Date:** 2026-07-25.
**Scope:** why the MDC2 array-stage NUTS runs (Stage B/C) would not converge, how
we diagnosed it, and the reparameterization that fixes it. This is the issue #109
"geometry wall" isolated at 2 parameters, so the fix applies directly to M3.

---

## 1. Setting

M1 recovers the injected GWB on IPTA MDC2 dataset 2b (33 pulsars, injected
`log10_A = -14.886`) using the two-stage-noise procedure:

- **Stage A** — single-pulsar noise runs → per-pulsar `(γ_p, σ_p)` posteriors.
- **Stage B** — array run, red noise **fixed** at the Stage A medians, sampling
  only the 2 GW parameters `(log10_ha, log10_gamma_a)`.
- **Stage C** — array run with **empirical priors** on the per-pulsar noise from
  Stage A (68 sampled dims), same 2 GW parameters on top.
- **Stage D** — HD-vs-CURN Bayes factor via `logz_lhm.py`.

Each stage is run under both the Hellings–Downs ORF (`run_analysis.py`) and the
identity ORF (`run_curn.py`, the CURN null). Stage A completed cleanly (33/33
pulsars, r_hat ≤ 1.010, ESS ≥ 390, no railing). The problem appeared in the
array stages.

Argus models the GWB as a single-corner **Ornstein–Uhlenbeck** process. The
recovered residual PSD is

```
S_r(f) = σ_a² / ( w² (γ_a² + w²) ),   w = 2π f,   σ_a² = (h_a²/12) γ_a
```

with `h_a = 10^log10_ha`, `γ_a = 10^log10_gamma_a`.

---

## 2. Symptom

The array-stage jobs first appeared to be a **walltime** problem: job after job
was cancelled at its time limit. We chased that for several rounds:

| Job | Config | Outcome |
|-----|--------|---------|
| 14572769 | Stage A, 4 chains, 30 min | all 33 array tasks TIMEOUT |
| 14593114 | Stage B curn, depth 10, 4 h | TIMEOUT — one chain at 29 s/it |
| 14635408 | Stage B curn, depth 8, 12 h | TIMEOUT — one chain at 17 s/it |
| 14635407 | Stage B hd, depth 7, 12 h | TIMEOUT — chain 2 at 19 s/it |

The fixes to *those* symptoms were real and are retained (chains run
**sequentially** on a single GPU, so 4×A100 or a longer wall is needed; Argus
writes results only at the *end* of sampling, so a timeout loses everything).
But the pattern was suspicious: in every Stage B run, **three of four chains
finished in ~45 minutes and exactly one chain ran ~20–40× slower**, pinned at
the maximum tree depth on every iteration.

Capping `max_tree_depth` at 5 (measurement: healthy chains use only ~5 leapfrog
steps ≈ depth 3, so a low cap costs them nothing) finally let the runs
**complete** — B-hd in 4:45, B-curn in 3:10, C-curn in 37.8 h. But completing is
not converging.

---

## 3. The real diagnosis

Post-run diagnostics exposed the actual problem — not speed, but **convergence**:

```
run              rhat(ha)  rhat(ga)  div%    per-chain median log10_ha
mdc2_stageB_hd     1.587    1.100    0.0%    [-14.63, -11.41, -14.63, -14.62]
mdc2_stageB_curn   1.567    1.067   17.6%    [-14.58, -11.58, -14.61, -14.60]
mdc2_stageC_curn   2.267    2.145   10.0%    [-12.58, -10.99, -13.38, -13.37]
```

In each run, three chains cluster near the injected amplitude
(`log10_ha ≈ -14.6`, close to the truth `-14.886`) while **one chain is parked
at high amplitude** (`≈ -11.4`). That single stuck chain is what both (a) maxes
the tree depth (its trajectories never U-turn) and (b) wrecks r_hat.

Removing the stuck chain is decisive:

```
mdc2_stageB_hd, good chains [0,2,3]:  r_hat(ha) = 1.001, r_hat(ga) = 1.001
mdc2_stageB_curn, good chains [0,2,3]: r_hat(ha) = 1.002, r_hat(ga) = 1.000
```

**The geometry, not the sampler, is the blocker.** The healthy chains sample the
main mode perfectly; a fraction of random initializations fall into a region the
sampler cannot leave in reasonable time.

### Why the chain gets stuck

The `(log10_ha, log10_gamma_a)` posterior is a **curved, weakly-identified
ridge**. Measured on the good chains of Stage B hd:

```
Pearson corr(log10_ha, log10_gamma_a) = 0.343      (moderate linear correlation)
quadratic fit of ga vs ha: x² coeff  = 0.506        (real curvature)
log10(σ_a²) = 2·log10_ha + log10_gamma_a − log10(12):
    std = 1.65 dex   vs   std(log10_ha) = 0.55,  std(log10_gamma_a) = 0.91
corr(σ_a² combination, log10_ha) = 0.854
```

The reading:

- The band amplitude `σ_a²` is the **least**-constrained combination (std 1.65,
  larger than either raw parameter) — it is the *flat along-ridge direction*.
- The ridge is **curved** (nonzero quadratic term), so a dense mass matrix
  cannot straighten it: a dense mass matrix only rotates out a *linear*
  correlation. This is exactly why `dense_mass = true` and the
  `dense_mass_blocks` GW block did not help — and why this is the same wall that
  made the 68-pulsar full-array run unsamplable in issue #109, now reproduced at
  just 2 parameters.

Lowering the tree-depth cap bounds a stuck chain's *cost* but not its *mixing*:
whatever the cap, the stuck chain consumes the whole budget every iteration
because its trajectory never U-turns along the flat ridge.

---

## 4. The solution: sample in a straightened basis

The fix is to sample the coordinates the data actually constrains, instead of
the raw `(log10_ha, log10_gamma_a)`.

A PTA constrains the residual PSD in a ~1-decade band near `1/T`. So the
well-identified coordinate is the **band-referenced pivot log-PSD**,
`log10 S_r(f_piv)`, and the loose coordinate is the corner location,
`log10_gamma_a`. Sampling those two as **independent** coordinates decouples the
constrained direction from the flat one and removes the ridge.

`log10_ha` is then *derived* by inverting the OU PSD at the pivot (closed-form,
smooth, exact):

```
S_r(f_piv) = (h_a²/12) γ_a / ( w² (γ_a² + w²) )
⇒  log10_ha = 0.5 [ log10(12) + log10 S_r(f_piv) + 2 log10(w)
                     + log10(γ_a² + w²) − log10_gamma_a ],   w = 2π f_piv
```

The inversion round-trips exactly over the whole prior box (verified
numerically to < 1e-9 dex).

### What changes and what does not

- **The likelihood is untouched** — it still receives `(log10_ha,
  log10_gamma_a)`. Only the *prior/sampling basis* changes: the model now places
  a uniform prior on `(log10 S_r(f_piv), log10_gamma_a)` instead of on
  `(log10_ha, log10_gamma_a)`. This is a deliberate, physically better-motivated
  prior (flat in the observable band amplitude), not merely a change of
  coordinates, so it is **opt-in** and the runs are labelled `*_ridge`.
- **Downstream tooling is unaffected.** The two sampled sites end in `_prime`
  (`log10_pivot_psd_prime`, `log10_gamma_a_prime`), so the learned-harmonic-mean
  evidence script `logz_lhm.py` picks them up automatically. `log10_ha` and
  `log10_gamma_a` remain `numpyro.deterministic`, so the truth gate
  (`check_mdc2_truth.py`) and corner plots read them exactly as before.

### Implementation

Opt-in via `[PriorModel] gw_parameterization` (fallback `direct` = original
behavior; **absent key ⇒ every existing config and both goldens are
bit-identical**):

```ini
[PriorModel]
gw_parameterization = ridge
log10_pivot_psd_min  = -13.0        # brackets injected (~-6.3) and direct-run
log10_pivot_psd_max  = -5.0         #   recovered (~-9.2) pivot log-PSD with room
gw_pivot_freq_hz     = 6.3376e-09   # 1/(5 yr), near the sensitive band
```

Code (all behind the flag):

- `prior_models.get_gw_parameter_priors` — ridge branch builds the pivot-log-PSD
  and `log10_gamma_a` reparameterized (N(0,1)) priors and stores the angular
  pivot frequency `w`; `get_prior_model_specs` forwards the ridge keys.
- `parameter_sampling.sample_gw_parameters` — ridge branch samples the two
  `_prime` sites and derives `log10_ha` via the inversion above.
- `count_free_parameters` and `bayesian_inference.display_prior_summary` — handle
  the ridge case (2 free GW params).
- `scripts/check_mdc2_truth.py` — computes r_hat on the *sampled* `_prime` sites
  rather than the derived `log10_ha` deterministic.
- Configs `mdc2_stage_{b_hd,b_curn,c_hd,c_curn}_ridge.ini`; Stage C's
  `dense_mass_blocks` retargeted to `log10_pivot_psd_prime, log10_gamma_a_prime`;
  slurm scripts take `BASIS = ridge|direct` (default `ridge`).

---

## 5. Verification

- **Unit tests** (added): ridge prior-spec construction; a sampling-trace test
  that samples the two `_prime` sites and confirms the derived `log10_ha`
  reproduces the sampled pivot log-PSD through the forward OU PSD (round-trip);
  the free-parameter count (2); and pass-through of the ridge keys.
- **Full suite:** 281 passed, 1 skipped. Goldens `63618.93` (informative) and
  `59420.06` (diffuse) unchanged — the `direct` default path is untouched.
- **Config smoke:** the ridge Stage B config builds a 2-free-parameter model on
  the real 33-pulsar data with the two `_prime` sites and a finite derived
  `log10_ha`.

Runtime confirmation (ridge Stage B/C jobs 14820454–57) is pending on the
cluster; success criterion is `r_hat < 1.01` on the `_prime` sites with no stuck
chain, followed by the truth gate and the Stage D lnB.

---

## 6. Note on the science signal (preliminary, caveated)

On the *good 3-chain* direct-basis posterior, the band-referenced truth gate at
`f = 1/(5 yr)` gave recovered `log10 S_r ≈ -9.19 ± 1.40` against an injected
`≈ -6.32` (assuming γ = 13/3), i.e. the injected value sat just outside the
recovered 95% band (~2σ low). This is **preliminary and not a verdict**: the
posterior is very broad (±1.4 dex), and both the injected spectral index (the
MDC2 repo records only the amplitude — see `check_mdc2_truth.py` docstring) and
the pivot/baseline choice are assumptions. The canonical assessment is the truth
gate on the converged **ridge** runs, not this hand calculation.

---

## 7. Outcome (M1 result, 2026-07-28)

All four ridge Stage B/C runs sampled cleanly — the technical goal of the
reparameterization is met, at both 2-D and 68-D:

| run | worst r_hat (sampled sites) | divergences | runtime |
|-----|-----------------------------|-------------|---------|
| Stage B hd (2-D)   | 1.001 | 0.0% | 1:00 |
| Stage B curn (2-D) | 1.000 | 0.0% | 1:05 |
| Stage C hd (68-D)  | 1.004 | 0.0% | 9:34 |
| Stage C curn (68-D)| 1.005 | 0.0% | 9:36 |

Compare the direct basis, which did not converge (Stage B r_hat 1.5–1.6 with one
stuck chain; Stage C r_hat 2.3, 10% divergences). The ridge fix also cut the
68-D iteration cost ~4× (~8 s/it vs ~32–38 s/it), because the GW corner no longer
forces max-depth trajectories.

**Truth gate (band-referenced amplitude at f = 1/(5 yr), injected `log10_A = -14.886`):**

| stage | noise treatment | recovered log10 PSD | injected | bias | verdict |
|-------|-----------------|---------------------|----------|------|---------|
| Stage B hd | fixed at Stage A medians | −9.67 ± 1.08 | −6.32 | −3.11σ | **FAIL** |
| Stage C hd | empirical priors (sampled) | −6.94 ± 1.81 | −6.32 | −0.35σ | **PASS** |

Stage C **recovers the injected amplitude** and Stage B does not — exactly the
predicted behavior. Stage A characterized each pulsar's noise with the GW off, so
each single-pulsar OU fit absorbed that pulsar's share of the GW-induced red power
into its intrinsic noise. Fixed-noise Stage B then double-counts it and
under-recovers the common signal by ~3σ. Stage C keeps the noise sampled (empirical
priors, 2× inflated), so the array run re-attributes the power to the common GW
term — and the truth gate passes at 0.35σ. This validates the two-stage-noise
detection machinery on a case with known ground truth.

**Stage D Bayes factor (HD vs CURN):**

- Stage B pair (2-D, fixed noise): `lnB = −0.016 ± 0.020` — indistinguishable, as
  expected once the common signal has been absorbed into the fixed noise.
- Stage C pair (68-D, empirical priors): the learned-harmonic-mean estimator is
  **degenerate at this dimension** — no shrinkage value is contained
  (max-weight-fraction ~0.9), so no calibrated lnB is available (script returns
  nan). Every matched-shrinkage estimate is positive (+1.8 … +6.9 across the grid),
  so the *direction* favors HD, but the magnitude is not trustworthy.

This is the anticipated high-D breakdown of LHM (roadmap risk #3 — "LHM unreliable
at 33/68 pulsars; MDC is the canary"). M1 has therefore delivered its diagnostic
purpose twice over: it validated the amplitude-recovery machinery **and** confirmed
that the M3 Bayes factor needs a different estimator. **Decision (2026-07-28):**
bank the amplitude-recovery result as M1's success; defer the calibrated
HD-vs-CURN Bayes factor to a product-space / hypermodel estimator, to be built and
validated on this same MDC2 case as part of M3 preparation.

## 8. Implications for M3

The real NG15 68-pulsar run (M3) samples the same 2 GW parameters on top of the
per-pulsar noise and will hit the **same** h_a–γ_a ridge (this ridge was in fact
first seen there, issue #109). **Use `gw_parameterization = ridge` for M3.**

Operational lessons retained from the same episode (relevant to M3's larger,
longer runs):

- NUTS chains run **sequentially** per GPU — match `num_chains` to `--gres=gpu:N`
  to parallelize, or size the walltime as `chains × per-chain`.
- Argus writes results only **after** sampling completes, so a walltime timeout
  discards the entire run — the MCMC-checkpointing backlog item should land
  before M3.
- Per-chain parameter medians are the cheap tell for a stuck chain; always check
  them alongside the pooled r_hat.

# Spec: A one-pass state-space detection statistic for the SGWB on NANOGrav 15-yr data

**Status:** draft
**Created:** 2026-11-19  ·  **Last amended:** 2026-11-20 (Phase 0's Touches and task 0.3's path made consistent)
**Parent spec:** [`spec/spec.md`](spec.md). This spec replaces the parent's Phases 5–7 and
its headline claim (§1, K0(b)). The parent's Phases 0–4, its data, compression and reference
sections (§3 A1–A5, C1–C4; §4 D1–D3, D5; §5 inputs and environments) and its non-goals (§7)
are inherited and not restated.

## 1. The claim

> A time-domain state-space (Kalman-filter) likelihood, evaluated at a single best-fit noise
> model on the NANOGrav 15-yr narrowband data set (67 pulsars), yields a correlation
> statistic — the slope of the log-likelihood along the path from uncorrelated to
> Hellings–Downs correlations — that is significant against a sky-scramble null at
> **p ≤ 5 × 10⁻³**. The statistic needs no posterior sampling: one filter pass and its
> derivative per noise setting. The result is stable across the plausible noise settings, the
> null is shown to be calibrated on simulated data, and the common-process amplitude is
> consistent with NANOGrav's.

This is a detection-significance claim, the state-space analogue of NANOGrav's optimal
statistic. It is **not** a Bayes-factor claim. The parent's lnB(HD/CURN) ≥ 2.3 headline is
dropped from Argus unless the speed track (Phase S) reopens it. Discovery-OU's
lnB = 5.18 ± 0.05 is quoted in the paper as the reference pipeline's number, labelled as such.

We do not claim parity with NANOGrav's optimal-statistic p < 10⁻⁴: 5000 scrambles resolve p
only down to 2 × 10⁻⁴ (inherited from the parent's §1).

## 2. Background

Inherited from the parent, plus what has happened since Phase 4:

- **The 67-pulsar joint posterior does not sample at an affordable cost.** Three probes
  (v1, v2, v3 configs) missed every convergence threshold. The last, on v3 (γa ≤ −7.0,
  150 + 100, 4 × A100, 4 d 16.5 h), failed all five of its criteria
  (`workflows/ng15_nb/results/array/probe3_v3_criteria.json`).
- **The cause is cost per gradient, not one bad parameter.** Argus's value-plus-gradient is
  6.6–6.7 s per chain (0.84 s batched over 32 points). The Discovery-OU reference samples a
  nominally equivalent model at an estimated ~10 ms per gradient. A 1000 + 1000 rung at the
  probes' depth-8 trees is ≈ 37 d and ≈ 3,500 A100-h, and the ladder needs five.
- **The reference does not meet the parent's Phase 5 bar either.** Discovery OU-14 at ε = 0
  (v1 box, 4 × 1000 + 1000) has max r̂ 1.035 and min ESS 47
  (`results/summaries/ng15nb_ou14_eps000.json`), and its lnB was accepted as reliable.
- **The kernel carries the detection.** Discovery-OU on the v3 box gives
  lnB(HD/CURN) = 5.18 ± 0.05 (`results/lnb_hd_curn_ou14_v3.json`). The parent's K2 is
  answered: the OU kernel is not the limit.
- **The statistic and its null already exist for the stock filter.** At ε = 0 the
  path-sampling integrand is exactly linear in the ORF,
  d lnL/dε = Σᵢⱼ Gᵢⱼ (Γ − I)ᵢⱼ with G = ∂lnL/∂C at C = I. One reverse-mode gradient with
  respect to the correlation matrix scores any number of scrambles by dot product
  (`workflows/ng15_sgwb_demo/scripts/scramble_ensemble_eps0.py`, validated on MDC2 1b to
  1e-15 per draw; scramble generator `scripts/sky_scrambles.py`).
- **The sparse array filter takes the ORF as an input array**
  (`compression/union.py::array_loglike_sparse`, `inp["orf"]`), so G is reachable there. It
  has not been computed on the sparse path yet.
- **400 Argus draws exist at ε = 0**, produced without any Discovery input:
  `/fred/oz022/tkimpson/Argus-ng15nb-p5c/workflows/ng15_nb/outputs/ng15_nb_v3_probe_eps000/ng15_nb_v3_probe_eps000_results.nc`.
  They are not a converged posterior. Here they serve as a spread of plausible noise
  settings.
- **Why a filter cannot learn the hyperparameters in one pass.** The GWB changes second
  moments (variances and cross-pulsar covariances), not means. Amplitudes enter only the
  process noise, so a state augmented with them has zero first-order innovation sensitivity,
  and the red-noise posteriors are on/off bimodal. A particle version needs rejuvenation
  passes over past data. So the single pass is available for the *statistic at fixed noise*,
  not for the hyperparameter posterior.

## 3. Scientific validity

- **Model.** The parent's model and production box (`configs/frozen/ng15_nb_v3.ini`:
  ridge basis, diffuse timing prior, uniform red-noise boxes γp [−12, −4], σp [−22, −10],
  γa [−11, −7.0], fixed white noise in the compression). Nothing in the likelihood changes.
- **The statistic.** With θ the 136 noise and common-process parameters and Γ the ORF,
  S(θ; Γ) = ∂lnL/∂ε at ε = 0 along C(ε) = (1 − ε) I + ε Γ, which equals
  Σᵢⱼ Gᵢⱼ(θ) (Γ − I)ᵢⱼ. The observed value is S(θ̂; Γ_HD), with θ̂ the best-fit point under
  the uncorrelated model (ε = 0).
- **The null.** S(θ̂; Γ_k) for ≥ 5000 sky-scrambled ORFs Γ_k, each with match
  |M̄| < 0.17 against Γ_HD (the parent's D6 rule). The one-sided p is the fraction of
  scrambles with S ≥ the observed value, counted as (n + 1)/(N + 1), with its binomial
  uncertainty.

**Assumptions and approximations.**
- **B1. A fixed noise point stands in for the noise posterior.** It holds if S varies
  little, relative to the null's width, across the noise settings the data allow. Tested in
  Phases 1 and 2 (K1, K2). It breaks if the on/off red-noise states of a few pulsars move
  the statistic by more than the null's width.
- **B2. Sky scrambles give a calibrated null when the noise is fitted to the same data.**
  Tested on simulated data in Phase 3 (V3, K3). NANOGrav makes the same assumption for
  their optimal statistic.
- **B3. The slope at ε = 0 is the detection statistic, not the evidence.** It ranks
  geometries by d lnZ/dε at CURN. A percentile here is not an lnB percentile.
- **B4. The best-fit point may sit on a box edge** (e.g. γa at −7.0, σp of "off" pulsars
  anywhere on a flat plateau). The optimum is then not unique. This is why the claim
  requires stability across optima, not a single number.
- A1–A5 of the parent are inherited.

**Correctness checks** (each is a gate in §11):
- **V1. The contraction is exact on the sparse filter.** On ≥ 10 of the existing draws, the
  dot product Σ G (Γ − I) agrees per draw with a scalar forward derivative of lnL in ε
  (JVP or central finite difference), for Γ_HD and for one scramble, to a relative error
  ≤ 1e-6.
- **V2. The value path is unchanged.** lnL from the statistic code at ε = 0 equals
  `SparseArrayFilter.get_likelihood` at the same θ to ≤ 1e-8 nats, and the parent's C2
  goldens (63618.93, 59454.72) still pass after any library change.
- **V3. The null is calibrated.** On ≥ 200 data sets simulated from Argus's own model at θ̂
  with **no** cross-correlation, the scramble p-values are uniform: the false-alarm rate at
  nominal 5 % lies in [2 %, 9 %], and a KS test against uniform gives p > 0.01.
  - > ⚠️ DRAFT — revise: the 200 simulations are scored at the true noise. Refitting the
    > noise per simulation is what the real analysis does; 20 refitted simulations measure
    > how much that moves the p-values. Whether 20 is enough is the user's call once the
    > refit cost is measured in Phase 2.
- **V4. The statistic has power.** On ≥ 200 simulated data sets with HD correlations at
  θ̂'s amplitude, the median p is reported, and it is ≤ 0.05. If it is not, a real-data
  non-detection would be uninformative and the paper says so.
- **V5. Agreement with the reference (reported, loosely gated).** The mean of S(θ; Γ_HD)
  over the 400 Argus draws is compared with the Discovery OU-14 v3 ε = 0 integrand mean.
  - > ⚠️ DRAFT — revise: proposed gate is agreement within one draw-to-draw standard
    > deviation of the Discovery integrand (4.1 on the v1 box, where the mean is 24.6). The
    > two likelihoods differ by construction (14 common Fourier modes against a full-band
    > common process), so this number is a guess until Phase 1 measures it.

## 4. Approach

**D1. Detect with the ε = 0 score at a fixed noise point, not with a sampled Bayes factor.**
**Why:** the sampler route needs 10⁵–10⁶ gradients per rung at 6.6 s each, over five rungs,
and no approved lever closes that. The statistic needs one gradient per noise setting. It is
the parent's K0(a) made self-standing, and it uses what a state-space filter gives
naturally.
**Alternatives considered:**
- *Lever B (a sampled per-pulsar flat-PSD term) and a fourth probe.* Fixes one owner trade
  at best; does not supply the missing factor of 10–40. A 150 + 100 probe cannot resolve
  r̂ at 1.05.
- *State augmentation or a particle filter over the hyperparameters.* See §2: no first-order
  sensitivity to amplitudes, and particle rejuvenation re-runs the filter.
- *The long-baseline subset* (parent K4's pivot). Its cost saving is unmeasured, and it
  drops data the statistic route can keep.

**D2. Headline noise treatment: a best-fit point from a multi-start optimiser, with the
spread shown.** (User decision, 2026-11-19.)
- θ̂ maximises lnL at ε = 0 inside the v3 box, from 32 starts run in lockstep on one A100.
- The paper reports S and p at θ̂, their range across the other optima within ΔlnL ≤ 3 of
  the best (gated, K2), and the spread of S over the 400 existing Argus draws (shown as a
  band, not gated).
**Why:** an optimiser costs hundreds of gradient evaluations and uses batched throughput
(0.84 s per evaluation), which a NUTS chain cannot. The unconverged draws are honest as a
spread of settings and dishonest as a posterior.
**Alternatives considered:**
- *Average S over Argus chain draws* (the parent's D6). The chains fail every convergence
  threshold, so the average is over a non-posterior. Kept as K2's pivot.
> ⚠️ DRAFT — revise: "ΔlnL ≤ 3" as the definition of a plausible optimum is a guess; fix it
> after Phase 2 shows the spread of optima.

**D3. Argus is independent of Discovery.** (User decision, 2026-11-19.)
- Nothing from Discovery may seed, warm-start, reweight or tune the Argus result: no
  initial points, mass matrices, step sizes or draws.
- Optimiser starts come from the prior box or from Argus's own single-pulsar posteriors
  (parent Phase 3).
- Discovery is allowed as a comparison (V5, T1) and as a diagnostic (the appendix ΔlnL
  check in Phase 4).

**D4. Reuse the linear-in-ORF trick; port it to the sparse filter as workflow code.**
New code lives in `workflows/ng15_nb/stat/`. The library filter is not edited for the
statistic. `scramble_ensemble_eps0.py` and `sky_scrambles.py` are imported or copied with
their validation, not rewritten.
**Why:** the trick is validated, and the user's standing preference is to change analysis
scripts before library source.

**D5. Speed work is a separate, time-boxed track** (Phase S: 3 weeks, 100 A100-h, benchmarks
only; user decision 2026-11-19). It reopens the Argus Bayes factor only through a parent
amendment, and only if per-chain value-plus-gradient gets ≥ 30× faster.
**Why:** each lever also makes the statistic cheaper, but none is needed for the claim, so
the track must not be able to stall it.

## 5. Data and reproducibility

Inputs, environments, seeds, provenance and pinning are the parent's (§5). Additions:

- **Existing draws.** The v3 probe `.nc` (path in §2) lives in a pinned worktree that is
  not tracked. Phase 1 records its SHA-256 and copies the readout, not the chain, into
  `workflows/ng15_nb/results/stat/`. Read it with h5py; posterior keys carry a site prefix
  (reuse the loaders in `array/probe3_criteria.py`).
- **Scramble ensembles** are seeded, saved as `.npz` under `workflows/ng15_nb/data/scrambles/`
  (gitignored), with a tracked summary JSON carrying the seed, count, match threshold and
  hash.
- **G matrices** (one 67 × 67 array per noise setting) are saved, so new ensembles are free
  to rescore.
- **Compute.**
  - > ⚠️ DRAFT — revise: every figure below is an estimate from the Phase 2 benchmark
    > rates (6.7 s per value-plus-gradient per point, 0.84 s batched over 32). The cost of a
    > gradient with respect to the 67 × 67 ORF is unmeasured; it is assumed equal to one
    > value-plus-gradient.
  - Phase 1: 400 gradients ≈ 1 A100-h, scrambles on CPU. Budget 10 A100-h.
  - Phase 2: 32 starts × ~1000 lockstep steps × 27 s ≈ 8 h on one A100. Budget 30 A100-h.
  - Phase 3: ~400 simulated G evaluations plus 20 refits. Budget 50 A100-h.
  - Phase 4: an 11-point profile, each a warm-started re-optimisation. Budget 40 A100-h.
  - Phase S: 100 A100-h (fixed).
  - Total ≈ 230 A100-h. The parent's 1000 A100-h budget predates the three probes, whose
    spend has not been totalled here. The user confirms the new total before Phase 2.
- **GPUs:** A100 only (`--partition=milan-gpu --gres=gpu:a100:N`). Readouts that take more
  than a minute run on SLURM, not the login node.

## 6. The paper

**Contribution.** A Kalman-filter likelihood detects Hellings–Downs correlations in the
NANOGrav 15-yr data at a sky-scramble p ≤ 5 × 10⁻³ using one filter pass at a fixed noise
model, independently of any Fourier-basis pipeline.

**Story.**

1. Introduction — every published GWB detection uses a Fourier-basis likelihood; a
   time-domain confirmation is missing, and this paper supplies one.
2. Method — the state-space model, exact epoch compression, and why the correlation
   statistic is one pass and linear in the ORF while the hyperparameter posterior is not.
3. Validation — the null is calibrated and the statistic has power on simulated data; the
   likelihood matches a dense oracle and a Fourier reference pulsar by pulsar.
4. Result — the observed statistic against the scramble null on NG15, and its stability
   across noise settings.
5. Comparison — amplitude and significance against NANOGrav and against a kernel-matched
   Fourier reference; what the OU kernel does and does not allow us to say.
6. Discussion — the cost of a full posterior in the time domain (the honest negative
   result), and what would make it affordable.

**Figures, tables and key numbers.**

| ID | Shows | Caption claim it must support | Where | Produced in |
|----|-------|-------------------------------|-------|-------------|
| F1 | Scramble-null histogram of S at θ̂, observed value marked | The true sky geometry beats all but a fraction p ≤ 5 × 10⁻³ of ≥ 5000 scrambles | main §4 | Phase 3 |
| F2 | S and p at each optimum and across the 400 Argus draws | The detection does not depend on the noise point: across the plausible optima and the four chain means the statistic stays within K1/K2's stability bound (factor of 10 in p as approved; span of 1 in null-σ units proposed), with the per-draw spread shown as a band | main §4 | Phases 1, 2 |
| F3 | Scramble p-values on simulated data, without and with HD correlations | With no correlations the p-values are uniform (false-alarm rate 2–9 % at nominal 5 %); with HD at the fitted amplitude the median p is ≤ 0.05 | main §3 | Phase 3 |
| F4 | S at θ̂ recomputed on the data truncated to increasing time spans | The statistic grows with the data span as a real background should | main §4 | Phase 5 |
| T1 | p (Argus at θ̂, with range), NANOGrav's optimal-statistic and Bayes-factor p; common amplitude from Argus (profile), Discovery-OU and published; Discovery-OU and published lnB as reference rows | Argus's amplitude interval overlaps the published one, and its significance is of the same order as NANOGrav's scramble p = 1.6 × 10⁻³ | main §5 | Phases 3, 4 |
| F5 | Pair scores Gᵢⱼ binned by angular separation, with the HD curve | The pairs that drive S follow the Hellings–Downs pattern in sign | appendix A | Phase 3 |
| F6 | lnL(Argus) − lnL(Discovery-OU) on Discovery draws | The two likelihoods differ by a spread of X nats, tracked by Y (diagnostic only) | appendix B | Phase 4 |
| N1, N2, F4-parent | Compression accuracy, kernel systematic, per-pulsar PSD scatter (parent §6) | As in the parent | appendix C | done (parent Phases 1–3) |
| N3 | Cost per gradient before and after each speed lever | A full time-domain posterior costs Z A100-h per rung; lever W gives a factor V | appendix D, cited in §6 | Phase S |

> ⚠️ DRAFT — revise: F4 is the "online" figure and is the least certain. The diffuse
> timing-model term is fitted to the whole span, so a truncated-span rerun (noise held at
> θ̂ or refitted) is the honest construction, not a running sum. Whether it is main text,
> and whether the noise is refitted per span, is decided when Phase 5 is reached. F5's
> caption claim has no number yet.

**Appendix policy.** The main text carries the statistic, its null, its calibration and its
stability. Likelihood validation inherited from the parent, the Discovery diagnostics and
the speed benchmarks are appendices: they support trust in the method but are not steps in
the argument.

## 7. Non-goals

The parent's §7, with one exception: the parent lists #108 (the parallel filter) as a
non-goal, and Phase S's task S.4 benchmarks it. That exception covers a benchmark only.
Plus:

- An Argus Bayes factor, a path-sampling ladder, or a converged 136-dimensional posterior,
  unless Phase S reopens them through a parent amendment.
- Any further 4-chain 67-pulsar NUTS probe.
- Lever B (a sampled per-pulsar flat-PSD term) and any change to the likelihood or the box.
- Seeding, warm-starting, reweighting or tuning Argus from Discovery output (D3).
- State augmentation with hyperparameters, particle filters, SMC, or any claim of real-time
  detection. "One pass" means one pass at fixed noise.
- A null on lnB, phase-shift nulls, and per-frequency statistics.
- Fixing the MDC2 noise-JSON symlink and rerunning MDC2 results. That is a separate small
  PR and feeds nothing here.
- Resolving the J0645+5158 and B1855+09 questions from the v3 probe criteria. Those criteria
  are retired with the probe route.

## 8. Kill criteria

- **K1 (Phase 1, the opening test).** On the 400 existing draws, with ≥ 5000 scrambles:
  stop and return to the user if **either**
  - the pooled-mean statistic has scramble p > 0.05, **or**
  - the four chains' individual p-values span more than a factor of 10.
  (User decision, 2026-11-19. Passing shows the route can carry the claim, not that the
  claim holds. If every chain's p is at the resolution floor, the span is 1 and passes.)
  > ⚠️ DRAFT — revise: the factor-of-10 test is in awkward units. Near z ≈ 3 a factor of 10
  > in p is only about 0.7 null standard deviations, and near the 2 × 10⁻⁴ floor it is the
  > difference between 0 and 9 exceedances in 5000, i.e. counting noise. So it is most
  > likely to fire when the detection is strongest. Proposed replacement, for the user to
  > approve: express each chain's (and, in K2, each optimum's) statistic as
  > z = (S − null mean)/null sd, and trigger when the z values span more than 1. The
  > per-draw spread of S is reported as a band and never gated: single draws scatter far
  > more than chain means (Discovery's draw sd is 4.1). Until approved, the factor-of-10
  > wording stands and both quantities are reported.
- **K2 (Phase 2). The fixed noise point is ill-defined.** Fewer than 8 of 32 starts
  converge, or p across the optima within ΔlnL ≤ 3 of the best spans more than a factor of
  10. Pivot, with the user, to the draw-averaged statistic (D2's alternative) and say in the
  paper that it is not a posterior average.
- **K3 (Phase 3). The null is not calibrated.** V3 fails. The scramble p then means
  nothing at fixed noise. Stop; the fallback is a null built from simulations at θ̂, which
  the user must approve because it makes the significance model-dependent.
- **K4 (Phase 3). The claim is not reached.** The headline p at θ̂ is > 5 × 10⁻³. Report the
  achieved p honestly and return to the user. V4's measured power says whether the shortfall
  is the method or the data.
- **K5 (Phase S). The time box.** The track closes at 3 weeks or 100 A100-h, whichever
  comes first. Below a 30× gain in per-chain value-plus-gradient, the Argus Bayes factor
  stays out of scope. At or above it, the user decides whether to reopen the ladder.

## 9. Open questions

- [NEEDS CLARIFICATION: whether reverse-mode differentiation with respect to the ORF works
  through the sparse filter's checkpointed segments at the same memory and cost as the
  parameter gradient. Task 1.1.]
- [NEEDS CLARIFICATION: whether 5000 scrambles with match < 0.17 are obtainable for the
  67 NG15 sky positions in reasonable attempts. Task 1.3.]
- [NEEDS CLARIFICATION: the optimiser (L-BFGS or Adam in the unconstrained coordinates) and
  its convergence test. Task 2.1 fixes both before the 32-start run.]
- [NEEDS CLARIFICATION: whether a simulator for the compressed layout exists. If not, it is
  built in task 3.2: draw the states and the scalar observations from the model with the
  compression's H rows and σ_w, timing coefficients set to zero (they are marginalised
  under the diffuse prior).]
- [NEEDS CLARIFICATION: how to simulate the remainder. `build_sparse_inputs` carries
  initial accumulators (A₀, b₀, c₀) per pulsar, and b₀ and c₀ are computed from the real
  data's remainder. A simulation that redraws only the scalars would mix real and simulated
  data and V3 would mean nothing. The simulator must redraw the remainder's b₀ and c₀ from
  the same model (A₀ depends on the design only), or simulate raw TOAs and recompress.
  Decided in task 3.2 before any simulated statistic is computed.]
- [NEEDS CLARIFICATION: how the amplitude interval is defined at fixed noise. Draft: a
  profile likelihood in the pivot log-PSD with the other 135 parameters re-optimised.
  Decided at the start of Phase 4.]

## 10. Risks

| Risk | Early warning |
|---|---|
| S swings with the on/off state of a few pulsars (B1) | Per-chain p-values in Phase 1 differ by more than 10× (K1); S correlates draw by draw with J1713+0747's or J0645+5158's σp |
| The optimiser returns many inequivalent optima on flat plateaus (B4) | lnL spread among converged starts > 3 nats, or θ̂ on several box edges |
| A maximum-likelihood noise point overfits and inflates or deflates S relative to a posterior average | S at θ̂ lies outside the 5–95 % range of S over the 400 draws |
| The scramble null is too narrow at fixed noise (B2) | False-alarm rate above 9 % in V3 |
| Simulations from the OU model are kinder than real data | V3 passes but the real-data scramble histogram is visibly non-Gaussian or skewed where the simulated ones are not |
| The ORF gradient costs far more than a parameter gradient | Task 1.1's first timing exceeds 30 s per draw |
| The speed track absorbs attention | Phase S work appearing in a statistic-track PR; the time box in K5 |
| The independence rule is broken by convenience | Any Discovery path appearing in `stat/` code outside the V5 and F6 comparisons |

## 11. Task list

**Waves.** W0: P0 → W1: P1 (kill check) → W2: P2 ∥ PS → W3: P3 ∥ P4 → W4: P5 → W5: P6.
Nothing fans out before P1's K1 is evaluated.

### Phase 0 — Housekeeping and the parent's record

**Goal:** make the state of the old route durable before the new one starts.
**Feeds:** infra; §6 text (the negative result in the Discussion)
**Depends on:** —
**Touches:** `spec/spec.md` (§8, §11, §12 only), `spec/detection-statistic.md`,
`workflows/ng15_nb/results/stat/inputs.json`, remote branch _(amended 2026-11-20)_
**Done when:** `phase-5-array-pilot` exists on the remote, and the parent's §12 carries a
dated entry that the user has approved.
**PR:** #138 (merged 2026-11-20)

- [x] 0.1 Push `phase-5-array-pilot` (27 commits, local only) → infra — verify by
      `git branch -vv` showing an upstream with 0 ahead.
- [x] 0.2 Parent §12 entry: the v3 probe FAIL with its five numbers, K4 evaluated as
      triggered, the pivot taken (this spec, not the long-baseline subset), and Phases 5–7
      marked superseded with a link here → §6 text — verify by the user approving the
      wording before it is committed.
      _(done: wording approved with the Phase 0 plan, 2026-11-20; A1 caveat added after review)_
- [x] 0.3 Record the SHA-256 of the v3 probe `.nc` and the Discovery OU-14 v3 ε = 0 chain
      in `workflows/ng15_nb/results/stat/inputs.json` → infra — verify by recomputing both hashes.

### Phase 1 — The opening test on the existing draws

**Goal:** find out, for about one GPU-hour, whether the statistic is significant on real
NG15 and how much it moves with the noise setting.
**Feeds:** F2, V5 row of T1
**Depends on:** P0
**Touches:** `workflows/ng15_nb/stat/`, `workflows/ng15_nb/results/stat/opening/`,
`workflows/ng15_nb/data/scrambles/`
**Done when:** a committed readout gives the pooled and per-chain p-values with binomial
uncertainties, V1 and V2 pass, and K1 is evaluated in writing.
**PR:** _not started_

- [ ] 1.1 `stat/orf_gradient.py`: G(θ) = ∂lnL/∂C at C = I through
      `array_loglike_sparse`, with wall time and peak memory logged → infra — verify by V2
      (lnL equal to `SparseArrayFilter.get_likelihood` to ≤ 1e-8 nats at 3 draws).
- [ ] 1.2 Validate the contraction → infra — verify by V1 (per-draw relative error ≤ 1e-6
      on ≥ 10 draws, for Γ_HD and one scramble, against a scalar derivative in ε).
- [ ] 1.3 Generate ≥ 5000 scrambles for the 67 NG15 positions with match < 0.17, seeded
      → F1, F2 — verify by the summary JSON reporting count, maximum match and hash, and
      by regenerating the first 10 from the seed.
- [ ] 1.4 Compute and save G for all 400 draws (SLURM, one A100) → F2 — verify by 400
      finite 67 × 67 arrays and a rerun of 5 draws matching to 1e-8. (G need not come out
      symmetric if the filter reads one triangle of the ORF; the contraction is still
      exact. Symmetrise only for F5.)
- [ ] 1.5 Readout: S per draw; pooled-mean and per-chain-mean S against the scramble
      distribution; p-values; S against J1713+0747's and J0645+5158's σp draw by draw
      → F2 — verify by the per-draw S for Γ_HD reproducing task 1.2's values exactly.
- [ ] 1.6 Compare the pooled-mean S with the Discovery OU-14 v3 ε = 0 integrand mean
      → T1 — verify by V5, and by a written line fixing or replacing V5's tolerance.
- [ ] 1.7 Evaluate K1 → §6 text — verify by a dated note in `results/stat/opening/` that
      quotes both trigger quantities and the verdict.

### Phase 2 — The best-fit noise point

**Goal:** find θ̂ and show how well defined it is.
**Feeds:** F2, T1
**Depends on:** P1 (K1 passed)
**Touches:** `workflows/ng15_nb/stat/fit/`, `workflows/ng15_nb/results/stat/fit/`
**Done when:** ≥ 8 of 32 starts converge, the optima and their S and p are tabulated, and
K2 is evaluated.
**PR:** _not started_

- [ ] 2.1 Optimiser driver over the v3 box at ε = 0, vmapped over 32 starts drawn from the
      prior box and Argus's own single-pulsar posteriors; convergence test fixed in the
      script before the run → infra — verify by 8 starts on a 3-pulsar subset of the
      real data reaching lnL values that agree with each other to ≤ 0.1 nat.
- [ ] 2.2 The 32-start run on NG15 (one A100, checkpointed) → F2 — verify by the count of
      converged starts and the lnL of each, with no Discovery path in the job's inputs.
- [ ] 2.3 G, S and p at every converged optimum → F2, T1 — verify by task 1.5's readout
      code run unchanged on the optima.
- [ ] 2.4 Where θ̂ sits: box edges, the on/off state of each pulsar, and S at θ̂ against the
      5–95 % range of S over the 400 draws → §4 text — verify by a table in the results
      note listing every parameter within 1 % of a bound.
- [ ] 2.5 Evaluate K2 and fix D2's "plausible optimum" rule → F2 — verify by a dated note
      and a §12 entry if the rule changes.

### Phase 3 — The null, its calibration and the headline number

**Goal:** the paper's significance, and the evidence that it means what it says.
**Feeds:** F1, F3, F5, T1
**Depends on:** P2
**Touches:** `workflows/ng15_nb/stat/sim/`, `workflows/ng15_nb/results/stat/null/`,
`workflows/ng15_nb/data/scrambles/` (a second, separately named ensemble)
**Done when:** F1's p is reported at its resolution, V3 and V4 are evaluated, and K3 and K4
are evaluated.
**PR:** _not started_

- [ ] 3.1 Headline: S(θ̂; Γ_HD) against ≥ 5000 scrambles, one-sided p with binomial
      uncertainty → F1, T1 — verify by rescoring from the saved G with a second,
      independently seeded ensemble and the two p-values agreeing within their
      uncertainties.
- [ ] 3.2 Simulator for the compressed layout at θ̂, redrawing the remainder accumulators
      as well as the scalars (see §9) → F3 — verify by (i) the mean parameter gradient of
      lnL at θ̂ over 50 simulated data sets being zero within 3 Monte Carlo standard errors
      in every parameter, (ii) the normalised innovations having variance 1 within 2 %, and
      (iii) no real-data array being read on the simulated path.
- [ ] 3.3 200 simulations without cross-correlation, scored at the true noise; 20 of them
      also with the noise refitted → F3 — verify by V3.
- [ ] 3.4 200 simulations with HD correlations at θ̂'s amplitude → F3 — verify by V4.
- [ ] 3.5 Pair scores binned by angular separation → F5 — verify by the binned values
      summing back to S to 1e-10.
- [ ] 3.6 Evaluate K3 and K4 → §4 text — verify by a dated results note quoting every
      number with its source file.

### Phase 4 — Amplitude and comparison

**Goal:** the consistency half of the claim, and the reference rows of T1.
**Feeds:** T1, F6
**Depends on:** P2
**Touches:** `workflows/ng15_nb/stat/amplitude/`, `workflows/ng15_nb/results/stat/amplitude/`
**Done when:** T1's amplitude row is filled for Argus, Discovery-OU and the published
value in one convention, and the overlap is stated.
**PR:** _not started_

- [ ] 4.1 Decide the interval definition (§9) and record it in §12 → T1 — verify by the
      user's approval of the entry.
- [ ] 4.2 Profile in the pivot log-PSD over ≥ 11 points, warm-started from θ̂ → T1 —
      verify by the profile's maximum reproducing lnL(θ̂) to ≤ 0.1 nat.
- [ ] 4.3 Put Argus, Discovery-OU and the published amplitude in one convention (the
      library pivot is two-sided: add log10 2 for one-sided) → T1 — verify by the
      conversion reproducing the parent's N2 numbers.
- [ ] 4.4 Diagnostic only: lnL(Argus) on ~2000 Discovery OU-14 v3 draws, the spread of the
      difference and what it tracks → F6 — verify by the parameter mapping round-tripping
      5 draws through both parameterisations to 1e-10, and by the output directory being
      read by no other task.

### Phase S — Speed track (time-boxed: 3 weeks, 100 A100-h, benchmarks only)

**Goal:** measure what each structural lever buys per gradient.
**Feeds:** N3; §6 text (Discussion)
**Depends on:** P1 (K1 passed)
**Touches:** `python/argus/jax_kalman_filter.py`, `workflows/ng15_nb/compression/`,
`workflows/ng15_nb/results/speed/`, `test/`
**Done when:** each lever has one measured attempt or a written reason it was not
attempted, K5 is evaluated, and no sampling run has been submitted.
**PR:** _not started_

- [ ] S.1 Baseline at the production layout: value, value-plus-gradient, single and
      batched over 32, with a per-step profile → N3 — verify by reproducing 0.84 s batched
      and 6.7 s single to within 15 %.
- [ ] S.2 Forward-mode derivative for a single direction (the ε score at one Γ) → N3 —
      verify by agreement with the reverse-mode contraction to 1e-8 relative, and its wall
      time against S.1's value time.
- [ ] S.3 Timing columns out of the per-step path (the 268 × 1,607 gain traffic) → N3 —
      verify by lnL equal to the baseline to ≤ 1e-8 nats at 5 points, V2's goldens passing,
      and the measured speed ratio.
- [ ] S.4 The parallel-scan filter at the production step count (~15k). The existing
      square-root version (`feat/kalman-sqrt-parallel`, `_run_kalman_filter_marginal_parallel`)
      predates the diffuse prior and covers the dense marginal path only, not the sparse
      one. First scope the port (diffuse prior, one scalar observation per step); attempt
      it only if the scope fits the remaining box, otherwise record why not → N3 — verify
      by lnL equal to the sparse filter's to ≤ 1e-6 nats at 5 points and the measured speed
      ratio, or by the written scope with its estimated effort.
- [ ] S.5 Evaluate K5 → N3 — verify by a results note with the best combined factor and a
      one-line recommendation on reopening the ladder.

Stat-track phases run from a pinned worktree, so Phase S's library edits cannot reach them
until merged.

### Phase 5 — The statistic against data span

To be specified when reached (F4): S at θ̂ on the data truncated to increasing spans. Open
points are in the §6 draft flag.

### Phase 6 — Write-up

To be specified when reached. The introduction and abstract come last.

## 12. Amendment log

### 2026-11-20 — Phase 0's Touches and task 0.3's path made consistent

**Trigger:** starting Phase 0. Task 0.3 writes `results/stat/inputs.json`, which Phase 0's
**Touches** did not list, and the path was relative where Phase 1's **Touches** spells out
`workflows/ng15_nb/results/stat/`. Phase 0 also commits this spec file itself and annotates
the parent's §8 (K4) alongside its §11 and §12.
**Change:** Phase 0's **Touches** now lists `spec/spec.md` (§8, §11, §12),
`spec/detection-statistic.md` and `workflows/ng15_nb/results/stat/inputs.json`; task 0.3's
path is written in full. No change of scope.
**Sections touched:** §11

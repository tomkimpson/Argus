# Spec: First state-space detection of the SGWB on NANOGrav 15-yr data

**Status:** draft
**Created:** 2026-10-15  ·  **Last amended:** 2026-10-16
**Supersedes:** `openspec/changes/sgwb-detection-route` (to be deleted, task 0.1) and the
wideband M3 plan in issue #111 / `workflows/ng15_sgwb_demo/PLAN.md`.

## 1. The claim

> A time-domain state-space (Kalman-filter) likelihood, run on the same NANOGrav 15-yr
> narrowband data set (67 pulsars) that underlies NANOGrav's 2023 evidence for a
> gravitational-wave background, finds Hellings–Downs inter-pulsar correlations favoured over
> an uncorrelated common red process with a Bayes factor of at least 10 ("positive evidence";
> benchmarked against NANOGrav's published hdγ-vs-curnγ value of 200–1000). Its correlation detection statistic is significant
> against a sky-scramble null at p ≤ 5 × 10⁻³, the same order as NANOGrav's sky-scramble
> p = 1.6 × 10⁻³ on their Bayes factor, and the common-process amplitude is consistent with
> NANOGrav's.
>
> We do not claim parity with their optimal-statistic p < 10⁻⁴: ≥ 5000 scrambles resolve
> p only down to 2 × 10⁻⁴.

This is the first SGWB detection with a non-Fourier, state-space pipeline. It confirms the
NANOGrav result with an independent likelihood, i.e. Paper A of issue #110. Every number is
benchmarked twice:
- against the published value;
- against a same-data, same-estimator reference run in NANOGrav's own JAX code (Discovery).

So a shortfall can be attributed to the data, the noise model (kernel), or the method.

The success bar (B ≥ 10, lnB ≥ 2.3) was chosen by the user on 2026-10-15; see K0 in §8.
Reaching NANOGrav's order (lnB ≥ 4.6) would strengthen the headline but is not required.

## 2. Background

**Published target** (Agazie et al. 2023, ApJL 951 L8, arXiv:2306.16213; text extracted
2026-10-15):
- Data: 67 pulsars with baselines of at least 3 yr; T = 16.03 yr.
- Bayes factors:
  - curnγ vs intrinsic noise only: B > 10¹⁴.
  - hdγ vs curnγ: **B = 200 with 14 Fourier frequencies, 1000 with 5**. Both use variable-γ
    power laws.
- Background distributions:
  - phase-shift null: p ≈ 10⁻³;
  - sky-scramble null (App. F, 5000 scrambles, |M̄| < 0.17): **p = 1.6 × 10⁻³**.
- Amplitudes: A_HD = 6.4 (+4.2, −2.7) × 10⁻¹⁵ with γ_HD = 3.2 ± 0.6; A_HD,13/3 = 2.4 (+0.7, −0.6) × 10⁻¹⁵.
- Data model:
  - per-backend EFAC/EQUAD/ECORR, **fixed at single-pulsar MAP values** in the GWB analysis;
  - DM variations modelled with **DMX** (piecewise constant);
  - one power-law red-noise process per pulsar (log10 A ∈ U[−20, −11], γ ∈ U[0, 7]).
- The DMGP alternative changes the result negligibly (their §5.1).
- A 2026 reanalysis with custom chromatic models (arXiv:2606.28554) gets B = 1571 ± 14. That
  is out of scope (§7).

**What Argus has** (main at 9f5736f):
- **Validated on MDC2 under the current configuration:** diffuse timing prior, true-uniform
  red-noise prior, ridge GW basis, marginalised filter, frozen path-sampling evidence
  procedure (`configs/frozen/evidence_procedure_v1.json`).
  - 1b (PR #123): lnB(HD/CURN) = +0.69. The true ORF's eps0 integrand (0.993) sits at the
    88th percentile of 1000 sky scrambles (p = 0.12), so the +0.69 is about 1σ and not
    significant. The no-injection control passes.
  - 2b (PR #124): −0.25. The GW-vs-noise result matches Hazboun et al.
- **Filter:** one scalar measurement per pulsar per time step (`model.py:124-249`). R is
  diagonal with a single EFAC/EQUAD per pulsar (`model.py:110-121`), and there is no ECORR.
  Masks support union grids (`jax_kalman_filter.py:209-220`, `:429-437`).
- **Marginal timing filter:** propagates the 4N dynamic states plus accumulators A, b, c over
  the timing columns (`_run_kalman_filter_marginal`, `jax_kalman_filter.py:583-730`). Cost per
  step is O((4N)³ + (4N)² M_sum + N M_sum²).
- **Real-data history:**
  - 6-psr wideband NG15 amplitude recovery (T3.3).
  - The 68-psr wideband union grid exists (`data/aligned_full/`, 189 epochs, 41.6% occupancy,
    M_sum = 994). Its joint NUTS run never converged, but that predates the ridge, diffuse and
    uniform-prior fixes.
  - **The only cost measurements at this scale are on the wrong path.** They come from
    `notes/T3.5_status_and_next_steps.md` and were taken on the *sequential* augmented-state
    filter (nx = 1266), before the masked-marginal port:
    - 0.26 s per likelihood-only evaluation on one A100 (job 14173193);
    - the quick-look run (17.7 h for 800 iterations at tree depth ≤ 7), which implies
      ≳ 0.6 s per value-plus-gradient evaluation.
  - The marginal path is ~5.4× cheaper on masked data (log.md, masked-marginal entry).
    Neither number is a baseline for D2. Task 2.6 measures the real one.
  - Wideband ingest drops all DMX columns (`build_aligned_feathers.py:203-231`). After binning
    to one point per epoch DMX is degenerate, and dropping it hides red power
    (`notes/RESULTS_ou_adequacy_ng15.md`). That defect is one motivation for this spec.
- **Prior failure that constrains this spec:** PR #100 ran Argus at TOA level on Discovery's
  narrowband feathers. It ran out of memory (~202 GB at 6 pulsars) and, after trimming, took
  ~250 s/iter. D2 is the answer to that.
- **Discovery** (github.com/nanograv/discovery) ships the 67 NG15 narrowband pulsars as
  feathers:
  - TOAs, residuals, the full design matrix including DMX, backend flags and radio
    frequencies;
  - the published per-backend noise dictionaries;
  - the published posterior chains.
  - The pyarrow-only reader `scripts/ingest_discovery_feather.py` and an F0 catalog are on
    the unmerged PR #100 branch. `discovery` is not installed in the `Argus` env (jax is
    pinned at 0.4.38).

## 3. Scientific validity

**Model.**
- Per pulsar, Argus models the residual as the sum of:
  - a GW term: integrated OU with a common corner γa, HD-correlated across pulsars;
  - spin red noise: integrated OU with (γp, σp) per pulsar;
  - a linear timing model, marginalised under the diffuse (flat) prior;
  - white noise.
- The GWB evidence is lnB(HD/CURN), obtained by path sampling over the correlation path
  C(ε) = (1−ε)I + ε C_HD using the frozen estimator.
- This spec adds:
  - (i) **per-backend white noise and ECORR**, fixed at NANOGrav's published values;
  - (ii) **DMX** retained in the timing model;
  - (iii) an **exact epoch-compression** step (D2), which feeds the existing one-scalar-per-step
    filter.

**Assumptions and approximations.**
- **A1. The state is constant within a compression window.** Each window is one DMX window,
  or a grid cell of width Δ. The GW and red-noise states are taken as constant within it.
  - At nHz frequencies, where the GW corner sits below 1/T, the *fractional* power error scales
    as (2π f Δ)² with f ≲ 1/T. For Δ = 7 d, T = 16.03 yr that is ≈ 6 × 10⁻⁵, and it grows as Δ².
  - **That fraction does not bound the error in nats.** What C1 tests is the within-window
    signal drift compared with each window's white-noise σ_w, summed over ~10³ windows.
    - For a strong-red-noise pulsar (µs-level red noise) at Δ = 7 d, the drift can be tens of
      ns. That is comparable to σ_w, so the error can exceed C1's 0.01-nat tolerance.
    - So the passing Δ may be set by the reddest pulsar, not by the GW.
  - The assumption also breaks for red-noise power near 1/Δ. A shallow-spectrum pulsar has
    proportionally more such power, so C1 includes one explicitly.
  - Measured in Phase 2 (C1 against a continuous-time oracle). There is no a-priori estimate
    we trust.
- **A2. White noise is fixed** at NANOGrav's MAP noise dictionaries. This is exactly
  NANOGrav's own GWB practice. It is not a simplification relative to the target.
- **A3. Single-corner OU red noise** (residual PSD between f⁻² and f⁻⁴, i.e. a
  power-law-equivalent slope γ ∈ [2, 4]) stands in for NANOGrav's per-pulsar power laws
  (γ ∈ [0, 7]) and their common process (γ free).
  - For the common process the mismatch is mild: NANOGrav's measured γ_HD = 3.2 ± 0.6 lies
    inside [2, 4]. 13/3 lies just outside, so we make no spectral-index claim.
  - The real exposure is per-pulsar shallow spectra (γ_PL ≲ 1), which is 5 of 25 red-noise
    pulsars in the wideband study (`RESULTS_ou_adequacy_ng15.md`). There, red power the
    kernel can't absorb can leak into the common term.
  - This is the main scientific risk (§10). It is measured directly on real data in Phase 1 by
    running Discovery with OU-shaped spectra.
- **A4. The diffuse timing prior** matches NANOGrav's convention: enterprise uses ~10⁴⁰-variance
  priors on timing parameters, which is effectively flat.
- **A5. DMX windows are taken from NANOGrav's timing solution as-is.** We do not re-time or
  re-window.

**Correctness checks** (each is a gate in §11):
- **C1. Compression is accurate** (exact up to A1). For each of 3 pulsars, compare the Argus
  compressed likelihood with a dense brute-force Gaussian likelihood on the raw TOAs.
  - The dense calculation uses:
    - the **continuous-time** OU covariance evaluated at every raw TOA time, i.e. the analytic
      kernel, *not* the filter's gridded transitions, so that the oracle does not share A1;
    - per-backend white noise plus ECORR blocks;
    - the full M including DMX, after the same rank rule as D2;
    - the diffuse prior.
  - Tolerance: after removing one parameter-independent constant, the spread of
    ΔlnL across 20 random parameter points is **≤ 0.01 nats**.
  - Gradients agree to a relative error of **≤ 1e-4**.
  - The pulsars are:
    - J1909−3744 (high precision, little red noise);
    - one strong-red-noise pulsar (J1713+0747 or B1937+21);
    - one shallow-spectrum pulsar, J1012+5307 (γ_PL ≲ 1 in `RESULTS_ou_adequacy_ng15.md`),
      because A1 is most exposed there.
- **C2. The goldens are preserved.** With no compression and zero initial accumulators:
  - the MDC2 goldens pass their existing tests: 63618.93 (informative) and 59420.06
    (diffuse), in `test/test_likelihood_value.py`, `test/test_masked_marginal_filter.py` and
    `test/test_correlation_path.py`;
  - **and** a new test asserts exact equality (`==`) between the log-likelihood with default
    zero accumulators and without the new argument. The golden tests use atol = 1.0, so they
    alone cannot show bit-for-bit.
- **C3. The reference reproduces the publication.** Discovery with NANOGrav's model (power-law
  red noise, 30 modes; curnγ vs hdγ with 14 modes; fixed white noise), using our path-sampling
  estimator, gives lnB(HD/CURN) within **±0.7 nats (a factor of 2) of ln 200 = 5.30**.
  - It also gives A_HD,13/3 within 1σ of 2.4 × 10⁻¹⁵.
  - The ±0.7 tolerance was confirmed by the user on 2026-10-15.
- **C4. Single pulsars match the reference.** For every pulsar, Argus's posterior red-noise
  PSD at f = 1/T and 1/(5 yr) matches Discovery's under the OU-shaped red-noise model to
  within 1σ, for ≥ 90% of pulsars. Every outlier gets a written diagnosis.
  - The published power-law chains are a qualitative cross-check only: a different kernel
    means a different posterior.
- **C5. Method agreement.** Argus's 67-psr lnB(HD/CURN) agrees with Discovery under the
  kernel-matched model (OU spectra, same priors, same estimator) to within **1 nat**.
  - The two are not expected to agree exactly. The Fourier basis truncates the spectrum,
    imposes periodicity and misses power below 1/T; the time-domain process does none of
    these.
  - > ⚠️ DRAFT — revise: 1 nat is a guess at the Fourier-vs-time-domain discrepancy. Phase 1 can measure part of it (vary the mode count in the OU reference) before this is frozen.

## 4. Approach

**D1. Use the narrowband NG15 data from Discovery's feathers, and drop wideband.**
**Why:** these are the data behind the published claim. They carry the per-backend flags,
radio frequencies, DMX columns and noise dictionaries, and a published posterior baseline.
Wideband has no published GWB analysis to compare against, and our wideband ingest drops DMX.
**Alternatives considered:**
- *Wideband plus our own reference run.* No published target, and the DMX defect remains.
- *Re-ingest the narrowband par/tim through enterprise.* Duplicates Discovery's feathers and
  risks a version mismatch.
- **The PLAN.md rule "do not use PR #100's adapter" is superseded.** We write a new ingest. We
  may borrow PR #100's pyarrow schema handling and F0 catalog, but not its white-noise folding,
  because it drops ECORR.

**D2. Exact epoch compression with local DMX marginalisation, instead of per-TOA filtering.**
Within a window w, the model is

  z_w = 1·s_w + D_w d_w + M_w β + n_w,  with n_w ~ N(0, N_w) and N_w = diag(EFAC_b²(σ² + EQUAD_b²)) + ECORR blocks, all fixed.

- d_w are the window's DMX parameters. They are local, so they can be marginalised locally
  under the flat prior.
- s_w is the achromatic state contribution: the filter's existing H row times the dynamic
  state.
- After projecting out D_w, the data split exactly (in the projected-noise metric) into two
  parts:
  - a scalar ŝ_w = s_w + m_wᵀβ + e_w, where e_w ~ N(0, σ_w²) and ECORR is included in σ_w²;
  - an orthogonal remainder that depends only on β. It is static, so its contribution to the
    marginal filter's (A, b, c) accumulators is a **constant, precomputed once**.
- The filter keeps one scalar per pulsar per step. Its H row is [−1 on GW r, 1/f0 on phase, m_w].
  The only library change is that the marginal filter accepts initial accumulators
  (A₀, b₀, c₀). With zero defaults this preserves the goldens.
- M_sum stays at the non-DMX timing columns, roughly the ~994 we have today. That avoids the
  ~10⁴ dense columns that caused PR #100's out-of-memory failure.
- Windows of one pulsar that fall in the same grid cell are merged by the same algebra. The
  only approximation is A1.
- **What D2 makes exact is the DMX, ECORR and per-backend bookkeeping, not the time
  resolution.** The step count, and so the cost, is set by the grid width Δ. K3 will likely
  force Δ to 2–4 weeks, which is the same time-gridding approximation the wideband pipeline
  already makes.
- Four ingest edge cases need an explicit rule:
  - **Single-frequency windows: drop.** A window observed at a single radio frequency has
    D_w exactly ∝ 1. Projecting out DMX then also removes s_w, so the window carries no
    achromatic information.
    - Dropping it is exact, and matches what NANOGrav/Discovery effectively see.
    - Keeping it without DMX would change the model relative to the reference, so it is not
      allowed.
    - Near-single-frequency windows need no rule: their large σ_w² already down-weights them.
  - **Several ECORR epochs inside one window.** N_w is block-structured, and the projection
    must use the full N_w metric.
  - **An ECORR epoch straddling two windows or grid cells.** The noise is then correlated
    across windows, and the independent scalar per window is no longer exact. Merge the
    affected windows into one before compressing.
  - **Rank rule for the timing model.** A global chromatic column lies (nearly) in the span of
    the DMX columns when the windows tile all TOAs: DM exactly, DM1/DM2 approximately.
    - After local projection such a column is ~0, which makes A₀ + A singular and breaks the
      −½ log|A| term of the diffuse marginal.
    - Rule: after projection, drop M columns whose projected norm falls below a relative
      tolerance (SVD on the stacked projected M). Record the dropped columns per pulsar.
    - The dense oracle (C1) applies the same rule. A column that is fully degenerate with
      DMX is unconstrained by the data under the flat prior, so dropping it changes lnL only
      by a constant.
- **Lock-in:** fixed white noise is baked into the preprocessing. Sampling white noise later
  would need recompression at every sample.

**Why:** the information is the same as per-TOA filtering, up to A1, at the cost of the
current filter. It is effectively what wideband timing does at the profile level, but done
inside the likelihood with exact bookkeeping.
**Alternatives considered:**
- *Per-TOA sequential filtering.* About 10⁵–10⁶ steps, and the PR #100 out-of-memory failure.
- *Vector observations per window* (several TOAs per update). Cost is O(m_w³) per step, and DMX
  would still be in M_sum.
- *Keep DMX in M_sum.* About 10⁴ columns gives a dense 10⁴ × 10⁴ A carried through reverse-mode
  autodiff, which is that same failure again.
- *A DMGP state.* A new chromatic process, and NANOGrav's headline uses DMX anyway.

**D3. Use Discovery as a same-data, same-estimator reference, in a separate conda env.**
- Run it with two noise models:
  - (a) NANOGrav's power laws, to reproduce the publication (C3);
  - (b) OU-shaped PSDs for both per-pulsar and common processes, matching Argus's model (C4, C5).
- Both use our path-sampling estimator over the ε-ORF path, so the estimator is identical
  across pipelines.

**Why:**
- (a) checks the data and the estimator independently of Argus.
- (a) vs (b) is the **kernel systematic on real data**. It replaces OpenSpec group 5's MDC2
  injection study and is far cheaper.
- (b) vs Argus is the **method check**.

**Alternatives considered:**
- *A product-space hypermodel* like NANOGrav's. It would use a different estimator, so a
  disagreement could not be attributed.
- *The published numbers only.* A shortfall could not be interpreted.

**D4. Joint sampling of per-pulsar OU red noise with the GW, under true-uniform priors, with
fixed white noise.**
**Why:**
- This is NANOGrav's structure.
- It has been validated on MDC2 33-psr: clean, with 0 divergences.
- The empirical two-stage priors absorbed the GWB (dead, `project_two_stage_priors_absorb_gwb`).

There are 136 sampled dimensions at 67 psr: 2 × 67 red-noise + 2 GW.
**Alternatives considered:**
- *A long-baseline subset only.* Kept as the fallback in K4, not as the plan.

**D5. Fix issue #125 (Gaussian-not-uniform priors) for the GW sites only, before the array runs.**
- The affected sites are `log10_pivot_psd` and `log10_gamma_a`.
- Update `lnb_gw_vs_noise.py` for the bounded pivot.
- **EFAC/EQUAD stay fixed (A2), so those sites are out of scope.**

**Why:** with lnB around 5, the posterior is likelihood-dominated, but the curnγ-vs-noise
estimator and the amplitude intervals are not. The fix also aligns the NUTS prior with
NANOGrav's log-uniform amplitude prior.

**D6. Assess significance on the eps0-integrand statistic against a sky-scramble null.**
- The correlation detection statistic is ⟨∂lnL/∂ε⟩ at ε = 0 under the true ORF.
  - It is Argus's analogue of NANOGrav's noise-marginalised optimal statistic: linear in the
    ORF and averaged over the CURN posterior.
- Its null comes from ≥ 5000 sky scrambles with |M̄| < 0.17, scored by the same code
  (`scripts/scramble_ensemble_eps0.py`) on the same ε = 0 draws.

**Why:**
- This matches NANOGrav App. F, which reports both a Bayes-factor p and an optimal-statistic p.
- It is cheap (CPU).
- Phase shifts are a Fourier-basis construct with no natural time-domain analogue.

**Alternatives considered:**
- *A null on lnB itself.* It needs a full ladder per scramble, ~800 A100-h each (§5).
  Resolving p ≈ 10⁻³ needs thousands of scrambles, i.e. ≳ 10⁶ A100-h. Even 50 ladders is
  ~4 × 10⁴ A100-h. Unaffordable.
- The scrambled-lnB proxy is still unverified at lnB level (open handoff item), so it is not
  used for the claim.

**D7. Delete the OpenSpec machinery.**
`spec/spec.md` becomes the single source of truth. Issue #111 is updated to point here.

## 5. Data and reproducibility

- **Input data.** NG15 narrowband feathers from the Discovery repo.
  - Pin the exact repository commit and record SHA-256 hashes of every feather in
    `data/ng15_nb/MANIFEST.json`.
  - The data can be re-downloaded and are not irreplaceable.
  - Ingested caches go in `workflows/ng15_nb/data/` (gitignored); the scripts that regenerate
    them are committed.
- **Noise dictionaries:** from the same feathers, with no refitting.
- **F0:** from the catalog built off the NG15 par files (`reference_ng15_data_paths`).
- **Environments:**
  - `Argus` env: jax 0.4.38, unchanged.
  - A new `discovery` env with versions pinned in `envs/discovery.yml`. The two envs exchange
    only feathers, JSON and `.nc` files.
- **Seeds:** NUTS seeds are fixed per run in the config, and checkpointing is enabled
  (interval 250).
- **Provenance:** the run harness calls `stamp_provenance.py`. The frozen evidence procedure
  is unchanged; it is extended only via a v2 file if needed.
- **Library pinning:** every production run uses a pinned worktree, recorded in the run's
  output (`project_argus_editable_install_gotcha`).
- **Compute** (rough):
  - Discovery reference: ~50 A100-h.
  - Argus single-pulsar stage: ~70 jobs × ~1 A100-h.
  - Argus 67-psr ladder: 5 rungs × ~40 h × 4 A100 ≈ 800 A100-h. This assumes ≲ 0.5 s per
    likelihood-plus-gradient evaluation; Phase 2 measures it.
  - Not yet counted, and to be added once measured:
    - extra Romberg rungs if the readout asks (0.625, 0.875; ~2 × 160 A100-h);
    - the Phase 4 MDC2 reruns (2 rungs at 33 psr);
    - the Phase 3 Discovery single-pulsar runs.
  - Scrambles: CPU.
  - Total ≈ 1000 A100-h **at ≤ 0.5 s per evaluation, with no extra rungs**.
  - Cost scales linearly with evaluation time, and evaluation time scales with the number of
    grid steps. The grid width Δ is therefore the budget dial.
  - **There is no trustworthy per-step rate on the marginal path yet** (§2: the T3.5 numbers
    are sequential-filter numbers). If the per-step value-plus-gradient cost turns out
    comparable to the sequential quick-look (≳ 0.6 s at 189 steps), the 0.5 s target already
    binds at Δ ≈ 1 month. Task 2.6 settles this before any compute is committed.
  - K3 is set to match this budget.
  - **Budget: ~1000 A100-h, confirmed by the user on 2026-10-15.** K3 stops and re-budgets
    above 1 s per evaluation.

## 6. Outputs

| Output | Supports |
|---|---|
| T1. Table of lnB(HD/CURN), lnB(CURN/noise) and amplitude for: published, Discovery-PL, Discovery-OU, Argus | §1, all three comparisons |
| F1. Sky-scramble null distribution of the eps0-integrand statistic for Argus, with the observed value marked and NANOGrav's optimal-statistic and Bayes-factor p quoted | significance claim |
| F2. Correlation-path integrand ⟨∂lnL/∂ε⟩ vs ε, Argus vs Discovery-OU | method agreement (C5) |
| F3. Common-process PSD posterior vs frequency: Argus, Discovery-OU, Discovery-PL, published | amplitude consistency; kernel systematic |
| F4. Per-pulsar red-noise PSD at 1/T: Argus vs Discovery-OU (scatter, 67 points) | C4 |
| N1. Compression-accuracy note: C1 ΔlnL spread vs window width Δ, plus cost per evaluation | D2 validity |
| N2. Kernel-systematic note: Discovery-PL vs Discovery-OU lnB and amplitude | A3 scope statement |

## 7. Non-goals

- Custom chromatic or DMGP noise models, solar-wind models, or the J1713+0747 dip models.
  We use DMX as in the 2023 analysis and do not chase B = 1571.
- Sampling white noise. EFAC/EQUAD/ECORR are fixed, and the #125 EFAC/EQUAD sites are left
  alone.
- Claims about the GWB spectral index. OU cannot express γ = 13/3, so we claim amplitude and
  correlation only.
- The wideband data set, and any further work on the wideband pipeline.
- Re-timing pulsars or re-deriving DMX windows.
- Phase-shift nulls, free-spectrum or broken-power-law GW models, and monopole/dipole searches.
- CW searches, and #108 (parallel filter).
- Product-space or nested-sampling evidence.
- Per-TOA filtering, unless D2 fails.

## 8. Kill criteria

The user's stance is to refine capabilities until the claim is reached. Most criteria are
therefore **pivots** with a named next step, not stops.

- **K0 (success, in two parts).**
  - (a) **Method claim:** Argus agrees with Discovery-OU (C5), and its eps0-integrand
    statistic beats a fraction ≥ 1 − p of the scrambles, with p ≤ 5 × 10⁻³.
  - (b) **Headline detection claim:** (a), and Argus's lnB(HD/CURN) ≥ ln 10 = 2.3. Report
    lnB ≥ ln 100 = 4.6, the same order as NANOGrav, as the stronger result if it is reached.
  - If (a) holds but (b) fails *because Discovery-OU itself falls short of Discovery-PL* (K2
    territory), then the method is validated and the **kernel** is the limit. Extend the kernel
    (the K2 pivot); don't re-engineer the filter.
- **K1 (Phase 1). The reference fails to reproduce the publication** (C3 misses). Stop and debug
  data or estimator before any Argus work, since Argus cannot be expected to beat a
  reproduction that fails.
- **K2 (Phase 1). The kernel cannot carry the detection.** If Discovery-OU gives
  lnB(HD/CURN) < ln 10 while Discovery-PL passes C3, pivot to a kernel extension before Phase 3.
  - The extension is the phase-OU mixture, which spans residual slopes 0–4
    (`project_ou_adequacy_ng15`), or a CARMA sum of OUs.
  - It gets its own scoped spec.
- **K3 (Phase 2). Compression fails.** Two ways:
  - C1 fails even at Δ = the DMX window (the finest Δ the compression allows): pivot to vector
    observations per window.
  - The cost per 67-psr likelihood-plus-gradient evaluation on one A100 is > 0.5 s at every Δ
    that passes C1: coarsen Δ (re-measuring A1), or revisit the #108 parallel filter, whose
    crossover needs exactly the larger T that a fine grid produces.
  - The cost is > 1 s even at the coarsest passing Δ: stop and re-budget with the user.
- **K4 (Phase 5). The 67-psr joint posterior won't sample.** That means the pilot still fails
  the Phase 5 Done-when thresholds (r̂ ≤ 1.01, divergences ≤ 0.5%, bulk ESS ≥ 400, no frozen
  chain) after three levers. The thresholds are the same, so there is no limbo band.
  - The levers are: a dense GW block, target_accept 0.95, and tree depth 8.
  - T3.5 capped tree depth at 7 only to finish within walltime. Checkpointed segmented runs
    now remove that reason.
  - Pivot to the ≥ 10 yr long-baseline subset as an interim result, and open a geometry
    investigation.
- **K5 (no hard stop).** The user decided on 2026-10-15 to set no time box. The campaign keeps
  refining through the K1–K4 pivots. The fallback write-up is the Discovery-OU kernel result
  (N2) plus a long-baseline-subset Argus result, to be revisited at each phase boundary rather
  than triggered automatically.

## 9. Open questions

- [NEEDS CLARIFICATION: the Discovery repo commit to pin, and whether its feathers are NG15
  v1.1.0, the version in the paper. Resolved in task 1.1.]
- [NEEDS CLARIFICATION: DMX window lengths in NG15. If windows are routinely longer than ~1 d
  (several receivers on different days), A1's Δ is set by the windows, not by choice. Measured
  in task 2.1.]
- [NEEDS CLARIFICATION: Discovery's ability to take an arbitrary PSD (the OU form) and a
  fixed-ε mixed ORF. Expected yes, since PSDs and ORFs are plain JAX functions. Checked in
  task 1.3.]
- [NEEDS CLARIFICATION: whether Argus's common process should keep its free corner γa (≈ curnγ
  with free γ) or fix it to mimic hd13/3. Default: free, matching NANOGrav's headline hdγ/curnγ.]

## 10. Risks

| Risk | Early warning |
|---|---|
| OU kernel too stiff, so the HD evidence is diluted (A3) | Discovery-OU lnB ≪ Discovery-PL lnB in Phase 1 (K2) |
| Compression error at shallow-spectrum pulsars (A1) | C1 spread grows with red-noise slope; per-pulsar C4 outliers concentrated in γ_PL ≲ 1 |
| 136-D sampling geometry | Phase 5 pilot (ε=0, 500+500) step size < 0.01, or per-chain medians split |
| Per-step cost blows up at 67 psr on the union grid (step count × O((4N)³)) | Task 2.6 benchmark of value-plus-gradient on the marginal path. There is no trustworthy prior rate (§2); if it is near the sequential quick-look (≳ 0.6 s at 189 steps), Δ is forced to ≳ 1 month |
| Discovery env / jax version conflicts | Task 1.2 fails to install; fall back to a container |
| Path-sampling integrand steep near ε=1 at lnB ~ 5 | Frozen-procedure Romberg check flags extra rungs (0.625, 0.875) |
| Fourier-vs-time-domain gap makes C5 unattainable even when both are right | Discovery-OU lnB moves by > 1 nat when its mode count changes (task 1.5) |

## 11. Task list

### Phase 0 — Housekeeping

**Goal:** retire the superseded plans so this spec is the single source of truth.
**Done when:** `openspec/` is gone from main, issue #111 points at `spec/spec.md`, and PLAN.md
carries a superseded banner.
**PR:** #127 (merged 2026-10-16)

- [x] 0.1 Delete `openspec/` (change + archive) in one commit; record in its message which
      tasks were done (groups 1–4) — verify by `git ls-files openspec | wc -l` = 0 and tests still pass.
      _(Note, 2026-10-16: there was no archive, only the change and `config.yaml`. Group 4
      was done except 4.3/4.4/4.6/4.7, which were deferred. The commit message records the
      exact state.)_
- [x] 0.2 Add a "superseded by spec/spec.md" banner to `workflows/ng15_sgwb_demo/PLAN.md` and
      the top of `TASKS.md` — verify by grep for the banner.
- [x] 0.3 Comment on issue #111 pointing at the spec — verify by `gh api` read-back.

### Phase 1 — Reference baseline in Discovery (the cheapest killer)

**Goal:** establish the same-data target, and measure whether an OU kernel can carry the HD
evidence on real NG15 at all, before any Argus engineering.
**Done when:** C3 passes, and N2 records Discovery-OU lnB(HD/CURN) and lnB(CURN/noise), with K2
evaluated.
**PR:** _not started_

- [ ] 1.1 Fetch Discovery at a pinned commit; write `MANIFEST.json` (commit, feather hashes,
      67 pulsar names, data-set version) — verify by 67 feathers hashed and version string recorded.
- [ ] 1.2 Create and pin the `discovery` conda env (`envs/discovery.yml`) — verify by importing
      it and evaluating the published single-pulsar likelihood for J1909−3744.
- [ ] 1.3 Build the reference driver `workflows/ng15_nb/reference/` with: (a) the NANOGrav
      power-law model (30-mode RN, 14-mode common, fixed WN) and (b) the OU-PSD model using
      Argus's exact one-sided PSD conventions (sidedness note); fixed-ε ORF (1−ε)I + εHD.
      There is no `ou_psd` in `python/argus`. The references to match are:
      - the common process in the ridge basis:
        `workflows/ng15_sgwb_demo/scripts/diag_ou_powerlaw_capture.py::ou_psd`;
      - per-pulsar red noise: `scripts/ng15_red_noise_budget.py::ou_psd_onesided`.
      Before use, first check both against the filter's own stationary Q (the ground truth).
      — verify by the OU PSD functions matching those references to 1e-12, and ε=1
      reproducing Discovery's built-in HD likelihood to 1e-8.
- [ ] 1.4 Run the PL ladder (5 rungs, variable γ) and apply our path-sampling estimator. Add
      one fixed-γ = 13/3 HD run for the A_HD,13/3 check: conditioning the variable-γ posterior
      at 13/3 is not the same fit. — verify against C3 (lnB within ±0.7 of 5.30; A_HD,13/3
      within 1σ).
- [ ] 1.5 Run the OU ladder at 14 and 30 common modes; compute lnB(HD/CURN) and lnB(CURN/noise)
      — verify by the reliability gate passing; record the mode-count sensitivity (input to C5).
- [ ] 1.6 Write N2 (kernel systematic: PL vs OU lnB and amplitude at 1/T and 1/(5 yr)) and
      evaluate K2 — verify by the committed note stating the K2 outcome explicitly.

### Phase 2 — Exact epoch compression (S0 feasibility)

**Goal:** show the narrowband TOAs can be fed to the existing filter exactly (C1) and cheaply.
**Done when:** C1 passes for all three test pulsars at the chosen Δ, C2 passes, and the 67-psr cost
benchmark is ≤ 0.5 s per evaluation, with K3 evaluated.
**PR:** _not started_. **Runs in parallel with Phase 1**: the two are independent until K2 is
evaluated, and Phase 1 has an unverified dependency (Discovery accepting a custom PSD and a
mixed ORF) that could stall it.

- [ ] 2.1 Narrowband ingest from the Discovery feathers (pyarrow only) into an Argus-side cache
      that keeps per-TOA backend, radio frequency, ECORR epoch id, DMX window id and the full
      M with column names — verify by round-tripping two pulsars and checking TOA counts and
      column names against the feather. Report the DMX-window length distribution, the number
      of single-frequency windows (dropped, D2), the number of windows that span more than
      one ECORR epoch, the number of ECORR epochs that straddle a window boundary (merged,
      D2), and the timing-model columns dropped by the rank rule.
- [ ] 2.2 Implement compression (D2) as a pure numpy/JAX preprocessing module with unit tests on
      a synthetic pulsar where the exact answer is known. The synthetic pulsar includes a
      window with two ECORR epochs, an ECORR epoch straddling two windows, a single-frequency
      window, and a DM column degenerate with DMX — verify by matching the compressed vs dense
      likelihood on the synthetic case to 1e-10, with a constant state per window so A1 holds
      exactly.
- [ ] 2.3 Add initial accumulators (A₀, b₀, c₀) to `_run_kalman_filter_marginal` — verify by C2
      (the golden tests pass, plus a new exact-equality test of zero accumulators against the
      unmodified call) and a test showing nonzero A₀ equals appending the equivalent rows.
- [ ] 2.4 Build the dense brute-force oracle: raw TOAs, the **continuous-time** analytic OU
      covariance at each TOA time (not the gridded F/Q), WN + ECORR, full M after the rank
      rule, diffuse — verify by checking it on the synthetic pulsar against 2.2, and on a
      single-step grid against the filter.
- [ ] 2.5 Run C1 on J1909−3744, one strong-red-noise pulsar and J1012+5307 (shallow) at
      Δ ∈ {DMX window, 7 d, 14 d, 30 d} — verify by the ΔlnL spread and gradient error tables in N1; choose the largest Δ that passes.
- [ ] 2.6 Compress all 67, build the union grid with masks, benchmark the likelihood plus
      gradient on one A100 — verify by the recorded seconds per evaluation and memory; evaluate K3.

### Phase 3 — Per-pulsar validation

**Goal:** every pulsar's Argus noise posterior is consistent with the kernel-matched reference.
**Done when:** C4 passes (≥ 90% within 1σ), and every outlier has a one-line diagnosis.
**PR:** _not started_

- [ ] 3.1 Single-pulsar Argus NUTS for all 67 (SLURM array, flat OU priors, fixed WN) — verify
      by 100% meeting r̂ ≤ 1.01, ESS ≥ 400 and 0 divergences. No pulsar with
      detected red noise may rail against a prior edge. For pulsars without detected red noise,
      piling up at the lower amplitude edge is expected; report it, don't fail on it.
- [ ] 3.2 Matching single-pulsar Discovery-OU runs — verify by the same diagnostics.
- [ ] 3.3 Produce F4 and the outlier table, flagging shallow-spectrum pulsars — verify against C4.

### Phase 4 — GW priors to true uniforms (#125, GW sites only)

**Goal:** the NUTS GW prior equals the configured box.
**Done when:** prior-predictive draws of `log10_pivot_psd` and `log10_gamma_a` match U(lo, hi)
(KS p > 0.05 on 10⁵ draws), `lnb_gw_vs_noise.py` handles the bounded pivot, the goldens pass,
and the MDC2 1b eps0 and eps1 rungs rerun with the change in lnB reported.
**PR:** _not started_ (can run in parallel with Phases 2–3)
**Divergence from #125's stated acceptance:** this phase covers the GW sites only (EFAC/EQUAD
stay fixed, A2) and reruns only the eps0 and eps1 rungs, not both full MDC2 ladders. Edit #125
to match when the phase opens.

- [ ] 4.1 Convert both sites following the PR #120 pattern — verify by the KS test and goldens.
- [ ] 4.2 Update `lnb_gw_vs_noise.py` prior mass to (c − lo)/(hi − lo); refuse an unbounded
      prior — verify by a unit test with a known-answer toy.
- [ ] 4.3 Rerun MDC2 1b eps0 and eps1 rungs; report Δ in lnB(GW/noise) and amplitude interval —
      verify by a committed comparison JSON; flag old checkpoints as incompatible (PR #121 style).

### Phase 5 — 67-psr Argus pilot (sampling feasibility)

**Goal:** show the full-array joint posterior samples under the production configuration.
**Done when:** one ε=0 rung (1000+1000, 4×A100, checkpointed) meets max r̂ ≤ 1.01,
divergences ≤ 0.5% and min bulk ESS ≥ 400, with no frozen chain, and K4 evaluated.
**PR:** _not started_

- [ ] 5.1 Write the production config (ridge, diffuse, flat RN, uniform GW, fixed WN, compressed
      data, chosen Δ) and freeze it as `configs/frozen/ng15_nb_v1.ini` — verify by a config
      diff against the MDC2 diffuse config showing only intended differences.
- [ ] 5.2 Short probe (200+200) to measure the step size and seconds per iteration, then the full
      pilot — verify against the Done-when thresholds and a per-chain freeze check.

### Phase 6 — Detection run, significance and comparison

**Goal:** the headline numbers in T1 and F1–F3.
**Done when:** the frozen estimator reports `reliable: true`, K0 is evaluated, C5 is evaluated,
and the scramble p is reported at its resolution.
**PR:** _not started_

- [ ] 6.1 Remaining ladder rungs (ε = 0.25, 0.5, 0.75, 1; extra rungs if the readout asks) —
      verify by each rung meeting the Phase 5 thresholds.
- [ ] 6.2 Path-sampling readout plus `lnb_gw_vs_noise.py` — verify by `reliable: true` and the
      plateau/KS gates passing.
- [ ] 6.3 Sky-scramble ensemble (≥ 5000, |M̄| < 0.17) of the eps0-integrand statistic on the
      ε=0 draws, with the observed value computed by the same code on the true ORF — verify by the
      p-value with its binomial uncertainty, and the ensemble file hashed into provenance.
- [ ] 6.4 Assemble T1, F1–F3, and evaluate K0 and C5 — verify by a committed results note
      quoting every number with its source run.

### Phase 7 — Write-up

To be specified when reached: the Paper A draft, plus the result artefact carrying the
kernel statement (N2) and the scope bound.

## 12. Amendment log

### 2026-10-16 — pre-merge review fixes (PR #127)

**Trigger:** the `/check-PR` review of the PR that first lands this spec
(`PR_REVIEW_127_2026-10-16.md`) found 4 major and 13 minor issues. The main ones:
- the cost baseline came from the sequential filter;
- C1's oracle could share A1;
- task 3.1 was unpassable for pulsars with no red noise;
- DM is degenerate with DMX after projection.

**Change:**
- §1: the significance claim is set against NANOGrav's Bayes-factor scramble p, and we
  disclaim parity with their optimal-statistic p.
- §2: the 88th percentile is attributed to the eps0 integrand; the T3.5 timings are
  relabelled as sequential-filter numbers.
- A1: the estimate is corrected to ≈ 6e-5 and no longer read as a nats bound.
- C1: the oracle uses continuous-time OU; a shallow-spectrum pulsar (J1012+5307) is added.
- C2: an exact-equality test is added.
- D2: drop single-frequency windows, merge windows that share an ECORR epoch, and add a rank
  rule for degenerate timing columns.
- D6: scramble-cost arithmetic corrected.
- §5: no assumed per-step rate; uncounted compute listed.
- K3: the C1 trigger is the DMX-window Δ.
- K4: the same thresholds as the Phase 5 Done-when; tree depth 8 justified by checkpointing.
- §10: the cost-risk row is updated.
- Tasks 0.1 (annotated), 1.3, 1.4, 2.1–2.5, 3.1 and the Phase 2 and Phase 4 Done-when are
  updated to match.

**Defaults taken without a user decision** (the user said "go" without picking):
- cost re-baselining is deferred to task 2.6;
- §1 is toned down rather than adding scrambles;
- J1012+5307 is the shallow pulsar.

**Sections touched:** §1, §2, §3, §4, §5, §8, §10, §11

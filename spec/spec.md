# Spec: First state-space detection of the SGWB on NANOGrav 15-yr data

**Status:** draft
**Created:** 2026-10-15  ·  **Last amended:** 2026-10-22
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
  - **Caveat (2026-10-17):** every diffuse result from PR #107 (where the diffuse prior
    was added; it became the default in #121) to the Λ-jitter fix carries a
    parameter-dependent bias. A jitter of 1e-9·trace(Λ)/n shifted the MDC2 golden by
    −34.7 nat.
    - The shift barely depends on ε: lnL(ε=1) − lnL(ε=0) moves by ≤ 7e-3 nat at 8 fixed
      parameter points. So the lnB(HD/CURN) values above are *expected* to move by
      ≲ 0.01 nat. That is a fixed-parameter estimate, not verified by a rerun.
    - It does depend on the GW amplitude (0.26 nat across ha = 1–5e-15) and on the noise
      parameters: 28–40 nat across 60 MDC2 prior draws, a spread of ~10 nat.
    - So the GW-vs-noise and amplitude results are exposed until rerun: #122's diffuse 1b
      evidence for the #121 default switch, and #123/#124. See §12, 2026-10-17.
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
  - (iii) an **exact epoch-compression** step (D2), which feeds a sparse marginal filter
    taking one scalar observation per step (amended 2026-10-19; the stock filter takes every
    pulsar's slot at every step, see §12 2026-10-18 K3 entry).

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
  - **Measured 2026-10-18: a constant state fails.** With one scalar per DMX window, C1
    misses by 0.5–5 nat on all three pulsars. The spin phase drifts linearly within a window
    at the spin velocity, whose stationary variance σp²/2γp is large at small γp (B1937+21's
    error tracks γp with correlation 0.99). D2 therefore emits a second, velocity scalar per
    window, and A1 now reads: **the state is linear in time within a window** (constant
    velocity). It still fails where the red noise decorrelates within a window (J1012+5307,
    correlation time ≈ 12 d against 14 d windows). See §12, 2026-10-18.
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
  - Gradients agree to a relative error of **≤ 1e-4**. (amended 2026-10-18, third: the
    gradient error is reported, not gated, in the oracle check as in the epoch-cell rule;
    the compressed likelihood's gradient is exact for its own model, and the value spread
    measures how far that model is from the dense one.)
  - **The oracle's own precision** (amended 2026-10-18). Where the dense oracle is too
    ill-conditioned to resolve these bounds, the check passes when the residual is within
    the oracle's precision. That precision is measured by moving the oracle's time origin
    (exact in theory, conditioned differently): the shift in its spread and gradient is
    its noise floor. Measured: B1937+21 1.5e-2 nat / 2.3e-3, B1855+09 8e-6 / 1.1e-6.
    (2026-10-19: no pass at the adopted layout relies on this clause, since every residual is
    inside 0.01 nat outright; the origin-shift figures come from a scratch check that was not
    committed, so re-measure before relying on it. 2026-10-22: Phase 3's posterior-point
    check on B1937+21 does rely on it: residual spread 1.4e-2 nat, inside the floor
    re-measured there, 2.4e-2 nat, in `results/single_psr/argus_check_posterior/`.)
  - **Epoch cells as the per-pulsar reference** (amended 2026-10-18). Epoch cells (D2) are
    near-exact, so for pulsars outside the three test pulsars, and for those too large for
    the oracle (J1713+0747), C1 compares the window-compressed likelihood with the
    epoch-cell one. That comparison gates on the ΔlnL spread (≤ 0.01 nat) only; its
    gradient differences are reported but do not gate (amended 2026-10-18, second): the
    window likelihood's gradient is its own exact gradient, so a gradient difference is a
    model difference, which the value spread already measures.
  - The pulsars are:
    - J1909−3744 (high precision, little red noise);
    - one strong-red-noise pulsar (J1713+0747 or B1937+21);
    - one shallow-spectrum pulsar, J1012+5307 (γ_PL ≲ 1 in `RESULTS_ou_adequacy_ng15.md`),
      because A1 is most exposed there.
- **C2. The goldens are preserved.** With no compression and zero initial accumulators:
  - the MDC2 goldens pass their existing tests: 63618.93 (informative) and 59454.72
    (diffuse; 59420.06 before the Λ-jitter fix of 2026-10-17, which was biased), in `test/test_likelihood_value.py`, `test/test_masked_marginal_filter.py` and
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
  - The reference is Discovery-OU on 30 Fourier modes, as in Phase 1. Where an outlier's
    diagnosis is that truncation, a 100-mode Discovery-OU posterior confirms it; with the
    likelihood checked only at points, the diagnosis is marked likely (amended 2026-10-22).
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
  - a second, **velocity** scalar v̂_w = ṡ_w + m'_wᵀβ + e'_w along the projected (t − t_w)
    direction, with t_w the weighted mean time of ŝ_w. It is orthogonal to ŝ_w's direction
    by construction of t_w, so e'_w is independent of e_w (amended 2026-10-18; one scalar
    failed C1, see A1);
  - an orthogonal remainder that depends only on β. It is static, so its contribution to the
    marginal filter's (A, b, c) accumulators is a **constant, precomputed once**.
- The filter keeps one scalar per pulsar per step, taking each window as **two steps at t_w
  with dt = 0** between them. The position step's H row is [−1 on GW r, 1/f0 on phase, m_w];
  the velocity step's is [−1 on GW ṙ, 1/f0 on phase rate, m'_w]. This doubles the step count.
  A window with too few TOAs to fix a slope emits the position scalar only (the velocity step
  is masked).
  Two library changes (amended 2026-10-19): the marginal filter accepts initial
  accumulators (A₀, b₀, c₀), which with zero defaults preserves the goldens; and a sparse
  marginal filter, `_run_kalman_filter_marginal_sparse`, takes one scalar observation per
  step, so the step count is the observation count (15,372 at the adopted layout) and no
  step touches an unobserved pulsar (§12, 2026-10-18 K3 entry).
- M_sum holds the non-DMX timing columns, plus the DMX columns of pulsars on epoch cells
  (below): 1,080 with all windows, 1,607 at the adopted layout (amended 2026-10-19). That
  avoids the ~10⁴ dense columns that caused PR #100's out-of-memory failure.
- Windows of one pulsar that fall in the same grid cell are merged by the same algebra. The
  only approximation is A1.
- **What D2 makes exact is the DMX, ECORR and per-backend bookkeeping, not the time
  resolution.** On the dense union grid the step count, and so the cost, is set by the grid
  width Δ. K3 was expected to force Δ to 2–4 weeks. (2026-10-19: not so. The sparse filter's
  cost is set by the observation count, which barely moves with Δ, so the adopted Δ is the
  DMX window, t_w-stamped, with no time gridding beyond A1.)
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
- **Epoch cells** (amended 2026-10-18). A pulsar can instead be compressed per ECORR
  epoch: one position scalar per epoch, no velocity step, and its DMX columns kept in β
  (no DMX projection). A1 then only has to hold within an epoch (≲ 1 s), so the cell is
  near-exact. Rule: a pulsar uses epoch cells when its two-scalar window likelihood
  differs from its epoch-cell likelihood by more than 0.01 nat in ΔlnL spread at 20 of its
  posterior points (spread only, see C1).
  Epoch cells add that pulsar's DMX columns to M_sum, so they are for the pulsars that
  need them only (all 67 would add 7,410 columns).
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
  - Argus single-pulsar stage: ~70 jobs × ~1 A100-h. (Measured in Phase 3: ≈ 256 A100-h of
    sampling, median 2.4 h per pulsar, plus the jobs lost to the first 4 h limit.)
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
  - **Unresolved since Phase 2 (2026-10-19).** The 800 A100-h figure assumes 0.5 s of
    *per-chain* latency. Phase 2 measured 0.84 s as *batched throughput* (32 lockstep chains
    on one A100); per-chain latency is 6.7 s for one point, or 26.8 s per 32-point step. The
    ladder is ≈ 1,340 A100-h only if the draws can be split across many short lockstep
    chains with warmup a small share, and ≈ 10,700 A100-h if each chain needs §5's length.
    Settle before Phase 5 (§12, 2026-10-19).

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
    - Triggered 2026-10-18 for one scalar per window. The first pivot taken is two scalars
      per window (position and velocity, D2), which doubles the union-grid steps. Task 2.6
      measures that cost **before** deciding the remaining C1 misses (J1909−3744 and
      B1937+21 gradients; J1012+5307 value and gradient). The user then chooses among
      relaxing C1's gradient bound, per-ECORR-epoch steps with DMX carried as a per-window
      state, or that treatment only for pulsars whose red-noise correlation time is shorter
      than their windows. (Resolved 2026-10-18: epoch cells with DMX in β for the four
      pulsars the spread rule selects, and the gradient bound reported, not gated; §12.)
  - The cost per 67-psr likelihood-plus-gradient evaluation on one A100 is > 0.5 s at every Δ
    that passes C1: coarsen Δ (re-measuring A1), or revisit the #108 parallel filter, whose
    crossover needs exactly the larger T that a fine grid produces.
  - The cost is > 1 s even at the coarsest passing Δ: stop and re-budget with the user.
  - **Outcome 2026-10-18:** 0.84 s per batched evaluation at the epoch-rule layout, between
    target and stop. The user accepted it and closed Phase 2 on it; neither pivot is taken.
    Coarsening Δ buys nothing on the sparse filter, whose cost follows the observation count
    (7 d: 9.2 s for one point vs 9.6 s at the DMX window, same filter version,
    `results/k3/sparse_7d.json` and `sparse_dmx_window.json`). #108 was measured slower only
    at MDC2 scale (T = 183), never at the ~15k sequential steps where its crossover was
    expected, so it stays an untested speed option. (Corrected 2026-10-19; per-chain
    latency and the budget are in §5.)
  - **Cost is judged as batched throughput** (decided by the user 2026-10-18): wall time of
    one value-plus-gradient call vmapped over B parameter points (B NUTS chains on one
    A100), divided by B. Revisit if lockstep chains turn out to be a major blocker.
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
- [NEEDS CLARIFICATION: the production γp ceiling for Phase 5 (task 5.1). The −6 box
  (from the MDC2 ladders, and used by Phase 1's Discovery-OU reference) cuts the f⁻²
  ridge for f⁻²-like pulsars. With both ceilings raised (γp −4, σp −9) their posteriors
  run on to the new γp ceiling, so no finite ceiling holds all the mass; the in-band PSD
  moves ≤ 0.07 dex for two of three pulsars checked (§12, 2026-10-22, which corrects the
  2026-10-20 wording). In the array, the cut could still push red power into the common
  process (A3). Changing the box would need the Phase 1 OU reference rerun on the same
  box, so that C5 still compares like with like. Decide before 5.1.]
- [NEEDS CLARIFICATION: the Fourier mode count of the OU reference behind C5 (Phase 1's
  ladder). At 30 modes Discovery-OU misses red power Argus finds in J1012+5307, and likely
  in J0437−4715 (§12, 2026-10-22); J1012's posterior agrees with Argus at 100 modes.
  Whether other pulsars shift, and whether C5's reference should use 100 modes, is
  unchecked. Decide before 6.4.]

## 10. Risks

| Risk | Early warning |
|---|---|
| OU kernel too stiff, so the HD evidence is diluted (A3) | Discovery-OU lnB ≪ Discovery-PL lnB in Phase 1 (K2) |
| Compression error at shallow-spectrum pulsars (A1) | C1 spread grows with red-noise slope; per-pulsar C4 outliers concentrated in γ_PL ≲ 1. Realised 2026-10-18: one scalar per window failed everywhere; two scalars still missed J1012+5307 and two gradient bounds. Resolved: epoch cells for 4 pulsars by the spread rule, and gradients reported, not gated (§12) |
| 136-D sampling geometry | Phase 5 pilot (ε=0, 500+500) step size < 0.01, or per-chain medians split |
| Per-step cost blows up at 67 psr on the union grid (step count × O((4N)³)) | Task 2.6 benchmark of value-plus-gradient on the marginal path. There is no trustworthy prior rate (§2); if it is near the sequential quick-look (≳ 0.6 s at 189 steps), Δ is forced to ≳ 1 month. Realised 2026-10-18: the dense grid needed 28.6 s at the DMX window; the sparse filter gives 0.84 s batched, 6.7 s per chain, and the budget is open (§5) |
| Discovery env / jax version conflicts | Task 1.2 fails to install; fall back to a container |
| Path-sampling integrand steep near ε=1 at lnB ~ 5 | Frozen-procedure Romberg check flags extra rungs (0.625, 0.875) |
| Diffuse results since PR #107 carry the Λ-jitter bias (fixed 2026-10-17) | GW-vs-noise and amplitude numbers from #122/#123/#124 move by ≳ 0.1 nat on rerun; lnB(HD/CURN) is expected (fixed-parameter estimate) to move by ≲ 0.01 |
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
**PR:** #128 (merged 2026-10-16)

- [x] 1.1 Fetch Discovery at a pinned commit; write `MANIFEST.json` (commit, feather hashes,
      67 pulsar names, data-set version) — verify by 67 feathers hashed and version string recorded.
- [x] 1.2 Create and pin the `discovery` conda env (`envs/discovery.yml`) — verify by importing
      it and evaluating the published single-pulsar likelihood for J1909−3744.
- [x] 1.3 Build the reference driver `workflows/ng15_nb/reference/` with: (a) the NANOGrav
      power-law model (30-mode RN, 14-mode common, fixed WN) and (b) the OU-PSD model using
      Argus's exact one-sided PSD conventions (sidedness note); fixed-ε ORF (1−ε)I + εHD.
      There is no `ou_psd` in `python/argus`. The references to match are:
      - the common process in the ridge basis:
        `workflows/ng15_sgwb_demo/scripts/diag_ou_powerlaw_capture.py::ou_psd`;
      - per-pulsar red noise: `scripts/ng15_red_noise_budget.py::ou_psd_onesided`.
      Before use, first check both against the filter's own stationary Q (the ground truth).
      — verify by the OU PSD functions matching those references to 1e-12, and ε=1
      reproducing Discovery's built-in HD likelihood to 1e-8.
- [x] 1.4 Run the PL ladder (5 rungs, variable γ) and apply our path-sampling estimator. Add
      one fixed-γ = 13/3 HD run for the A_HD,13/3 check: conditioning the variable-γ posterior
      at 13/3 is not the same fit. — verify against C3 (lnB within ±0.7 of 5.30; A_HD,13/3
      within 1σ).
- [x] 1.5 Run the OU ladder at 14 and 30 common modes; compute lnB(HD/CURN) and lnB(CURN/noise)
      — verify by the reliability gate passing for lnB(HD/CURN), and lnB(CURN/noise) recorded
      as a 95% lower bound (amended 2026-10-17); record the mode-count sensitivity (input to C5).
- [x] 1.6 Write N2 (kernel systematic: PL vs OU lnB and amplitude at 1/T and 1/(5 yr)) and
      evaluate K2 — verify by the committed note stating the K2 outcome explicitly.

### Phase 2 — Exact epoch compression (S0 feasibility)

**Goal:** show the narrowband TOAs can be fed to the existing filter exactly (C1) and cheaply.
**Done when:** C1 passes for all three test pulsars at the chosen Δ (2026-10-19: the adopted layout, DMX windows plus epoch cells by rule), C2 passes, and the 67-psr cost
benchmark is ≤ 0.5 s per evaluation, with K3 evaluated. (amended 2026-10-18: the cost
criterion is < 1 s per batched evaluation, K3's stop; measured 0.84 s. Speed work carries
into Phase 5's sampling test — see §12)
**PR:** #131 (merged 2026-10-19). **Runs in parallel with Phase 1**: the two are independent until K2 is
evaluated, and Phase 1 has an unverified dependency (Discovery accepting a custom PSD and a
mixed ORF) that could stall it.

- [x] 2.1 Narrowband ingest from the Discovery feathers (pyarrow only) into an Argus-side cache
      that keeps per-TOA backend, radio frequency, ECORR epoch id, DMX window id and the full
      M with column names — verify by round-tripping two pulsars and checking TOA counts and
      column names against the feather. Report the DMX-window length distribution, the number
      of single-frequency windows (dropped, D2), the number of windows that span more than
      one ECORR epoch, the number of ECORR epochs that straddle a window boundary (merged,
      D2), and the timing-model columns dropped by the rank rule.
- [x] 2.2 Implement compression (D2) as a pure numpy/JAX preprocessing module with unit tests on
      a synthetic pulsar where the exact answer is known. The synthetic pulsar includes a
      window with two ECORR epochs, an ECORR epoch straddling two windows, a single-frequency
      window, and a DM column degenerate with DMX — verify by matching the compressed vs dense
      likelihood on the synthetic case to 1e-10, with a constant state per window so A1 holds
      exactly.
- [x] 2.3 Add initial accumulators (A₀, b₀, c₀) to `_run_kalman_filter_marginal` — verify by C2
      (the golden tests pass, plus a new exact-equality test of zero accumulators against the
      unmodified call) and a test showing nonzero A₀ equals appending the equivalent rows.
- [x] 2.4 Build the dense brute-force oracle: raw TOAs, the **continuous-time** analytic OU
      covariance at each TOA time (not the gridded F/Q), WN + ECORR, full M after the rank
      rule, diffuse — verify by checking it on the synthetic pulsar against 2.2, and on a
      single-step grid against the filter.
- [x] 2.5 Run C1 on J1909−3744, one strong-red-noise pulsar and J1012+5307 (shallow) at
      Δ ∈ {DMX window, 7 d, 14 d, 30 d} — verify by the ΔlnL spread and gradient error tables in N1; choose the largest Δ that passes.
      (amended 2026-10-18: one scalar per window fails at every Δ; rerun C1 with the
      two-scalar D2 at the DMX window, and leave the remaining misses to the user's
      decision after 2.6 — see §12)
      (amended 2026-10-18, second: pulsars that miss use epoch cells, chosen by the D2
      rule over all 67; a residual within the oracle's precision passes — see §12)
- [x] 2.6 Compress all 67, build the union grid with masks, benchmark the likelihood plus
      gradient on one A100 — verify by the recorded seconds per evaluation and memory; evaluate K3.
      (amended 2026-10-18: on the two-scalar union grid, two steps per window; run before
      the 2.5 decision)
      (amended 2026-10-18, later: benchmarked on the sparse filter at the epoch-rule layout,
      which replaced the union grid; the dense union grid is kept as the test reference)

### Phase 3 — Per-pulsar validation

**Goal:** every pulsar's Argus noise posterior is consistent with the kernel-matched reference.
**Done when:** C4 passes (≥ 90% within 1σ), and every outlier has a one-line diagnosis.
**PR:** #133 (merged 2026-10-22)

- [x] 3.1 Single-pulsar Argus NUTS for all 67 (SLURM array, flat OU priors, fixed WN) — verify
      by 100% meeting r̂ ≤ 1.01, ESS ≥ 400 and 0 divergences. No pulsar with
      detected red noise may rail against a prior edge. For pulsars without detected red noise,
      piling up at the lower amplitude edge is expected; report it, don't fail on it.
      (amended 2026-10-20: "prior edge" means the σp edges. Mass at either γp edge is
      reported, not failed; the ceiling is flagged as OU's f⁻² limit. See §12)
      (amended 2026-10-22: a first-pass failure is rerun at target_accept 0.99, with 8000
      draws where ESS or r̂ needs it (J0437−4715, J0645+5158); the rerun is what is
      judged. See §12)
- [x] 3.2 Matching single-pulsar Discovery-OU runs — verify by the same diagnostics.
- [x] 3.3 Produce F4 and the outlier table, flagging shallow-spectrum pulsars — verify against C4.

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

### 2026-10-17 — lnB(CURN/noise) is a lower bound (PR #128)

**Trigger:** the Phase 1 readout. All three lnB(CURN/noise) estimates (PL, OU14, OU30) report
`reliable: false`. The `/check-PR` review (`PR_REVIEW_128_2026-10-17.md`) flagged that task 1.5's
verify clause could not be met as written.

**Change:**
- Task 1.5: the reliability gate applies to lnB(HD/CURN) only. lnB(CURN/noise) is recorded as
  a 95% lower bound: 5.53 (PL), 5.03 (OU14), 4.16 (OU30). A density-ratio estimate from a
  posterior run cannot reach the published B > 10¹⁴, so a bound is the most it can give.
- T1 (assembled in task 6.4) will report lnB(CURN/noise) for the Discovery rows as a lower bound.

**Approved by the user** on 2026-10-17. PR #128's other proposed amendments (§9 Q1/Q3, the
`ou.ini` boxes as binding priors, the C5 tolerance and the §5 budget) are not applied here.

**Sections touched:** §11

### 2026-10-17 — the diffuse Λ jitter biased the diffuse golden

**Trigger:** task 2.5's C1 on B1855+09, a small pulsar run first on CPU to shake down the
driver before the GPU runs on the three C1 pulsars (it does not replace any of them). Against the dense oracle, the stock filter's ΔlnL
spread was 4.2e-2 nat at the DMX-window Δ, and 3.0e-4 once the diffuse branch's jitter of
1e-9·trace(Λ)/n was zeroed. On MDC2 at the golden parameters:
- stock value: 59420.06;
- jitter zeroed: 59454.725;
- independent check, the informative filter as α → ∞: 59454.547 (α = 1e6), 59454.722
  (1e8) and 59454.725 (1e10).

Λ's equilibrated condition number is 1.6e12 (smallest eigenvalue 2.9e-12 relative to its
diagonal), so the jitter swamped real directions. The C2 golden was itself the biased value.

**Change:**
- The diffuse branch of `_run_kalman_filter_marginal` takes logdet Λ from a Cholesky of
  the unit-diagonal equilibrated Λ, with no jitter, and returns −inf if Λ is not PD. This
  shipped as its own PR (`fix/diffuse-lambda-jitter`) before Phase 2's.
- C2: the diffuse golden is 59454.72. A new test, `test/test_diffuse_mdc2_limit.py`, checks
  it against the α → ∞ limit on MDC2 (5e-3 nat at α = 1e10).
- The diffuse value now carries ~1e-3 nat of rounding noise from Λ's conditioning (it was
  ~6e-6 with the jitter). The gradient is unaffected. Two test tolerances were widened to
  match (plus one test's sparse mask made identifiable), and C1's 0.01 nat bound has 10×
  margin over it.
- §2 and §10: diffuse results since PR #107 (the jitter's origin, `git log -S`) are flagged
  as exposed. Rerunning #122/#123/#124 is left to the user.

**Sections touched:** §2, §3, §10, §12

### 2026-10-18 — two scalars per compression window

**Trigger:** task 2.5's C1 at the DMX-window Δ (20 Discovery-OU posterior draws per pulsar,
spread / max gradient relative error; bounds 0.01 nat / 1e-4):

| Pulsar | one scalar (D2 as written) | two scalars (position + velocity) |
|---|---|---|
| B1855+09 (control) | 3.0e-4 / 1.4e-4 | 1.5e-6 / 2.5e-7 |
| J1909−3744 | 1.58 / 0.90 | 5.3e-4 / 3.5e-4 |
| B1937+21 | 5.2 / 1.5 | 8.6e-3 / 1.1e-2 |
| J1012+5307 | 0.54 / 0.27 | 5.4e-2 / 3.8e-2 |

The 7, 14 and 30 d grids are worse than the DMX window with one scalar. The cause is the
within-window drift (t − t_w)·ṡ, which a constant-state compression puts into the β-only
remainder. It is largest at small γp, where the stationary spin-velocity variance σp²/2γp is
large: B1937+21's error correlates 0.99 with γp. J1012+5307's red-noise correlation time
(≈ 12 d) is shorter than its 14 d windows, so even the velocity changes within a window.

Separately, `model.get_Q_block`'s q11 loses precision at small γ·dt (5% at γ = 1e-12 s⁻¹,
dt = 1 d). Fixing it in a diagnostic copy moved C1 by ≈ 0.03 nat; the library fix needs its
own approval and is not part of this amendment. (Approved and merged as PR #130, 2026-10-18.)

**Change:**
- D2 emits a second, velocity scalar per window along the projected (t − t_w) direction,
  fed as a second filter step at dt = 0. A1 now assumes a state linear in time within a
  window.
- K3: the two-scalar pivot is taken; it doubles the union-grid steps. Task 2.6 runs on that
  grid before the user decides the remaining C1 misses.
- Tasks 2.5 and 2.6 annotated accordingly. §10's A1 row records the outcome.

**Approved by the user** on 2026-10-18.

**Sections touched:** §3, §4, §8, §10, §11, §12

### 2026-10-18 — K3 measured; grid stamping, union whitening and a sparse filter

**Trigger:** task 2.6's benchmark of value plus gradient (all 136 parameters) on one
A100, 67 psr, two scalars per cell, against K3's 0.5 s target and 1 s stop.

**What was measured:**
- **Dense union grid** (the existing marginal filter, every pulsar's slot at every step):

  | Δ | steps | value | value + gradient | C1 worst posterior spread (4 pulsars) |
  |---|---|---|---|---|
  | DMX window | 14,820 | 8.7 s | 28.6 s | 5.4e-2 nat |
  | 7 d | 1,150 | 0.68 s | 2.3 s | 9.6e-2 nat |
  | 14 d | 690 | 0.41 s | 1.3 s | 0.37 nat |
  | 30 d | 376 | 0.22 s | 0.74 s | 2.7 nat |

  No Δ both passed C1 and came in under 1 s. The stock gradient's reverse mode needs
  220 GB at the DMX-window Δ; it was timed on a benchmark-only copy with checkpointed
  scan segments (same value to 3e-8 nat).
- **Grid stamping (a D2 detail).** With two scalars, a cell can be stamped at its weighted
  mean time t_w (exact position, but no two pulsars share a step) or at the grid centre
  t_c, with the position row carrying −(t_w − t_c) on the rate states (exact for a state
  linear in the cell, and pulsars share steps). Centre stamping is what makes coarse Δ
  cheap, and it costs accuracy: J1012+5307 at 14 d went from 2.5e-2 nat (t_w) to 0.37.
- **Union whitening.** On shared steps the library's innovation jitter (1e-9 × the mean S
  of the observed pulsars) is set by poorly measured observations and perturbs their
  neighbours: 3.8 nat on a synthetic pair at 30 d. The union-grid inputs whiten every
  observation to unit noise, with the Jacobian in the offset; the joint then equals the sum
  of pulsars to 1e-8.
- **`get_Q_block` small-γ·dt fix** merged separately as PR #130 (approved 2026-10-18).
- **Sparse marginal filter** (approved 2026-10-18 as the K3 re-budget):
  `_run_kalman_filter_marginal_sparse` takes one scalar observation per step, so it never
  touches unobserved pulsars and builds the timing information A as one product after the
  scan. Its cost depends on the number of observations, not on Δ, so the accurate
  t_w-stamped DMX-window cells cost no more than a grid. At the DMX-window Δ (14,811
  observations) it gives value 0.79 s and value plus gradient 6.2 s for one point, and
  **0.87 s per evaluation when 32 points are vmapped together** (32 NUTS chains on one
  GPU). It equals the jitter-free dense path to 1e-9 on synthetic data. The dense MDC2
  goldens are bit-identical after the shared-helper refactor.

**Change:**
- D2: grid cells are stamped at t_c with the position-row shift; DMX-window cells at t_w.
- ~~Proposed, not yet decided: the array likelihood for Phases 3–6 is the sparse filter at
  the DMX-window Δ.~~ — decided 2026-10-18: it is, with epoch cells by rule (later entries).
- ~~Open: whether K3 is judged on batched throughput (per evaluation it is 6.2 s).~~ —
  resolved 2026-10-18: the user accepts batched throughput for now (K3, §8), so the sparse
  filter at 0.87 s/eval sits between K3's target and stop. Speed work continues.
- The remaining C1 misses: the user asked (2026-10-18) to try both fallbacks — epoch cells
  with DMX in β for the pulsars that need it, and, if needed, a relaxed gradient bound.

**Approved by the user** on 2026-10-18 (the sparse filter, the `get_Q_block` fix and this
note).

**Sections touched:** §12

### 2026-10-18 — epoch cells by rule, and C1 within the oracle's precision

**Trigger:** the C1 fallbacks the user asked for. Posterior spread / gradient error, two
scalars per DMX window vs epoch cells (DMX in β), 20 posterior draws each:

| Pulsar | two scalars per window | epoch cells |
|---|---|---|
| J1012+5307 | 5.4e-2 / 3.8e-2 | 3.9e-8 / 4.2e-9 |
| J1909−3744 | 5.3e-4 / 3.5e-4 | 1.4e-5 / 1.9e-6 |
| B1937+21 | 8.6e-3 / 1.1e-2 | 4.2e-3 / 2.3e-3 |
| B1855+09 | 1.5e-6 / 2.5e-7 | 1.5e-6 / 2.5e-7 |

Moving the oracle's time origin by 1000 d (exact in theory) shifts the oracle alone by
1.5e-2 nat / 2.3e-3 on B1937+21, so its epoch-cell residual is at the oracle's precision.
B1855+09's 0.10 nat on the prior stress set is also oracle-side: its two filter layouts
agree there to 3e-5.

**Change:**
- C1 (§3): a residual within the oracle's precision (origin-shift test) passes; epoch
  cells are the near-exact per-pulsar reference where the oracle is not used.
- D2 (§4): epoch cells, and the rule choosing which pulsars use them.
- Task 2.5 annotated: next, run the rule over all 67 and record how many pulsars need
  epoch cells and the resulting M_sum.
- Speed: the timing block Ξ is ≈ 45% of batched time (0.55 s/eval with 16 columns vs
  0.98 s with 1,080), so the next speed step is a chunked Ξ update in the sparse filter.

**Approved by the user** on 2026-10-18 (this amendment and the chunked Ξ restructure).

**Sections touched:** §3, §4, §11, §12

### 2026-10-18 — the epoch-cell rule gates on the value spread only

**Trigger:** the rule over all 67 pulsars (`results/c1/epoch_rule.json`). At both of C1's
bounds 33/67 pulsars need epoch cells and M_sum grows 1,080 → 5,072, almost all flagged by
the 1e-4 gradient bound. On the 0.01-nat spread alone 4 do (J1012+5307, J1643−1224,
J1705−1903, J1903+0327; M_sum 1,607); the other 63 spreads sum to ≤ 0.05 nat.

**Change:** the window-vs-epoch rule gates on the ΔlnL spread only and reports gradient
differences (§3 C1, §4 D2). The oracle check on the three C1 pulsars is unchanged
(superseded the same day: see "the oracle check gates on the spread too").

Also recorded: a chunked Ξ update in the sparse filter (approved as a speed step) was exact
but not faster on its own (32 vmapped points: 0.93 vs 0.99 s/eval; one point 11.3 vs
9.4 s). A GPU profile then showed the backward pass launch-bound on `get_Q_block`'s
per-step series residuals; checkpointing the Q blocks as well made both pay.

**K3 at the rule layout** (67 psr, 15,372 observations, M_sum 1,607, one A100,
`results/k3/sparse_dmx_window_epoch_rule.json`): before, 1.27 s/eval at 32 vmapped points
(10.6 s for one point), over K3's 1 s stop; with chunked Ξ (chunk 8) and checkpointed Q,
**0.84 s/eval** (6.7 s for one point), lnL equal to 1e-8 nat and gradient to 2e-9.
Between K3's 0.5 s target and its 1 s stop.

**Approved by the user** on 2026-10-18.

**Sections touched:** §3, §4, §12

### 2026-10-18 — Phase 2 closes at 0.84 s per batched evaluation

**Trigger:** K3 at the epoch-rule layout measured 0.84 s/eval (32 vmapped points, one
A100), between the 0.5 s target and the 1 s stop. K3's pivots for that zone are both poor:
coarsening Δ does not cut the sparse filter's cost (corrected 2026-10-19; this said it costs
C1 accuracy), and #108's parallel filter was measured slower at MDC2 scale.

**Change:** Phase 2's cost criterion is < 1 s per batched evaluation (K3's stop); speed
work continues alongside Phase 5's sampling test (§8 K3 outcome, §11 Phase 2).

**Budget consequence:** ~~§5's ladder estimate assumed ≲ 0.5 s; scaled linearly, 0.84 s
puts the 67-psr ladder at ≈ 1,340 A100-h (from 800) and the total at ≈ 1,540 A100-h.~~
Wrong (corrected 2026-10-19): that scales batched throughput, but §5 assumed per-chain
latency. The budget is unresolved; see §5 and the 2026-10-19 entry.

**Approved by the user** on 2026-10-18.

**Sections touched:** §8, §11, §12

### 2026-10-18 — the oracle check gates on the spread too

**Trigger:** under the spread-only epoch-cell rule, J1909−3744 and B1937+21 stay on window
cells, where they pass C1's oracle spread bound (5.3e-4 and 8.6e-3 nat) but not its
gradient bound (3.5e-4 and 1.1e-2; N1). Putting them on epoch cells too would take M_sum
to ≈ 2,190 and likely past K3's 1 s stop.

**Change:** C1's oracle comparison gates on the ΔlnL spread only and reports gradient
errors, for the same reason as the rule (§3 C1). At the adopted layout all three C1
pulsars pass: J1909−3744 5.3e-4, B1937+21 8.6e-3, J1012+5307 (epoch cells) 3.9e-8 nat.

**Approved by the user** on 2026-10-18.

**Sections touched:** §3, §12

### 2026-10-19 — review corrections: the budget, the D2 text, the dense reference

**Trigger:** /check-PR on PR #131 (5 major findings).

**Change:**
- **Budget (§5).** §5's 800 A100-h assumes 0.5 s of per-chain latency: 40 h per rung is
  ≈ 288k sequential evaluations per chain. Phase 2 measured 0.84 s as batched throughput.
  Warmup cannot be divided across lockstep chains, so per-chain latency is what sets wall
  time: 6.7 s for one point (13× the assumption) or 26.8 s per 32-point step (54×). The
  ladder ranges from ≈ 1,340 A100-h, if the draws split across many short lockstep chains
  and warmup is a small share, to ≈ 10,700 A100-h, if each chain needs §5's length on one
  GPU. Phase 2 still closes: the user chose to merge on 2026-10-19 with the budget recorded
  as open. It must be settled before Phase 5, by a chain-design test or by speed work.
- **D2 (§4) and §3.** The text now says what was adopted: two library changes (accum0 and
  the sparse filter), M_sum 1,607 including DMX columns for epoch-cell pulsars, one
  observation per step, and the t_w-stamped DMX-window Δ.
- **K3 outcome (§8).** Coarsening Δ is not taken because it does not cut cost, not because
  of C1: under the spread-only gate 7 d passes for J1909−3744 and B1937+21, and J1012+5307
  is on epoch cells at any Δ. A 7 d layout would need its own epoch-rule pass, which was
  not run. #108 was measured only at MDC2 scale.
- **The dense union path** (`union.array_loglike`) is ~0.1 nat low on real data. The PD
  jitter 1e-9·s̄·I also lands on masked unit slots, for ½Σ log(1 + 1e-9·s̄) = 0.097 nat at
  the DMX window (predicted 0.0975, observed 0.0974). It is documented, not fixed: only
  the dense reference and the K3 cross-check use it, and C1, the epoch rule and Phases 3–6
  do not.
- **C1 results.** The committed `results/c1/*.json` had `passes_c1` set by both bounds; it
  is re-applied as spread-only, and the result writers now record the git SHA.
- §10's A1 and cost rows, §8 K3's open choice, and task 2.6's text are annotated.

**Approved by the user** on 2026-10-19 ("apply the fixes then merge").

**Sections touched:** §3, §4, §5, §8, §10, §11, §12

### 2026-10-20 — the γp edges are reported, not gated; C4's "within 1σ"

**Trigger:** task 3.1/3.2's first results. Under a rule that also gated the γp ceiling
(log10 γp = −6), about 10 pulsars per pipeline "railed" there, mostly the same ones in
Argus and Discovery-OU: J0645+5158, J1745+1017, J1802−2124, J1853+1303, J1946+3417,
J2145−0750, and the shallow J1705−1903 and J1903+0327.

**What the ceiling does:** above its corner the integrated-OU residual PSD tends to
σp²/(f0² γp² ω²), an f⁻² spectrum, so a pile-up at the ceiling means the data want
OU's shallowest spectrum or shallower (A3's shallow-spectrum exposure). It was first
argued (and approved) that the likelihood is flat along γp → ∞, so the pile-up would
follow any ceiling. **The spot check refuted that.** (Corrected 2026-10-22:
it did not; the check was confounded by the σp ceiling. See below.) Discovery-OU reruns with the
ceiling at −4 (`run_discovery.py --gamma-p-max -4`):

| Pulsar | γp median, −6 → −4 | ceiling mass, −6 → −4 | log10 PSD at 1/T, median | at 1/(5 yr) |
|---|---|---|---|---|
| J1745+1017 | −7.07 → −6.66 | 0.07 → 0.00 | −0.01 → −0.41 (16–84% width ≈ 2 dex) | −1.29 → −1.48 |
| J2145−0750 | −6.32 → −5.29 | 0.20 → 0.06 | −3.40 → −3.42 | −4.42 → −4.44 |
| J1903+0327 | −6.28 → −5.92 | 0.22 → 0.00 | −1.55 → −1.61 | −2.57 → −2.62 |

The γp posteriors close off between −6.7 and −5.3, so −6 truncates real posterior mass.
The in-band PSD that C4 compares moves by 0.02–0.06 dex in two cases and by 0.4 dex
(≈ 0.4σ) in the third, and C4 compares two pipelines on the same box, so the truncation
does not bias C4. It may matter for the array runs: §9 now asks for the Phase 5 ceiling.
(Corrected 2026-10-22: the posteriors do not close off, and the 30-mode Fourier truncation,
a different effect, biases C4 for J1012+5307 and likely for J0437−4715. See below.)

**Change:**
- Task 3.1: the gated edges are σp's. Both γp edges are reported. A detected pulsar with
  > 5% of its draws in the top 2% of the γp box is flagged `at_f2_limit` and listed in
  the C4 diagnosis table as truncated by the −6 ceiling. Phase 3 keeps the −6 box,
  matching Phase 1's Discovery-OU reference.
- C4's "within 1σ" (§3) is read as z = |m_A − m_D| / √(σ_A² + σ_D²) ≤ 1 on the log10
  one-sided spin PSD, with σ half the 16–84% width, at both f = 1/T and 1/(5 yr); a pulsar
  piling up at the σp floor is compared the same way (taken in the Phase 3 plan,
  2026-10-19).
- The single-pulsar runs draw 4000 samples per chain after 1000 warmup (the frozen block
  has 1000), because the γp floor is a slowly mixing flat direction.
- §9: a new open question on the production γp ceiling, to settle before task 5.1.

**Approved by the user** on 2026-10-20 (the rule first; then, after the spot check, this
corrected rationale and the §9 question).

**Sections touched:** §3, §9, §11, §12

### 2026-10-22 — the γp spot check redone; the 30-mode truncation; reruns at target 0.99

**Trigger:** the `/check-PR` review of PR #133, and two checks it asked for.

**The γp ceiling.** The 2026-10-20 spot check raised only γp's ceiling. Two of its three
pulsars then piled onto the σp ceiling instead (J1745+1017 12% of mass, J1903+0327 17%),
because along the f⁻² ridge σp rises with γp. Rerun with both ceilings raised
(`run_discovery.py --gamma-p-max -4 --sigma-p-max -9`, job 18955930):

| Pulsar | γp median [16–84%] | γp ceiling mass | log10 PSD at 1/T, −6 box → both raised | at 1/(5 yr) |
|---|---|---|---|---|
| J1745+1017 | −5.74 [−8.03, −4.55] | 0.05 | −0.01 → −0.64 (16–84%: −0.96 to +0.68) | −1.29 → −1.66 |
| J2145−0750 | −5.27 [−6.16, −4.43] | 0.06 | −3.40 → −3.42 | −4.42 → −4.43 |
| J1903+0327 | −5.31 [−6.14, −4.44] | 0.06 | −1.55 → −1.62 | −2.57 → −2.63 |

The posteriors run on to the new ceiling, so the likelihood is near-flat along the ridge
and the first rationale of 2026-10-20 (the pile-up follows the ceiling) was close to right;
the "closes off between −6.7 and −5.3" correction was wrong. The in-band PSD moves by
≤ 0.07 dex for two pulsars and 0.6 dex, inside a 1.6 dex-wide posterior, for J1745+1017.
The rule is unchanged: γp edges are reported, not gated. 14 Argus and 12 Discovery-OU
pulsars are `at_f2_limit` (not "about 10"); N3 lists them.

**The 30-mode truncation biases C4.** J1012+5307's outlier is Discovery-OU's 30 Fourier
modes missing red power that Argus's continuous-time kernel sees, and J0437−4715's likely
is:
- J1012+5307: a 100-mode Discovery-OU posterior (`--rn-components 100`, job 18957720)
  agrees with Argus, z = 0.03 at both frequencies, against 1.21 at 30 modes. Its earlier
  diagnosis ("the box edge sets the PSD") could not explain a gap between two pipelines on
  the same box.
- J0437−4715: Discovery's lnL at Argus's posterior point is −3.0 nat relative to no red
  noise at 30 modes and +8.1 at 100 (+8.3 at 300). This is a likelihood-level diagnosis;
  a 100-mode posterior was not run.

C4 keeps the 30-mode reference (it passes, 65/67), and §3 C4 now names the 100-mode run as
an outlier's evidence. §9 asks whether C5's reference needs more modes.

**Reruns.** Seven first-pass failures (5 Argus, 2 Discovery-OU; divergences, r̂ or ESS)
were rerun at target_accept 0.99, J0437−4715 and J0645+5158 (Argus) with 8000 draws; all
pass, and C4 reads the rerun wherever one exists. B1937+21 and J1909−3744 diverged in both
pipelines at 0.9.

**C1's oracle-precision clause** is now used: B1937+21's posterior-point residual
(1.4e-2 nat) passes inside the oracle's re-measured floor (2.4e-2 nat).

**Change:**
- §3 C1: the 2026-10-19 note that no pass relies on the oracle-precision clause is updated.
- §3 C4: the 30-mode reference, and the 100-mode run as an outlier's evidence.
- §5: the measured Argus single-pulsar cost, ≈ 256 A100-h.
- §9: the γp-ceiling question reworded; a new question on C5's mode count.
- §11 task 3.1: the rerun procedure.
- §12 2026-10-20: its spot-check rationale and "does not bias C4" marked as corrected here.

**Approved by the user** on 2026-10-22.

**Sections touched:** §3, §5, §9, §11, §12

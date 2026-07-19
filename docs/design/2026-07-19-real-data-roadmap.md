# Design: applying Argus to real data — roadmap to a full-array NG15 SGWB detection

- **Date:** 2026-07-19
- **Status:** approved design, pre-implementation
- **Tracking issue:** [#111](https://github.com/tomkimpson/Argus/issues/111)
- **Context:** [#106](https://github.com/tomkimpson/Argus/pull/106) (pipeline record, being dismantled),
  [#109](https://github.com/tomkimpson/Argus/issues/109) (full-array NUTS wall),
  [#110](https://github.com/tomkimpson/Argus/issues/110) (strategic framing: non-Fourier cross-check pipeline)

## 1. Goal and success criteria

**Goal.** Demonstrate that Argus detects the stochastic gravitational-wave background, first on
controlled mock data, then on the real NANOGrav 15yr array.

Two success gates, in order:

1. **M1 (near-term):** on IPTA MDC2 dataset 2b (33 pulsars, injected GWB with
   log₁₀A = −14.89), a decisively positive HD-vs-CURN log-Bayes factor and a GW posterior whose
   90% credible interval covers the injected amplitude.
2. **M3 (ultimate):** the same procedure on the real 68-pulsar NG15 wideband dataset, yielding a
   detection-grade lnB(HD/CURN) and an amplitude consistent with NANOGrav's published wideband
   analysis.

This roadmap is the path to the "Paper A" result framed in #110 (first non-Fourier, time-domain
confirmation of the HD-correlated background), but no paper boundary is fixed yet.

## 2. Background: where we are and why two-stage

Validated so far (PR #106 and earlier): MDC2 golden likelihoods and GW-parameter posteriors; T3.3
amplitude recovery on real 6-psr NG15 consistent with published; T3.4 HD-vs-CURN lnB = +2.1 ± 0.1
(6-psr) via the learned harmonic mean (LHM); a missing-observation Kalman filter so short-baseline
pulsars need not be truncated.

Blocked: full-joint NUTS on the ~142-D full-array posterior (per-pulsar red noise + hierarchical +
GW) is **unsamplable** — three tuning levers (diagonal mass; dense GW block at accept 0.95; dense GW
block at 0.99) triangulate a divergence-vs-mixing wall that no `target_accept_prob` threads. This is
posterior geometry, not a pipeline defect (details in #106/#109).

**Why not Rao-Blackwellize the red noise (the original #109 prescription)?** The marginalized
timing-model filter (#107) works because the timing model is linear-Gaussian: conditional on
everything else, its coefficients have a Gaussian posterior the filter integrates analytically. The
per-pulsar red-noise hyperparameters (γ_p, σ_p) enter the filter **nonlinearly through F and Q**
(`python/argus/model.py`: `get_F`, `get_Q`); their conditional posterior has no conjugate structure,
so there is no analytic marginalization. The red-noise *realization* is already marginalized — it is
the Kalman state. What remains is the hyperparameter geometry, and the near-term answer is to
constrain those hyperparameters from single-pulsar runs rather than out-sample the ridge.

**What is field-standard.** NANOGrav headline analyses fix per-pulsar *white* noise from
single-pulsar runs and sample per-pulsar *red* noise jointly (feasible for them via PTMCMC with
empirical-distribution jump proposals — not available to NUTS). The published precedent for
constraining red noise from single-pulsar runs is the factorized-likelihood / empirical-priors route
(used in NG15 as the CURN-factorized cross-check). Hence the design below pilots fixed noise
(cheapest) but expects to land on empirical priors (defensible), with MDC ground truth as the
referee. The fully joint formulation is deferred to its own methods issue ("joint later").

## 3. W0 — extraction and housekeeping

Dismantle PR #106 (branch `t3.5-full-array`, 68 files) into two successors, then close it with a
comment linking both. Nothing merges unchecked: each extracted piece must pass tests and preserve
golden likelihoods.

**W0.1 — pipeline branch/PR (critical path).** Extract, review, and land:

- Library edits + tests: `python/argus/bayesian_inference.py` (incl. `dense_mass_blocks`),
  `python/argus/data_loader.py`, `python/argus/gravitational_waves.py`,
  `python/argus/jax_kalman_filter.py` (missing-observation support), `python/argus/workflow.py`,
  and the matching `test/` updates.
- Ingest + injection tooling: `scripts/ingest_par_tim.py`,
  `workflows/ng15_sgwb_demo/scripts/build_aligned_feathers.py`, `check_epoch_alignment.py`,
  `inject_powerlaw_gwb.py`, `reduce_ng15_white_noise.py`, `stage_symlinks.py`, injection-truth
  JSONs, `ng15_psr_noise*.json`.
- Evidence machinery: `workflows/ng15_sgwb_demo/scripts/logz_lhm.py`.
- Run harnesses + configs: `run_analysis.py`, `run_curn.py`, the `ng15_*.ini` configs and SLURM
  scripts that M1/M3 will adapt. Drop dead experiments (e.g. the failed
  `target_accept=0.99` qlblock lever ships only as documentation, not as a default config).
- Narrative worth keeping on main: `workflows/ng15_sgwb_demo/notes/` decision records, `PLAN.md`,
  `TASKS.md`, `log.md` additions — review for accuracy during extraction.

**W0.2 — nested-sampling reference PR (not on critical path).** Spin out as a separate PR that can
sit as a citable record: `blackjax_ns_analytic_check.py`, `ns_*.py` scripts,
`mdc2_blackjax_ns.ini`, `blackjax_ns_run.sh`, `ns_scaling_run.sh`, NS notes
(`DECISION_nested_sampling_parked.md`, `t2.6_blackjax_ns_verdict.md`, `ns_numerical_hygiene.md`),
and `research-evaluations/2026-07-09-blackjax-nested-sampling-model-selection.md`.

**W0.3 — issue housekeeping.**

- Close #104 (superseded by #108).
- Correct #109: keep the diagnosis and the negative-result table; strike the Rao-Blackwell
  prescription (see §2); repoint at this roadmap (#111).
- File a deferred "joint noise+GW formulation" issue (see §7).

**Done when:** #106 closed; pipeline PR merged with tests + goldens green; NS PR open; issues tidied.

## 4. M1 — MDC2 dress rehearsal (33 pulsars, known answer)

Dataset: `workflows/data/IPTA_MockDataChallenge2/dataset_2b` (33 pulsars, unevenly sampled, "all
the usual noises"); truth in `group1_gw_parameters.json` (log₁₀A = −14.886) and
`group1_psr_noise.json`.

**Stage A — single-pulsar noise runs.** For each of the 33 pulsars independently: sample its
intrinsic noise parameters (γ_p, σ_p; white-noise scaling as the model exposes it) with no common
process. Cheap and embarrassingly parallel (SLURM array job). Output: per-pulsar posterior samples
saved per pulsar. A secondary check falls out for free: compare recovered per-pulsar noise against
`group1_psr_noise.json`.

**Stage B — fixed-noise pilot array run.** Array-level run with each pulsar's (γ_p, σ_p) fixed at
its Stage-A posterior median; sample only the GW block (h_a, γ_a) + HD correlation. The sampled
dimension collapses to a handful, so this tests wiring, wall-clock, and whether the GW-corner
geometry is benign once the per-pulsar ridge is gone.

**Stage C — empirical-priors run.** Replace the flat/hierarchical per-pulsar priors with priors
built from the Stage-A posteriors, keeping (γ_p, σ_p) sampled but informatively constrained.
Implementation choice (decide at implementation time, simplest first): fit a Gaussian (or truncated
normal) in log-space to each Stage-A marginal; escalate to histogram/KDE numpyro distributions only
if the Gaussian fit is visibly poor. Compare B vs C against injected truth:

- Both pass → adopt C for M3 (propagates noise uncertainty; closer to field practice).
- Only B passes → investigate C's geometry before deciding; C failing truth-recovery outright is
  a red flag worth its own diagnosis.
- Neither passes → stop and diagnose; do not proceed to M3.

**Stage D — evidence.** HD-vs-CURN lnB via the T3.4 LHM machinery (`logz_lhm.py`) on the winning
stage's posteriors, with a CURN-model twin run. Because the sampled dimension is now small, the
"LHM breaks at high-D" failure mode from the joint formulation should not apply — MDC is the
calibration case that confirms this. Report lnB with an uncertainty from repeat estimates
(seeds/bootstrap).

**Acceptance (gates M3):** r̂ < 1.01, no divergence cliff; injected log₁₀A inside the 90% CI;
lnB(HD/CURN) ≥ 3 (≳20:1 odds) with the repeat-estimate scatter small compared to the value. Record the frozen procedure (winning stage, configs, evidence
settings) in a note — M3 runs it unchanged.

## 5. M2 — steep-spectrum de-risk (parallel to M1; gates M3's claims)

Risk (#110): an OU process has a Lorentzian spectrum and may not express steep γ = 13/3 power-law
red noise; if the state-space model cannot express the consensus signal model, cross-check
posteriors are not comparable to NANOGrav's.

- Single simulated pulsar; γ = 13/3 power-law injection using the existing injector
  (`inject_powerlaw_gwb.py` / `data/inject_powerlaw/`).
- Fit 1: plain OU. Fit 2 (if OU fails): small sum-of-OU / CARMA-style mixture (celerite-style
  rational approximation of the power-law kernel) as an extension of the state block.
- Reference: the same dataset through ENTERPRISE/discovery (feathers + tooling already on OzSTAR).
- **Pass:** recovered (log₁₀A, γ) consistent with injection and with the reference run.
- **Kill criterion:** ~1 month of honest effort (per #110). Failure does not kill the roadmap —
  it narrows M3's claims to amplitude + HD correlation (no spectral-index claims) and promotes the
  kernel question to future work.

## 6. M3 — real NG15 full array (68 pulsars, wideband)

Run the M1-frozen procedure, unchanged, on the real data:

- NG15 **wideband** ingest at 68 pulsars (ingest + epoch-alignment tooling from W0.1; wideband
  sidesteps most of the ECORR/per-backend gap, and NANOGrav's own wideband analysis is the published
  comparison point). Keep every pulsar — the missing-observation filter exists so short baselines
  need not be truncated.
- Stage A ×68 (SLURM array, cheap per pulsar) → Stage B/C as decided by M1 → Stage D evidence.
- Kernel per M2's outcome (plain OU or mixture).
- Compute: A100s (milan-gpu); array-level runs sized from the M1 pilot's wall-clock.

**Acceptance:** converged array run (r̂ < 1.01, no divergence cliff); detection-grade positive
lnB(HD/CURN); amplitude consistent with the published NG15 wideband result. That is the headline:
an independent, non-Fourier, time-domain confirmation of the SGWB.

## 7. Deferred — explicitly off the critical path

- **Joint noise+GW sampling** (the "joint later" of the two-stage decision): reparameterization of
  (γ_p, σ_p), Gibbs-within-NUTS, or delayed-acceptance screening with the per-pulsar-parallel CURN
  filter. Gets its own issue in W0.3; potentially its own methods paper.
- **#108 sqrt-parallel associative-scan filter:** parked pending its T-scaling crossover study;
  irrelevant to the geometry wall (same likelihood, same gradients).
- **Evidence-at-scale alternatives** (Savage-Dickey, product-space hypermodel): only if LHM
  misbehaves in M1 Stage D.
- **Narrowband + ECORR hardening:** optional stretch after M3 if the result warrants an
  apples-to-apples narrowband comparison in a paper.

## 8. Risk register

| Risk | Detector | Response |
|---|---|---|
| OU cannot express γ = 13/3 | M2 | Mixture kernel; else narrow M3 claims to amplitude + HD |
| Empirical priors leave a sampling wall | M1 Stage C diagnostics | Bank fixed-noise MDC result + comparison; promote joint track |
| LHM unreliable at 33/68 psr | M1 Stage D calibration | Product-space fallback, tested on MDC first |
| Wideband ingest surprises at 68 psr (formats, epochs) | M3 ingest dry-run before Stage A | Fix in tooling; missing-obs filter already handles gaps |
| Fixed-noise pilot biased by noise/GW covariance | M1 B-vs-C comparison against truth | That is exactly what Stage C exists to absorb |

## 9. Decision log (this design session, 2026-07-19)

1. **MDC detection first; full-array real NG15 is the ultimate goal.** Supersedes the 2026-07-18
   "bank 6-psr and pivot to differentiator" leaning; consistent with #110's Paper A.
2. **Two-stage noise now, joint sampling later.**
3. **#106 extracted and closed** (pipeline PR + NS reference PR), not merged as-is.
4. **Noise flavor decided empirically in M1** (fixed-noise pilot vs empirical priors), leaning
   empirical priors since fixing red noise at point estimates is *not* field-standard.
5. **Wideband is the primary M3 dataset.**
6. **#109's Rao-Blackwell prescription rejected** on technical grounds (nonlinear hyperparameters —
   no analytic marginalization); its diagnosis stands.

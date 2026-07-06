# Research log

## 2026-07-07 — Discovery review → first CW detection on NG15 (F-statistic)

**Goal.** Review NanoGrav's Discovery for anything useful to Argus, then build a first
continuous-wave (CW) detection demo on NANOGrav 15yr data.

**What was tried.**
- Reviewed Discovery (cloned read-only). It's a JAX Gaussian-process (Woodbury) PTA
  code; we keep Argus's Kalman-filter likelihood and take from Discovery only: its
  bundled NG15 dataset, the feather schema, and its CW waveform for cross-checking.
  Wrote `docs/discovery_review.md`.
- Built a Discovery→Argus feather adapter (`scripts/ingest_discovery_feather.py` +
  `scripts/ng15_f0_catalog.json`, F0 sourced from NG15 par files since Discovery
  feathers omit it), folding per-backend white noise into effective errors and mapping
  power-law red noise → OU.
- Cross-check test (`test/test_cw_discovery_crosscheck.py`) found the exact convention
  map: **Argus(ψ) = −0.5·Discovery(−ψ)**.
- Injection-recovery demo (`workflows/ng15_cw_demo/`). First tried NUTS on Discovery's
  **narrowband** feathers → OOM (~202 GB at 6 pulsars) and, when trimmed, ~250 s/iter
  (~11-day ETA). Pivoted to NG15 **wideband** data (~40× fewer TOAs, Argus's native
  scale) via `prepare_demo_wideband.py` (PINT ingest + SVD-recondition the DMX-heavy
  design matrix). NUTS then ran (2h58m) but did not converge (r̂≈1.85, ESS≈3); jaxns
  nested sampling was too slow (>3 h, no result).
- Since the goal is *detection* not parameter estimation, built an analytic
  **F-statistic (F_e) + B-statistic** module (`python/argus/cw_fstatistic.py`, tests in
  `test/test_cw_fstatistic.py`, driver `run_fstatistic.py`): a Kalman whitener reusing
  the filter's building blocks → whitened inner products → `2F_e = XᵀM⁻¹X` and a
  closed-form amplitude-marginalised Bayes factor. All-sky freq×sky scan.

**What was learned.**
- Whitener inner products equal the full Kalman likelihood ratio to machine precision
  (validated) — the F-stat is correct by construction.
- On NG15 wideband, a loud injection is cleanly recovered (2F_e≈5800, SNR≈76) at the
  injected f_gw/RA (Dec at the antenna sky-degeneracy image).
- **Key finding:** the empirical null (no-injection data) has max 2F_e≈900 — real
  **common red noise / the GWB** that the coherent F_e picks up and our per-pulsar-only
  OU model does not remove. A faint injection (h0=5e-14) does not stand out above it; a
  loud one (5e-13) does.
- NG15 wideband par files still carry ~169 DMX params over ~400 epochs → raw design
  matrix is catastrophically ill-conditioned (σ-range ~1e16); SVD-orthonormalising it
  fixes P_eps (cond ~1e4) while preserving the marginalised likelihood.
- Cluster: `milan-gpu` partition flag was "down"; A100 `gina*` nodes reachable via
  `milan-c`.

**Decisions / dead ends.**
- Narrowband data for Argus CW NUTS: ruled out (memory + compute both blow up). Use
  wideband.
- NUTS and nested sampling for CW *detection*: ruled out for now (non-convergent /
  slow on the multimodal posterior). Frequentist F-statistic is the chosen detector.
- Savage-Dickey on a NUTS run: rejected in favour of the analytic B-statistic (avoids
  the h0=0 nesting problem and needs no converged chain).

**Open threads.**
- Add a common (Hellings–Downs / common-uncorrelated) red-noise term to the whitening
  covariance — the ingredient needed to detect faint CWs above the null excess.
- CW strain upper limits vs f_gw; scramble-based background/false-alarm; scale to
  30–40+ pulsars (breaks the sky degeneracy); add F_p and the pulsar term.
- Work is uncommitted on branch `discovery-ng15-cw-demo`; committing + PR next.

## 2026-06-09 — /simplify pass on the CW inference PR (quality only, behavior-preserving)

**Goal.** Reduce duplication/complexity in the continuous-waves branch's CW additions
(~2.5k lines across `python/argus/`) without changing numerical results, inference, or speed.

**What was tried.** Ran a 4-angle review (reuse / simplification / efficiency / altitude) over
`git diff main...HEAD`, verified every finding against the live source, then applied only the
safe, behavior-preserving ones. Built a strict equivalence harness that captured 21 reference
arrays before any edit (both `compute_cw_signal_single_pulsar` variants × 3 branches, all
numpyro sample-site values from a seeded `Predictive` draw, and the jaxns CW prior-name order
obtained by manually driving the generator and intercepting `yield`ed `Prior.name`s).

Changes applied: (1) collapsed the CW scalar reparameterize-or-fix block — copy-pasted 3× in
`parameter_sampling.py` (numpyro `sample_cw_parameters`, jaxns `build_jaxns_cw_prior_model`,
and the key list in `count_free_parameters`) — into one module-level `CW_SCALAR_PARAMS` spec +
a `_sample_cw_scalar_numpyro` helper; removed a dead `import math`. (2) Extracted shared
`_cw_earth_term` from the two CW waveform functions in `gravitational_waves.py`. (3) Deleted the
dead SMC / replica-exchange diagnostics block in `workflow.py` (modules removed in c553794;
`smc_results`/`re_results` never assigned). (4) Minor: `KPC_TO_SECONDS` → module constant,
hoisted two in-`@jax.jit`-body imports, deduped `is_cw` in `utils.py`.

**What was learned.** Equivalence held exactly (21/21 arrays at rtol=1e-12, atol=0); full test
suite 227 passed (CW subset 62). Warm-jit A/B microbenchmark of the waveform functions:
before 259.8/255.3 µs, after 259.3/264.0 µs per call — within ±2-3% run-to-run noise, i.e. no
hot-path cost (jit inlines the helper to the same XLA graph).

**Decisions / dead ends.** Deliberately shipped **none** of the efficiency-agent's hot-path
rewrites. Precomputing polarization tensors is a no-op (XLA LICM already hoists the
loop-invariant out of the antenna vmap). Precomputing `n_hat`/`geometric_factors` at init is
impossible — they depend on `alpha_gw`/`delta_gw`, which are *sampled*, so they change every
likelihood call (the efficiency agent's premise was wrong). `R_scalars` masking and
`pulsar_direction` precompute are immeasurable against the per-pulsar Kalman core that
dominates `_cw_likelihood`. Doing nothing there is what guarantees "as fast or faster." Larger
altitude ideas (a `KalmanFilter`/`Likelihood` base class, a `PriorSpecification` dataclass)
were left out as out-of-scope architecture.

**Open threads.** Pre-existing working-tree edits to `test/conftest.py` and
`test/test_data_loader.py` were present at session start and are unrelated to this pass. PR on
`continuous-waves` not yet merged to main.

## 2026-06-02 — Validated the Argus CW likelihood; f_gw grid recovers the injection

**Goal.** Settle whether the Argus Kalman-filter CW likelihood agrees with the standard
PTA tool (enterprise) on IPTA MDC2 dataset_3b, then attack the pulsar-term f_gw
multimodality with a fixed-f_gw grid search — as a step toward running on real data.

**What was tried.**
1. Built a "Level-1" agreement gate (`workflows/cw_shared/level1_likelihood_agreement.py`):
   sweep each of the 7 CW params through the injection in both codes (Argus earth-term vs a
   local fixed-noise earth-term enterprise PTA) and compare normalized 1-D profile shapes.
2. The gate appeared to show Argus failing (shallow profiles, peaks off injection) while
   enterprise peaked at injection. Treated as a bug and ran systematic debugging.
3. Ruled out hypotheses one at a time: (a) my flat default noise — refuted, the existing
   Level-3 NUTS posterior misses the injection identically and fit sigma_p->~1e-18;
   (b) red-noise modelling — refuted by `diag_noise_hypothesis.py`: enterprise with red
   noise OFF still recovers the injection; (c) waveform amplitude — verified correct
   (Argus CW signal RMS ~0.45us, matches analytic h0/(2*Omega) x antenna patterns).
4. Looked at scales: data residual RMS ~10us, white noise ~4.6us, CW signal ~0.45us ->
   matched-filter SNR ~7. Argus profile depth (~30) matched that; my enterprise profiling
   depth (~7000) was ~100x too confident. Checked the official Level-2 PTMCMC posteriors:
   BOTH Argus and enterprise detect, nail log10_f_gw, and land on the SAME biased sky mode
   (~alpha=1.46), missing the injection together.
5. Wrote a controlled injection test (`workflows/cw_shared/controlled_injection_test.py`):
   inject a loud known CW (SNR~50) into clean white-noise residuals on the real geometry,
   same residuals to both codes, check peak recovery + detection statistic vs SNR^2/2.
6. Built and ran the f_gw grid (`workflows/cw_level4b_fgw_grid/`): 40 fixed-f_gw,
   pulsar-term, phase-reparam NUTS runs on A100s (SLURM array 12746776), then aggregated
   into a max-logL profile + detection statistics.

**What was learned.**
- **Argus's CW likelihood is correct and correctly normalized.** In the controlled test it
  recovers all 7 params essentially exactly and gives a detection statistic = 0.95 x the
  theoretical SNR^2/2 (the 5% is timing-model marginalization). The Kalman-filter likelihood
  is sound.
- The MDC2 "failure" is **real-data hardness, not a bug**: marginal SNR (~7) + injected
  power-law red noise + genuine multimodality of single-source CW sky/incl/psi localization.
  Both official codes agree with each other and miss the injection together.
- **The f_gw grid works**: the max-logL profile shows a sharp clean peak at the injection
  (peak at log10_f=-8.28 vs injection -8.215), ~80-120 logL above a flat baseline, with
  divergences lowest at the peak and high off-peak (sampler thrashes at wrong f).

**Decisions / dead ends.**
- The Level-1 1-D-profile gate is the WRONG validation for a multimodal likelihood: sweeping
  through the injection only tests whether the injection is a *local* mode, missing deeper
  modes. Superseded by the controlled injection test as the validation of record.
- My enterprise comparison harness (fixed-noise white-only PTA + overriding `psr._residuals`)
  is UNRELIABLE — in the controlled test it gave a *negative* detection statistic
  (impossible), because overriding residuals leaves enterprise's internal caching/design
  matrix inconsistent. Do not trust it; the trustworthy enterprise reference is the official
  Level-2 PTMCMC run on real data.
- The per-point Savage-Dickey BF in the grid aggregator is confounded by pulsar-term chi-phase
  overfitting (reads inf at low f). Use the max-logL profile as the detection signal.
- Grid index 15 (-8.231, the exact-injection point) timed out twice (slowest geometry) but is
  bracketed by 14 and 16, so the peak is fully resolved; not worth backfilling.
- Path gotcha: grid configs live in runs/ (one level deeper than level4a's config), so the
  template's `../data` relative path failed; `generate_configs.py` now absolutises data paths.

**Open threads.**
- The sky/inclination/polarization multimodality is still unaddressed (inherent to single
  source at this SNR; the grid fixes frequency only). Would need restarts or a tempered
  angular kernel for full recovery.
- Real-data pivot (the actual goal): target NANOGrav 15yr. Prerequisites surfaced —
  ECORR/jitter (Argus white noise = (efac*err)^2 + equad^2, no jitter term), noise
  marginalization (OU process can't fit steep power-law red noise -> fix noise from
  single-pulsar runs), and the detected GWB foreground (must be modelled or it biases the CW
  search). Recommended first step: single-pulsar NANOGrav 15yr ingestion test to check the
  loader handles real ECORR/DMX before a full search.

# Research log

## 2026-07-08 — SGWB injector (T2.1), a get_Q_block bug, and MDC2 GPU re-validation (T0.1)

**Goal.** Build the Stage-2 GWB injector (`workflows/ng15_sgwb_demo`, T2.1): inject a
synthetic HD-correlated SGWB into the epoch-aligned NG15 feathers to de-risk the central
question — can Argus's single-corner OU recovery absorb a true power-law GWB without biasing
the amplitude?

**What was tried.** Designed `scripts/inject_powerlaw_gwb.py` (CPU, numpy-only) with two
modes: `powerlaw` (true `f^-13/3` via a frequency-domain Fourier-sum GP, enterprise PSD
convention) and `ou` (forward-sim of Argus's own OU generative model as the self-consistent
control). Both replace residuals with pure-synthetic signal on the real geometry, keep all
other feather fields, add fixed-value white noise, and record injected truth (PSD at the
Fourier freqs + pivot amplitudes) for a shape-agnostic comparison. Framing agreed with the
user first: neither power-law nor OU is "the truth"; a PTA only constrains ~1 decade of
frequency, so the robust observable is the band-referenced amplitude, not the spectral index.

Building the OU control exposed a bug: its residuals came out ~500× larger than the
power-law injection (75–294 µs vs 200–520 ns). Traced to `python/argus/model.py`
`get_Q_block`: the integrated-OU **position** process-noise `q11` was divided by `γ**3`
instead of `γ**2`, inflating it by exactly `1/γ` (~1e9 at PTA `γ~1e-9`). Confirmed against the
exact integral `∫₀^dt[(1-e^{-γτ})/γ]²dτ` three ways (series, quadrature, `dt³/3` limit);
`q12`/`q22` were correct. Fixed on an isolated branch `fix-qblock-q11-normalization` (PR #101
to `main`), merged into `ng15-sgwb-demo`; added regression tests and bumped the MDC2 golden
log-likelihood (55963.86→63618.93). With the corrected `q_block`, the OU control naturally
matched the power-law RMS at the default `log10_ha=-14.35`.

Then re-validated on GPU (T0.1): a lite MDC2 GWB+HD+NUTS run (only `log10_ha`/`log10_gamma_a`
sampled; red+white noise fixed). Took four SLURM submissions to get a valid run.

**What was learned.** The fix is confirmed on GPU: likelihood 63618.81, 0 divergences,
`r_hat` 1.00–1.01, robust interior posterior `log10_ha≈-12.88`, `log10_gamma_a≈-8.08` (narrow
and widened priors agree → genuine mode, not a runaway). Crucially, the recovered amplitude
shifted from the buggy run's −15.5 to −12.88 because the fix changes the `ha`→residual-amplitude
scaling: r-noise is now `∝ ha²·γa` (was `∝ ha²`, `γa` cancelled). This is exactly the
`(ha,γa)`↔physical-amplitude mapping the PLAN said Stage 2 must establish — the bug had been
corrupting it. Consequence: all Stage-2/3 `log10_ha` priors must re-centre to ~−12…−13.

**Decisions / dead ends.** (1) First SLURM job failed instantly (0% resource, 63 s) — `set -e`
aborts on the benign non-zero return from `~/.bashrc`/conda-init before any output; removed it
(existing example scripts omit it too). (2) `milan-gpu` briefly flapped `PartitionDown`; a
`--partition=milan-c` override was silently ignored (site GPU-routing forces `milan-gpu`), but
it came back up. (3) The first "COMPLETED" run was a FALSE PASS: it printed the *old* likelihood
55963.87 because `argus` is pip-installed **editable** pointing at the main checkout
`/fred/oz022/tkimpson/Argus` (no fix), and `run_analysis.py` only `sys.path.append`s the repo
`python/` dir — the append loses to the editable install. So GPU runs silently ignore
treehouse-worktree edits. Worked around with `export PYTHONPATH=<worktree>/python` (prepends) +
a log line proving `argus.model.__file__` and the q11 divisor. This contradicts PLAN §3's
"Argus is not pip-installed" claim (to be corrected). The clean long-term fix is merging PR #101
so the editable target serves the fix.

**Open threads.** PR #101 awaiting merge. Next task T2.2 (lite injection-recovery config →
`data/inject_powerlaw`/`data/inject_ou`) needs the re-centred `log10_ha` prior and, until #101
merges, the PYTHONPATH hack in its SLURM script. Then T2.3/T2.4 are the actual OU-vs-power-law
decision gate. The injector's red-noise mode is built but off by default (needs per-pulsar
γp/σp, not in `ng15_psr_noise.json`).

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

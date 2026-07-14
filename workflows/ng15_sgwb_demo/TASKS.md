# TASKS — NG15 wideband SGWB demo

> Read [`PLAN.md`](./PLAN.md) first (context, environment, data paths, code reuse map).
> Then do **the next unchecked task** below, in order. Each task lists its
> dependency, whether it needs a GPU, the concrete work, and how to know it's done.
> Check the box (`[x]`) and add a one-line result note when a task is complete.

## Conventions for whoever picks this up
- Branch: **`ng15-sgwb-demo`**. Commit only via the **`/commit`** skill. **No AI attribution
  anywhere** — no `Co-Authored-By` trailer in commits, no "Generated with Claude Code" footer in
  PR bodies/comments.
- Env: activate **`Argus`** (`conda activate Argus`), never `argus-env`. See PLAN §3.
- CPU work: prefix with `JAX_PLATFORMS=cpu`. GPU/NUTS work: **SLURM only** (A100, `oz022`, `milan-gpu`/`milan-c`).
- Prefer new scripts under `scripts/` over editing `python/argus/*` (see PLAN §7).

---

## CURRENT STATUS / NEXT STEP

- ✅ Strategic direction decided (SGWB-first; PR #100 held). Branch `ng15-sgwb-demo` created.
- ✅ Environment resolved (`Argus` env; GPU-init hangs interactively → SLURM only).
- ✅ Stage 0 attempted on CPU and killed (too slow); to be run properly on SLURM or folded in.
- ✅ T1.1 done — workflow skeleton created (`run_analysis.py` clone imports cleanly; feather gitignore added).
- ✅ T1.2 done — `stage_symlinks.py`; 6-pulsar subset staged to `data/staging_subset/` (gitignored).
- ✅ T1.3 done — 6 feathers ingested (PINT backend + zero-col drop; both fixes in `ingest_par_tim.py`,
  MDC2-safe). Ragged TOA counts (364–1493) → RISK A is next.
- ✅ T1.4 done — `check_epoch_alignment.py`; **RISK A cleared, verdict FEASIBLE** at 30-day cadence
  → 78 joint epochs (>50 floor), per-pulsar retention ≥47%. Common window MJD 54401→58940 (12.4 yr).
- ✅ T1.5 done — `build_aligned_feathers.py`; **RISK A RESOLVED.** 6 aligned feathers in
  `data/aligned/` (78 joint epochs each); `get_processed_residuals(mode="gwb")` returns clean
  `(78,6)` residual/error matrices + unit-diagonal `(6,6)` HD matrix. Key finding: dropped per-epoch
  `DMX_*` cols (149–397/pulsar → 9–26 astrophysical cols) so `P_eps` is finite/full-rank and signal
  survives the timing marginalization (full binned design matrix has rank == 78 = nepoch, would else
  eat the GWB). DMX-drop is the accepted "no DM-noise" simplification (PLAN §2/§4) — flag in README (T4.2).
- ✅ T1.6 done — `scripts/reduce_ng15_white_noise.py`; per-backend white noise collapsed to
  `data/ng15_psr_noise.json` (6 pulsars, TOA-count-weighted variance-preserving; efac 1.00–1.19,
  log10_equad −6.32→−7.20). Loads clean via `utils.get_efac_equad_injections`. **Stage 1 complete.**
- ✅ T2.1 done — `scripts/inject_powerlaw_gwb.py` (CPU, numpy-only). Two modes on the real aligned geometry:
  `powerlaw` (frequency-domain Fourier-sum GP, log10_A_gw=−14.6, γ=13/3) and `ou` (forward-sim of Argus's own
  OU generative model, the T2.3 control). White noise on by default (from `ng15_psr_noise.json`); red noise a flag
  (off). Both write to `data/inject_{powerlaw,ou}/` + `injection_truth.json` (records injected PSD at the Fourier
  freqs + pivot amplitudes at f=1/yr and 1/(5yr) → shape-agnostic comparison per PLAN framing). Both verify clean
  via `get_processed_residuals(mode="gwb")` → (78,6) + unit-diag HD. RMS: powerlaw 211–520 ns, OU 99–527 ns
  (matched at default log10_ha=−14.35).
- ⚠️ **Library bug found + fixed while building the OU control** (committed on branch
  `fix-qblock-q11-normalization`, merged into `ng15-sgwb-demo`): `model.get_Q_block` divided the integrated-OU
  position process-noise `q11` by `γ**3` instead of `γ**2`, inflating it by `1/γ` (~1e9 at PTA γ). Fixed to `γ**2`
  (verified vs exact quadrature + dt³/3 limit; q12/q22 were correct). Added regression tests; updated the MDC2
  golden log-likelihood 55963.86→63618.93. Full suite 234 passed, 1 skipped. **Full MDC2 NUTS *recovery*
  re-validation still pending on GPU/SLURM** — fold into T0.1 or the first T2.3 run before trusting Stage 3 numbers.
- ✅ T2.2 done — `configs/ng15_config_lite.ini` (abs paths; white noise FIXED, red noise FREE/hierarchical;
  GW priors bracket the OU truth log10_ha≈−14.35). Parses + finite dry likelihood on BOTH inject dirs
  (inject_ou 6407.54, inject_powerlaw 6414.72). `excluded_psrs=__NONE__` sentinel needed (empty value would
  exclude all pulsars via `'' in psr` in `utils.get_noise_parameters`).
- ✅ T2.3 done — `slurm_scripts/ng15_slurm_run.sh` (parameterized `MODE=ou|powerlaw`). OU control
  PASSED on A100 (job 14018847): recovered log10_ha=-14.90±0.47 (truth -14.35, 1.16σ),
  log10_gamma_a=-8.71±0.58 (truth -8.5, 0.36σ), 0 divergences. **Harness validated.** GPU note:
  OzSTAR reroutes all GPU jobs to milan-gpu (milan-c is NOT a GPU fallback).
- ✅ **T2.4 done + hi-res CONFIRMED (DECISION GATE PASSED).** Power-law→OU: band-referenced amplitude
  recovered **within ~0.5σ** (bias −0.20 dex/−0.46σ at f=1/(5yr); +0.12 dex/+0.25σ at f=1/yr), stable
  lite→hi-res, and no worse than the OU→OU control's own −0.74σ. OU absorbs the power-law band
  amplitude; the *spectral shape/index is not recovered* and the power-law→OU posterior is
  **pathological** (hi-res r_hat 1.051, ESS 51, 50 divergences vs control 1.004/2496/0). New:
  `scripts/compare_ou_recovery.py`, `configs/ng15_config_confirm.ini`, `slurm_scripts/ng15_confirm_run.sh`.
- ✅ **T2.6 done — blackjax NS evidence engine PASS (2026-07-09).** Runs on the pinned jax 0.4.38
  (blackjax-devs main, no cascade, `jax.shard_map` shim); analytic logZ correct; on MDC2 the NS
  posterior reproduces NUTS exactly and yields logZ. Backend behind `sampler=blackjax`.
  See `notes/t2.6_blackjax_ns_verdict.md`. **⚠️ SUPERSEDED for production — NS parked (below).**
- 🛑 **DECISION (2026-07-10): slice nested sampling PARKED as the production evidence engine.**
  The NS cost-scaling study found NS is ~10–30× slower than NUTS at *lower* dimension (accurate
  full-32 hierarchical evidence run ≈ 2.5–7 weeks vs NUTS ~2 days), because slice-NS discards the
  gradient NUTS exploits. Sampler dimensional scaling is benign and accuracy is tunable
  (`num_inner_steps~6D`) — runtime is the killer. **Evidence/Bayes factors now route via NUTS +
  a posterior-reuse estimator (learned harmonic mean first), NOT NS** — this changes T3.4's method.
  Also fixed a Kalman near-singular-covariance pathology NS exposed (`_log_likelihood`, PR #102 to
  main; golden preserved). Full record: `notes/DECISION_nested_sampling_parked.md`,
  `notes/ns_numerical_hygiene.md`. Do **not** re-investigate slice-NS scaling.
- ✅ **T3.1 done — production config `configs/ng15_config.ini`** on the *real* aligned NG15
  feathers. Cloned `ng15_config_confirm.ini`; NUTS 2000/1000/4 @ target_accept=0.95, `dense_mass`,
  `log10_ha`∈[-17,-11], `output_id=ng15_real`. **`log10_gamma_a` kept FREE** — re-verified the
  `fixg` "invariant" claim is false (fixing γ_a biases the band amplitude 1.3-1.9 dex low). Aligned
  feathers regenerated (gitignored; T1.2→T1.3→T1.5). Verified: parses + dry likelihood finite (6462.27).
- ✅ **T3.2 done — production SLURM script `slurm_scripts/ng15_production_run.sh`** (4 × A100).
  New variant adapted from `ng15_confirm_run.sh`, simplified: no `MODE`/`sed` derived-config
  (the T3.1 config is self-contained and lives outside `outputs/`), runs `configs/ng15_config.ini`
  directly. `--gres=gpu:4` matched to `num_chains=4`; `Argus` env, `oz022`, `milan-gpu`,
  `PYTHONPATH`→worktree (q11 fix), full env/q11 pre-flight block. Walltime 8 h (confirm's 2000
  iters took ~24 min; production is 3000 iters @ target_accept=0.95). Verified: `bash -n` clean +
  `sbatch --test-only` schedules on milan-gpu (gina17, 4 GPUs).
- ✅ **T3.3 done — real-data subset run** (job 14108726, 4 × A100 gina14, 70m49s wall, COMPLETED).
  **Converged cleanly:** max r_hat 1.003, log10_ha ESS 2434, only **4 divergences / 8000 (0.05%)**
  — target_accept=0.95 crushed the ~50 seen at confirm. Recovered **log10_ha = −13.40 ± 0.20**,
  off both prior edges (2.4 dex clear of the −11 top); log10_gamma_a = −7.73, not railed. **Band
  amplitude consistent with the published NG15 SGWB** (fixed γ=13/3, log10_A=−14.6): bias
  −0.02 dex/−0.09σ at f=1/yr and −0.19 dex/−0.86σ at f=1/(5yr) — i.e. <1σ at both pivots; log10_ha
  brackets the T2.4-mapped −13.5 to ~0.5σ. Outputs in `outputs/ng15_real/` (+ `comparison.json`,
  spectral overlay, corner). Verified via extended `scripts/compare_ou_recovery.py --published`.
- 👉 **NEXT: T3.4** (HD-vs-CURN contrast). Evidence/Bayes factor via NUTS + posterior-reuse
  (learned harmonic mean first, validated against the MDC2 anchor logZ=63780); NS parked. Fallback:
  amplitude/fit contrast between HD and identity-correlation runs (RISK-B caveat).

---

## Stage 0 — Baseline sanity (optional; GPU/SLURM)

- [x] **T0.1 — Smoke-test the GWB+HD+NUTS path on MDC2 via SLURM.** *(done — became high-value after the q11 fix)*
  - ✅ Result: `configs/mdc2_smoke_lite.ini` (abs paths; red+white noise FIXED → only log10_ha/log10_gamma_a
    sampled) + `slurm_scripts/mdc2_qfix_smoke.sh` (A100, `Argus` env, milan-c fallback). **PASS with the q11 fix:**
    likelihood 63618.81 (corrected value), 0 divergences, r_hat 1.00–1.01, n_eff ~245–300; converges to a robust
    INTERIOR mode log10_ha=−12.88±0.05, log10_gamma_a=−8.08±0.13 (narrow & widened priors agree → not a runaway).
    **Two gotchas found & handled:** (1) SLURM `set -e` aborts on ~/.bashrc/conda-init before any output — removed.
    (2) argus is pip-installed EDITABLE→`/fred/oz022/tkimpson/Argus` (main checkout, no fix); `run_analysis`'s
    `sys.path.append` loses to it, so the FIRST run silently used old buggy code (likelihood 55963.87). Fixed by
    `export PYTHONPATH=<worktree>/python` in the SLURM script (script now logs `argus.model from:` + the q11 line to
    prove provenance). See [[project_argus_editable_install_gotcha]]. **Consequence:** all Stage-2/3 log10_ha priors
    must be re-centred for the corrected ha-scaling (peak near −12 to −13, not −15), and the fix should land on `main`
    so GPU runs stop needing the PYTHONPATH hack.
  - Depends on: nothing. GPU: yes (1 × A100).
  - Purpose: confirm the mature GWB path is healthy in this checkout before adding NG15 complexity. Low value because the branch has no library changes and the path is JOSS-validated — **skip or fold into the first real GPU run (T2.3) unless something looks off.**
  - Do: write a corrected SLURM script (activate `Argus`, `--account=oz022`,
    `--partition=milan-gpu`, `--gres=gpu:1`) that runs
    `workflows/example_workflow_lite/run_analysis.py configs/example_config_lite.ini`.
    Submit with `sbatch`; monitor `squeue -u $USER`.
  - Done when: run exits 0; log shows a finite likelihood (`test_likelihood_performance`),
    NUTS `print_summary` with `r_hat ≈ 1` for `log10_ha`/`log10_gamma_a`, no divergence
    blow-up; a corner plot renders under `outputs/01/`.

---

## Stage 1 — Ingest NG15 wideband + resolve epoch alignment (CPU)

- [x] **T1.1 — Create the workflow skeleton.**
  - Depends on: nothing. GPU: no.
  - Do: create `configs/`, `scripts/`, `data/`, `slurm_scripts/` under
    `workflows/ng15_sgwb_demo/`. Clone `run_analysis.py` from
    `workflows/example_workflow/run_analysis.py` (keeps `use_gw=True`, x64). Add a
    `.gitignore` entry (or rely on repo root) if the ingested feathers/data are large.
  - Done when: the directory tree in PLAN §8 exists and `run_analysis.py` imports cleanly
    (`JAX_PLATFORMS=cpu python -c "import run_analysis"` from the dir, or a `--help`).
  - ✅ Result: dirs created (`.gitkeep` in each empty one); `run_analysis.py` cloned with
    NG15 banner, imports OK under `Argus` env. Root `.gitignore` gets
    `workflows/ng15_sgwb_demo/data/**/*.feather` (small JSON stays tracked, large feathers ignored).

- [x] **T1.2 — Stage co-located par+tim for the pulsar subset.**
  - Depends on: T1.1. GPU: no.
  - Do: write `scripts/stage_symlinks.py` (or a documented shell step) that symlinks the
    subset's `.wb.par` and `.wb.tim` from the NG15 wideband `par/` and `tim/` dirs (PLAN
    §4) into a single staging dir, excluding `...ao`/`...gbt` variants. Subset = J1909-3744,
    J1713+0747 first, then J1744-1134, J0613-0200, B1855+09, J1600-3053.
  - Done when: staging dir has matched `.par`+`.tim` for each subset pulsar, no telescope
    duplicates, equal counts.
  - ✅ Result: `scripts/stage_symlinks.py` created; staged 6 pulsars (6 par + 6 tim symlinks)
    into `data/staging_subset/`. Canonical `{PSR}_PINT_*.wb.{par,tim}` glob excludes `ao`/`gbt`
    variants. NG15 uses **B1855+09** (no J1857+0943 file). Two wrinkles handled: (a) symlinks
    named canonically `{PSR}.wb.par`/`.tim` so `ingest_par_tim.py`'s positional-zip pairs them
    despite J1600-3053's mismatched par (20230202) / tim (20230224) date stamps; (b) staging
    dir gitignored (symlinks → machine-specific cluster paths).

- [x] **T1.3 — Ingest the subset to feathers.**
  - Depends on: T1.2. GPU: no (needs `enterprise` — Argus env has it).
  - Do: `JAX_PLATFORMS=cpu python scripts/ingest_par_tim.py <staging_dir> data/` (from repo
    root, adjust relative paths). Verify each feather round-trips via
    `LoadWidebandPulsarData.read_feather` with finite design matrix, `P_eps`, RA/DEC, F0.
  - Done when: one `<pulsar>.feather` per subset pulsar in `data/`, all round-trip clean.
  - ✅ Result: 6 feathers written; all round-trip clean (finite `M_scaled`/`P_eps`/RA/DEC/F0,
    no RuntimeWarnings). TOA counts are ragged (364/423/481/1493/433/833 → RISK A for T1.4/T1.5).
    **Two backward-compatible fixes to the shared `scripts/ingest_par_tim.py` (a data-prep
    script, not `python/argus/*` library):**
    - `--timing-package pint`: NG15 par/tim are PINT-format (`CLOCK TT(BIPM2019)`, `EPHEM DE440`);
      enterprise's default tempo2/libstempo backend aborts at C level with
      `ERROR [CLK4]: Date -nan out of range of TDB-TDT table`. PINT loads them cleanly. Default
      `None` keeps prior tempo2 behavior, so MDC2 is unaffected.
    - `drop_degenerate_columns()`: NG15 design matrices have all-zero columns (`DMJUMP*`, one cut
      `DMX_0239` epoch) → singular `MᵀN⁻¹M` → non-finite `P_eps` that poisons the Kalman filter.
      Dropped them (2–7 cols/pulsar, all DM-related). Likelihood-preserving (zero column adds
      nothing to `Mβ`; Argus models only timing residuals, no DM). Verified **no-op on MDC2**
      (8→8 cols). Command: `... ingest_par_tim.py data/staging_subset workflows/ng15_sgwb_demo/data
      --timing-package pint --overwrite`.

- [x] **T1.4 — Epoch-alignment diagnostic (RISK A gate).**
  - Depends on: T1.3. GPU: no.
  - Do: write `scripts/check_epoch_alignment.py` — load each subset feather; report TOA
    counts per pulsar, epoch spacing distribution, and overlap across pulsars; state
    whether a common epoch grid is feasible and at what cadence (~monthly is the
    starting guess).
  - Done when: the script prints a clear feasibility verdict + suggested binning cadence.
    **If alignment looks infeasible even after binning, STOP and flag for review** (this
    is the make-or-break risk; a masked-epoch library extension is a last resort — PLAN §7).
  - ✅ Result: `scripts/check_epoch_alignment.py` created (reuses `read_multiple_feather`, which
    skips the shape check; TOAs sec→MJD; cadence sweep computes joint epochs = grid cells
    occupied by *all* pulsars). Run: `JAX_PLATFORMS=cpu python
    workflows/ng15_sgwb_demo/scripts/check_epoch_alignment.py`. **Verdict FEASIBLE.** Common
    baseline overlap MJD 54401→58940 (12.4 yr). Cadence sweep (joint epochs / min-retention):
    7 d→18/6%, 14 d→45/19%, **30 d→78/47%**, 60 d→59/66%. Recommended **30-day grid → 78 joint
    epochs** (sets joint filter `nepoch`); above the 50-epoch viability floor. Alignment cost is
    real but reported (≈1801/4027 TOAs dropped at 30 d) — coarser 60 d retains more per-pulsar
    data but yields fewer joint epochs. Feeds T1.5.

- [x] **T1.5 — Build epoch-aligned feathers (RISK A resolver).**
  - Depends on: T1.4 (feasible). GPU: no.
  - Do: write `scripts/build_aligned_feathers.py` — bin each pulsar's wideband TOAs onto
    the common epoch grid from T1.4, producing equal-length, epoch-aligned residual/error
    series, re-written as feathers the stock GWB loader accepts (`save_feather`).
  - Done when: `LoadWidebandPulsarData(...).get_processed_residuals(mode="gwb")` on the
    aligned `data/` returns **without error**, with `(nepoch, Npsr)` residual/error
    matrices and an `(Npsr, Npsr)` HD matrix (unit diagonal).
  - ✅ Result: `scripts/build_aligned_feathers.py` (CPU; no library edits). Reuses T1.4 grid math
    (30-day union-window grid → **78 joint epochs**), inverse-variance epoch-averages
    residuals/TOAs/errors/design-rows within each joint bin (identical weights preserve the linear
    timing model `r=Mβ+n`; combined err = 1/√Σ(1/σ²)), and writes to `data/aligned/*.feather` via the
    stock `save_feather` schema. **The load-bearing subtlety:** after binning to 78 epochs the *full*
    design matrix has rank exactly 78 (verified), so an unreduced timing model would absorb the entire
    GW signal and `MᵀN⁻¹M` is singular (non-finite `P_eps`). Fix = drop per-epoch `DMX_*` columns by
    fitpars name (149–397/pulsar → **9–26 astrophysical cols**; the accepted PLAN §2/§4 "no DM-noise"
    simplification, flag in README T4.2) + all-zero drop. Result: `P_eps` finite/full-rank for all 6
    (cond 1e2–6e11; J1713/J1600 stiff but tolerated); a likelihood-exact SVD fallback is coded but not
    triggered. Verification (built into the script) passes: `get_processed_residuals(mode="gwb")` →
    residuals/errors `(78,6)`, HD `(6,6)` unit-diagonal; `process_pulsar_residuals_by_epoch` no longer
    raises. Run: `JAX_PLATFORMS=cpu python workflows/ng15_sgwb_demo/scripts/build_aligned_feathers.py
    --overwrite`. Aligned feathers gitignored via `data/**/*.feather`.

- [x] **T1.6 — Reduce NG15 per-backend white noise to a per-pulsar JSON.**
  - Depends on: T1.3. GPU: no.
  - Do: write `scripts/reduce_ng15_white_noise.py` — parse the subset's
    `noise/{PSR}.wb.pars.txt`, collapse per-backend EFAC/EQUAD to one effective per-pulsar
    `efac`/`equad` (e.g. TOA-error-weighted), and write `data/ng15_psr_noise.json` in
    Argus schema `{psr: {"efac":…, "equad":…}}` (`equad` stored as log10; consumed by
    `utils.get_efac_equad_injections`). Drop ECORR/DM-noise (Argus has none).
  - Done when: `data/ng15_psr_noise.json` loads via `utils.get_efac_equad_injections`
    without error for the subset.
  - Note: red noise starts **free/hierarchical** (omit `spin_injections_path`) — no
    power-law→OU conversion needed yet.
  - ✅ Result: `scripts/reduce_ng15_white_noise.py` (CPU; no library edits). **Key finding:**
    `.wb.pars.txt` holds only param *names* (column labels); values live in the PTMCMC
    `.wb.chain_1.txt` (`N_rows×(N_params+4)`, trailing 4 = logpost/loglik/accept/swap). Per
    backend takes the **posterior median** of `_efac` (linear) and `_log10_t2equad` (log10 s;
    delogged) after 25% burn-in, dropping `dmefac`/`log10_dmequad`/`red_noise_*`. Collapse is
    **TOA-count-weighted, variance-preserving** to match Argus's `R=(efac·σ)²+equad²`
    (`model.py:112`): `efac_eff=Σwₑ·efacₑ/Σwₑ`, `equad_eff=√(Σwₑ·equadₑ²/Σwₑ)`, stored as
    `log10`. Weight `wₑ` = per-backend TOA count from the tim `-f` flag (the `{flag}` token
    equals the param-name backend token exactly). JSON keys inserted in
    `sorted(glob("*.feather"))` order (`data_loader.py:259`) so the positionally-consumed
    efac/equad arrays align with the residual matrix. Ran
    `JAX_PLATFORMS=cpu python workflows/ng15_sgwb_demo/scripts/reduce_ng15_white_noise.py
    --overwrite` → `data/ng15_psr_noise.json` (6 pulsars, efac 1.00–1.19, log10_equad
    −6.32→−7.20, all in the 0.1–3 / −9→−5 sanity range). Verified: loads via
    `utils.get_efac_equad_injections` → two length-6 finite positive arrays, keys in sorted
    order. Small JSON is git-tracked (only `*.feather` gitignored). TOA-count-weighting is an
    accepted demo approximation (single scalar/pulsar, no ECORR/DM-noise) — flag in README T4.2.

---

## Stage 2 — De-risk OU-vs-power-law (GPU/SLURM, cheap)

- [x] **T2.1 — Power-law GWB injector.**
  - Depends on: T1.5. GPU: no (generation is cheap; run on CPU).
  - Do: write `scripts/inject_powerlaw_gwb.py` — inject an HD-correlated **true power-law**
    GWB (`log10_A_gw=-14.6`, `γ=13/3`) plus per-pulsar red noise into the aligned feathers'
    real sampling geometry (real cadences/errors/RA-DEC → real `Γ_HD`). Also support an
    **OU-GWB** injection mode for the control (T2.3). Write injected feathers to a separate
    dir (e.g. `data/inject_powerlaw/`, `data/inject_ou/`).
  - Done when: injected feathers load and process in GWB mode; injected truth values are
    recorded alongside for comparison.
  - ✅ Result: `scripts/inject_powerlaw_gwb.py` (CPU, numpy-only, no `argus` runtime path except loaders).
    `--mode powerlaw`: Fourier-sum GP, per-mode sin/cos coeffs ~N(0,(P(f_k)/T)·Γ_HD), P(f)=A²/(12π²)(f/f_yr)^−γ
    f_yr^−3 (enterprise convention; f_yr=Julian yr), evaluated at each pulsar's own TOAs. `--mode ou`: forward-sim
    of Argus's interleaved `[r,a]` GW state (σa2=(ha²/12)γa Γ_HD, init a~N(0,ha²Γ/24), residual=−r) via numpy ports
    of `get_F_block`/`get_Q_block`. HD from RA/DEC (same fn as recovery). White noise on by default (indexed by
    name from `ng15_psr_noise.json`, not positional); red noise a flag needing explicit (γp,σp). Residuals REPLACE
    the real ones (pure synthetic); all other feather fields kept → real geometry preserved. `injection_truth.json`
    records the injected PSD at Fourier freqs + pivot amplitudes (f=1/yr, 1/(5yr)) + seed. Both modes verify clean:
    `get_processed_residuals(mode="gwb")` → (78,6) residuals/errors, (6,6) unit-diag HD, all finite. RMS powerlaw
    211–520 ns, OU 99–527 ns (variance-matched at default log10_ha=−14.35). Commands:
    `JAX_PLATFORMS=cpu python workflows/ng15_sgwb_demo/scripts/inject_powerlaw_gwb.py --mode {powerlaw,ou} --overwrite`.
    **NB:** building the OU control surfaced the `get_Q_block` `q11` `γ**3` bug (fixed — see status note above); the
    injector's numpy `q_block_np` matches the corrected `γ**2`.

- [x] **T2.2 — Lite injection-recovery config.**
  - Depends on: T2.1. GPU: no.
  - Do: write `configs/ng15_config_lite.ini` (clone `example_config_lite.ini`): `data_path`
    → injected feather dir, `noise_params_path` → `ng15_psr_noise.json`, red noise
    free/hierarchical, lite NUTS (200/100/2). Widen `log10_ha_min/max` to bracket the
    injected amplitude.
  - Done when: config parses and a dry likelihood eval is finite (CPU ok for the eval).
  - ✅ Result: `configs/ng15_config_lite.ini` (cloned `mdc2_smoke_lite.ini`'s patterns —
    ABSOLUTE paths, corrected-q11 framing). `data_path` defaults to `data/inject_ou/` (T2.3
    control); flip one line to `data/inject_powerlaw/` for T2.4. White noise FIXED via
    `ng15_psr_noise.json` (matches injected truth); red noise FREE/hierarchical (empty
    `spin_injections_path`). GW priors bracket the OU truth: `log10_ha` [-17,-11] (value
    -14.35), `log10_gamma_a` [-10.5,-7] (value -8.5) — centred on the *known synthetic* OU
    amplitude, with upward room for the T2.4 power-law→OU mapping. Lite NUTS 200/100/2.
    **Gotcha found:** `excluded_psrs` must be a NON-MATCHING sentinel (`__NONE__`), not blank —
    `utils.get_noise_parameters` passes it raw (no blank-filtering, unlike `workflow.py`), so
    an empty value → `['']` and `'' in psr` matches every pulsar → all excluded → empty noise
    arrays → broadcast error. Verified (CPU, worktree/q11-fixed code via `PYTHONPATH`): config
    parses, loads 6 pulsars → (78,6) residual/error + (6,6) unit-diag HD, dry likelihood finite
    for BOTH injection dirs (inject_ou 6407.54, inject_powerlaw 6414.72). No library edits.

- [x] **T2.3 — Control run: OU-injected → OU-recovered (SLURM).**
  - Depends on: T2.2. GPU: yes (1 × A100).
  - Do: write `slurm_scripts/ng15_slurm_run.sh` (activate **`Argus`**, `oz022`,
    `milan-gpu`, `gpu:1`) running `run_analysis.py configs/ng15_config_lite.ini` on the
    OU-injected data. Submit + monitor.
  - Done when: recovers injected `log10_ha`/`log10_gamma_a` at truth within ~1σ → confirms
    the injection+recovery harness is correct.
  - ✅ Result: `slurm_scripts/ng15_slurm_run.sh` — one parameterized script for T2.3/T2.4
    (`MODE=ou|powerlaw`, default `ou`); derives a per-mode config (data_path/output_id) via
    line-anchored sed into `outputs/derived_configs/` so the two runs don't clobber. Keeps the
    T0.1 fixes (no `set -e`, `Argus` env, PYTHONPATH→worktree so the q11 fix is used — log
    confirms `argus.model` from worktree + `q11 ... /γ**2`). **PASS (job 14018847, milan-gpu
    gina3, 10m54s, COMPLETED):** recovered `log10_ha=-14.90±0.47` (truth -14.35, 1.16σ; truth
    inside 94% HDI [-15.75,-14.05]) and `log10_gamma_a=-8.71±0.58` (truth -8.5, 0.36σ); **0
    divergences**, max r_hat 1.031 (GW log10_ha r_hat 1.006), ESS ~178 (low, expected at
    200/100/2). Free/hierarchical red noise recovered small (`log10_σp≈-16.5→-17`, consistent
    with no injected red noise) — mild ~1σ low-bias in log10_ha likely a little GW power leaking
    into it. Corner plot rendered (`outputs/ng15_inject_ou/plots/`). **Harness validated →
    greenlight T2.4.**
  - ⚠️ **GPU/partition correction:** the OzSTAR job_submit plugin canonicalizes ANY GPU request
    to `milan-gpu` regardless of the requested partition — `milan-c` does NOT provide a GPU route
    (all T0.1 jobs likewise ran on milan-gpu despite the script saying milan-c). If milan-gpu is
    administratively down, GPU jobs just queue (Reason=PartitionDown) until it returns. Committed
    script sets `--partition=milan-gpu` (documentation; the reroute enforces it anyway).

- [x] **T2.4 — The real test: power-law-injected → OU-recovered (SLURM). DECISION GATE.**
  - Depends on: T2.3 (harness validated). GPU: yes (1 × A100).
  - Do: same lite run on the power-law-injected data. Compare recovered amplitude
    (mapped to `log10_A_gw`) and HD component against injected truth.
  - Done when: bias quantified. **Gate:** within ~1σ → greenlight Stage 3. If biased
    (expected failure: one OU corner can't match `f^-13/3` across the band), record the
    bias vs. the `log10_gamma_a` prior — this is a load-bearing scientific result and
    decides whether a spectral-model extension is needed before real data. Flag for review.
  - ✅ Result (job 14020177, milan-gpu gina11, 31m32s, COMPLETED; likelihood 6414.72, worktree
    q11-fixed code confirmed): recovered `log10_ha=-13.47±0.62`, `log10_gamma_a=-7.43±0.48`,
    **0 divergences**. New analysis `scripts/compare_ou_recovery.py` does the shape-agnostic
    **band-referenced PSD comparison** (`injection_truth.json` pivots; OU residual PSD vs the
    injected power-law PSD). **GATE ~PASSES: band-referenced amplitude recovered within ~1σ** —
    bias **−0.21 dex (amp) / −0.28σ at f=1/(5yr)** and **+0.06 dex / +0.08σ at f=1/yr** (both
    smaller than the OU→OU control's own −1.08σ baseline). The spectral overlay
    (`outputs/ng15_inject_powerlaw/plots/spectral_overlay.png`) shows the expected failure of
    *shape*: OU sits below the steep power law at the lowest freqs, above it at the highest, but
    tracks it through the sensitive mid-band → the band amplitude is the robust observable, the
    index is not (as PLAN framing predicted). `log10_gamma_a` leans on the high prior edge (−7;
    20% of draws within 0.25 dex) — the OU corner straining against `f^-4.33`.
    ⚠️ **CAVEAT — under-converged at lite settings:** max r_hat 1.050, min ESS **34** (vs control's
    178); the wide PSD posterior (~1.5 dex) means "within 1σ" is a loose statement. Numbers are
    directionally solid (the mid-band crossover is sampling-noise-robust). Machine-readable verdict
    in `outputs/ng15_inject_powerlaw/comparison.json`.
  - ✅ **CONFIRMATION (hi-res, user-requested) — GATE CONFIRMED PASS.** New
    `configs/ng15_config_confirm.ini` + `slurm_scripts/ng15_confirm_run.sh` (4×A100, 1000/1000/4,
    `dense_mass`, `max_tree_depth=10`, `log10_gamma_a` upper widened −7→−6). Both modes re-run:
    * **OU control** (job 14023052, 25m, 4 chains parallel): cleanly converged — max r_hat **1.004**,
      min ESS **2496**, **0 divergences**; `log10_ha=-14.77±0.47`, `log10_gamma_a=-8.30±0.70`
      (interior). Its OWN band-amplitude bias @f=1/(5yr) is **−0.40 dex / −0.74σ**.
    * **Power-law gate** (job 14023051, 42m): band-amplitude bias @f=1/(5yr) **−0.20 dex / −0.46σ**,
      @f=1/yr **+0.12 dex / +0.25σ** — i.e. the OU recovers the power-law band amplitude *as well as
      (better in dex than)* it recovers a self-consistent OU injection. Point estimate stable across
      lite→hi-res (−0.21→−0.20 dex amp) → robust. `log10_ha=-13.45±0.35`, `log10_gamma_a=-7.31±0.33`
      — the corner does **not** rail even with the widened prior (settles ~−7.3, high/near the band);
      the spectral *shape/index is not recovered* (expected — no power-law model).
    * ⚠️ **Load-bearing property — the power-law→OU posterior is pathological:** even at hi-res it gives
      max r_hat **1.051**, min ESS **51** (barely above lite's 34), **50 divergences** (control: 0).
      The ha↔γa band-amplitude degeneracy + the OU/power-law shape mismatch make a curved posterior
      that more compute barely improves. **Carry into Stage 3:** on real data expect divergences on
      the GWB corner, push `target_accept≥0.95`, and treat the *band-referenced amplitude* (not the
      corner/index) as THE observable. Confirmed verdict: `outputs/ng15_confirm_powerlaw/comparison.json`,
      overlay `outputs/ng15_confirm_powerlaw/plots/spectral_overlay.png`. **Gate PASSED → Stage 3 greenlit.**

- [x] **T2.6 — blackjax nested-sampling feasibility spike (methods track; parallel to Stage 3).**
  - 🛑 **OUTCOME (2026-07-10): feasibility PASSED, but NS PARKED for production** after the
    cost-scaling study (too slow: ~10–30× NUTS). Evidence now via NUTS + posterior-reuse, not NS.
    See `notes/DECISION_nested_sampling_parked.md`. The PASS details below stand as the record.
  - **PASS (2026-07-09).** blackjax NS runs on the pinned jax 0.4.38 (installed blackjax-devs main
    `1.6.dev --no-deps`, no jax cascade; 1-line `jax.shard_map` shim). Analytic logZ correct
    (d=2,5,15). On MDC2 dataset_2b the NS posterior **reproduces the NUTS baseline exactly**
    (log10_ha −12.88±0.05, log10_gamma_a −8.1±0.13) and gives logZ=63781±0.2. Backend:
    `bayesian_inference.run_blackjax_nested_sampling` (+`_blackjax_ns_evidence`,
    `_import_blackjax_ns`); dispatch `sampler=blackjax`. Caveats: ~1 h/run on the 32-psr
    likelihood (tune `num_delete`/`num_live`); a tail numerical pathology needed a bounded 6σ
    latent prior; OU cross-check deferred (feathers unrecoverable). Full writeup:
    `notes/t2.6_blackjax_ns_verdict.md`. **→ unlocks T3.4.**
  - Depends on: nothing (can run now, in parallel with T3.1). GPU: optional (CPU ok for the
    small validation problems).
  - **Why:** RISK B is the milestone's scoping ceiling — NUTS yields no evidence, so the best
    claim is "amplitude under an assumed HD template," NOT a Bayes-factor HD-vs-CURN *detection*.
    blackjax now ships a native nested sampler (https://blackjax-devs.github.io/sampling-book/
    algorithms/nested-sampling/, arXiv:2601.23252); Argus's likelihood is already a differentiable
    JAX Kalman filter, and `bayesian_inference.py:805 run_nested_sampling` is a stub raising
    `NotImplementedError` for GWB — a clean extension point behind the config's existing `sampler`
    field. A working JAX evidence engine would lift the deliverable from *demonstration* to
    *detection* (HD-vs-CURN + CURN-vs-noise Bayes factors) and is reusable across CW / common-red-noise.
  - **Riskiest assumption (test first, cheapest):** does blackjax NS give *trustworthy, reproducible
    evidence* at acceptable cost? Evidence is prior-sensitive (unlike posterior shape) and our priors
    are reparameterized `U→N(0,1)` — the transform Jacobians must be handled correctly or Z is
    silently wrong. Validate Z on (1) an analytic Gaussian (known Z), then (2) the **OU-injected
    synthetic** (correctly-specified model; cross-check the posterior against the existing NUTS run).
    Benchmark cost at the ~15–20D hierarchical model.
  - Do: integrate a blackjax-NS backend behind `run_nested_sampling` (JAX-native, gradient-augmented
    within-shell kernel). Check dependency/version compatibility (Argus env pins jax 0.4.38 — blackjax
    NS may need newer; verify like the argus-env vs Argus jax issue). **Kill-gated:** set a kill date.
  - Done when: Z reproduced on the analytic + OU-synthetic checks within tolerance and cost recorded
    → PASS unlocks the T3.4 upgrade; else record why NS is not viable here (so we don't retry) and
    keep NUTS + the documented RISK-B framing.
  - Follow-on (lower priority, NOT scoped here): blackjax's composable Markov kernels could address
    the `h_a↔γ_a` ridge pathology (a structured/blocked kernel) — note only.
  - 📄 Reasoning + strategy eval: `research-evaluations/2026-07-09-blackjax-nested-sampling-model-selection.md`.

---

## Stage 3 — Real-data SGWB recovery (GPU/SLURM, production)

- [x] **T3.1 — Production config.**
  - Depends on: T2.4 (greenlit). GPU: no (authoring).
  - Do: write `configs/ng15_config.ini` (clone `example_config.ini`): `data_path` →
    aligned real feathers, `noise_params_path` → `ng15_psr_noise.json`, red noise
    free/hierarchical, production NUTS (≥2000/1000, 4 chains, `dense_mass=true`,
    `target_accept≈0.85`, `max_tree_depth≥10`). Set `log10_ha` prior to bracket the
    published amplitude.
  - Done when: config parses; dry likelihood finite.
  - ✅ Result: `configs/ng15_config.ini` (cloned `ng15_config_confirm.ini` — keeps abs paths,
    `__NONE__` sentinel, corrected-q11 framing). `data_path`→`data/aligned/` (real), white noise
    FIXED via `ng15_psr_noise.json`, red noise FREE/hierarchical. NUTS **2000/1000/4,
    target_accept=0.95** (raised from confirm's 0.90 per the T2.4 real-data caveat — expect
    GWB-corner divergences), `dense_mass=true`, `max_tree_depth=10`; `num_chains=4` pairs with
    `--gres=gpu:4` (T3.2, chains parallel across GPUs). `log10_ha` prior [-17,-11] (brackets
    published A_gw≈-14.6 → OU log10_ha≈-13.5 via the T2.4 band-amplitude map); `output_id=ng15_real`.
  - **Load-bearing decision — `log10_gamma_a` kept FREE (not fixed).** Re-checked the Stage-2
    `fixg` sweep: its "band amplitude invariant to fixed gamma_a" claim is **FALSE**. Free run
    recovers γ_a≈-7.3 with band-amp bias -0.20 dex/-0.46σ; FIXING γ_a below that starves the OU
    corner of in-band power and biases the observable LOW by 1.3-1.9 dex (m80@-8.0 → -1.30 dex;
    m85@-8.5 → -1.94 dex; `outputs/ng15_fixg_m8{0,5}_powerlaw/comparison.json`). So fixing γ_a is
    NOT a valid cross-check on real data — production keeps it free (config header records this).
  - **Prereq done:** aligned feathers are gitignored/session-local and were absent from this
    worktree + main checkout → regenerated via T1.2→T1.3(pint)→T1.5 (raw NG15 wideband reachable
    at `/fred/oz022/.../NANOGrav15yr_PulsarTiming_v2.1.0/wideband`). `data/aligned/` rebuilt
    (6×78 joint epochs); `get_processed_residuals(mode="gwb")` → (78,6) + unit-diag (6,6) HD.
  - **Verified** (CPU, worktree `PYTHONPATH` for the q11 fix): config parses, 6 pulsars load →
    (78,6) + unit-diag HD, dry likelihood **finite = 6462.27** (~6.4e3, matches T2.2 injected evals).

- [x] **T3.2 — Production SLURM script.**
  - Depends on: T3.1. GPU: no (authoring).
  - Do: extend `slurm_scripts/ng15_slurm_run.sh` (or add a production variant) with
    `--gres=gpu:4` and `num_chains=4` matched, activating `Argus`.
  - Done when: script is correct (right env, account, partition, gres).
  - Result: added `slurm_scripts/ng15_production_run.sh`, a new 4 × A100 variant adapted from
    `ng15_confirm_run.sh`. Simplified — no `MODE` arg / no `sed` derived-config, since the T3.1
    config is self-contained and sits outside `outputs/` (no `shutil.SameFileError`); runs
    `configs/ng15_config.ini` directly. Carries the full proven setup: `--account=oz022`,
    `--partition=milan-gpu`, `--gres=gpu:4` matched to `num_chains=4`, `--mem=32G`,
    `--cpus-per-task=8`, `--time=8:00:00`, no `set -e`, `Argus` conda env,
    `PYTHONPATH`→worktree `python/` (editable-install / q11-fix gotcha), and the env/q11 pre-flight
    block. Job/log names → `ng15_real`. **Verified** (authoring only, no GPU submit): `bash -n`
    clean; `sbatch --test-only` parses and would schedule on milan-gpu (gina17, 8 procs / 4 GPUs).

- [x] **T3.3 — Real-data subset run (SLURM).**
  - Depends on: T3.2. GPU: yes (4 × A100).
  - Do: submit on the real aligned NG15 subset. Monitor to completion.
  - Done when: converges (`r_hat ≲ 1.01`, low divergences); recovered `log10_ha`→strain
    overlaps the published `log10_A_gw ≈ -14.6`; posterior off prior edges.
  - ✅ **Result (job 14108726, 4 × A100 on gina14, 70m49s wall, COMPLETED, GPU 85.6% avg).**
    Env pre-flight passed: `argus.model` from the worktree, q11 fix `uses γ**2: True`, 4 CUDA devices.
    - **Convergence:** max r_hat **1.003**; log10_ha ESS bulk 2434 / tail 1188; **4 divergent
      transitions / 8000 (0.05%)** — the target_accept=0.95 mitigation crushed the ~50 seen at the
      T2.4 confirm. All 42 sampled params r_hat ≤ 1.003, ESS ≥ 400.
    - **Recovered:** `log10_ha = −13.40 ± 0.20` (corner −13.398 +0.083/−0.091), `log10_gamma_a =
      −7.73 ± 0.25`. Both **off the prior edges** (log10_ha 3.6/2.4 dex from [−17,−11] edges;
      γ_a not railed against [−10.5,−6.0]).
    - **Amplitude vs published NG15 SGWB (band-referenced, both refs):** vs fixed γ=13/3
      (log10_A=−14.6) — bias **−0.02 dex / −0.09σ at f=1/yr**, −0.19 dex / −0.86σ at f=1/(5yr)
      (i.e. **<1σ at both pivots**). vs free-γ (γ=3.2, log10_A=−14.19) — −0.43 dex / −1.94σ at 1/yr,
      −0.20 dex / −0.93σ at 1/(5yr). log10_ha −13.40 brackets the T2.4-mapped published ≈ −13.5 to
      ~0.5σ. The strict 94%-HDI-overlap flag is False at 1/(5yr) only because the log-PSD core is
      tight (published value 0.055 dex outside a narrow HDI); the discrepancy is sub-σ and negligible.
    - Outputs: `outputs/ng15_real/` — `ng15_real_results.nc`, `comparison.json`, `numpyro_diagnostics/`,
      `plots/` (corner + `spectral_overlay.{png,pdf}`). Corner plot rendered fine (no low-dim
      `utils.corner_plot` error this run). Verified with extended `scripts/compare_ou_recovery.py
      --published --run-prefix ng15_real` (added a published-reference mode: no injection truth,
      compares recovered OU band amplitude vs both NG15 references).

- [x] **T3.4 — HD-vs-CURN contrast (evidence via NUTS + posterior-reuse; NS parked).**
  - Depends on: T3.3. GPU: yes (4 × A100 for the CURN NUTS run; estimator is CPU-only).
  - Do: rerun with `data["hd_correlation"]` overridden to identity (a diagnostic-script
    override — no library edit) to represent a common-uncorrelated red process.
  - **Method (revised 2026-07-10 — slice-NS parked, see DECISION note):** get the **Bayes factor**
    HD-vs-CURN by computing evidence from the NUTS posteriors we already pay for, via a
    **posterior-reuse estimator — learned harmonic mean (LHM)** (cheapest; ~free on top of an
    existing NUTS run). Scope decided HD-vs-CURN only (CURN-vs-noise deferred). Fallback on gate
    failure = thermodynamic integration (not needed — gate passed). Do **not** use the blackjax NS
    backend (parked).
  - **Honesty flag:** a *decisive* HD factor needs the full array (T3.5) — NANOGrav's HD evidence
    came from 67 pulsars (2211 pairs), not 6 (15 pairs); on the subset this proves the *method* and
    gives a weak-but-present factor.
  - ✅ **Result — HD favoured over CURN, lnB = +2.1 ± 0.1 (odds ≈ 8:1; Kass–Raftery "positive", NOT
    decisive).** Method-proving as expected on 6 pulsars.
    - **Estimator** (`scripts/logz_lhm.py`, self-contained CPU; arviz/numpy/scipy, no `harmonic`
      dependency): LHM in the model's unit-Gaussian latent space (the `.nc` already stores the
      per-draw `log_likelihood` group + `*_prime`/`*_raw` latents, so logZ is pure post-processing —
      no Kalman re-eval). Target φ = shrunk full-cov Gaussian on a train fold; estimate on a disjoint
      test fold; shrinkage sweep + 2-fold swap as reliability probes.
    - **GATE (2-D MDC2 anchor) PASSED:** LHM logZ = **63781.09 ± 0.02** vs NS anchor **63780.97 ± 0.16**
      (Δ = +0.11, within 3σ; flat shrinkage plateau). `outputs/mdc2_smoke_wide/*_lhm_gate.json`.
    - **Route-B correctness check PASSED (18-D):** reconstructed `log_density` matches
      `logprior + loglik` to < 5e-7 across all 4 chains — validates the 18-D latent extraction that
      the 2-D anchor cannot exercise.
    - **CURN run (job 14128813, 4 × A100, COMPLETED, 2h23m):** `run_curn.py` overrides the HD ORF with
      the identity (6×6; HD diag = 1 so identity = correct CURN) at runtime — no library edit. Env
      pre-flight passed (worktree `argus.model`, q11 `uses γ**2: True`). Converged: r_hat 1.002 on GW
      params, 0.03% divergences. Recovers `log10_ha = −13.99 ± 0.74` (broader/lower than HD's −13.40 ±
      0.20 — without cross-correlations more power is attributed to per-pulsar noise).
    - **Bayes factor:** logZ_HD = 6476.1 ± 0.25, logZ_CURN = 6474.0 ± 0.07 (18-D; low per-model ESS as
      expected). **lnB reported at MATCHED shrinkage** (s = 0.6–0.9 plateau; the 16 shared red-noise/
      hierarchical dims cancel in the difference, so lnB is ~2.5× tighter than either absolute logZ):
      **lnB = 2.11 ± 0.12**, stable across the plateau. At the posterior median HD also fits +4.3 nat
      better than CURN (consistency check). `outputs/ng15_curn/hd_vs_curn_bayes_factor.json`.
    - Artifacts: `scripts/logz_lhm.py`, `run_curn.py`, `configs/ng15_curn_config.ini`,
      `slurm_scripts/ng15_curn_run.sh`; outputs in `outputs/ng15_curn/` and `outputs/ng15_real/ng15_real_evidence.json`.
  - Done when: Bayes factor reported with estimator validated on the anchor. ✅

- [ ] **T3.5 — Scale to the full NG15 array.** ⏳ *machinery validated + first-look amplitude; converged run pending.*
  - Depends on: T3.3 working on the subset. GPU: yes (4 × A100).
  - Do: repeat T1.2–T1.6 + T3.3 for the full 68-pulsar set (watch memory/compute;
    epoch alignment gets harder). Log any pulsars dropped and why (no silent truncation).
  - Done when: full-array recovery completes and is compared to the subset + published.
  - **Key enabler — missing-observation support in the joint GWB filter (no truncation).** The
    intersection epoch grid collapses at 68 pulsars (baselines span 2.4→15.9 yr; all-overlap
    window ≈ 3.2 yr → below the 50-epoch floor and low-freq-blind). Instead the joint Kalman
    filter now conditions each measurement update on only the pulsars observed at that epoch:
    - `jax_kalman_filter._update` takes a per-epoch `(Npsr,)` mask; absent pulsars' `H` rows and
      `R` rows/cols are zeroed / identity-augmented, threaded through the `lax.scan`. Exact
      Kalman marginalization (each `H` row touches only its own pulsar; cross-pulsar coupling is
      all in `Pp`). **Golden-preserving:** all-ones mask reproduces the MDC2 likelihood bit-for-bit
      (63618.93). New tests: all-absent no-op, append-absent-epoch exactness, absent-data-ignored.
    - `build_aligned_feathers.py --grid union` bins onto the **union** grid + emits a `mask`
      column (grid-cell reference times); mask flows through `save/read_feather` →
      `process_pulsar_residuals_by_epoch` → filter (default all-ones, so subset/MDC2 unchanged).
    - **Latent HD bug fixed** (`gravitational_waves.pairwise_angular_separation`): 15/68 pulsars
      had self-correlation 0.5 not 1.0 (float noise in the self-pair separation slipped past the
      `np.isclose` zero-test → halved GW auto-power). Diagonal now pinned to 0. Golden-safe.
  - **Full-array prep (all glob-driven scripts scaled cleanly):** `stage_symlinks.py --all`
    (68 canonical pulsars, ao/gbt excluded) → ingest (`--timing-package pint`) → `build_aligned
    _feathers.py --grid union` → `reduce_ng15_white_noise.py`. **189 union epochs, ZERO TOAs
    dropped, ZERO pulsars dropped** (all `P_eps` finite; J0437/J1713 stiff but tolerated),
    41.6% of cells observed. State dim nx=1266 (M_sum=994). Dry likelihood finite (46920.5).
    Artifacts: `configs/ng15_config_full{,_probe,_quicklook}.ini`, `slurm_scripts/ng15_full_{run,
    probe,quicklook}.sh`, `data/ng15_psr_noise_full.json`.
  - **Sizing (1-GPU probe, job 14173193):** 0.26 s/likelihood-eval on A100; the 1266-D
    `h_a↔γ_a` ridge forces long trajectories. **max_tree_depth capped 10→7** (a depth-10 chain
    hit ~1000-step trajectories, ~400 s/it, and stalled the whole parallel run — job 14178734
    died at walltime with no output). Depth 7 bounds each iter to ≤127 steps (~80 s/it steady).
  - ⏳ **Quick-look (job 14195666, 4×A100, 300 warmup + 500 samples, depth 7, 17.7 h, COMPLETED):**
    recovered **log10_ha median = −13.40** (3/4 chains agree ~−13.4; chain 1 outlier at −12.65),
    **band-referenced amplitude OVERLAPS published NG15** at f=1/(5yr) (−0.74σ vs fixed γ=13/3),
    matching the 6-psr subset (−13.40 ± 0.20). **NOT converged:** r_hat ≈ 10, min ESS ≈ 4,
    3.05% divergences — the ridge + short 300-warmup leave the chains unmixed. Machinery +
    ballpark amplitude validated; a converged measurement needs the production run
    (`ng15_full_run.sh`: 1000 warmup + 2000 samples, depth 7, 4 chains, ~2.5–3 days) and likely
    a non-centered reparameterization of the GW corner to tame the ridge. Outputs in
    `outputs/ng15_full_quicklook/` (comparison.json, spectral_overlay.png).

---

## Stage 4 — Deliverable packaging

- [ ] **T4.1 — Results plots.**
  - Depends on: T3.3/T3.4. GPU: no.
  - Do: write `scripts/plot_hd.py` — HD-vs-CURN amplitude comparison + the OU-amplitude→
    strain conversion, overlaid on the published NG15 value. Corner plots come free from
    `utils.corner_plot`.
  - Done when: figures render from the saved posterior(s).

- [ ] **T4.2 — README.**
  - Depends on: most of Stage 3. GPU: no.
  - Do: write `README.md` documenting: which NG15 release/dir was ingested; the exact
    ingest + alignment + noise-reduction commands; the stage gates; and the **honest scope
    of the HD claim** (RISK B) — "common HD-correlated SGWB amplitude recovered, consistent
    with the published amplitude," with HD-vs-CURN as supporting evidence, not a Bayes
    factor.
  - Done when: a reader can reproduce the result end-to-end from the README.

- [ ] **T4.3 — Finalize + commit.**
  - Depends on: all above. GPU: no.
  - Do: tidy configs/scripts/slurm; ensure `PLAN.md`/`TASKS.md` reflect final state; commit
    via **`/commit`**. Open a PR when the user asks (do not push/PR unprompted).
  - Done when: branch is clean and the deliverable matches PLAN §8.

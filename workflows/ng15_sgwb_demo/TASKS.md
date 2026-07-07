# TASKS — NG15 wideband SGWB demo

> Read [`PLAN.md`](./PLAN.md) first (context, environment, data paths, code reuse map).
> Then do **the next unchecked task** below, in order. Each task lists its
> dependency, whether it needs a GPU, the concrete work, and how to know it's done.
> Check the box (`[x]`) and add a one-line result note when a task is complete.

## Conventions for whoever picks this up
- Branch: **`ng15-sgwb-demo`**. Commit only via the **`/commit`** skill. No `Co-Authored-By` lines.
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
- 👉 **NEXT: T1.6** (reduce per-backend white noise to per-pulsar JSON) — CPU, no GPU; depends only on T1.3.
  Then Stage 2 (T2.1 injector, GPU work behind it).

---

## Stage 0 — Baseline sanity (optional; GPU/SLURM)

- [ ] **T0.1 — Smoke-test the GWB+HD+NUTS path on MDC2 via SLURM.** *(optional — may fold into T2.3)*
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

- [ ] **T1.6 — Reduce NG15 per-backend white noise to a per-pulsar JSON.**
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

---

## Stage 2 — De-risk OU-vs-power-law (GPU/SLURM, cheap)

- [ ] **T2.1 — Power-law GWB injector.**
  - Depends on: T1.5. GPU: no (generation is cheap; run on CPU).
  - Do: write `scripts/inject_powerlaw_gwb.py` — inject an HD-correlated **true power-law**
    GWB (`log10_A_gw=-14.6`, `γ=13/3`) plus per-pulsar red noise into the aligned feathers'
    real sampling geometry (real cadences/errors/RA-DEC → real `Γ_HD`). Also support an
    **OU-GWB** injection mode for the control (T2.3). Write injected feathers to a separate
    dir (e.g. `data/inject_powerlaw/`, `data/inject_ou/`).
  - Done when: injected feathers load and process in GWB mode; injected truth values are
    recorded alongside for comparison.

- [ ] **T2.2 — Lite injection-recovery config.**
  - Depends on: T2.1. GPU: no.
  - Do: write `configs/ng15_config_lite.ini` (clone `example_config_lite.ini`): `data_path`
    → injected feather dir, `noise_params_path` → `ng15_psr_noise.json`, red noise
    free/hierarchical, lite NUTS (200/100/2). Widen `log10_ha_min/max` to bracket the
    injected amplitude.
  - Done when: config parses and a dry likelihood eval is finite (CPU ok for the eval).

- [ ] **T2.3 — Control run: OU-injected → OU-recovered (SLURM).**
  - Depends on: T2.2. GPU: yes (1 × A100).
  - Do: write `slurm_scripts/ng15_slurm_run.sh` (activate **`Argus`**, `oz022`,
    `milan-gpu`, `gpu:1`) running `run_analysis.py configs/ng15_config_lite.ini` on the
    OU-injected data. Submit + monitor.
  - Done when: recovers injected `log10_ha`/`log10_gamma_a` at truth within ~1σ → confirms
    the injection+recovery harness is correct.

- [ ] **T2.4 — The real test: power-law-injected → OU-recovered (SLURM). DECISION GATE.**
  - Depends on: T2.3 (harness validated). GPU: yes (1 × A100).
  - Do: same lite run on the power-law-injected data. Compare recovered amplitude
    (mapped to `log10_A_gw`) and HD component against injected truth.
  - Done when: bias quantified. **Gate:** within ~1σ → greenlight Stage 3. If biased
    (expected failure: one OU corner can't match `f^-13/3` across the band), record the
    bias vs. the `log10_gamma_a` prior — this is a load-bearing scientific result and
    decides whether a spectral-model extension is needed before real data. Flag for review.

---

## Stage 3 — Real-data SGWB recovery (GPU/SLURM, production)

- [ ] **T3.1 — Production config.**
  - Depends on: T2.4 (greenlit). GPU: no (authoring).
  - Do: write `configs/ng15_config.ini` (clone `example_config.ini`): `data_path` →
    aligned real feathers, `noise_params_path` → `ng15_psr_noise.json`, red noise
    free/hierarchical, production NUTS (≥2000/1000, 4 chains, `dense_mass=true`,
    `target_accept≈0.85`, `max_tree_depth≥10`). Set `log10_ha` prior to bracket the
    published amplitude.
  - Done when: config parses; dry likelihood finite.

- [ ] **T3.2 — Production SLURM script.**
  - Depends on: T3.1. GPU: no (authoring).
  - Do: extend `slurm_scripts/ng15_slurm_run.sh` (or add a production variant) with
    `--gres=gpu:4` and `num_chains=4` matched, activating `Argus`.
  - Done when: script is correct (right env, account, partition, gres).

- [ ] **T3.3 — Real-data subset run (SLURM).**
  - Depends on: T3.2. GPU: yes (4 × A100).
  - Do: submit on the real aligned NG15 subset. Monitor to completion.
  - Done when: converges (`r_hat ≲ 1.01`, low divergences); recovered `log10_ha`→strain
    overlaps the published `log10_A_gw ≈ -14.6`; posterior off prior edges.

- [ ] **T3.4 — HD-vs-CURN contrast.**
  - Depends on: T3.3. GPU: yes (1–4 × A100).
  - Do: rerun with `data["hd_correlation"]` overridden to identity (a diagnostic-script
    override — no library edit) to represent a common-uncorrelated red process; compare
    recovered amplitude / fit quality against the HD run.
  - Done when: the contrast is quantified (supports "the correlated component matters"),
    with the RISK B caveat documented (this is not a Bayes factor).

- [ ] **T3.5 — Scale to the full NG15 array.**
  - Depends on: T3.3 working on the subset. GPU: yes (4 × A100).
  - Do: repeat T1.2–T1.6 + T3.3 for the full 68-pulsar set (watch memory/compute;
    epoch alignment gets harder). Log any pulsars dropped and why (no silent truncation).
  - Done when: full-array recovery completes and is compared to the subset + published.

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

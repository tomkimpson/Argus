# PLAN — SGWB detection on real NG15 wideband data with Argus

> **For a fresh Claude instance:** read this file for the *why* and the *how it fits
> together*, then open [`TASKS.md`](./TASKS.md) in this directory and do the next
> unchecked task. Everything you need (environment, data paths, code reuse map) is in
> this file — you should not need to rediscover it. Do not depend on any prior
> conversation.

---

## 1. Context — why this work exists

Argus is a JAX/Kalman-filter state-space library for Bayesian inference on pulsar
timing array (PTA) data. We paused during work on PR #100 (branch
`discovery-ng15-cw-demo`, a *continuous-wave* detection demo on NANOGrav 15-year data)
to ask whether continuous waves (CW) were the right next step, or whether demonstrating
**stochastic gravitational-wave background (SGWB)** detection using Argus's existing,
mature machinery was more natural.

**Decision (made and confirmed with the user):**

- **The next milestone is SGWB detection on real NG15 wideband data**, reproducing the
  published NANOGrav 15yr Hellings–Downs result, using Argus's default GWB path. CW
  detection remains the longer-term goal, but not now.
- **PR #100 is held as-is** — left open and untouched. This work must use Argus's
  *native* ingestion path only. **Do NOT** use, edit, or depend on PR #100's
  Discovery→Argus feather adapter, F-statistic module, or its branch.

**Why SGWB-first is correct (not merely easier):**

1. **Lowest method risk.** The GWB path — Hellings–Downs (HD) spatial correlation + a
   shared Ornstein–Uhlenbeck (OU) process + NUTS — is Argus's default, JOSS-paper-backed
   capability. CW is validated only on loud injections and is multimodal on real data.
2. **A real, checkable target.** NANOGrav 15yr's headline result is the HD-correlated
   SGWB (`log10_A_gw ≈ -14.6`, spectral index `γ = 13/3`). There is *no* confirmed CW in
   NG15, so a CW demo could only be a self-injected recovery or an upper limit.
   Reproducing a known real detection is a far stronger first real-data claim.
3. **It de-risks the shared bottleneck.** Whether Argus's single-corner OU red-noise
   model can represent a steep power law is the load-bearing question for *both* SGWB and
   CW. (PR #100's CW F-statistic is contaminated by exactly the common red noise / GWB an
   SGWB run models.) An SGWB reproduction is the natural place to answer it.

**Wideband is settled/forced.** Argus is wideband-only on `main` (single
`LoadWidebandPulsarData` loader; no narrowband code path). PR #100 already found
narrowband NG15 intractable (~202 GB OOM under NUTS). Keep it wideband.

---

## 2. The two real technical risks (found during planning)

"Reuse the mature GWB path" is **not** turnkey on real data. Two obstacles reshape the
work:

- **RISK A — epoch alignment (the true gating risk).** GWB mode calls
  `process_pulsar_residuals_by_epoch` (`python/argus/data_loader.py:99`), which **raises
  a `ValueError` unless every pulsar has identical shape and 1:1 epoch correspondence**.
  The IPTA Mock Data Challenge 2 (MDC2) data the GWB path was validated on is idealized —
  all 33 pulsars have exactly 185 synchronously-sampled TOAs. Real NG15 pulsars are
  ragged (different TOA counts, unaligned epochs) and the joint GWB Kalman filter has
  **no missing-observation mechanism**. This must be solved with an epoch-binning
  pre-processor *before any sampling runs*. It is the make-or-break step.
  - Note: this gap is specific to the *joint* GWB filter. The per-pulsar CW filter does
    not need cross-pulsar alignment — so this is GWB-milestone infrastructure.

- **RISK B — HD is assumed, not measured.** The HD correlation matrix `Γ_HD` is fixed
  inside the covariance (`_compute_sigma_matrix`, `jax_kalman_filter.py:175`). NUTS
  yields no evidence, and nested sampling raises `NotImplementedError` for GWB. So Argus
  recovers a *common HD-correlated amplitude under an assumed template*; it **cannot**
  produce a Bayes-factor HD-vs-CURN (common-uncorrelated-red-noise) detection. The
  deliverable must be scoped honestly (see §6).

- **Minor — noise model mismatch.** NG15 white noise is *per-backend* (many EFAC/EQUAD
  per pulsar) and includes ECORR and DM-noise. Argus applies a single scalar EFAC/EQUAD
  per pulsar and has **no ECORR or DM-noise**. Per-backend values must be collapsed to an
  effective per-pulsar pair; ECORR/DM-noise are dropped for this demo.

---

## 3. Environment & execution facts (verified this session — do not re-derive)

**Machine / repo.** Work happens in an Argus git worktree; the reference checkout used
here is `/home/tkimpson/.treehouse/Argus-891104/1/Argus`. Feature branch:
**`ng15-sgwb-demo`** (already created off `main`).

**Conda environment — use `Argus`, NOT `argus-env`.**
- Correct env: `/fred/oz022/tkimpson/conda_envs/Argus` — has `flax` 0.8.5, `jax` 0.4.38
  (in the declared `jax[cuda12]>=0.4.35,<0.5.0` range), `numpyro`,
  `tensorflow-probability`, and `enterprise` 3.4.4 (needed for par/tim ingestion).
- **Avoid `argus-env`**: it is missing `flax` and has `jax` 0.6.2 (outside the declared
  range). The committed SLURM scripts (`example_workflow*/slurm_scripts/*.sh`) wrongly do
  `conda activate argus-env` — any new SLURM script must activate `Argus` instead.
- **CORRECTION (verified 2026-07-08):** Argus **IS** pip-installed **editable** in the
  `Argus` env, pointing at the **main checkout** `/fred/oz022/tkimpson/Argus/python` (which
  tracks `main`). `run_analysis.py` only `sys.path.append`s the repo `python/` dir, and that
  append **loses** to the editable install already on `sys.path`. So **any `run_analysis`
  (GPU/SLURM) run uses the main-checkout code, NOT whatever treehouse worktree you are
  editing** — a treehouse edit is invisible to GPU runs. To force a worktree's code, prepend
  `PYTHONPATH="$WORKTREE/python"` (prepends → wins over the editable install), or land the
  change on `main` so the editable target serves it. This bit us once (an MDC2 run silently
  used old code). See memory `project_argus_editable_install_gotcha`.

**GPU vs CPU.**
- Interactive/login-node GPU init **hangs** (CUDA contention with other jobs). Do **not**
  run NUTS interactively.
- **All sampling (NUTS) runs go through SLURM on A100 GPUs.** Per user standing
  preference: A100, account `oz022`, partition `milan-gpu` (if down, submit via
  `milan-c` — same `gina*` nodes).
- CPU is fine (and used) for imports, ingestion, epoch-alignment diagnostics, and other
  data prep. Force CPU with `JAX_PLATFORMS=cpu` to avoid the CUDA hang.

**Parallel chains.** `run_nuts_sampling` uses `chain_method="parallel"` when
`n_devices >= num_chains` (`bayesian_inference.py:684`). For production (4 chains),
request `--gres=gpu:4`.

---

## 4. Data

**NG15 wideband release (verified present on disk):**
`/fred/oz022/tkimpson/pta-solar-wind-2/data/NANOGrav15yr_PulsarTiming_v2.1.0/wideband/`
- `par/` — `{PSR}_PINT_YYYYMMDD.wb.par`
- `tim/` — `{PSR}_PINT_YYYYMMDD.wb.tim`
- `noise/` — per-pulsar `{PSR}.wb.pars.txt` (+ `{PSR}.wb.chain_1.txt`)
- 68 canonical pulsars after excluding telescope-suffixed `...ao` / `...gbt` duplicates
  (e.g. skip `B1937+21ao_PINT_*.wb.par`, keep `B1937+21_PINT_*.wb.par`).
- **`par/` and `tim/` are separate directories**, but `scripts/ingest_par_tim.py` globs a
  *single* `input_dir` for both → stage co-located symlinks in one dir first.
- The NG15 `noise/*.pars.txt` files list **per-backend** params
  (`{PSR}_{rcvr}_{backend}_efac`, `_log10_t2equad`, `_dmefac`, `_log10_dmequad`) plus
  `{PSR}_red_noise_gamma`, `{PSR}_red_noise_log10_A`. These must be reduced (see RISK
  minor) — Argus has no per-backend / ECORR / DM-noise support.

**Baseline demo data (for Stage 0 sanity), already in-repo:**
`workflows/data/IPTA_MockDataChallenge2/dataset_2b/` — 33 par/tim pairs (185 TOAs each,
synchronously sampled); `group1_psr_noise.json` (symlink); pulsar red noise fixed via
`workflows/example_workflow_lite/data_files/approximate_spin_injections.pkl`. No feathers
are checked in — the loader falls back to par/tim via `enterprise` automatically.

**Pulsar subset to start with** (science-motivated: dominate the NG15 HD S/N, keep the
alignment problem small): **J1909-3744, J1713+0747** first (highest precision, longest
baseline), then **J1744-1134, J0613-0200, B1855+09 (a.k.a. J1857+0943), J1600-3053** →
a ~6-pulsar array. Do **not** include J0437-4715 (southern/PPTA, not in NG15). Scale to
the full array only after Stage 3 works on the subset.

---

## 5. Code reuse map (what to build on, with paths)

| Need | Reuse | Notes |
|---|---|---|
| par/tim → feather | `scripts/ingest_par_tim.py` | CLI: `python scripts/ingest_par_tim.py <input_dir> <output_dir> [--excluded-psrs …] [--max-files N] [--overwrite]`. Writes `<pulsar>.feather` only (no noise JSON). Needs `enterprise` (Argus env has it). |
| load + GWB processing | `python/argus/data_loader.py` `LoadWidebandPulsarData`, `get_processed_residuals(mode="gwb")` | Globs `*.feather` in `data_path`; builds HD from RA/DEC at `:316`; **`process_pulsar_residuals_by_epoch` (`:99`) is RISK A** — raises on ragged data. `save_feather` (`:534`), `read_feather` (`:583`). |
| run inference | `python/argus/workflow.py` `run_inference(config_path=…, use_gw=True)` | `signal_model` fallback `"gwb"` (`:103`); `sampler` fallback `"nuts"` (`:152`). Clone `workflows/example_workflow*/run_analysis.py` (it sets `use_gw=True`, x64). |
| GWB / noise priors | `python/argus/prior_models.py` | `get_gw_parameter_priors` (`log10_ha_*`, `log10_gamma_a_*`); `get_pulsar_noise_priors` (hierarchical red noise; `spin_injections_path` **fixes** it); `get_measurement_noise_priors` (`efac_*`, `log10_equad_*`; `noise_params_path` **fixes** white noise). |
| noise-file loaders | `python/argus/utils.py` | `get_efac_equad_injections` (`:152`, JSON `{psr:{efac,equad}}`, `equad` stored as log10); `get_psr_noise_injections` (`:182`, pandas pickle cols `psr`/`optimal_sigma`/`optimal_gamma`); `get_noise_parameters`; `corner_plot`. |
| Kalman / HD / OU | `python/argus/jax_kalman_filter.py` | `_compute_sigma_matrix` (`:175`) threads HD + OU state model — the power-law/OU question lives here. |
| sampler internals | `python/argus/bayesian_inference.py` | `run_nuts_sampling` (`:598`, parallel chains at `:684`), `test_likelihood_performance` (`:905`), `run_nested_sampling` (raises for GWB `:805`). |

**Config schema** (clone `workflows/example_workflow_lite/configs/example_config_lite.ini`
for lite / `workflows/example_workflow/configs/example_config.ini` for production):

```ini
[Data]
data_path = <path to feather dir>
excluded_psrs = <comma-separated>
# signal_model = gwb   (optional; gwb is the default)
# sampler = nuts       (optional; nuts is the default)

[NUTS]
num_samples, num_warmup, num_chains, target_accept_prob, max_tree_depth, dense_mass

[PriorModel]
# GWB: log10_ha (strain amplitude) and log10_gamma_a (OU corner, NOT the power-law index)
log10_ha_fixed/value/min/max
log10_gamma_a_fixed/value/min/max
# Pulsar red noise: omit spin_injections_path -> free hierarchical sampling
spin_injections_path            # if set, red noise FIXED from pickle
log10_gamma_p_min/max, log10_sigma_p_min/max
log10_gamma_p_mean_min/max, log10_gamma_p_std_min/max         # hyperpriors (always on)
log10_ratio_mean_min/max, log10_ratio_std_min/max            # log10(σp)=log10(γp)+log10(ratio)
# White noise: point noise_params_path at JSON to FIX efac/equad
noise_params_path
efac_min/max, log10_equad_min/max

[Logging]  level, enable_file_logging
[Output]   output_dir, output_id, base_dir
```

**Important interpretation note:** Argus's `log10_gamma_a` is the **OU corner frequency
(Hz)**, *not* the power-law spectral index `γ`. The mapping between the OU corner/amplitude
and the published power-law `(log10_A_gw=-14.6, γ=13/3)` is exactly what Stage 2 establishes.

---

## 6. Staged approach (riskiest assumption first)

Full task-level detail is in [`TASKS.md`](./TASKS.md). Summary of the arc:

- **Stage 0 — Baseline sanity.** Confirm the GWB+HD+NUTS path is healthy in this checkout
  (lite config on MDC2). *May be folded into the first real GPU job rather than run as its
  own queue cycle — see TASKS.*
- **Stage 1 — Ingest NG15 wideband + resolve epoch alignment (RISK A).** Mostly CPU: stage
  symlinks, ingest the subset to feathers, run the epoch-alignment diagnostic, build
  epoch-binned aligned feathers, reduce the per-backend white noise to a per-pulsar JSON.
  Red noise starts free/hierarchical (no power-law→OU conversion needed yet).
- **Stage 2 — De-risk OU-vs-power-law (GPU, cheap).** On the real NG15 sampling geometry
  but a *synthetic* signal: (1) OU-injected → OU-recovered control (validates harness);
  (2) power-law-injected → OU-recovered (the real test). **Decision gate:** if OU recovers
  the injected amplitude within ~1σ, greenlight Stage 3; if it biases the amplitude,
  quantify the bias (a load-bearing result either way).
- **Stage 3 — Real-data SGWB recovery (GPU, production).** NUTS on the real NG15 subset,
  then full array. Validate recovered amplitude against the published value; run the
  HD-vs-CURN amplitude contrast (override `data["hd_correlation"]` with identity — a
  diagnostic-script trick, no library edit).
- **Stage 4 — Deliverable.** `plot_hd.py`, `README.md`, finalized configs + SLURM scripts.

**Honest framing of the result (RISK B).** The defensible claim is: *"a common
HD-correlated SGWB amplitude recovered on real NG15 wideband data, consistent with the
published NANOGrav 15yr amplitude,"* with the HD-vs-CURN amplitude contrast as supporting
evidence — **not** a Bayes-factor HD detection. Do not overclaim.

---

## 7. Constraints / working agreements

- Work on branch **`ng15-sgwb-demo`**. **Commit only via the `/commit` skill.** Never add
  `Co-Authored-By: Claude` lines to commits.
- **Prefer new scripts** under `workflows/ng15_sgwb_demo/scripts/` over editing library
  source. Edit `python/argus/*` only if the change is likelihood-identical and
  as-fast-or-faster (benchmark-gated). A masked-epoch extension to
  `process_pulsar_residuals_by_epoch` is a *last-resort* option for RISK A and should be
  flagged for review, not done silently.
- Ingestion (`enterprise`) runs offline on CPU, not on the GPU node.
- All NUTS sampling runs go through SLURM on A100 (`oz022`, `milan-gpu`/`milan-c`).

---

## 8. Deliverable layout (target)

```
workflows/ng15_sgwb_demo/
  PLAN.md                     # this file
  TASKS.md                    # the task checklist
  run_analysis.py             # clone of example_workflow/run_analysis.py (use_gw=True)
  configs/
    ng15_config_lite.ini      # subset, lite NUTS — Stage 1/2 iteration
    ng15_config.ini           # full array, production NUTS — Stage 3
  scripts/
    stage_symlinks.py         # co-locate par+tim for a pulsar subset
    check_epoch_alignment.py  # RISK A diagnostic
    build_aligned_feathers.py # RISK A resolver (epoch binning)
    reduce_ng15_white_noise.py# per-backend -> per-pulsar EFAC/EQUAD JSON
    inject_powerlaw_gwb.py    # Stage 2 synthetic-signal injector
    plot_hd.py                # HD-vs-CURN contrast + strain conversion plot
    # powerlaw_to_ou.py       # optional: fixed-OU red-noise pickle
  data/                       # aligned NG15 feathers + ng15_psr_noise.json (may be gitignored if large)
  slurm_scripts/
    ng15_slurm_run.sh         # A100, conda activate Argus, oz022, milan-gpu
  README.md
```

---

## 9. Verification (end-to-end acceptance)

1. Stage 0 GWB run on MDC2 completes with finite likelihood + rendered corner plot,
   `r_hat ≈ 1`, no divergence blow-up.
2. `check_epoch_alignment.py` shows the subset can be aligned; after
   `build_aligned_feathers.py`, `get_processed_residuals(mode="gwb")` returns clean
   `(nepoch, Npsr)` residual/error matrices and an `(Npsr, Npsr)` HD matrix.
3. Stage 2 control (OU→OU) recovers injected truth; the power-law→OU test quantifies any
   amplitude bias (the Stage 3 gate).
4. Stage 3 real-data NUTS converges (`r_hat ≲ 1.01`, low divergences) with recovered
   `log10_ha`→strain overlapping the published NG15 value; HD-vs-CURN contrast shows the
   correlated component matters.
5. The workflow reproduces the result from a clean checkout via
   `python run_analysis.py configs/ng15_config.ini` (data ingested per README).

---

## 10. Explicitly out of scope (future work)

- Bayes-factor HD-vs-CURN model comparison (needs GWB evidence / a new sampler).
- A faithful power-law red-noise state model, ECORR, DM-noise, or per-backend white noise.
- Returning to CW detection (F-statistic + common-red-noise foreground) on PR #100.

# NG15 continuous-wave (CW) demo

A first demonstration of Argus searching for a **continuous gravitational wave**
(single SMBH binary) in real **NANOGrav 15yr** data, using the data shipped by
NanoGrav's [Discovery](https://github.com/nanograv/discovery).

See [`../../docs/discovery_review.md`](../../docs/discovery_review.md) for the full
review of Discovery and the design rationale, data/noise caveats, and the CW
waveform convention map (notably **ψ_argus = −ψ_discovery**).

## Pipeline

```
Discovery NG15 feathers ──(scripts/ingest_discovery_feather.py)──► Argus feathers
        + noise dicts                                              + noise_params.json
        + F0 (scripts/ng15_f0_catalog.json)                       + spin_injections.pkl
                                              │
                         prepare_demo.py inject Earth-term CW
                                              │
                                              ▼
                         run_analysis.py + configs/ng15_cw_injection.ini
                                  (CW Kalman filter + NUTS)
                                              │
                                              ▼
                              recover injected CW parameters
```

## Two data paths — use WIDEBAND for the NUTS run

Discovery ships NG15 as **narrowband** feathers (tens of thousands of TOAs, hundreds
of DMX columns per pulsar). Argus's CW Kalman filter cost scales as
`n_obs × dim_M²`, and NUTS needs reverse-mode gradients through the whole scan, so
narrowband is **intractable**: 6 pulsars OOM'd at **202 GB**, and 5 pulsars ran at
~250 s/iteration (~11-day ETA). NG15 **wideband** data (one achromatic TOA + DM per
epoch) has ~40× fewer TOAs — exactly the scale Argus (`LoadWidebandPulsarData`) was
built for — so the NUTS run becomes tractable (~0.5 GB, hours).

| Path | Prep script | Config | Use |
|---|---|---|---|
| **Wideband** (recommended for NUTS) | `prepare_demo_wideband.py` | `configs/ng15_cw_injection_wb.ini` | The actual CW recovery run |
| Narrowband (Discovery feathers) | `prepare_demo.py` | `configs/ng15_cw_injection.ini` | Adapter/cross-check demo; fast likelihood scan only — do **not** run full NUTS |

## Run it (wideband injection-recovery)

```bash
cd workflows/ng15_cw_demo
conda activate Argus
# 1. Prepare: PINT-read wideband par/tim, SVD-recondition the design matrix,
#    fix noise, inject a known Earth-term CW.
python prepare_demo_wideband.py \
    --wb-dir /fred/.../NANOGrav15yr_PulsarTiming_v2.1.0/wideband \
    --discovery-dir /path/to/discovery/data \
    --out-dir ./demo_data_wb \
    --pulsars J1944+0907 B1855+09 J1455-3330 J0613-0200 J1744-1134

# 2. (fast, optional) confirm the likelihood peaks at the injection — no NUTS:
python verify_injection.py --data-dir ./demo_data_wb

# 3. Recover (NUTS) on an A100:
sbatch slurm_scripts/run_ng15_cw.sh              # defaults to the wideband config
```

Compare the posterior (corner plot in `outputs/`) against `demo_data_wb/injection.json`.

> **Wideband ingestion notes.** NG15 wideband files are PINT products, so ingestion
> uses `timing_package='pint'` (tempo2 chokes on the `DMDATA` keyword). The wideband
> par still fits ~169 per-epoch DMX parameters over only ~hundreds of epochs, so the
> raw design matrix is catastrophically ill-conditioned (singular-value range ~1e16);
> `prepare_demo_wideband.py` replaces it with its SVD orthonormal basis, which spans
> the same timing subspace (so the marginalised likelihood is unchanged) but is
> well-conditioned (cond ~1e4). This mirrors discovery's `makegp_timing(svd=True)`.

## Narrowband path (adapter + fast scan only)

```bash
python prepare_demo.py --discovery-dir /path/to/discovery/data --out-dir ./demo_data \
    --pulsars J2317+1439 J0030+0451           # keep it small; narrowband is heavy
python verify_injection.py --data-dir ./demo_data   # fast likelihood scan (peaks at injection)
```
Full NUTS on narrowband is not recommended (see the OOM/ETA numbers above).

### 3. Real-data upper limit (Stage 4, follow-on)

Re-prepare without injecting and search the unmodified residuals:

```bash
python prepare_demo.py --discovery-dir /path/to/discovery/data --out-dir ./realdata --no-inject
# point configs/ng15_cw_injection.ini [Data] data_path at ./realdata/ and run
```

Produces a CW strain upper limit vs f_gw; sanity-check against NanoGrav's published
NG15 CW limits and the bundled `NG15yr-*-chain.feather` posteriors.

## Detection statistic: all-sky F-statistic (F_e) + B-statistic

For **detection** (rather than parameter estimation) the recommended tool is the
coherent Earth-term **F-statistic**, `argus.cw_fstatistic` — it is *sampling-free*
and so immune to the NUTS non-convergence and slow nested sampling seen on the
multimodal CW posterior. It analytically maximises over the 4 extrinsic amplitudes
(h0, Φ0, ψ, cos ι); its Bayesian twin the **B-statistic** analytically *marginalises*
them (a proper Bayes factor, no h0=0 nesting issue). Both come from the same
Kalman-whitened inner products and are evaluated all-sky over a frequency × sky grid.
The whitener is validated against the full Kalman likelihood in
`test/test_cw_fstatistic.py`.

```bash
# clean machinery demo: a loud injection recovered above the real-data background
python run_fstatistic.py --data-dir ./demo_data_wb_loud --null-dir ./demo_data_wb_noinj
# outputs/fstatistic/: fstat_profile.png, fstat_skymap.png, fstat_summary.json
```

**Result on NG15 wideband (5 pulsars):**
- A loud injection (`h0=5e-13`) is recovered at `2F_e ≈ 5800` (SNR ≈ 76), peaking at
  the injected `f_gw` (to grid resolution) and RA; the recovered Dec sits at the
  **antenna-pattern sky degeneracy** (`F₊,Fₓ` are invariant under `α→α+π` and a
  `δ` reflection) — expected and unbroken with only 5 pulsars.
- The **empirical null** (no-injection data) has `max 2F_e ≈ 900` — *not* χ²₄-small,
  because the real NG15 residuals carry **common red noise / the GWB** that the
  coherent F_e picks up and our per-pulsar-only noise model does not remove. The
  loud injection sits far above this; a faint injection (`h0=5e-14`, the
  `demo_data_wb` default) does **not**, illustrating why a common-red-noise/GWB term
  is the key next ingredient for a real CW search (see roadmap in the review doc).

Datasets used: `demo_data_wb_loud` (loud injection), `demo_data_wb` (faint
injection), `demo_data_wb_noinj` (null) — all built by `prepare_demo_wideband.py`.

## Caveats (see the review doc)

- **ECORR dropped** — Argus has no epoch-correlated white-noise model; with
  narrowband TOAs this under-estimates the per-epoch white-noise floor.
- **OU ≈ power-law red noise** — the adapter maps NG15 power-law red noise to
  Argus's single-corner OU process by matching the residual PSD amplitude at 1/T;
  the slope is approximate. Noise is *fixed* (not sampled) for this demo.
- **Earth-term only** in the default config (no pulsar-term `chi` nuisance params);
  flip `include_pulsar_term = true` to exercise the full phase-parameterized model.

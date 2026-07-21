# Discovery (NanoGrav) review — relevance to Argus and a first CW demo on NG15

*Prepared while scoping a first continuous-wave (CW) demonstration of Argus on real
NanoGrav data.*

## TL;DR

NanoGrav's [Discovery](https://github.com/nanograv/discovery) is a JAX
Gaussian-process PTA library. It solves the **same data problem** as Argus by a
**different inference route** (frequency-domain Woodbury-marginalized GP likelihood
vs. Argus's time-domain state-space Kalman filter). We do **not** adopt its
likelihood — the state-space approach is the point of Argus. What is genuinely
valuable from Discovery is:

1. **A ready-made NG15 dataset.** Discovery ships all 67 NG15 pulsars as per-pulsar
   `.feather` files with embedded **published noise dictionaries** (per-backend
   EFAC/EQUAD/ECORR + power-law red noise) and the **published NG15yr posterior
   chains** — i.e. data *and* a validation baseline, no enterprise/PINT run needed.
2. **A schema to adapt.** Argus and Discovery independently converged on the same
   architecture choice (per-pulsar feather cache, enterprise as data-prep-only).
   A thin adapter bridges the two schemas.
3. **An independent CW waveform** to cross-check Argus's against.

This review backs a staged build: **(0)** this writeup, **(1)** a Discovery→Argus
feather adapter, **(2)** a CW waveform cross-check test, **(3)** a CW
injection-recovery demo on NG15, **(4)** a real-data upper-limit follow-on.

## What Discovery is, structurally

| Concern | Discovery | Argus |
|---|---|---|
| Inference | GP marginalization; Woodbury/Sherman-Morrison reduction (`likelihood.py`, `matrix.py`) | State-space Kalman filter (`jax_kalman_filter.py`, `cw_kalman_filter.py`) |
| Domain | Frequency (Fourier basis red noise) | Time (OU process in the state) |
| Red noise | Power-law (and free-spectrum) GP over a Fourier basis | Ornstein–Uhlenbeck (integrated-frequency) process in the state |
| Deterministic signals | Subtracted from residuals before GP marginalization (`deterministic.py`) | Subtracted from observations (CW mode), per-pulsar scalar filter |
| White noise | Per-backend EFAC/EQUAD + **ECORR** (epoch GP) | Single per-pulsar EFAC/EQUAD; **no ECORR** |
| GWB correlation | Hellings–Downs ORF `globalgp` | Hellings–Downs covariance (GWB mode) |
| Stack | JAX, numpyro, pyarrow; enterprise data-prep-only | JAX, numpyro, pyarrow; enterprise data-prep-only |

## What we take from Discovery (and what we don't)

| Item | Use? | How |
|---|---|---|
| Bundled NG15 feathers + noise dicts + posterior chains | **Yes** | Demo data + validation baseline |
| `Pulsar.read_feather` schema | **Yes** | Reference for `scripts/ingest_discovery_feather.py` |
| `deterministic.makedelay_binary` CW residual | **Yes (cross-check)** | `test/test_cw_discovery_crosscheck.py` |
| Per-backend WN / ECORR / power-law RN conventions | **Yes (conventions)** | Inform noise handling + roadmap |
| GP/Woodbury likelihood (`likelihood.py`, `matrix.py`) | **No** | Argus keeps its Kalman filter |
| numpyro NUTS pattern | Already in Argus | — |

## The CW waveform: equivalent, with a documented convention map

Both implement the circular-SMBH-binary CW residual of Ellis et al. (2012, 2013).
The Earth-term forms differ only in internal conventions (Discovery rotates the
plus/cross *amplitudes* by ψ and uses a ψ-free antenna pattern; Argus rotates the
polarization *tensors*). The cross-check test verifies, to ~1e-15 across randomized
geometry/inclination/phase/frequency, that

```
residual_argus(ψ)  =  -0.5 · residual_discovery(-ψ)
```

i.e. they are physically identical up to **(a)** a global convention constant
`C = -1/2` (Discovery's amplitude is 2× Argus's, with an overall sign — absorbed
when fitting `h0`), and **(b)** a **polarization-angle sign flip**,
`ψ_argus = -ψ_discovery` (the well-known handedness ambiguity in defining ψ).

**Practical consequence:** when comparing Argus CW parameters to Discovery (or to
NanoGrav's published CW analyses that use Discovery's/enterprise's convention),
**flip the sign of ψ**. Parameter conventions are otherwise aligned
(`h0`, `f_gw`, `cos_iota`, `Phi0`, sky `(α, δ)`; Argus's per-pulsar phase `chi`
corresponds to Discovery's `phi_psr`).

## The real gap: the noise model

NG15 real data carries three noise ingredients that Argus does not natively model:

1. **Per-backend white noise.** NG15 pulsars have several receiver/backend
   combinations, each with its own EFAC/EQUAD. *Handled* in the adapter by folding
   per-backend white noise into effective per-TOA errors using Discovery's t2equad
   convention `σ_eff² = EFAC_b²·(σ_toa² + EQUAD_b²)`, which Argus's single-EFAC/EQUAD
   model then consumes exactly (with EFAC=1, EQUAD≈0).
2. **ECORR (epoch-correlated white noise).** Argus has no ECORR. **Dropped** for the
   first demo. With narrowband TOAs (many per epoch) this *under-estimates* the
   per-epoch white-noise floor and over-counts independent information. Acceptable
   for a high-SNR injection demo; a real obstacle for quantitative real-data limits.
3. **Power-law red noise.** Argus models red noise as a single-corner OU
   (integrated-frequency) process; its residual PSD goes as f^-2 (below corner) to
   f^-4 (above), whereas NG15 red noise is a power law f^-γ with pulsar-specific γ.
   The adapter's `powerlaw_to_ou` matches the residual PSD *amplitude* at the lowest
   sampled frequency (1/T) with a steep corner; the **slope is only approximate**.

**Top roadmap item:** give Argus a faithful power-law red-noise representation
(e.g. a sum-of-OU / multi-corner state-space approximation to a power law) and an
ECORR-equivalent epoch term. Until then, the first demo *fixes* noise at Discovery's
published values rather than sampling it, and treats real-data results as
approximate.

## Data note: narrowband vs wideband

Discovery's bundled NG15 feathers are the **narrowband** TOAs (e.g. J1713+0747 has
~59k TOAs, design-matrix dimension ~423). Argus's machinery is built around
wideband-style (≈one residual per epoch) data. The per-pulsar scalar Kalman filter
*works* on narrowband data but pads to the largest TOA count and design matrix,
which is memory- and compute-heavy (best run on the A100 GPUs, not a login node).
For lighter demos, prefer pulsars with fewer TOAs/parameters, or epoch-average the
narrowband TOAs (a possible adapter extension). The NG15 release also ships its own
`wideband/` par/tim and `noise/` directories, a future alternative to Discovery's
feathers.

## Deliverables produced from this review

- `scripts/build_ng15_f0_catalog.py` + `scripts/ng15_f0_catalog.json` — F0 catalog
  (Discovery feathers omit the absolute spin frequency, which Argus needs as
  `h[0]=1/F0`; sourced once from the NG15 par files).
- `scripts/ingest_discovery_feather.py` — Discovery→Argus feather adapter; emits
  Argus-native feathers + `noise_params.json` (white) + `spin_injections.pkl` (OU red).
- `test/test_cw_discovery_crosscheck.py` — the CW waveform cross-check above.
- `workflows/ng15_cw_demo/` — the CW injection-recovery demo on NG15 (Stage 3).

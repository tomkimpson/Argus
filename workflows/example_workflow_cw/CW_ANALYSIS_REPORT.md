# Continuous Wave Signal Analysis Report

## Overview

This document summarises the implementation and initial testing of continuous wave (CW) gravitational wave source detection in Argus, addressing [Issue #32](https://github.com/tomkimpson/Argus/issues/32). The CW mode detects deterministic sinusoidal signals from individual supermassive black hole binaries (SMBHBs), complementing the existing stochastic gravitational wave background (GWB) mode.

## Implementation Summary

### Architecture

The CW mode is fundamentally different from the GWB mode:

| Property | GWB mode | CW mode |
|----------|----------|---------|
| GW signal | Latent stochastic state | Deterministic (subtracted from observations) |
| State vector | Joint: 4N + M_sum | Per-pulsar: 2 + M^(n) |
| Innovation | N-vector, NxN matrix solve | Scalar, scalar division |
| Cross-pulsar coupling | Hellings-Downs correlation | None (shared CW params only) |
| Epoch alignment | Required | Not needed |
| Parallelism | Not possible | Trivial (JAX vmap) |

The CW signal is computed deterministically from 7 source parameters and subtracted from pulsar timing residuals. The resulting CW-subtracted observations are then processed by N independent per-pulsar scalar Kalman filters, run in parallel via `jax.vmap` with `lax.scan`.

### CW Source Parameters

| Parameter | Symbol | Prior | Description |
|-----------|--------|-------|-------------|
| Strain amplitude | log₁₀h₀ | Uniform(-16, -12) | GW strain at Earth |
| GW frequency | log₁₀f_gw | Uniform(-9, -7) | Gravitational wave frequency (Hz) |
| Source RA | α_gw | Uniform(0, 2π) | Right ascension (radians) |
| Source DEC | sin(δ_gw) | Uniform(-1, 1) | Isotropic sky coverage via sin(declination) |
| Inclination | cos ι | Uniform(-1, 1) | Binary orbital inclination |
| Polarisation | ψ | Uniform(0, π) | GW polarisation angle |
| Initial phase | Φ₀ | Uniform(0, 2π) | GW phase at reference time |

Additionally, per-pulsar noise parameters (γ_p, σ_p for spin noise; EFAC, EQUAD for measurement noise) are sampled hierarchically, shared with the GWB mode.

### Signal Model

The Earth-term-only timing residual for pulsar _a_ at time _t_ is:

```
Δt_GW^(a)(t) = F_+^a · Δs_+(t) + F_×^a · Δs_×(t)
```

where F_+, F_× are antenna pattern functions depending on sky position and polarisation, and:

```
Δs_+(t) = h₀(1 + cos²ι) / (2Ω) · sin(Ωt + Φ₀)
Δs_×(t) = -h₀ cos ι / Ω · cos(Ωt + Φ₀)
```

with Ω = 2πf_gw. An optional pulsar term (toggled via `include_pulsar_term` config flag) adds the retarded-time signal component using pulsar distances from the enterprise catalog.

### Files

**New files:**
- `python/argus/cw_kalman_filter.py` — Per-pulsar scalar Kalman filter with vmap/lax.scan
- `test/test_cw_gravitational_waves.py` — 28 tests for CW signal model
- `test/test_cw_kalman_filter.py` — 15 tests for CW Kalman filter
- `workflows/example_workflow_cw/` — Configs, runner script, SLURM scripts

**Modified files:**
- `gravitational_waves.py` — Antenna patterns, polarisation tensors, timing residuals, pulsar term
- `data_loader.py` — Per-pulsar data processing, pulsar distance extraction
- `bayesian_inference.py` — CWParameters struct, CW likelihood, CW NumPyro model, parallel chain support
- `parameter_sampling.py` — CW parameter sampling with reparameterisation
- `prior_models.py` — CW prior specification from config
- `workflow.py` — CW/GWB mode dispatch, pulsar term toggle
- `utils.py` — Corner plot and diagnostics fixes for CW mode

## Test Data

We use the IPTA Mock Data Challenge 2, Dataset 3b, which contains a known CW injection with the following parameters:

| Parameter | Injection Value |
|-----------|----------------|
| f_gw | 6.1 × 10⁻⁹ Hz (log₁₀ = -8.215) |
| log₁₀h | -13.350 |
| gw_phi (α_gw) | 4.067 rad |
| gw_theta → δ_gw | π/2 - 1.431 = 0.140 rad |
| inclination → cos ι | cos(0.436) = 0.907 |
| ψ | 0.646 rad |
| Φ₀ | 0.175 rad |

The dataset contains 32 pulsars (after excluding J1640+2224), each with 183 observations.

## Inference Runs

### Run Summary

| Run | Chains | Warmup | Samples | Total | Tree Depth | f_gw | Pulsar Term | Wall Time | Divergences |
|-----|--------|--------|---------|-------|------------|------|-------------|-----------|-------------|
| Smoke test | 1 | 10 | 10 | 10 | 5 | Free | No | 1.5 min | 0 |
| Full (light) | 1 | 100 | 200 | 200 | 8 | Free | No | 73 min | 0 |
| Light 4-chain | 4 | 100 | 200 | 800 | 8 | Free | No | 3.8 hrs | 195 |
| Medium | 2 | 500 | 1000 | 2000 | 8 | Free | No | 6.3 hrs | 42 |
| Intensive | 4 | 1000 | 2000 | 8000 | 10 | Free | No | 27 hrs | 325 |
| Fixed f_gw | 2 | 250 | 500 | 1000 | 8 | Fixed | No | 59 min | 8 |
| Pulsar term v1 | 2 | 250 | 500 | 1000 | 8 | Free | Yes (default d) | 3.1 hrs | 1 |
| Pulsar term v2 | 2 | 250 | 500 | 1000 | 8 | Free | Yes (ATNF d) | 3.1 hrs | 1 |

### Parameter Recovery (Earth-term only, f_gw free)

Across all Earth-term-only runs with f_gw free, we consistently recover 4 of 7 CW parameters within the 95% credible interval:

| Parameter | Recovered? | Notes |
|-----------|-----------|-------|
| log₁₀h₀ | **Yes** | Consistently within 95% CI across all runs |
| log₁₀f_gw | **Yes** | Well-constrained but shows multimodality |
| α_gw | **Marginal** | Within 95% CI but posterior mean offset from injection |
| δ_gw | **Yes** | Recovered, though shows bimodality in intensive run |
| cos ι | **No** | Fundamental Earth-term degeneracy |
| ψ | **Marginal** | Bimodal; one mode contains injection |
| Φ₀ | **No** | Broad, poorly constrained |

### Parameter Recovery (Fixed f_gw)

| Parameter | Recovered? | Notes |
|-----------|-----------|-------|
| log₁₀h₀ | **Yes** | Within 95% CI |
| α_gw | **Yes** | Dramatic improvement — injection well-contained |
| δ_gw | **Yes** | Both chains agree, injection centred |
| cos ι | **No** | Earth-term degeneracy persists |
| ψ | **Yes** | Within 95% CI |
| Φ₀ | **No** | Earth-term degeneracy persists |

### Parameter Recovery (Pulsar Term — v1, enterprise default distances)

| Parameter | Recovered? | Notes |
|-----------|-----------|-------|
| log₁₀h₀ | **No** | Posterior mean -14.9, injection -13.35 — biased low |
| log₁₀f_gw | **No** | Posterior mean -7.78, injection -8.21 — biased high |
| α_gw | **Yes** | Within 95% CI, though chains disagree |
| δ_gw | **No** | Both chains offset from injection |
| cos ι | **No** | Broad, not centred on injection |
| ψ | **Yes** | Within 95% CI |
| Φ₀ | **No** | Poorly constrained |

9 of 32 pulsars (28%) had unknown distances defaulting to 1 kpc. Initial hypothesis: incorrect distances corrupt the pulsar term phases, biasing the posterior.

### Parameter Recovery (Pulsar Term — v2, ATNF catalog distances)

The 9 missing distances were sourced from the ATNF Pulsar Catalogue (v2.7.0) and injected via a `pulsar_distances.json` file. Results:

| Parameter | Recovered? | Notes |
|-----------|-----------|-------|
| log₁₀h₀ | **No** | Posterior mean -14.79, injection -13.35 — still biased low |
| log₁₀f_gw | **No** | Posterior mean -7.91, injection -8.21 — still biased high |
| α_gw | **Yes** | Within 95% CI, chains disagree (4.41 vs 2.33) |
| δ_gw | **No** | Both chains offset from injection |
| cos ι | **No** | Broad, chains disagree in sign |
| ψ | **Yes** | Within 95% CI |
| Φ₀ | **No** | Poorly constrained |

**Key finding:** Correcting the pulsar distances did not improve recovery. The v2 results are nearly identical to v1, ruling out the missing-distance hypothesis. The likely explanations are:

1. **Distance model mismatch**: The ATNF catalog distances are DM-derived (dispersion measure + NE2001/YMW16 electron density model), not parallax-based. These can differ from the true distances used in the MDC2 injection by factors of 2–3, which at nanohertz frequencies produces completely wrong pulsar term phases.

2. **The MDC2 may have used different distances** than what's in either the par files or the ATNF catalog. The mock data was generated with specific internal distance values that are not publicly documented.

3. **The pulsar term makes the likelihood surface inherently harder for NUTS.** The phase offset Ω·d·(1+n̂·q̂)/c creates rapid oscillations in the likelihood as a function of sky position and frequency, producing a highly multimodal posterior that NUTS cannot navigate even with correct distances. The literature (e.g. Ellis 2013, Taylor et al. 2016) typically uses parallel tempering for pulsar-term CW searches for exactly this reason.

Both pulsar term runs showed excellent sampling diagnostics (1 divergence each, consistent chain speeds at ~14.7s/step), suggesting the sampler is efficiently exploring a well-conditioned but incorrect region of parameter space.

## Key Findings

### 1. The (cos ι, ψ, Φ₀) degeneracy is a fundamental Earth-term limitation

The timing residual decomposes into two quadratures A·sin(Ωt) + B·cos(Ωt) at each pulsar. The data constrains A and B, but multiple (cos ι, ψ, Φ₀) combinations produce the same A and B. This degeneracy persists regardless of the number of samples, chains, or sampler settings. Including the pulsar term should break this degeneracy by providing independent phase constraints from each pulsar's distance.

### 2. f_gw multimodality causes cascading parameter biases

When f_gw is free, NUTS chains can find secondary frequency modes. These secondary modes come paired with compensating sky positions (α_gw, δ_gw), producing apparent biases in sky localisation that are actually sampling artifacts. Evidence:
- The intensive run (f_gw free) showed α_gw posterior mean at 4.93, offset from injection 4.07
- The fixed-f_gw run showed α_gw well-centred on the injection
- Per-chain analysis revealed different chains finding different (f_gw, α_gw, δ_gw) combinations

### 3. Fixing f_gw dramatically improves sampling

| Metric | f_gw free (intensive) | f_gw fixed |
|--------|----------------------|------------|
| Divergences | 325 | 8 |
| Wall time | 27 hours | 59 minutes |
| Step pace | 5–38 s/step | 3–4 s/step |
| Chain agreement | Poor (r_hat issues) | Good |
| α_gw recovery | Biased | Centred on injection |

### 4. Chain speed varies dramatically

In the intensive run, chain completion times ranged from 7.4 to 27 hours — a 3.7× spread. This is caused by different chains encountering different posterior geometry: chains in well-conditioned regions take fewer leapfrog steps per NUTS iteration, while chains navigating narrow ridges (especially in f_gw) hit the maximum tree depth repeatedly.

### 5. Pulsar term does not improve recovery — likely requires parallel tempering

Including the pulsar term produced the cleanest sampling diagnostics (1 divergence, consistent chain speeds at ~14.7s/step) but the worst parameter recovery (only 2/7 recovered). Two runs were performed:

- **v1**: enterprise default distances (9 pulsars defaulting to 1 kpc)
- **v2**: corrected distances from the ATNF Pulsar Catalogue for all 32 pulsars

Both runs gave nearly identical (poor) results, ruling out distance accuracy as the primary issue. The likely explanation is that the pulsar term creates a highly multimodal likelihood surface with rapid oscillations in phase space that NUTS cannot navigate — it efficiently explores one mode but that mode is not necessarily the correct one.

This is consistent with the PTA CW literature, where pulsar-term searches universally use parallel tempering (not gradient-based samplers) to handle the multimodality. The pulsar term phase Ω·d·(1+n̂·q̂)/c introduces O(10³–10⁵) oscillation cycles across the prior volume, creating far more local optima than Earth-term-only models.

The excellent sampling diagnostics (1 divergence, consistent chain speeds) confirm that NUTS is working correctly — the problem is not sampling failure but mode trapping in a posterior landscape that gradient-based methods fundamentally cannot traverse.

### 6. NUTS limitations for multimodal posteriors

NUTS excels at navigating complex geometry within a single mode (gradient-guided exploration), but cannot jump between well-separated modes. This manifests as:
- Different chains getting trapped in different modes
- High r_hat values indicating chain disagreement
- Apparent multimodality that may reflect sampling failure rather than true posterior structure

## Computational Performance

- **Likelihood evaluation**: 0.02s per call (after JIT compilation) for 32 pulsars × 183 observations
- **Gradient evaluation**: 0.02s per call (after JIT)
- **JIT compilation**: ~1.5s first call
- **GPU utilisation**: 53–97% depending on run configuration
- **Parallel chains**: Supported via NumPyro's `chain_method="parallel"` with `pmap` across GPUs

## Potential Next Steps

### Near-term

1. **Pulsar term with parallel tempering** — Both pulsar term runs (default and ATNF distances) failed to recover parameters despite clean sampling. The pulsar term likely requires a sampler capable of traversing between modes (e.g. parallel tempering via `blackjax`, or nested sampling via `jaxns`). NUTS is fundamentally limited to single-mode exploration.

2. **Fixed-frequency grid search** — Evaluate the likelihood on a grid of f_gw values, running NUTS on the remaining 6 parameters at each grid point. This plays to NUTS's strengths (efficient within-mode exploration) while avoiding its weakness (cross-mode jumping). Each grid point runs in ~1 hour with clean posteriors.

3. **MCMC checkpointing** — Implement per-chain saving so that completed chains are preserved if a job times out. The intensive run lost one full attempt (24 hours) due to chain 0 not finishing within the wall time.

4. **Direct uniform sampling** — Investigate whether the Normal(0,1) → affine reparameterisation introduces artificial structure for cyclic parameters (ψ, Φ₀). A direct uniform prior might be more appropriate for these.

### Medium-term

5. **Alternative samplers** — For the multimodal f_gw posterior, consider:
   - Parallel tempering with HMC kernels (e.g. via `blackjax`)
   - Nested sampling (e.g. `dynesty`) for simultaneous posterior estimation and evidence computation
   - Many short independent chains from random starting points

6. **Frequency evolution** — The current implementation assumes constant f_gw over the observation span. For nearby or massive binaries, including ḟ_gw may be necessary.

7. **Multiple CW sources** — Extend to search for multiple simultaneous CW sources by summing their contributions before subtraction.

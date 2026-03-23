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
| Pulsar term | 2 | 250 | 500 | 1000 | 8 | Free | Yes | 3.1 hrs | 1 |

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

### Parameter Recovery (Pulsar Term Included)

| Parameter | Recovered? | Notes |
|-----------|-----------|-------|
| log₁₀h₀ | **No** | Posterior mean -14.9, injection -13.35 — biased low |
| log₁₀f_gw | **No** | Posterior mean -7.78, injection -8.21 — biased high |
| α_gw | **Yes** | Within 95% CI, though chains disagree |
| δ_gw | **No** | Both chains offset from injection |
| cos ι | **No** | Broad, not centred on injection |
| ψ | **Yes** | Within 95% CI |
| Φ₀ | **No** | Poorly constrained |

**Critical caveat:** 9 of 32 pulsars (28%) had unknown distances defaulting to 1 kpc. The pulsar term phase offset depends sensitively on distance (Ω·d·(1+n̂·q̂)/c), so incorrect distances introduce a misspecified signal component. The sampler compensates by shifting all parameters to accommodate the phase mismatch, corrupting the entire posterior. These results do **not** reflect the expected performance of a correctly specified pulsar term model.

Pulsars with default (likely incorrect) distances: J0034-0534, J0218+4232, J0621+1002, J0751+1807, J0900-3144, J1614-2230, J1944+0907, J2010-1323, J2229+2643.

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

### 5. Pulsar term requires accurate distances

Including the pulsar term (Earth + pulsar term model) produced the cleanest sampling (1 divergence, consistent chain speeds) but the worst parameter recovery (only 2/7 recovered). This is caused by 9/32 pulsars having unknown distances defaulting to 1 kpc — the incorrect distances introduce misspecified phase offsets that bias the entire posterior.

This demonstrates that the pulsar term is highly sensitive to distance accuracy. The phase offset Ω·d·(1+n̂·q̂)/c means even modest distance errors (factor of a few) produce completely wrong pulsar term contributions at nanohertz frequencies. To properly exploit the pulsar term for breaking the (cos ι, ψ, Φ₀) degeneracy, we need either:
- Accurate distances for all pulsars (e.g. from ATNF catalog parallax measurements)
- Exclusion of pulsars with unknown distances
- Marginalisation over distance uncertainties (sampling pulsar distances as additional parameters)

Notably, the sampling itself was excellent — 1 divergence and both chains running at identical pace (~14.7s/step). The pulsar term adds useful constraining power to the likelihood surface, making it better-conditioned for NUTS even though the constraints point to the wrong solution with incorrect distances.

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

1. **Pulsar term with accurate distances** — The initial pulsar term run demonstrated that incorrect distances corrupt the posterior. Next steps: source accurate parallax distances from the ATNF catalog, or exclude the 9 pulsars with unknown distances and re-run. Alternatively, sample pulsar distances as additional parameters with Gaussian priors around catalog values.

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

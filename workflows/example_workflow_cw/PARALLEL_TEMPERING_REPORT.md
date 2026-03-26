# Parallel Tempering & Alternative Samplers for CW Inference

## Problem Statement

We need to sample the posterior for continuous wave (CW) gravitational wave sources, including free GW frequency (`f_gw`) and pulsar terms via the phase reparameterisation (`chi`). This creates a high-dimensional (~107 parameters for 31 pulsars) and multimodal posterior. Standard NUTS sampling can get trapped in local modes. We explored three alternative sampling approaches to complement the existing NUTS implementation.

## Parameter Space

For the IPTA MDC2 dataset 3b with 31 pulsars (excluding J1640+2224):

| Parameter group | Count | Description |
|----------------|-------|-------------|
| CW source | 7 | log10_h0, alpha_gw, sin_delta_gw, log10_f_gw, cos_iota, psi, Phi0 |
| Pulsar phases (chi) | 31 | Phase reparameterisation for pulsar term |
| Hierarchical gamma_p | 2 + 31 | Hyperparameters + per-pulsar red noise damping |
| Hierarchical ratio (sigma_p) | 2 + 31 | Hyperparameters + per-pulsar red noise amplitude |
| EFAC | fixed | From noise params JSON |
| EQUAD | fixed | From noise params JSON |
| **Total** | **107** | |

## Approach 1: Blackjax Tempered SMC

### Implementation

Tempered Sequential Monte Carlo using `blackjax.smc.adaptive_tempered`. Particles move from the prior (lambda=0) to the posterior (lambda=1) through a sequence of tempered distributions, with NUTS mutation kernels at each temperature rung.

**Files:**
- `python/argus/tempered_smc.py` — parameter registry, log-prob functions, SMC execution, ArviZ conversion, diagnostics
- `python/argus/bayesian_inference.py` — `run_tempered_smc()` wrapper
- `python/argus/workflow.py` — routing for `sampler = smc`

**Config:** `[TemperedSMC]` section with `num_particles`, `num_mcmc_steps`, `adaptive`, `target_ess`, `step_size`

### Results

**Light run (100 particles, 5 MCMC steps/rung):**
- Completed successfully in ~3.8 hours (of which ~1.7 hours was JIT compilation)
- 38 adaptive temperature steps to reach lambda=1.0
- Evidence: log_Z = 63394.89
- 97.5% GPU utilisation after JIT compilation
- 100 posterior samples (thin, but pipeline verified end-to-end)

### Issues: Memory & JIT Compilation

| Run | Particles | MCMC steps | Memory request | Outcome |
|-----|-----------|------------|---------------|---------|
| Light | 100 | 5 | 3 GB peak | Success |
| Heavy | 2000 | 20 | 37 GB | OOM |
| Heavy | 1000 | 20 | 19 GB | OOM |
| Heavy | 500 | 20 | 9.3 GB | OOM |
| Heavy | 250 | 20 | 44 GB | OOM (JIT compilation overhead) |
| Heavy batch | 250 | 5 | ~7.5 GB (est.) | Running |

**Root cause:** Blackjax vmaps the entire NUTS state (position, momentum, gradients, mass matrix, binary tree trajectory) across all particles simultaneously. Memory scales as `O(particles × mcmc_steps × tree_depth × ndim)`. The NUTS tree-building is particularly memory-hungry because it stores the full trajectory for each particle.

**JIT compilation time:** ~1.7 hours for 100 particles × 107 dims. This is because JAX traces through `vmap(scan(nuts_step))` — nested vectorisation and sequential operations creating a massive XLA computation graph. Standard NUTS (via NumPyro) avoids this by tracing a single chain and handling parallelism at a higher level.

**JAX compilation cache:** Enabled via `JAX_COMPILATION_CACHE_DIR`. Compiled kernels are saved to disk and reused across runs with the same computation graph. However, changing `num_particles` changes the vmap shape and requires recompilation.

### Current status

Batch run with 4 × 250 particles × 5 MCMC steps submitted (job 10807272). If successful, will yield 1000 total posterior samples via concatenation.


## Approach 2: Dynesty Nested Sampling

### Implementation

Pure Python nested sampler (`dynesty`) that calls the JAX likelihood as a black box. Avoids JIT compilation overhead entirely — the likelihood is compiled once for a single evaluation (seconds) and then called repeatedly.

**Files:**
- `python/argus/dynesty_sampler.py` — parameter layout, prior_transform, likelihood wrapper, sampling, ArviZ conversion
- `python/argus/bayesian_inference.py` — `run_dynesty()` wrapper
- `python/argus/workflow.py` — routing for `sampler = dynesty`

**Config:** `[Dynesty]` section with `nlive`, `dlogz`, `bound`, `sample`, `dynamic`

### Results

**Light run (nlive=100, dlogz=1.0):**
- Completed in ~10.5 hours
- Evidence: log_Z = 63351.97 ± 1.77 (consistent with SMC: 63394.89)
- 5000 posterior samples (resampled from nested sampling weights)
- No JIT compilation issues — started sampling immediately
- Efficiency decreased as bounds tightened in 107 dims (~5s/iteration near convergence)

### Issues

- **nlive=100 is critically undersampled for 107 dims** — dynesty warned `nlive <= 2 * ndim`. Posteriors for some parameters (h0, cos_iota) don't recover the injections well due to mode trapping.
- **Runtime scales roughly linearly with nlive** — production run with nlive=1000 would take ~100 hours (~4 days).
- **No GPU acceleration of the sampling loop** — dynesty is CPU-bound for the sampling logic, only the likelihood evaluation runs on GPU.


## Approach 3: Standard Parallel Tempering MCMC (Not Yet Implemented)

### Motivation

The standard approach in PTA CW analyses (NANOGrav, EPTA, IPTA) uses parallel tempering MCMC with sequential chains, not vmapped particles:

- **NANOGrav 15yr CW:** QuickCW with 100M iterations, 8 PT chains, temperatures 0–3
- **PTMCMCSampler:** Geometric temperature ladder, 25% swap acceptance rate, Tskip=100

This avoids the OOM issues of blackjax (no vmap over particles) and the slow convergence of nested sampling in high dimensions. Each chain runs independently with periodic swap proposals between adjacent temperatures.

### How it would differ from our tempered SMC

| | Tempered SMC (current) | Replica Exchange MCMC (proposed) |
|-|----------------------|----------------------------------|
| Architecture | Population of particles | Fixed number of persistent chains |
| Temperature | Gradually increased from 0→1 | Fixed per chain, swaps between chains |
| Memory | O(particles × ndim) all at once | O(chains × ndim) — much smaller |
| GPU usage | vmap over particles (memory-heavy) | Sequential or pmap over chains |
| Evidence | Yes (byproduct of SMC) | No (need Savage-Dickey or thermodynamic integration) |
| Multimodality | Resampling helps | Swaps between hot/cold chains |


## Evidence Comparison

| Method | log_Z | Uncertainty | Runtime |
|--------|-------|-------------|---------|
| Tempered SMC (100 particles) | 63394.89 | — | 3.8 hrs |
| Dynesty (nlive=100) | 63351.97 | ± 1.77 | 10.5 hrs |

Both methods agree on the order of magnitude. The ~43 unit difference is expected given the light settings — both runs are undersampled for 107 dimensions.


## Recommendations

1. **Short term:** Use the 4×250 particle SMC batch for approximate posteriors + evidence. Use NUTS (4 chains × 2000 samples) as the primary posterior sampler.

2. **Medium term:** Implement proper replica exchange MCMC (sequential chains, not vmapped) to match the PTA field standard. This avoids the memory scaling issues and can run 8+ temperature chains with negligible overhead per chain.

3. **Production dynesty:** Run with nlive=500–1000 if evidence estimates are needed. Accept ~2–4 day runtime.

4. **HMC mutation kernel for SMC:** Replace NUTS with HMC (fixed trajectory length) in the tempered SMC to reduce both JIT compilation time and per-particle memory. The tree-building in NUTS is the main memory driver.

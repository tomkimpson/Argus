# Strategies for Handling Nuisance Parameters in CW Inference

## Problem Statement

The CW parameter space has 107 dimensions, but only 7 are the CW source parameters we care about. The remaining ~100 are nuisance parameters:
- 31 per-pulsar phase parameters (chi) — needed for the pulsar term
- 4 noise hyperparameters (gamma_p_mean, gamma_p_std, ratio_mean, ratio_std)
- 62 per-pulsar noise parameters (31 gamma_p + 31 sigma_p)
- EFAC/EQUAD (currently fixed from noise files)

These nuisance parameters must be marginalised over for correct CW posteriors, but we don't need detailed posteriors for them. Currently, all 107 parameters are sampled jointly, meaning every HMC proposal requires a full Kalman filter pass through all dimensions.

For context, PTA collaborations face the same problem (~100-130 total parameters, ~7 of interest). QuickCW handles it with a Metropolis-within-Gibbs scheme that updates CW and noise parameters in separate blocks, with a ~10,000x speedup for extrinsic CW updates via cached inner products. Our Kalman filter likelihood doesn't have that decomposition — every parameter change requires a full forward pass.

## Structural Properties of Our Likelihood

Key facts (from the CW Kalman filter analysis):
- **Noise-CW decoupling**: The CW signal is subtracted deterministically before the KF processes residuals. Noise parameters only affect the KF process noise, not the signal. Empirically confirmed: noise ESS is 1000-2300 even when CW params have ESS of 5.
- **Near-Gaussian noise posteriors**: All NUTS runs show noise parameters are well-constrained, unimodal, and approximately Gaussian (high ESS, r_hat < 1.02).
- **Per-pulsar independence**: Given CW source params, each pulsar's noise inference is independent (already exploited via `jax.vmap`).

## Options

### Option 1: Fix Noise at MAP

**Approach:** Run a short NUTS or optimisation pass to find the noise parameter MAP, then fix them and only sample the 38-dim CW+chi subspace.

**Pros:**
- Simplest to implement — just set noise params as fixed in the config
- Reduces dimensionality from 107 to 38 immediately
- This is what many PTA analyses actually do (fix noise from a separate noise-only run)
- Our NUTS runs show the noise MAP is robust across all chain configurations

**Cons:**
- Loses noise parameter uncertainty (no marginalisation)
- The fixed noise values could bias CW posteriors if there's any residual noise-CW coupling
- Not fully Bayesian

**Assessment:** Safe for our problem given the empirical evidence of noise-CW decoupling. A good pragmatic first step if we need quick results.

### Option 2: Block Gibbs Sampling (Recommended)

**Approach:** Alternate between two HMC blocks within each replica exchange iteration:
- **Block A** (every iteration): Fix noise parameters, run HMC in the 38-dim CW+chi subspace
- **Block B** (every Nth iteration, e.g. N=10-20): Fix CW+chi parameters, run HMC in the 69-dim noise subspace

This is valid MCMC — Gibbs sampling preserves the correct stationary distribution. The full joint posterior is correctly sampled, but each block is cheaper than a 107-dim HMC step.

**Pros:**
- Fully Bayesian — correctly marginalises over noise parameters
- CW block runs at 38 dims: shorter trajectories, higher acceptance rates, faster per-step
- Noise block is easy (well-conditioned, near-Gaussian) and only needs occasional updates
- Mirrors QuickCW's Metropolis-within-Gibbs structure but with HMC
- Naturally compatible with parallel tempering (each temperature chain does Gibbs updates)

**Cons:**
- Gibbs introduces correlation between blocks — mixing depends on how often noise is updated
- Implementation requires splitting the parameter vector and building separate HMC kernels per block
- Need to manage two mass matrices and step sizes (one per block)

**Assessment:** Best balance of correctness and efficiency. The noise block at 69 dims is well-conditioned and can use large step sizes (the noise posterior is nearly quadratic). The CW+chi block at 38 dims is where the interesting multimodal structure lives, and HMC with gradients handles it well.

### Option 3: Profile Likelihood for CW Parameters

**Approach:** For each proposed CW+chi parameter set, quickly optimise the noise parameters to their conditional mode via a few Newton steps. Use this "profile" likelihood for HMC proposals over the 38-dim CW+chi subspace.

Unlike the Laplace approximation (archived), this involves **no nested autodiff** — the noise optimisation is forward-only computation. HMC gradients are computed only w.r.t. the 38 CW+chi parameters, and the noise optimisation doesn't need to be differentiated through.

**Pros:**
- Reduces effective dimensionality to 38 for the sampler
- No nested autodiff (unlike Laplace) — the inner optimisation is just forward computation
- Per-pulsar independence means 31 independent 2D Newton solves, vmapped in parallel

**Cons:**
- The profile likelihood ≠ the marginal likelihood — introduces a bias compared to full marginalisation
- The HMC gradient of the profile likelihood w.r.t. CW params uses the implicit function theorem — this requires computing cross-derivatives (d²L/dCW·dNoise) which may be non-trivial
- Could be combined with occasional full Gibbs noise updates to correct the bias

**Assessment:** Promising but requires careful implementation to get the gradients right. The implicit function theorem approach would give exact gradients without backpropagating through the Newton iterations, but adds implementation complexity.

### Option 4: Subspace HMC with Occasional Full-Space Jumps

**Approach:** Run HMC proposals in a low-dimensional subspace (the CW + a few principal noise directions identified from the mass matrix), with occasional full 107-dim proposals to explore the noise directions.

**Pros:**
- Adaptive — the subspace can be tuned based on the warmup mass matrix
- Keeps the sampler fully general
- No need to explicitly partition parameters

**Cons:**
- Identifying the right subspace requires analysis of the mass matrix
- The "occasional full-space" jumps may have very low acceptance in 107 dims
- More complex to implement than Gibbs blocking

**Assessment:** Elegant but harder to implement and tune than explicit Gibbs blocking. Better suited as a future refinement.

### Option 5: Mixed Proposal Types

**Approach:** Use HMC for the CW+chi parameters (where gradients are essential for navigating the multimodal landscape) but use cheap Metropolis-Hastings or adaptive Metropolis for the noise parameters (where the posterior is nearly Gaussian and well-conditioned).

**Pros:**
- Uses the right tool for each parameter block
- MH for noise is very cheap (~0.006s per proposal, no gradient needed)
- HMC for CW+chi gets the gradient benefit where it matters most
- Mirrors the PTA field approach but leverages gradients selectively

**Cons:**
- MH acceptance in 69 noise dims may be low without careful tuning
- Need to manage two different proposal types within the same sampler
- Slightly more complex than pure HMC Gibbs

**Assessment:** A good variant of Option 2. If we do Gibbs blocking, the noise block could use MH instead of HMC to avoid the gradient cost.

## Recommendation

**Option 2 (Block Gibbs)** is the recommended next step. The implementation plan:

1. Split the parameter registry into two blocks: CW+chi (38 dims) and noise (69 dims)
2. Build separate HMC kernels for each block (different step sizes, mass matrices, trajectory lengths)
3. Within each replica exchange iteration:
   - Always run the CW+chi HMC block (38 dims, needs gradients for multimodality)
   - Every N iterations, run the noise HMC block (69 dims, well-conditioned)
4. The noise block can optionally use MH instead of HMC (Option 5 variant)

Expected impact:
- CW block at 38 dims: ~3x shorter trajectory needed, ~2-3x higher acceptance rate
- Noise block at 69 dims: runs infrequently (1 in 10-20 iterations), always accepts easily
- Net speedup: ~2-4x per effective sample, plus better mixing from the lower-dimensional CW proposals

This can be implemented incrementally — start with the Gibbs blocking within a single-temperature chain, verify it works, then combine with replica exchange for mode-hopping.

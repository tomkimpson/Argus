# Alternative Sampling Strategies for CW Inference

## Motivation

NUTS cannot handle the multimodal posterior created by the CW frequency parameter and the 32-dimensional per-pulsar phase (χ) space. Likelihood profiling confirms the signal model is correct and the pulsar term provides genuine constraints — the problem is purely sampling. This document outlines alternative inference strategies under consideration.

## Current Status

- **NUTS with fixed f_gw**: Recovers sky position (α_gw, δ_gw) well. Partially constrains (cos ι, ψ, Φ₀) via pulsar term but biased — likely trapped in one mode of the joint χ space. Not viable for real data where f_gw is unknown.
- **NUTS with free f_gw**: Chains find different f_gw modes, r_hat ~1.9, poor recovery across the board.
- **jaxns nested sampling**: Pipeline works (validated on 2-dim problem) but JIT compilation is prohibitive at >40 dimensions with our Kalman filter likelihood. Feasible if noise parameters are fixed (~39 dims), pending investigation of compilation scaling.

## Option 1: Parallel Tempering via blackjax

**Status:** Active development on a separate branch/worktree.

**Approach:** Run multiple MCMC chains at different temperatures T₀ = 1 (cold/posterior) through T_K (hot/prior-like). Hot chains explore broadly and can jump between f_gw modes; replica exchange swaps between adjacent temperatures, propagating mode discoveries to the cold chain.

**Why it should work:**
- This is the standard approach in PTA CW analyses (enterprise + PTMCMCSampler, QuickCW)
- Directly addresses the mode-hopping limitation of NUTS
- blackjax is JAX-native — can reuse our existing likelihood without modification
- Can combine HMC proposals (efficient within-mode) with parallel tempering (cross-mode)

**Considerations:**
- Need to tune the temperature ladder (number of temperatures, spacing)
- Each temperature requires its own chain — computational cost scales linearly with number of temperatures
- blackjax's parallel tempering API may require custom adaptation for our high-dimensional problem
- The (f_gw, χ) coupling means hot chains need to be hot enough to flatten the f_gw multimodality while still being informative for the χ parameters

**Key references:**
- Vousden et al. (2016) — adaptive temperature selection for parallel tempering
- Ellis (2013), Taylor et al. (2014) — parallel tempering for PTA CW searches
- QuickCW (Bécsy et al. 2022) — Fisher-matrix-enhanced proposals for PTA CW

## Option 2: flowMC (Normalising Flow Enhanced MCMC)

**Approach:** Use normalising flows as learned proposal distributions within MCMC. The flow learns the posterior geometry during an initial training phase, then proposes jumps that respect the posterior structure — enabling efficient mode-hopping without the temperature ladder overhead.

**Why it could work:**
- Normalising flows can learn multimodal distributions and propose cross-mode jumps
- JAX-native (github.com/kazewong/flowMC) — compatible with our likelihood
- Potentially more efficient than parallel tempering for high-dimensional multimodal problems
- Combines the flexibility of learned proposals with the exactness guarantees of MCMC (detailed balance)

**Considerations:**
- Flow training requires an initial exploration phase — quality depends on seeing enough modes during training
- May struggle if the modes are very narrow relative to the prior volume (the flow needs to discover them)
- Relatively new method — less battle-tested than parallel tempering for PTA problems
- Hyperparameter tuning (flow architecture, training schedule, number of training samples)
- Our 107-dim parameter space is high for normalising flows — may need to train on a reduced representation

**Key references:**
- Wong et al. (2022) — flowMC: normalising flow enhanced sampling
- Gabrie et al. (2022) — adaptive Monte Carlo with normalising flows

## Option 3: Stochastic Variational Inference (SVI) via NumPyro

**Approach:** Abandon sampling entirely. Instead, fit a parametric approximation q(θ) to the posterior p(θ|d) by maximising the Evidence Lower Bound (ELBO). NumPyro provides SVI with automatic differentiation through our JAX likelihood.

**Why it could be useful:**
- Much faster than MCMC — optimisation rather than sampling
- NumPyro has built-in SVI with flexible guide families (multivariate normal, normalising flows via AutoIAFNormal)
- Can handle high dimensions efficiently since it uses gradients (like NUTS) but doesn't need to converge chains
- Provides a lower bound on the evidence (ELBO) — useful for model comparison
- Could serve as an initialisation strategy: run SVI to find the posterior region, then start NUTS chains from the SVI solution

**Considerations:**
- Variational inference is approximate — the posterior q(θ) may not capture the true posterior shape, especially multimodality
- Standard mean-field or full-rank Gaussian guides cannot represent multimodal posteriors. Would need normalising flow guides (AutoIAFNormal) or mixture guides
- ELBO optimisation can get stuck in local optima — the same multimodality problem, just in optimisation space rather than sampling space
- No asymptotic exactness guarantee (unlike MCMC)
- May underestimate posterior uncertainty (a known VI failure mode)

**Practical use case:** Even if SVI doesn't give publication-quality posteriors, it could be valuable as:
1. A fast diagnostic — does the ELBO-optimal solution agree with NUTS?
2. An initialiser — use the SVI posterior to warm-start NUTS chains near the correct mode
3. A model comparison tool — compare ELBO across models (signal vs noise-only) as a quick detection statistic

**Key references:**
- Blei et al. (2017) — Variational inference: a review
- NumPyro SVI documentation — autoguides (AutoNormal, AutoIAFNormal, AutoDAIS)

## Option 4: Fixed-f_gw Grid Search + NUTS

**Approach:** Evaluate the posterior on a grid of f_gw values (e.g., 50–100 points across the [-9, -7] prior range). At each grid point, run NUTS on the remaining parameters (6 CW source + 32 χ + noise) with f_gw fixed. Combine the per-grid-point posteriors via importance weighting to recover the full joint posterior including f_gw.

**Why it should work:**
- We already know NUTS works well with fixed f_gw (Run 11: 2 divergences, sky position recovered, pulsar term constraining cos ι)
- Sidesteps the f_gw multimodality entirely — each grid point is a unimodal problem
- Embarrassingly parallel — all grid points are independent, can run simultaneously on the cluster
- No new code required beyond a submission script and a post-processing combination step
- With 100 grid points at ~1–3 hours each on A100, total wall time is ~1–3 hours if fully parallelised

**Considerations:**
- f_gw is not treated as truly continuous — resolution limited by grid spacing. With 100 points over [-9, -7], spacing is 0.02 in log₁₀f_gw, which is finer than the posterior width (~0.01 based on the fixed-f_gw runs)
- Total compute is high (100 × 1–3 GPU-hours = 100–300 GPU-hours), though parallelisable
- Importance weighting assumes the per-grid-point posteriors are representative — may be unreliable if the posterior shape changes rapidly between grid points
- Standard approach in parts of the PTA literature for CW searches
- Does not provide a single unified chain — post-processing combination adds complexity

**Key references:**
- Taylor et al. (2014) — frequency grid approach for PTA CW searches
- Ellis et al. (2012) — frequentist F-statistic on frequency grid

## Structural Properties of the Kalman Filter Likelihood

Our Kalman filter formulation has properties that generic samplers do not exploit. Understanding these may guide the choice of inference method and identify opportunities for novel approaches.

### 1. Per-pulsar independence

Given the CW source parameters, each pulsar's likelihood is independent. We exploit this computationally via `jax.vmap`, but a sampler could exploit it structurally — for example, **data-tempered SMC** could start with 1 pulsar, obtain a posterior, then sequentially add pulsars one at a time. Each addition sharpens the posterior incrementally, and SMC resampling handles mode death gracefully. This is naturally suited to our `vmap` decomposition and would be a novel contribution to PTA CW inference.

### 2. Analytic Rao-Blackwellisation

The Kalman filter analytically marginalises over the latent pulsar state (spin phase, spin frequency, timing model parameters). We only need to sample the "outer" parameters (CW source + noise). This is a massive variance reduction that we get for free — most PTA analyses using the frequency-domain likelihood do not have this property. The effective dimensionality of the sampling problem is already reduced before any inference method is applied.

### 3. Full gradient availability

The likelihood is fully JAX-differentiable via autodiff through `lax.scan`. This provides exact gradients at negligible cost beyond the likelihood evaluation itself (~0.02s each). This rules out vanilla Metropolis-Hastings approaches (wasteful when gradients are available) and favours gradient-informed methods: HMC/NUTS, Langevin dynamics, gradient-informed SMC proposals, or variational inference.

### 4. Sequential time-domain structure

The data has a natural time ordering processed by `lax.scan`. This could enable online/streaming posterior updates when new observations are added without reprocessing the entire dataset — relevant for real PTA analyses where new TOAs arrive regularly.

### Methods that exploit these properties

**Data-tempered SMC:** Use the per-pulsar independence to build up the posterior by sequentially adding pulsars, rather than using likelihood tempering (which heats the entire likelihood uniformly). Available in blackjax. Handles multimodality by construction and naturally uses our `vmap` decomposition.

**Laplace approximation for noise parameters:** The noise parameters (γ_p, σ_p) are well-constrained and approximately Gaussian (high ESS in all NUTS runs). A Laplace approximation (mode + Hessian via `jax.hessian`) could analytically marginalise them, reducing the sampling problem from ~107 dimensions to ~39 (7 CW + 32 χ). Directly addresses the dimensionality bottleneck for NUTS, jaxns, and other samplers.

**Block Gibbs with HMC:** Update CW source parameters, per-pulsar χ parameters, and noise parameters in separate blocks using HMC within each block. The χ parameters are conditionally unimodal (as shown by likelihood profiling), so NUTS works well for them given fixed CW params. The CW params given fixed χ is a lower-dimensional problem where mode-hopping is easier.

### Comparison with QuickCW

QuickCW achieves a ~10,000× speedup for extrinsic parameter updates by caching frequency-domain inner products and reusing them when only (h₀, cos ι, ψ, Φ₀) change. Our Kalman filter likelihood does not have this decomposition — every parameter change requires a full forward pass through the filter. Plugging our likelihood into QuickCW's parallel tempering machinery is possible but would not benefit from the projection/shape speed trick. The parallel tempering and Fisher-matrix proposals would still work, just at ~0.02s per likelihood evaluation rather than QuickCW's ~microsecond extrinsic updates.

## Comparison Summary

| | Parallel Tempering | flowMC | SVI | f_gw Grid Search |
|---|---|---|---|---|
| **Handles multimodality** | Yes (temperature exchanges) | Yes (learned proposals) | Partially (with flow guides) | Yes (by construction) |
| **Exactness** | Exact (given convergence) | Exact (MCMC guarantee) | Approximate | Exact per grid point |
| **JAX-native** | Yes (blackjax) | Yes | Yes (NumPyro) | Yes (existing NUTS) |
| **Reuses our likelihood** | Yes | Yes | Yes | Yes |
| **Computational cost** | High (N_temps × N_chains) | Medium (training + sampling) | Low (optimisation) | High but parallelisable |
| **Maturity for PTA CW** | Standard tool | Novel | Novel | Standard tool |
| **Evidence computation** | No (needs thermodynamic integration) | No | Yes (ELBO, approximate) | No |
| **Development effort** | Medium (already started) | Medium | Low (NumPyro built-in) | Low (scripting only) |
| **f_gw treatment** | Continuous | Continuous | Continuous | Discrete grid |

## Option 5: Laplace Marginalisation of Noise Parameters (Attempted, Archived)

**Status:** Implemented 2026-03-26, tested with NUTS and dynesty, archived 2026-03-27.

**Approach:** Analytically marginalise the 62 per-pulsar red noise parameters (gamma_p, sigma_p) via Laplace approximation, reducing the sampled dimensionality from ~107 to ~43. For each pulsar, a 2D Newton solve finds the MAP of (log10_gamma_p, log10_ratio) and a Hessian-based Gaussian integral replaces explicit sampling. All 32 pulsars are vmapped in parallel.

**Why it didn't work in practice:**

- **NUTS:** Each leapfrog step requires backpropagating through the Newton iterations (nested autodiff through `jax.grad`/`jax.hessian` of `lax.scan`). Per-step cost ~4.5s vs 0.02s for the standard likelihood (~200x slower). A 500-sample run would take ~67 hours per chain.

- **Dynesty:** Each likelihood evaluation runs 32 × 8 Newton steps with grad + Hessian through the KF scan. Per-eval cost ~250s vs ~5s. The 107-dim standard dynesty-500 run converges faster in wall time despite the higher dimensionality.

The per-likelihood cost increase of 50–200x dominates over sampling efficiency gains from halving the dimensions. The concept of reducing dimensionality is sound, but the inner optimisation cost is prohibitive with current implementation.

**Possible future improvements:** Implicit differentiation via `jax.custom_jvp` at the Newton solution would eliminate nested autodiff for gradient-based samplers, potentially making this viable. The core code is preserved in `archive/laplace_marginalisation/`.

## Option 6: Replica Exchange MCMC with HMC (Implemented 2026-03-27)

**Status:** Implemented, first run in progress (job 10831402).

**Approach:** K=8 persistent chains at fixed inverse temperatures (geometric ladder from β=1 to β=0.01), each running HMC proposals (fixed trajectory, not NUTS). Periodic swap proposals between adjacent temperatures via Metropolis-Hastings acceptance. NUTS warmup on the cold chain adapts step size + diagonal mass matrix; hot chain step sizes scaled by β^(-0.25).

**Implementation:** `python/argus/replica_exchange.py`. Config via `[ReplicaExchange]` section, `sampler = replica_exchange` or `pt`.

**First-run observations (2026-03-27):**

- NUTS warmup took ~1.8 hours (includes JIT of NUTS kernel). Adapted step size: 0.048.
- JIT compilation of the main `lax.scan` loop is very slow (~2+ hours). This is because the entire 5000-iteration loop with nested `vmap(scan(hmc))` inside an outer `scan` is compiled as a single XLA program.

**Known issue — JIT compilation time:**

The current implementation puts the entire sampling loop inside `jax.lax.scan`, creating a massive XLA graph. This contrasts with NumPyro's NUTS which uses a Python outer loop calling a JIT-compiled single-step kernel — fast compile (~seconds), negligible Python overhead per iteration given the ~5-10s/step cost.

**Proposed fix — hybrid Python/JAX loop:**

Refactor to JIT-compile only one iteration (10 HMC steps + swap = small graph, fast compile) and use a Python loop for the outer 5000-iteration sample collection. This is the same pattern NumPyro uses. Expected impact: compilation drops from hours to minutes, with ~10-20% slower sampling from Python loop overhead — a clear net win. The inner `vmap(scan(10 × hmc_step))` stays fully compiled for GPU efficiency.

## Recommended Priority

1. **Parallel tempering** — highest priority, already in development, proven approach for this exact problem in the PTA literature
2. **SVI as diagnostic/initialiser** — low effort via NumPyro, immediately useful even if not a final solution. Can also be used to warm-start NUTS chains near the correct mode
3. **f_gw grid search** — pragmatic fallback guaranteed to work with existing infrastructure, no new code needed. Can produce results immediately while other approaches are developed
4. **flowMC** — promising but highest risk, consider if parallel tempering proves insufficient
5. **jaxns on reduced model** — still viable for evidence computation with fixed noise (~39 dims), pending compilation scaling investigation

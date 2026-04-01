# Laplace Marginalisation of Per-Pulsar Noise Parameters (Archived)

## What this was

Laplace approximation to analytically marginalise the 62 per-pulsar red noise
parameters (gamma_p, sigma_p) from the CW likelihood, reducing the sampled
dimensionality from ~107 to ~43.

For each pulsar, a 2D Newton solve finds the MAP of (log10_gamma_p, log10_ratio)
and a Hessian-based Gaussian integral replaces the explicit sampling of those
parameters.

## Why it was archived (2026-03-27)

The concept is sound — the noise parameters are structurally decoupled from the
CW signal, well-constrained, and approximately Gaussian. However, the
implementation proved impractical for both samplers tested:

- **NUTS**: Each leapfrog step requires backpropagating through the Newton
  iterations (nested autodiff), making per-step cost ~200x more expensive
  (~4.5s vs 0.02s). A 500-sample run would take ~67 hours per chain.

- **Dynesty**: Each likelihood evaluation runs 32 × 8 Newton steps with
  grad + Hessian through the Kalman filter scan. Per-eval cost ~250s vs ~5s
  for standard dynesty. The 107-dim standard run with nlive=500 converges
  faster in wall time despite the higher dimensionality.

The per-likelihood cost increase of ~50-200x dominates over sampling efficiency
gains from halving the dimensions.

## How to revive

The core engine (`laplace_marginalisation.py`) is self-contained. To re-integrate:

1. Copy `laplace_marginalisation.py` back to `python/argus/`
2. The integration points in `bayesian_inference.py`, `cw_kalman_filter.py`,
   `parameter_sampling.py`, `tempered_smc.py`, and `dynesty_sampler.py` used
   a `marginalise_noise = true/false` config flag in `[CWModel]`.
3. The key bottleneck is the nested autodiff. An **implicit differentiation**
   approach (using `jax.custom_jvp` at the Newton solution) would eliminate
   the need to backprop through the solver, potentially making this viable
   for gradient-based samplers.

## Files

- `laplace_marginalisation.py` — Core engine (per-pulsar objective, Newton solver, vmapped Laplace likelihood)
- `test_laplace_marginalisation.py` — 10 unit/integration tests (all passing at time of archive)
- `laplace_smoke_test_config.ini` — NUTS config with `marginalise_noise = true`
- `dynesty_laplace_config.ini` — Dynesty config with `marginalise_noise = true`

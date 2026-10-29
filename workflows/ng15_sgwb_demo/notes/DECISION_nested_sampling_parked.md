# DECISION: park nested sampling; use NUTS for inference (2026-07-10)

**Status: DECIDED.** Slice-based nested sampling (NS) is **parked** as the evidence engine for
Argus at PTA scale. Proceed with **NUTS** for posterior inference. Do **not** re-investigate
slice-NS scaling without a new idea (see "Revisit only if" below).

## Context

T2.6 built a blackjax nested-slice-sampling (`blackjax.nss`) GWB evidence engine
(`run_blackjax_nested_sampling`) to unlock **Bayesian evidence / logZ → HD-vs-CURN Bayes
factors** (task T3.4), because NUTS gives a posterior but **no evidence**. This session (the
"NS cost-scaling study") assessed whether NS is viable for a full PTA analysis. The answer is no.

## What we found

1. **Sampler dimensional scaling is benign.** On a likelihood-free analytic target, NS step
   count grows only ~linearly with dimension (`n_steps ~ D^0.77`). Not the problem.
2. **High-D evidence accuracy is tunable, not fundamental.** The default `num_inner_steps = 2D`
   biases logZ for D≳10 (confirmed by our diagnostic *and* blackjax's own docstring); `~6D`
   inner steps restores accuracy (held to D=120). A tuning cost, not a wall.
3. **A numerical-robustness gap was exposed — and fixed.** NS's global prior exploration drives
   the Kalman innovation covariance near-singular in the tails (triggered by high-precision
   pulsars, e.g. J0437-4715), giving spurious huge log-likelihoods that NS locks onto. Fixed in
   `_log_likelihood` (`jax_kalman_filter.py`) with a magnitude-relative jitter + PD guard; golden
   likelihood preserved, 81 core tests pass. **This fix is kept** (it hardens the likelihood for
   any sampler). See `ns_numerical_hygiene.md`.
4. **Runtime is the killer — and it's fundamental.** Each Kalman likelihood eval is *cheap*
   (~0.03 s at N=2, ~0.07 s at N=32, jitted, GPU; scales ~N²). But NS needs
   `n_steps × num_inner_steps × num_delete` of them, and that product explodes:

   | run | dim | wall (1× A100) |
   |---|---|---|
   | NUTS, 2-D MDC (recorded) | 2 | ~5–13 min |
   | NUTS, full parameters | ~134 | **~2 days** |
   | NS, accurate full-32, red-noise-only | 70 | **~2.5–7 weeks** |

   NS is **~10–30× slower than NUTS**, at a *lower* dimension, and the gap grows with D.

## Root reason

NUTS uses the likelihood **gradient** (free from the JAX Kalman filter) to take long, informed
trajectories → few evaluations. Slice-NS **discards the gradient** and explores by blind
prior-volume shrinkage → many evaluations (`num_inner_steps` must scale with D for accuracy,
compounding `n_steps ∝ D`). Same cheap likelihood; NS just calls it far too many times.
(Note: SwiG / `blackjax.nsswig` goes the *wrong* way — ~`dim`× more evals per step. Confirmed.)

## Decision

- **Posterior inference: NUTS.** Fast, gradient-based, numerically clean (it stays in the typical
  set, so it never hit the covariance pathology above).
- **Evidence / Bayes factors (when needed): NOT slice-NS.** Pursue gradient- or posterior-reuse
  estimators instead — in rough order of cheapness to try:
  1. **Learned harmonic mean** (or similar) on the NUTS posterior we already pay for — logZ at
     near-zero extra cost.
  2. **Thermodynamic integration / stepping-stone** over a temperature ladder (gradient-based).
  3. **Gradient-based nested sampling** (`blackjax.ns.from_mcmc` with an HMC inner kernel).
- **Keep** the NS engine code (works, validated at small scale) and the `_log_likelihood`
  robustness fix, but neither is the path to T3.4 at scale.

## Revisit only if

- A **gradient-based NS inner kernel** is implemented (could change the runtime story), or
- The target problem shrinks enough (few pulsars / low D) that slice-NS is affordable, or
- A materially faster likelihood or NS variant appears.

Otherwise: **do not re-run the slice-NS scaling investigation.** The numbers above are the answer.

## Artifacts (this study)

Scripts under `scripts/`: `ns_scaling_dimension.py`, `gen_scaling_configs.py`,
`ns_likelihood_microbench.py`, `ns_kernel_compare.py`, `ns_pathology_probe.py`,
`ns_scaling_analyze.py`. Data under `outputs/scaling/`. Notes: `ns_numerical_hygiene.md`.
Related memory: `project_ns_scaling_verdict`, `project_ns_numerical_hygiene`.

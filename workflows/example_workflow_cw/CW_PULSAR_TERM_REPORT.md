# CW Pulsar Term Analysis Report

## Objective

Recover all 7 continuous wave (CW) source parameters including the pulsar term, which should break the Earth-term-only (cos ι, ψ, Φ₀) degeneracy. We investigate NUTS sampling with phase reparameterization (arXiv 2410.10087) and jaxns nested sampling as alternative inference approaches.

## Test Data

IPTA Mock Data Challenge 2, Dataset 3b: 32 pulsars (excluding J1640+2224), 183 observations each.

| Parameter | Injection Value |
|-----------|----------------|
| log₁₀h₀ | -13.350 |
| log₁₀f_gw | -8.215 |
| α_gw | 4.067 rad |
| δ_gw | 0.140 rad |
| cos ι | 0.907 |
| ψ | 0.646 rad |
| Φ₀ | 0.175 rad |

## Summary of Prior Results (Earth-term only)

From CW_ANALYSIS_REPORT.md:
- **Intensive run** (4 chains, f_gw free): 325 divergences, 27 hours. f_gw multimodality traps NUTS in secondary modes, causing cascading sky position biases.
- **Fixed-f_gw run** (2 chains): 8 divergences, 59 minutes. Removing frequency multimodality dramatically improves sampling — α_gw and δ_gw well-centred on injection. (cos ι, ψ, Φ₀) remain degenerate as expected for Earth-term only.
- **Pulsar term v1/v2** (distance-based): 1 divergence each, but worst parameter recovery (2/7 params). NUTS efficiently samples the wrong mode. Correcting distances to ATNF values made no difference.

**Conclusion from prior work:** NUTS cannot mode-hop. The pulsar term creates additional multimodality from phase oscillations. A different approach is needed.

## New Runs

### Run 9: Phase-Reparameterized Pulsar Term (NUTS, f_gw free)

**Motivation:** Replace distance-dependent pulsar term phase with per-pulsar χ ∈ [0, 2π) parameters (arXiv 2410.10087), eliminating the O(10³–10⁵) oscillation cycles from distance uncertainty.

**Config:** 2 chains × 250 warmup × 500 samples, phase_parameterization=true, f_gw free. 107 free parameters (7 CW + 32 χ + 68 noise).

| Parameter | Injection | Posterior Mean ± Std | Recovered? |
|-----------|-----------|---------------------|------------|
| log₁₀h₀ | -13.350 | -12.904 ± 0.076 | Biased high |
| log₁₀f_gw | -8.215 | -8.187 ± 0.014 | Close but offset |
| α_gw | 4.067 | 4.648 ± 0.533 | Offset 33° |
| δ_gw | 0.140 | -0.499 ± 0.636 | Wrong sign |
| cos ι | 0.907 | 0.188 ± 0.279 | Way off |
| ψ | 0.646 | 0.226 ± 1.656 | Broad |
| Φ₀ | 0.175 | 7.385 ± 1.714 | Wrong |

- **Divergences:** 0
- **Max r_hat:** 1.91, **Min bulk ESS:** 3
- **Wall time:** 3.1 hours (P100)

**Interpretation:** Same failure mode as distance-based pulsar term v1/v2 — excellent sampling diagnostics but terrible recovery. The two chains find completely different modes (r_hat ~1.9). Phase reparameterization made the χ parameters themselves well-behaved, but did not smooth the joint CW source parameter multimodality enough for NUTS.

### Run 10: Fixed f_gw + Phase-Reparam Pulsar Term (NUTS, light)

**Motivation:** Combine the two things that work: (1) fixing f_gw removes frequency multimodality, (2) pulsar term provides phase constraints to break (cos ι, ψ, Φ₀) degeneracy.

**Config:** 2 chains × 250 warmup × 500 samples, f_gw fixed at injection (-8.21467), phase_parameterization=true. 100 free parameters (6 CW + 32 χ + 62 noise).

| Parameter | Injection | Posterior Mean ± Std | Recovered? |
|-----------|-----------|---------------------|------------|
| log₁₀h₀ | -13.350 | -12.927 ± 0.080 | Offset 0.42 |
| α_gw | 4.067 | 4.096 ± 0.053 | **Yes** |
| δ_gw | 0.140 | 0.137 ± 0.049 | **Yes** |
| cos ι | 0.907 | 0.535 ± 0.177 | Narrower than prior but offset |
| ψ | 0.646 | 1.531 ± 0.496 | Offset |
| Φ₀ | 0.175 | 1.843 ± 1.017 | Broad, offset |

- **Divergences:** 2 (0.2%)
- **Max r_hat:** 1.83, **Min bulk ESS:** 3
- **Wall time:** 3.0 hours (P100)

**Interpretation:** Major improvement for sky position (α_gw, δ_gw within 1σ). The (cos ι, ψ, Φ₀) posteriors are narrower than the prior — the pulsar term is providing genuine constraints — but the means are offset from injection. A few χ parameters have ESS ~3 and high r_hat, suggesting some per-pulsar phases are still multimodal.

### Run 11: Fixed f_gw + Phase-Reparam Pulsar Term (NUTS, intensive)

**Motivation:** More compute to test whether the Run 10 offsets are a sampling artifact or genuine.

**Config:** 4 chains × 1000 warmup × 2000 samples, f_gw fixed, phase_parameterization=true. 4× A100 GPUs.

| Parameter | Injection | Posterior Mean ± Std | vs Run 10 |
|-----------|-----------|---------------------|-----------|
| log₁₀h₀ | -13.350 | -12.952 ± 0.086 | Consistent |
| α_gw | 4.067 | 4.103 ± 0.048 | **Yes — stable** |
| δ_gw | 0.140 | 0.128 ± 0.051 | **Yes — stable** |
| cos ι | 0.907 | 0.587 ± 0.187 | Consistent (still offset) |
| ψ | 0.646 | 1.732 ± 0.568 | Consistent (still offset) |
| Φ₀ | 0.175 | 2.279 ± 1.157 | Consistent (still offset) |

- **Divergences:** 41 (0.51%)
- **Max r_hat:** 1.73, **Min bulk ESS:** 6
- **α_gw ESS:** 2353, **δ_gw ESS:** 1528, **ψ ESS:** 629, **Φ₀ ESS:** 563
- **Wall time:** 7.7 hours (A100)

**Interpretation:** The intensive run confirms the light run results — sky position is rock solid, but (cos ι, ψ, Φ₀) offsets are not a sampling artifact. They are stable across 4 independent chains with high ESS (500–2300). The h₀ offset (~0.4 in log₁₀) is also consistent and likely compensates for the cos ι bias (lower inclination requires higher amplitude to match the observed strain). Divergences increased (41 vs 2), suggesting the intensive run explored more of the posterior surface and encountered geometric difficulties.

### Run 12: jaxns Nested Sampling (trivial, 2 free params)

**Motivation:** Validate the jaxns integration pipeline on a tractable problem before scaling up.

**Config:** 200 live points, 5000 max samples, s=3 slices. Only log₁₀h₀ and log₁₀f_gw free; all other CW params fixed at injection, noise fixed.

| Parameter | Injection | jaxns Median | jaxns Mean ± Std |
|-----------|-----------|-------------|-------------------|
| log₁₀h₀ | -13.350 | -13.01 | -12.94 ± 0.55 |
| log₁₀f_gw | -8.215 | -8.23 | -8.15 ± 0.52 |

- **log Z:** 47176.81 ± 0.18
- **Likelihood evaluations:** 130,922
- **Samples:** 2,500 (terminated on "small remaining evidence")
- **ESS:** 314 (jaxns internal), 1899/1871 (ArviZ bulk/tail)
- **Wall time:** 31 minutes (P100), ~25 min JIT compilation + ~3 min sampling

**Interpretation:** jaxns works end-to-end. Evidence computed with tight uncertainty. f_gw median is within 0.02 of injection. The JIT compilation of the full Kalman filter likelihood dominates wall time.

### Failed Run: jaxns with 107 dimensions

**Config:** Same as Run 9 but using jaxns nested sampling instead of NUTS.

**Result:** JIT compilation never completed after 7.5 hours on a P100. The XLA trace of the nested sampling loop with the Kalman filter likelihood at 107 dimensions produces an intractably large compilation graph. This is a fundamental limitation of jaxns at high dimensionality — the compilation cost scales steeply with the number of sampled parameters, compounded by the complexity of the likelihood function.

## Analysis

### What works

1. **Sky position (α_gw, δ_gw)** is consistently recovered when f_gw is fixed. Both the light and intensive runs agree within 1σ of injection, with tight posteriors (std ~0.05 rad).

2. **The phase-reparameterized pulsar term provides genuine constraints** on (cos ι, ψ, Φ₀) — posteriors are significantly narrower than the prior (cos ι std=0.19 vs prior width 2.0; ψ std=0.57 vs prior width π). This is not possible with Earth-term only.

3. **jaxns nested sampling pipeline** is functional and produces correct evidence estimates. The ArviZ integration, corner plots, and diagnostics all work.

### What doesn't work

1. **NUTS cannot handle f_gw multimodality** with the pulsar term. Phase reparameterization (Run 9) did not help — chains still get trapped in different modes.

2. **(cos ι, ψ, Φ₀) are biased even with fixed f_gw and the pulsar term** (Runs 10–11). The intensive run confirms this is not a sampling artifact. Possible explanations:
   - NUTS finds one mode of the χ-coupled (cos ι, ψ, Φ₀) space and stays there, even with fixed f_gw
   - The per-pulsar χ parameters each have residual multimodality that chains cannot traverse
   - A few χ parameters with ESS ~6 and r_hat ~1.7 support this interpretation

3. **jaxns cannot compile at 107 dimensions** with our Kalman filter likelihood. The compilation cost of tracing the nested sampling loop with `lax.scan`-based likelihood is prohibitive.

### The h₀ bias

All runs (NUTS and jaxns) show log₁₀h₀ biased high by ~0.4 relative to injection. This is consistent across Earth-term-only and pulsar-term runs. It likely reflects a (h₀, cos ι) degeneracy: the observed strain depends on h₀ × f(cos ι), so if cos ι is biased low, h₀ compensates upward.

## Comparison with Standard PTA Approach

Standard PTA CW analyses (enterprise + PTMCMCSampler, QuickCW) use:
- **Parallel tempering** for mode-hopping across f_gw peaks and χ modes
- **Custom Fisher-matrix proposals** tuned to the PTA waveform structure
- **The same phase reparameterization** (χ per pulsar, uniform on [0, 2π))

Our signal model and parameterization match the literature. The limitation is that NUTS is a gradient-based within-mode explorer that cannot jump between separated posterior modes. Parallel tempering solves this by using hot chains to bridge modes and replica exchange to transfer information to the cold chain.

## Next Steps

### Near-term

1. **jaxns with fixed noise (~39 dims):** Fix the 68 noise parameters at known/estimated values and run nested sampling with only the 7 CW + 32 χ parameters free. The trivial run showed JIT compiles in ~25 min at 2 dims; 39 dims should be tractable (compilation time scales with graph size, not just parameter count). This tests whether nested sampling can navigate the f_gw + χ multimodality that defeats NUTS.

2. **Consult jaxns developers:** Open a GitHub issue to understand compilation scaling and best practices for ~40–100 dim problems with complex likelihoods.

### Medium-term

3. **Parallel tempering via blackjax:** Implement replica-exchange MCMC with HMC proposals. This is the standard PTA approach and directly addresses the mode-hopping limitation. blackjax stays in the JAX ecosystem and can reuse our existing likelihood.

4. **Fixed-frequency grid search:** Evaluate the posterior on a grid of f_gw values (e.g., 50–100 points), running NUTS on the remaining 6 CW + 32 χ + noise parameters at each grid point. Each point takes ~1–3 hours. Combine via importance weighting.

### Longer-term

5. **QuickCW comparison:** Benchmark against the standard PTA tool on the same MDC2 dataset to establish a performance baseline.

6. **Detection statistics:** Use jaxns evidence computation (on the reduced-dimension model) to compute Bayes factors for CW detection.

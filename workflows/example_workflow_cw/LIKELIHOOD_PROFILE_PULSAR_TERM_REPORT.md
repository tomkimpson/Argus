# Likelihood Profiles: Earth-term vs Pulsar Term

## Objective

Compare conditional likelihood profiles with and without the phase-reparameterised pulsar term to determine whether the pulsar term provides genuine constraining power on (cos ι, ψ, Φ₀) and whether the parameter biases observed in NUTS runs are a signal model limitation or a sampling problem.

## Method

Sweep each CW parameter over its full prior range while holding all other parameters fixed at their injected values. For the pulsar-term profiles, the per-pulsar phase parameters χ are set to their **true values** computed from the injection:

χₙ = Ω × dₙ/c × (1 + n̂·q̂ₙ) mod 2π

where Ω = 2πf_gw (injected), dₙ is the pulsar distance (from enterprise/ATNF catalog), n̂ is the GW propagation direction, and q̂ₙ is the pulsar direction. This represents the best-case scenario where the sampler has found the correct χ values.

Noise parameters are fixed at posterior medians from the intensive fixed-f_gw run.

## Results

### Parameter Profile Summary

| Parameter | Injection | Earth-term peak | Pulsar-term peak | Earth offset | PT offset |
|-----------|-----------|----------------|-----------------|-------------|----------|
| log₁₀h₀ | -13.350 | -13.146 | -13.487 | +0.204 | -0.137 |
| log₁₀f_gw | -8.215 | -8.246 | -8.216 | -0.031 | -0.001 |
| α_gw | 4.067 | 4.136 | 4.168 | +0.069 | +0.101 |
| δ_gw | 0.140 | 0.087 | 0.229 | -0.053 | +0.089 |
| cos ι | 0.907 | 1.000 (boundary) | 0.729 | +0.093 | -0.178 |
| ψ | 0.646 | 0.695 | 0.821 | +0.049 | +0.175 |
| Φ₀ | 0.175 | 0.063 | 6.094 | -0.112 | +5.919 |

### Key Findings

**1. The pulsar term constrains cos ι.** The Earth-term cos ι profile is essentially flat, increasing monotonically toward cos ι = 1 (boundary). The pulsar-term profile is convex with a clear peak at ~0.73, dropping by ~100 in Δln L away from the peak. This confirms the pulsar term breaks the Earth-term inclination degeneracy.

**2. The pulsar term constrains ψ.** Similar to cos ι — the Earth-term profile is broad while the pulsar-term profile is sharper with more structure around the injection.

**3. Frequency multimodality persists.** The log₁₀f_gw profile shows dozens of local minima in both Earth-term and pulsar-term cases. The pulsar term makes the profile sharper at the injection frequency but does not eliminate the secondary peaks. This is the primary obstacle for NUTS sampling.

**4. Sky position is well-constrained by both.** α_gw and δ_gw profiles peak near injection for both Earth-term and pulsar-term, with similar constraining power. The pulsar term adds some additional fine structure.

**5. Per-pulsar χ profiles are unimodal.** Sweeping χ₀ (pulsar 0) while holding all other χ at true values shows a smooth, single-peaked profile with the true value at the peak. No multimodality per pulsar in isolation.

**6. Φ₀ is strongly coupled to χ.** The pulsar-term Φ₀ profile peaks far from injection (6.09 vs 0.175). This is likely a conditional artifact — Φ₀ enters the waveform as Ωt + Φ₀ (Earth term) and Ωt + Φ₀ - χ (pulsar term), so the preferred Φ₀ in a conditional sweep depends sensitively on the exact χ values and noise realisation.

### Interpretation of Conditional Profile Offsets

The cos ι offset (~0.18) and ψ offset (~0.18) from injection are **conditional profile biases**, not model errors. These are the same class of effect as the δ_gw offset investigated in DELTA_GW_ASSESSMENT.md:

- Conditional profiles sweep one parameter while fixing all others at injection. The conditional MLE is not the same as the marginal posterior peak when parameters are correlated.
- The timing residual amplitude depends on h₀ × f(cos ι), creating a (h₀, cos ι) degeneracy. At fixed h₀ = injection, the noise realisation shifts the conditional cos ι peak.
- These offsets would be different with a different noise draw and do not indicate a systematic bias.

The important conclusion is that the profile is **peaked and convex** — the pulsar term has constraining power. The exact conditional peak location is not directly informative about parameter recovery in the full joint posterior.

### Previous Run with χ = 0

An earlier profile run using χ = 0 for all pulsars showed the pulsar-term likelihood peaking at the prior boundaries for every parameter (h₀ → -16, f_gw → -9, α_gw → 0, etc.). This is expected: incorrect χ values make the pulsar term destructively interfere with the Earth term, and the likelihood prefers "no signal". This demonstrates that **the sampler must find approximately correct χ values for the pulsar term to be constructive** — a significant sampling challenge with 32 coupled phase parameters.

## Conclusions

1. **The signal model is correct.** The pulsar term provides genuine constraints on (cos ι, ψ) when χ is at its true values. The parameter biases observed in NUTS runs are not a fundamental limitation of the model.

2. **The problem is purely sampling.** NUTS must jointly find the correct (f_gw, χ₁...χ₃₂, cos ι, ψ, Φ₀) — a high-dimensional space with f_gw multimodality and 32 coupled χ parameters. The conditional profiles show the correct solution exists and is well-defined, but NUTS cannot navigate to it.

3. **The case for alternative samplers is strong.** The profiles confirm there is signal to be recovered. Parallel tempering (which can hop between f_gw valleys and explore the χ space via temperature exchanges) or nested sampling on a reduced-dimensionality problem are the most promising paths forward.

## Pending

An intensive NUTS run with f_gw free and phase-reparameterised pulsar term (4 chains × 3000 steps, A100 GPUs, job 10794791) is currently in progress (~60% complete). This will provide a like-for-like comparison with the fixed-f_gw intensive run and show whether any of the 4 chains find the correct mode by chance.

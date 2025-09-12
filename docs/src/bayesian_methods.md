# Advanced Bayesian Parameter Estimation in Argus

This document provides pedagogical explanations of the sophisticated Bayesian parameter estimation techniques implemented in Argus for pulsar timing array (PTA) analysis. These methods address key computational and statistical challenges that arise when performing high-dimensional Bayesian inference on PTA datasets.

!!! info "Target Audience"
    This guide is intended for researchers and students who want to understand the **why** behind Argus's advanced parameter estimation methods, not just the how. Basic familiarity with Bayesian inference and MCMC sampling is assumed.

---

## The Challenge: High-Dimensional Parameter Spaces

Pulsar timing array analysis involves estimating dozens to hundreds of parameters simultaneously:

- **Gravitational wave parameters**: Amplitude ($h_a$) and spectral index ($\gamma_a$)
- **Per-pulsar noise parameters**: Red noise amplitude ($\sigma_p$) and slope ($\gamma_p$) for each of $N$ pulsars  
- **Measurement noise parameters**: Error scaling factors (EFAC) and additional white noise (EQUAD) for each pulsar

This creates parameter spaces with dimensions ranging from ~10 parameters for small arrays to >500 parameters for next-generation PTAs. Standard MCMC sampling can struggle in these high-dimensional spaces due to:

1. **Poor mixing**: Parameters may be strongly correlated, leading to inefficient exploration
2. **Divergent transitions**: NUTS sampler may encounter numerical difficulties with poorly scaled parameters
3. **Computational cost**: Gradient calculations become expensive as dimensionality increases

Argus addresses these challenges through three sophisticated reparameterization techniques.

---

## Technique 1: h_a Parameter Reparameterization

### The Problem

The gravitational wave amplitude parameter $h_a$ is typically assigned a uniform prior $h_a \sim \text{Uniform}(10^{a}, 10^{b})$, which in log space becomes $\log_{10} h_a \sim \text{Uniform}(a, b)$. However, uniform priors can be problematic for NUTS sampling because:

- The sampler must adapt its step size to the prior bounds
- Gradients near boundaries can be poorly conditioned
- The geometry of uniform distributions is not well-suited to Hamiltonian dynamics

### The Solution: Standardized Normal Reparameterization

Instead of sampling $\log_{10} h_a$ directly from its uniform prior, Argus samples an auxiliary variable and applies a transformation:

1. **Sample**: $\log_{10} h_a' \sim \mathcal{N}(0, 1)$ (standard normal)
2. **Transform**: $\log_{10} h_a = \mu + \sigma \cdot \log_{10} h_a'$

where:
- $\mu = \frac{a + b}{2}$ (midpoint of uniform range)
- $\sigma = \frac{b - a}{6}$ (using 3-sigma rule: 99.7% of samples within bounds)

### Mathematical Intuition

This transformation leverages the fact that NUTS sampling works most efficiently with standardized parameters. The **3-sigma rule** ensures that the vast majority of samples from $\mathcal{N}(0,1)$ map to values within the original uniform bounds, while still allowing occasional exploration outside these bounds (which can help with mode discovery).

### Computational Benefits

1. **Better gradient conditioning**: Normal distributions have well-behaved gradients everywhere
2. **Automatic step size tuning**: NUTS can efficiently adapt to $\mathcal{N}(0,1)$ geometry
3. **Reduced divergences**: Avoids boundary effects that cause sampler failures

### When to Use

Enable h_a reparameterization when:
- Using uniform priors for $\log_{10} h_a$ 
- Experiencing slow convergence or divergent transitions
- Working with wide prior ranges (>2 orders of magnitude)

**Configuration**:
```ini
[PriorModel]
log10_ha_fixed = false
log10_ha_min = -18.0
log10_ha_max = -12.0
# Reparameterization is automatic when using uniform priors
```

---

## Technique 2: Hierarchical Modeling for Pulsar Noise Parameters

### The Problem

Each pulsar in a PTA has individual red noise parameters $\gamma_p^{(i)}$ and $\sigma_p^{(i)}$. The naive approach assigns independent priors to each:

$$\gamma_p^{(i)} \sim \text{Uniform}(\gamma_{\text{min}}, \gamma_{\text{max}}) \quad \text{for } i = 1, \ldots, N$$

This leads to several problems:
1. **Curse of dimensionality**: Parameter space grows linearly with number of pulsars
2. **Information waste**: No sharing of knowledge between similar pulsars
3. **Prior sensitivity**: Individual pulsar constraints depend heavily on prior choices

### The Solution: Population-Level Hierarchical Priors

Hierarchical modeling assumes that individual pulsar parameters are drawn from a **population distribution** characterized by hyperparameters:

#### Level 1: Population Hyperparameters
$$\begin{align}
\log_{10} \gamma_{\text{pop}} &\sim \text{Uniform}(\gamma_{\text{mean,min}}, \gamma_{\text{mean,max}}) \\
\log_{10} \sigma_{\gamma,\text{pop}} &\sim \text{Uniform}(\sigma_{\text{std,min}}, \sigma_{\text{std,max}})
\end{align}$$

#### Level 2: Individual Pulsar Parameters
$$\log_{10} \gamma_p^{(i)} \sim \mathcal{N}(\log_{10} \gamma_{\text{pop}}, \sigma_{\gamma,\text{pop}})$$

### Gradient Balancing for High-Dimensional Spaces

To prevent gradient pathologies in high-dimensional spaces, individual parameters are rescaled:

$$\log_{10} \gamma_p^{(i)} = \log_{10} \gamma_{\text{pop}} + \frac{\gamma_{\text{raw}}^{(i)} \cdot \sigma_{\gamma,\text{pop}}}{\sqrt{N}}$$

where $\gamma_{\text{raw}}^{(i)} \sim \mathcal{N}(0, 1)$ and the $1/\sqrt{N}$ factor balances gradients as the number of pulsars increases.

### Statistical Benefits

1. **Information sharing**: Well-constrained pulsars inform poorly-constrained ones
2. **Automatic regularization**: Population prior prevents extreme individual values
3. **Reduced effective dimensionality**: $(2N)$ individual parameters → $(2 + N)$ hierarchical parameters
4. **Physical interpretability**: Population parameters have astrophysical meaning

### Computational Benefits

1. **Better mixing**: Hierarchical structure breaks parameter correlations
2. **Gradient balancing**: $1/\sqrt{N}$ scaling prevents gradient explosion
3. **Fewer divergences**: Smoother posterior geometry

### When to Use

Enable hierarchical modeling when:
- Analyzing >5 pulsars simultaneously
- Individual pulsar data quality varies significantly  
- Interested in population-level astrophysical inference
- Standard sampling shows poor mixing for noise parameters

**Configuration**:
```ini
[PriorModel]
hierarchical_noise = true
log10_gamma_p_mean_min = -10.0
log10_gamma_p_mean_max = -6.0
log10_gamma_p_std_min = 0.1
log10_gamma_p_std_max = 2.0
```

---

## Technique 3: Log-Ratio Parameterization

### The Problem

Red noise parameters $\gamma_p$ and $\sigma_p$ are often strongly correlated because they both characterize the same underlying stochastic process. Standard parameterizations can lead to:

1. **Parameter correlations**: $\gamma_p$ and $\sigma_p$ exhibit strong covariance
2. **Inefficient sampling**: NUTS must navigate curved correlation structure  
3. **Interpretation difficulties**: Physical meaning obscured by correlations

### The Solution: Decorrelating Transformation

Instead of sampling $\gamma_p$ and $\sigma_p$ independently, the log-ratio parameterization samples:

1. $\log_{10} \gamma_p$ (as usual)
2. $\log_{10} \text{ratio}$ where $\text{ratio} = \sigma_p / \gamma_p$

Then $\sigma_p$ is computed deterministically:
$$\log_{10} \sigma_p = \log_{10} \gamma_p + \log_{10} \text{ratio}$$

### Physical Motivation

The ratio $\sigma_p / \gamma_p$ has a cleaner physical interpretation than $\sigma_p$ alone - it represents the **strength** of timing noise relative to its correlation timescale. This ratio is often more stable across pulsars than the individual parameters.

### Hierarchical Log-Ratio Implementation

Combined with hierarchical modeling:

$$\begin{align}
\log_{10} \text{ratio}_{\text{pop}} &\sim \text{Uniform}(\text{ratio}_{\text{mean,min}}, \text{ratio}_{\text{mean,max}}) \\
\sigma_{\text{ratio,pop}} &\sim \text{Uniform}(\sigma_{\text{ratio,min}}, \sigma_{\text{ratio,max}}) \\
\log_{10} \text{ratio}^{(i)} &\sim \mathcal{N}(\log_{10} \text{ratio}_{\text{pop}}, \sigma_{\text{ratio,pop}})
\end{align}$$

### Benefits

1. **Reduced correlations**: $\gamma_p$ and ratio are typically less correlated than $\gamma_p$ and $\sigma_p$
2. **Physical interpretability**: Ratio parameter has clearer astrophysical meaning
3. **Improved mixing**: Decorrelated parameters are easier for NUTS to sample
4. **Computational efficiency**: Faster convergence due to better geometry

### When to Use

Enable log-ratio parameterization when:
- Red noise parameters show strong correlations in standard analysis
- Interested in population-level ratio statistics
- Standard parameterization shows poor NUTS performance
- Working with hierarchical noise models

**Configuration**:
```ini
[PriorModel]
log_ratio_parameterization = true
log10_ratio_mean_min = -2.0
log10_ratio_mean_max = 2.0  
log10_ratio_std_min = 0.1
log10_ratio_std_max = 1.0
```

---

## Practical Guidelines

### Choosing the Right Combination

| Scenario | h_a Reparam | Hierarchical | Log-Ratio |
|----------|-------------|--------------|-----------|
| Small PTA (≤5 pulsars) | ✓ | ✗ | ✗ |
| Medium PTA (6-20 pulsars) | ✓ | ✓ | Optional |
| Large PTA (>20 pulsars) | ✓ | ✓ | ✓ |
| Parameter correlations observed | ✓ | ✓ | ✓ |
| Poor NUTS performance | ✓ | ✓ | ✓ |

### Monitoring Effectiveness

Signs that advanced parameterizations are helping:
- **Increased effective sample size (ESS)** for difficult parameters
- **Reduced number of divergent transitions**  
- **Better R̂ convergence diagnostics** across chains
- **Faster warmup adaptation**

Signs of problems:
- ESS decreases compared to standard parameterization
- New divergences appear
- Posterior distributions become unreasonably wide

### Computational Considerations

- **Memory**: Hierarchical models require storing population parameters
- **Runtime**: Gradient calculations become more complex with transformations
- **Convergence**: May need longer warmup to adapt to new geometry
- **Interpretation**: Transform samples back to original parameterization for plotting

---

## Advanced Topics

### Gradient Balancing Theory

The $1/\sqrt{N}$ scaling in hierarchical models prevents a phenomenon called **gradient explosion** where:

$$\frac{\partial \log p(\theta)}{\partial \theta_{\text{pop}}} \propto N \quad \text{(problematic)}$$

The rescaling ensures gradients remain $O(1)$ regardless of $N$:

$$\frac{\partial \log p(\theta)}{\partial \theta_{\text{pop}}} \propto \sqrt{N} \quad \text{(well-conditioned)}$$

### Custom Transformations

Advanced users can implement custom parameter transformations by:
1. Subclassing the base parameter model
2. Implementing forward and inverse transformations  
3. Ensuring proper Jacobian corrections for probability densities

### Integration with Other Tools

These techniques are compatible with:
- **ArviZ**: All transformations preserve MCMC diagnostics
- **Corner plots**: Automatic back-transformation for visualization
- **Model comparison**: Bayes factors computed correctly with proper Jacobians

---

## Further Reading

- [Kalman Mathematics](kalman_mathematics.md): Mathematical foundations of the state-space approach
- [Mathematical Background](mathematical_background.md): Detailed derivations and proofs
- [Examples](examples/index.md): Practical applications of these techniques
- **Betancourt (2017)**: "A Conceptual Introduction to Hamiltonian Monte Carlo" - theoretical background
- **Gelman et al. (2013)**: "Bayesian Data Analysis" - hierarchical modeling principles

---

## Summary

Argus's advanced Bayesian techniques address the fundamental challenges of high-dimensional parameter estimation in PTA analysis:

1. **h_a reparameterization** improves NUTS sampling through standardized normal geometry
2. **Hierarchical modeling** enables information sharing and reduces effective dimensionality  
3. **Log-ratio parameterization** decorrelates strongly correlated noise parameters

Together, these methods enable robust, efficient Bayesian inference even for next-generation PTAs with hundreds of parameters. The key insight is that **how** you parameterize the problem is often as important as **what** model you choose.
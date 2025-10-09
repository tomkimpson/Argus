# Bayesian Inference in Argus

A general overview of Bayesian inference in `Argus` was provided on the [Bayesian Inference](bayesian_inference.md) page. This page goes into some additional detail on some of the techniques implemented in Argus for Bayesian inference with PTAs. 


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

Argus addresses these challenges through three reparameterization techniques.

---

## Technique 1: $h_a$ Parameter Reparameterization

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


See also [Pulsar Timing Arrays require hierarchical models
](https://arxiv.org/html/2406.05081v2).

### Fixed Parameter Override

When spin injection files are provided, Argus can fix pulsar red noise parameters to specific values instead of sampling them. This is typically used for:

- Testing with known injected signals
- Validation studies with predetermined parameter values
- Development and debugging scenarios

The hierarchical modeling is automatically disabled for parameters that are explicitly fixed via injection files.

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


!!! note "Parameter Counting"
    The effective number of parameters with hierarchical modeling and log-ratio parameterization is: **4 hyperparameters + 2×N individual parameters** (where N is the number of pulsars), compared to **2×N parameters** with independent priors.

---

## Further Reading

- [Kalman Mathematics](kalman_mathematics.md): Mathematical foundations of the state-space approach
- **Betancourt (2017)**: "A Conceptual Introduction to Hamiltonian Monte Carlo" - theoretical background
- **Gelman et al. (2013)**: "Bayesian Data Analysis" - hierarchical modeling principles


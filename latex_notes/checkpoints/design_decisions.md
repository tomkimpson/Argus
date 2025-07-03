# Design Decisions and Technical Implementation Notes

## Overview

This document outlines key design decisions and technical implementations in the Argus Bayesian inference pipeline for pulsar timing array gravitational wave detection. These decisions were made to optimize NUTS sampling performance for high-dimensional hierarchical models.

## Parameter Reparameterization

### The Problem with Direct Uniform Priors

**Original approach**: Sample parameters directly from uniform distributions like `log10_ha ~ Uniform(-16.0, -14.0)`

**Why NUTS struggles with uniform priors**:
- Uniform distributions have **flat, discontinuous gradients** (zero inside, undefined at boundaries)
- NUTS relies on **gradient information** to navigate parameter space efficiently
- At the boundaries, gradients become ill-defined, causing sampling issues
- The sampler can get "stuck" at boundaries or have poor mixing
- Can lead to divergent transitions and poor R-hat values

### The Reparameterization Solution

**Improved approach**: 
1. Sample `log10_ha_prime ~ Normal(0, 1)` 
2. Transform: `log10_ha = mean + log10_ha_prime × std`

**Why this works better**:
- **Smooth gradients**: Normal distributions have well-defined gradients everywhere
- **No boundaries**: Normal distributions have infinite support (no hard edges)
- **Standard space**: NUTS is optimized for parameters that look like N(0,1)
- **Better geometry**: The parameter space has better curvature properties

### Implementation Details

The transformation uses the **3-sigma rule** to map uniform bounds to normal parameters:

```python
# Config specifies: log10_ha ∈ [min_val, max_val]
mean = (min_val + max_val) / 2.0
std = (max_val - min_val) / 6.0  # 3-sigma rule: 99.7% coverage

# Sample and transform
log10_ha_prime ~ N(0, 1)
log10_ha = mean + log10_ha_prime × std
```

**Example transformations**:
- Run 016: `[-16.0, -14.0]` → `log10_ha = -15.0 + log10_ha_prime × 0.333`
- Run 020: `[-18.0, -14.0]` → `log10_ha = -16.0 + log10_ha_prime × 0.667`

### Expected Sample Behavior

The 3-sigma rule means ~99.7% of samples fall within the specified bounds, but **tail samples beyond these bounds are normal and expected**. With 1000 samples, finding 2-3 samples at values like -16.2 or -16.5 (when bounds are [-16.0, -14.0]) is statistically correct.

**This is not a bug** - it preserves proper statistical properties while giving NUTS the smooth parameter space it needs.

## Hierarchical Modeling

### Motivation

Pulsar timing arrays involve analyzing dozens of pulsars, each with individual noise parameters (`log10_γp`, `log10_σp`). Traditional approaches sample each parameter independently, leading to:

- **High dimensionality**: 32 pulsars × 2 parameters = 64+ dimensions
- **Poor scaling**: NUTS performance degrades exponentially beyond ~35-50 parameters
- **Parameter correlations**: Individual pulsar parameters are often correlated
- **Overfitting**: Independent priors don't capture population-level structure

### Hierarchical Solution

**Population-level modeling**: Instead of independent priors, model parameters hierarchically:

```python
# Traditional (problematic)
for i in range(n_pulsars):
    log10_γp[i] ~ Uniform(-11.0, -6.0)  # Independent

# Hierarchical (improved)
population_mean ~ Uniform(-9.0, -7.0)      # Population hyperparameters
population_std ~ Uniform(0.1, 1.0)
for i in range(n_pulsars):
    log10_γp[i] ~ Normal(population_mean, population_std)  # Shared structure
```

### Benefits

1. **Dimensionality reduction**: 32 independent parameters → 2 hyperparameters + 32 constrained parameters
2. **Information sharing**: Data from all pulsars informs the population model
3. **Better regularization**: Extreme outlier values are naturally penalized
4. **Improved convergence**: NUTS can focus on learning the population structure
5. **Scientific insight**: Population parameters have astrophysical meaning

### Implementation in Argus

**Successful hierarchical modeling** (Runs 013, 015):
- `log10_γp`: Hierarchical population model
- `log10_σp`: Fixed values (not sampled)
- **Result**: ~35 parameters, excellent convergence

**Failed full hierarchical** (Run 014):
- `log10_γp`: Hierarchical 
- `log10_σp`: Independent hierarchical 
- **Result**: ~69 parameters, exponential performance degradation

**Current compromise** (Runs 016, 017, 020):
- `log10_γp`: Hierarchical
- `log10_σp`: Log-ratio parameterization (`log10_σp = log10_γp + log10_ratio`)
- **Goal**: Reduce correlations while keeping dimensionality manageable

## Log-Ratio Parameterization for σp

### The σp Problem

Each pulsar has intrinsic timing noise characterized by both `γp` (red noise amplitude) and `σp` (white noise amplitude). These parameters are often correlated in real pulsars, but modeling them independently creates challenging geometry for NUTS.

### Log-Ratio Solution

Instead of sampling `log10_σp` directly, parameterize as:

```python
log10_γp[i] ~ Hierarchical(...)           # As before
log10_ratio[i] ~ Hierarchical(...)        # New: ratio population model  
log10_σp[i] = log10_γp[i] + log10_ratio[i]  # Deterministic relationship
```

**Benefits**:
- **Reduced correlations**: The ratio captures the relationship between γp and σp
- **Physical interpretation**: `ratio = σp/γp` has astrophysical meaning
- **Dimension efficiency**: Sample ratios instead of absolute values
- **Better geometry**: Linear relationships are easier for NUTS to navigate

### Current Status

This approach is being tested in runs 016, 017, and 020. Early results show promise but require validation against simpler models.

## NUTS Configuration Optimizations

### Conservative Settings for High-Dimensional Spaces

Based on extensive testing, the following NUTS settings work best for hierarchical models:

```ini
target_accept_prob = 0.85     # Conservative (default 0.8)
max_tree_depth = 10           # Reduced from default 12
dense_mass = true            # Essential for correlated parameters
num_chains = 2               # Adequate for convergence checking
```

**Rationale**:
- **High acceptance probability**: Reduces divergences in complex parameter spaces
- **Limited tree depth**: Prevents runaway trajectories that can occur in 60+ dimensions
- **Dense mass matrix**: Captures parameter correlations automatically
- **Multiple chains**: Essential for R-hat convergence diagnostics

### Parameter Count Guidelines

From empirical testing:
- **≤ 35 parameters**: NUTS works reliably with proper tuning
- **35-50 parameters**: Potentially workable with advanced techniques
- **≥ 70 parameters**: Requires alternative approaches or fundamental reparameterization

## Gradient Balancing

### The Problem

In high-dimensional models, different parameter types can have vastly different gradient magnitudes:
- GW amplitude gradients: O(10²)
- Pulsar noise gradients: O(10⁴)
- Hierarchical hyperparameter gradients: O(10¹)

This creates **poorly conditioned geometry** where NUTS struggles to balance exploration across parameter types.

### Solution: Standardized Parameterization

All parameters are transformed to have similar gradient magnitudes:

```python
# Scale hierarchical parameters by number of pulsars
log10_γp_raw[i] ~ Normal(0, 1/√n_pulsars)

# Use consistent 3-sigma transformations
# All parameters end up with comparable gradient magnitudes
```

This ensures NUTS spends roughly equal effort exploring each parameter dimension.

## Scientific Implications

### Trade-offs in Model Complexity

1. **Run 015 (35 parameters)**:
   - **Pros**: Stable, convergent, practical runtime
   - **Cons**: σp parameters fixed, not learned from data
   - **Scientific impact**: May bias GW detection by assuming incorrect noise levels

2. **Run 020 (69 parameters)**:
   - **Pros**: All noise parameters inferred from data
   - **Cons**: Potential sampling difficulties, longer runtimes
   - **Scientific impact**: Unbiased GW detection if sampling succeeds

### The Parameter Count Dilemma

There's a fundamental tension between:
- **Statistical rigor**: Inferring all parameters from data
- **Computational tractability**: NUTS limitations in high dimensions
- **Scientific validity**: Ensuring unbiased GW detection

The log-ratio parameterization represents an attempt to thread this needle by reducing parameter correlations while maintaining full parameter inference.

## Future Directions

### If Current Approaches Fail

1. **Principal component analysis**: Reduce (γp, σp) to dominant population modes
2. **Physical constraint models**: Use theoretical relationships between noise parameters
3. **Block sampling strategies**: Coordinate updates of related parameters  
4. **Alternative inference methods**: Variational inference, specialized MCMC variants

### Long-term Research

1. **Time-varying parameters**: Allow noise properties to evolve over observation periods
2. **Pulsar grouping**: Share parameters across astrophysically similar pulsars
3. **Multi-scale modeling**: Hierarchical models at multiple levels (individual, group, population)

---

## Corner Plot Visualization and Parameter Ranges

### The Reparameterization Plotting Challenge

When using Normal(0,1) reparameterization, **samples naturally extend beyond the config "prior bounds"** due to the 3-sigma rule. This creates a visualization dilemma when creating corner plots.

### Sample Range vs Prior Range Trade-offs

**Using sample-based ranges** (`smooth_sigma=None`):
- **Pros**: Shows true posterior width and uncertainty
- **Cons**: May truncate valid tail samples from reparameterization
- **Behavior**: Axes fit tightly to actual sample ranges

**Using prior-based ranges** (`smooth_sigma>0`):
- **Pros**: Shows full parameter support including reparameterization tails
- **Cons**: Can make posteriors look artificially narrow and "spike-like"
- **Behavior**: Axes span config bounds (or extended ranges)

### The "Pointy Posterior" Problem

Setting plot ranges too wide creates misleading visualizations:
- Posterior appears overly constrained relative to the prior
- True parameter uncertainty is visually minimized
- Can give false impression of strong constraints

### Current Implementation

The corner plot script provides two modes:

1. **No smoothing**: Uses data-driven ranges showing realistic posterior width
2. **With smoothing**: Uses extended prior ranges that may make posteriors appear artificially narrow

**Extended ranges** (used when smoothing):
- `log10_ha`: [-18.5, -13.5] (extended from config [-18.0, -14.0])
- `log10_sigma_p`: [-20.5, -11.5] (extended from config [-20.0, -12.0]) 
- `log10_gamma_p`: [-11.5, -5.5] (extended from config [-11.0, -6.0])

### Interpretation Guidelines

When viewing smoothed corner plots:
1. **Remember the scale**: Wide plot ranges can make posteriors look artificially constrained
2. **Compare modes**: Check both smoothed and unsmoothed versions
3. **Focus on shape**: The posterior shape matters more than its apparent width relative to plot boundaries
4. **Tail behavior**: Samples beyond config bounds are expected and statistically valid

**Key insight**: The "true" posterior width is better represented by the unsmoothed plots, while smoothed plots are useful for visualizing posterior shape and correlations.

---

**Last updated**: 2025-07-03  
**Context**: Runs 021-022 with enhanced sampling; corner plot smoothing improvements implemented
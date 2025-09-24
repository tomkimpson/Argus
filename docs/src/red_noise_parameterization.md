# Red Noise Parameterization in Argus

This document explains the parameterization approaches used for pulsar red noise parameters in Argus Bayesian inference models.

## Overview

Pulsar red noise is characterized by two key parameters:
- **γp**: The spectral index (power-law slope) of the red noise power spectrum
- **σp**: The amplitude (root-mean-square) of the red noise

Argus uses a sophisticated parameterization scheme that combines hierarchical modeling for γp with log-ratio parameterization for σp to improve sampling efficiency and capture physically motivated parameter correlations.

## Hierarchical Modeling for γp (Spectral Index)

Argus always applies hierarchical modeling to the spectral index parameter log₁₀(γp) to improve sampling efficiency.

### Mathematical Formulation

Instead of sampling individual γp values independently for each pulsar, the hierarchical approach introduces population-level hyperparameters:

```
log₁₀(γp,i) ~ Normal(μ_γ, σ_γ)
```

where:
- `μ_γ` ~ Uniform(γp_mean_min, γp_mean_max) is the population mean
- `σ_γ` ~ Uniform(γp_std_min, γp_std_max) is the population standard deviation
- `i` indexes individual pulsars

### Benefits

1. **Information sharing**: Pulsars with similar characteristics inform each other's parameter estimates
2. **Population inference**: Enables study of the distribution of spectral indices across the pulsar population
3. **Improved convergence**: Reduces parameter correlations and improves sampling efficiency
4. **Physical realism**: Reflects the expectation that pulsars may share common red noise characteristics

### Configuration Parameters

```ini
# Hyperprior ranges for hierarchical modeling of gamma_p (always enabled)
log10_gamma_p_mean_min = -9.0
log10_gamma_p_mean_max = -7.0
log10_gamma_p_std_min = 0.1
log10_gamma_p_std_max = 1.0
```

## Log-Ratio Parameterization for σp (Amplitude)

Argus always uses log-ratio parameterization for the red noise amplitude to improve sampling efficiency and capture parameter correlations.

### Mathematical Formulation

Instead of sampling σp independently, it is derived deterministically from γp:

```
log₁₀(σp,i) = log₁₀(γp,i) + log₁₀(ratio_i)
```

where the log-ratio follows a hierarchical model:
```
log₁₀(ratio_i) ~ Normal(μ_ratio, σ_ratio)
```

with population hyperparameters:
- `μ_ratio` ~ Uniform(ratio_mean_min, ratio_mean_max) 
- `σ_ratio` ~ Uniform(ratio_std_min, ratio_std_max)

### Physical Motivation

This parameterization is motivated by astrophysical considerations:

1. **Empirical correlations**: Observations suggest correlations between red noise spectral properties and amplitudes
2. **Reduced parameter space**: The correlation reduces the effective dimensionality of the parameter space
3. **Improved sampling**: By parameterizing the ratio rather than σp directly, MCMC sampling becomes more efficient
4. **Physical interpretability**: The ratio parameter has a direct physical interpretation as the relationship between spectral shape and amplitude

### Configuration Parameters

```ini
# Hyperprior ranges for log-ratio parameterization (always enabled)
# log10(σp) = log10(γp) + log10(ratio)
log10_ratio_mean_min = -8.0
log10_ratio_mean_max = -4.0
log10_ratio_std_min = 0.5
log10_ratio_std_max = 3.0
```

## Fixed Parameter Override

### Injection-Based Parameter Fixing

When spin injection files are provided, Argus can fix pulsar red noise parameters to specific values instead of sampling them. This is typically used for:

- Testing with known injected signals
- Validation studies with predetermined parameter values
- Development and debugging scenarios

The hierarchical modeling is automatically disabled for parameters that are explicitly fixed via injection files.

## Implementation Details

### Gradient Balancing

Argus implements gradient balancing techniques to improve MCMC sampling:

1. **Standardized parameterization**: Raw parameters are sampled from N(0,1) and transformed
2. **3-sigma rule**: Uniform prior ranges are converted to Normal distributions with std = (high-low)/6
3. **Scaled gradients**: Individual pulsar parameters are scaled by 1/√(n_pulsars) to balance gradients

### Parameter Counting

The effective number of parameters depends on the parameterization:

- **Hierarchical γp + log-ratio σp**: 4 hyperparameters + 2×n_pulsars individual parameters
- **Independent priors**: 2×n_pulsars parameters

## Configuration

The hierarchical modeling and log-ratio parameterization are always enabled. Configure the hyperprior ranges:

```ini
# Hyperprior ranges for hierarchical modeling of gamma_p
log10_gamma_p_mean_min = -9.0
log10_gamma_p_mean_max = -7.0
log10_gamma_p_std_min = 0.1
log10_gamma_p_std_max = 1.0

# Hyperprior ranges for log-ratio parameterization of sigma_p
log10_ratio_mean_min = -8.0
log10_ratio_mean_max = -4.0
log10_ratio_std_min = 0.5
log10_ratio_std_max = 3.0
```

This approach provides:
- Efficient sampling for large pulsar arrays
- Physically motivated parameter correlations
- Population-level inference capabilities
- Robust convergence properties

## References

This parameterization scheme is based on established practices in pulsar timing array analyses and gravitational wave detection methodologies. The hierarchical approach follows similar implementations in enterprise and other PTA analysis software packages.
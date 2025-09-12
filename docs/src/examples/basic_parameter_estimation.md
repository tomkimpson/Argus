# Basic Parameter Estimation

This tutorial demonstrates how to perform basic parameter estimation using Argus with a simple mock dataset.

## Overview

We'll cover:
- Loading timing data
- Setting up a basic model
- Running Bayesian inference
- Interpreting results

## Prerequisites

Make sure you have Argus installed:

```bash
pip install argus
```

## Example Code

```python
import numpy as np
import argus
from argus.data_loader import DataLoader
from argus.model import ArgusModel
from argus.workflow import run_inference

# Load mock data
data_loader = DataLoader()
timing_data = data_loader.load_mock_data("IPTA_Challenge1_open")

# Set up the model
model = ArgusModel(
    data=timing_data,
    include_gw_signal=True,
    noise_model="white_red"
)

# Configure priors - Advanced Bayesian techniques are available
# Basic configuration (suitable for small PTAs)
priors = {
    "log10_A_gw": {"type": "uniform", "min": -18, "max": -12},  # Automatic reparameterization applied
    "gamma_gw": {"type": "uniform", "min": 1, "max": 7},
    "log10_white_noise": {"type": "uniform", "min": -10, "max": -4}
}

# Run NUTS sampling
output_dir = run_inference(
    config_path="config.ini",
    use_gw=True
)

# Analyze results
print(f"Best-fit log10(A_gw): {results.posterior['log10_A_gw'].mean():.2f}")
print(f"Posterior std log10(A_gw): {results.posterior['log10_A_gw'].std():.2f}")
```

## Expected Output

```
Best-fit log10(A_gw): -15.23
Posterior std log10(A_gw): 0.45
Number of MCMC samples: 2000
Number of chains: 4
```

## Advanced Bayesian Techniques

For improved sampling performance, especially with larger PTAs, consider enabling Argus's advanced parameterization methods:

### Hierarchical Noise Modeling

For PTAs with >5 pulsars, enable hierarchical modeling to share information across the pulsar population:

```python
# Enable hierarchical priors for pulsar red noise parameters
advanced_priors = {
    "log10_A_gw": {"type": "uniform", "min": -18, "max": -12},
    "gamma_gw": {"type": "uniform", "min": 1, "max": 7},
    
    # Hierarchical pulsar noise parameters
    "hierarchical_noise": True,
    "log10_gamma_p_mean": {"type": "uniform", "min": -10, "max": -6},
    "log10_gamma_p_std": {"type": "uniform", "min": 0.1, "max": 2.0},
    "log10_sigma_p_mean": {"type": "uniform", "min": -20, "max": -10},
    "log10_sigma_p_std": {"type": "uniform", "min": 0.1, "max": 2.0}
}
```

### Log-Ratio Parameterization

For highly correlated red noise parameters, enable log-ratio parameterization:

```python
# Decorrelate sigma_p and gamma_p using ratio parameterization
ratio_priors = {
    "log10_A_gw": {"type": "uniform", "min": -18, "max": -12},
    "gamma_gw": {"type": "uniform", "min": 1, "max": 7},
    
    # Standard gamma_p priors
    "log10_gamma_p": {"type": "uniform", "min": -10, "max": -6},
    
    # Log-ratio parameterization for sigma_p
    "log_ratio_parameterization": True,
    "log10_ratio_mean": {"type": "uniform", "min": -2, "max": 2},
    "log10_ratio_std": {"type": "uniform", "min": 0.1, "max": 1.0}
}
```

### Configuration File Approach

These techniques can also be enabled via configuration files:

```ini
[PriorModel]
# GW parameters (reparameterization automatic for uniform priors)
log10_ha_fixed = false  
log10_ha_min = -18.0
log10_ha_max = -12.0

# Enable hierarchical modeling for large PTAs
hierarchical_noise = true
log10_gamma_p_mean_min = -10.0
log10_gamma_p_mean_max = -6.0
log10_gamma_p_std_min = 0.1
log10_gamma_p_std_max = 2.0

# Enable log-ratio parameterization for correlated parameters
log_ratio_parameterization = true
log10_ratio_mean_min = -2.0  
log10_ratio_mean_max = 2.0
log10_ratio_std_min = 0.1
log10_ratio_std_max = 1.0
```

### When to Use Advanced Techniques

| PTA Size | Recommended Techniques | Benefits |
|----------|----------------------|----------|
| 1-5 pulsars | h_a reparameterization only | Improved NUTS sampling |
| 6-20 pulsars | h_a reparam + hierarchical | Population inference, better mixing |
| 20+ pulsars | All techniques | Essential for convergence |

### Monitoring Effectiveness

Check if advanced techniques are helping by monitoring:

- **Effective Sample Size (ESS)**: Should increase for difficult parameters
- **R̂ diagnostics**: Should approach 1.0 faster  
- **Divergent transitions**: Should decrease or disappear
- **Sampling time**: May initially increase but leads to better convergence

```python
# Check sampling diagnostics
import arviz as az

print("ESS for log10_A_gw:", az.ess(results.posterior['log10_A_gw']))
print("R-hat for log10_A_gw:", az.rhat(results.posterior['log10_A_gw']))
print("Number of divergences:", results.sample_stats.diverging.sum())
```

## Next Steps

- Learn more about [Advanced Bayesian Methods](../bayesian_methods.md) for detailed explanations
- Try [multi-parameter estimation](multi_parameter_estimation.md) for more complex models
- Explore [Mathematical Background](../mathematical_background.md) for theoretical foundations
- See [gravitational wave detection](gw_detection.md) techniques

!!! tip "Performance Guidelines"
    - **Small PTAs (≤5 pulsars)**: 2000-4000 samples usually sufficient
    - **Medium PTAs (6-20 pulsars)**: 4000-8000 samples recommended with hierarchical priors
    - **Large PTAs (>20 pulsars)**: 8000+ samples may be needed, use all advanced techniques
    
!!! warning "Parameter Interpretation"
    When using hierarchical or log-ratio parameterizations, remember that posterior samples are automatically transformed back to physical parameters for analysis. Population-level parameters provide additional astrophysical insights.
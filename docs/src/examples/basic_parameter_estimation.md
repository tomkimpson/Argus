# Basic Parameter Estimation

This tutorial demonstrates how to perform basic parameter estimation using Argus with a simple mock dataset.

## Overview

We'll cover:
- Loading timing data
- Setting up a basic model
- Running Bayesian inference
- Interpreting results

## Prerequisites

Make sure you have Argus installed. See the [Getting Started](../getting_started.md) guide for detailed installation instructions.

**Quick install from source:**

```bash
# Set up a virtual environment (recommended)
python -m venv argus-env
source argus-env/bin/activate

# Clone and install
git clone https://github.com/tomkimpson/Argus.git
cd Argus
pip install .
```

## Using Example Workflows

The easiest way to get started is to use the provided example workflows in the `workflows/` directory. These contain complete, working examples that you can run immediately.

### Quick Start: Example Workflow Lite

For rapid prototyping and testing:

```bash
cd workflows/example_workflow_lite
python run_analysis.py configs/example_config.ini
```

This workflow uses:
- IPTA Mock Data Challenge dataset
- Reduced MCMC samples (200) for faster execution
- 2-4 chains for basic convergence checking
- All pulsar noise and GW parameters

Expected output:
```
=== EXAMPLE WORKFLOW LITE - RAPID PROTOTYPING ===
JAX version: 0.4.x
Default device: gpu
...
Inference complete! Results saved to: outputs/results_dev_lite/20250930_120000/
```

### Production Analysis: Full Workflow

For publication-quality results:

```bash
cd workflows/example_workflow
python run_analysis.py configs/example_config.ini
```

This uses more MCMC samples (2000+) and chains (4+) for robust convergence diagnostics.

## Using the Python API

You can also use Argus programmatically in your own scripts:

```python
from argus import workflow

# Run Bayesian inference using a configuration file
output_dir = workflow.run_inference(
    config_path="path/to/config.ini",
    use_gw=True,
    timestamp="20250930_120000"
)

print(f"Results saved to: {output_dir}")
```

The workflow handles:
- Loading pulsar timing data
- Setting up the Bayesian model with priors
- Running NUTS sampling with NumPyro
- Saving posterior samples and diagnostics
- Generating summary plots

## Configuration Files

Analysis parameters are specified in `.ini` configuration files. Example structure:

```ini
[Data]
data_path = ../../data/IPTA_MockDataChallenge2/dataset_2b/
excluded_psrs = J1640+2224

[NUTS]
num_samples = 200
num_warmup = 100
num_chains = 4
target_accept_prob = 0.855

[PriorModel]
# GW parameters
log10_ha_min = -18.0
log10_ha_max = -14.0

# Pulsar red noise parameters
log10_gamma_p_min = -11.0
log10_gamma_p_max = -6.0
```

See `workflows/example_workflow*/configs/` for complete configuration templates.

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
*Developer notes: Nice to have additional example tutorials*
- Multi-parameter estimation for more complex models
- Explore [Mathematical Background](../mathematical_background.md) for theoretical foundations
- Gravitational wave detection techniques

!!! tip "Performance Guidelines"
    - **Small PTAs (≤5 pulsars)**: 2000-4000 samples usually sufficient
    - **Medium PTAs (6-20 pulsars)**: 4000-8000 samples recommended with hierarchical priors
    - **Large PTAs (>20 pulsars)**: 8000+ samples may be needed, use all advanced techniques
    
!!! warning "Parameter Interpretation"
    When using hierarchical or log-ratio parameterizations, remember that posterior samples are automatically transformed back to physical parameters for analysis. Population-level parameters provide additional astrophysical insights.
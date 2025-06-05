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
from argus.bayesian_inference import run_nested_sampling

# Load mock data
data_loader = DataLoader()
timing_data = data_loader.load_mock_data("IPTA_Challenge1_open")

# Set up the model
model = ArgusModel(
    data=timing_data,
    include_gw_signal=True,
    noise_model="white_red"
)

# Configure priors
priors = {
    "log10_A_gw": {"type": "uniform", "min": -18, "max": -12},
    "gamma_gw": {"type": "uniform", "min": 1, "max": 7},
    "log10_white_noise": {"type": "uniform", "min": -10, "max": -4}
}

# Run nested sampling
results = run_nested_sampling(
    model=model,
    priors=priors,
    nlive=1000,
    sample="rwalk"
)

# Analyze results
print(f"Evidence: {results.logz:.2f} ± {results.logzerr:.2f}")
print(f"Best-fit log10(A_gw): {results.samples['log10_A_gw'].mean():.2f}")
```

## Expected Output

```
Evidence: -2345.67 ± 0.12
Best-fit log10(A_gw): -15.23
Number of likelihood evaluations: 125000
Sampling efficiency: 15.3%
```

## Next Steps

- Try [multi-parameter estimation](multi_parameter_estimation.md) for more complex models
- Learn about [custom noise models](custom_noise_models.md) 
- Explore [gravitational wave detection](gw_detection.md) techniques

!!! tip "Performance"
    For production runs, increase `nlive` to 2000-5000 for better parameter constraints.
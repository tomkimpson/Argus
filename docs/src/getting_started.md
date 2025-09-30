# Getting Started

This guide will help you install Argus and run your first pulsar timing array analysis.

## Prerequisites

Argus requires Python 3.11 or 3.12. We strongly recommend setting up a fresh virtual environment to avoid dependency conflicts.

### Setting Up a Virtual Environment

**Using venv:**
```console
$ python -m venv argus-env
$ source argus-env/bin/activate  # On Windows: argus-env\Scripts\activate
```

**Using conda:**
```console
$ conda create -n argus-env python=3.11
$ conda activate argus-env
```

## Installation

:::{note}
PyPI distribution is coming soon! For now, please install from source.
:::

### From Source

Clone the repository and install:

```console
$ git clone https://github.com/tomkimpson/Argus.git
$ cd Argus
$ pip install .
```

### Development Installation

If you plan to contribute or modify the code:

```console
$ git clone https://github.com/tomkimpson/Argus.git
$ cd Argus
$ pip install -e ".[dev]"
```

This installs Argus in editable mode with development dependencies including testing and documentation tools.

## Running Your First Analysis

Argus includes example workflows to help you get started quickly.

### Example Workflows

Two example workflows are provided:

1. **`example_workflow_lite`** - For rapid prototyping and testing
2. **`example_workflow`** - For production-quality analysis

### Quick Start with Example Workflow Lite

```console
$ cd workflows/example_workflow_lite
$ python run_analysis.py configs/example_config.ini
```

This will:
- Load pulsar timing data from the IPTA Mock Data Challenge
- Run Bayesian inference with reduced MCMC samples for faster execution
- Save posterior samples, diagnostic plots, and analysis results
- Display a summary when complete

Expected output:
```
=== EXAMPLE WORKFLOW LITE - RAPID PROTOTYPING ===
JAX version: 0.4.x
Default device: gpu
...
Inference complete! Results saved to: outputs/results_dev_lite/20250930_120000/
```

### Production Analysis

For publication-quality results, use the full workflow:

```console
$ cd workflows/example_workflow
$ python run_analysis.py configs/example_config.ini
```

This uses more MCMC samples and chains for better convergence diagnostics.

## Configuration Files

The example workflows use `.ini` configuration files to specify:

- Data paths and pulsars to include
- MCMC sampling parameters (samples, warmup, chains)
- Prior ranges for gravitational wave and pulsar noise parameters
- Output directories

Example configuration structure:

```ini
[Data]
data_path = ../../data/IPTA_MockDataChallenge2/dataset_2b/
excluded_psrs = J1640+2224

[NUTS]
num_samples = 200
num_warmup = 100
num_chains = 4

[PriorModel]
log10_ha_min = -18.0
log10_ha_max = -14.0
```

See the example config files in `workflows/example_workflow*/configs/` for complete templates.

## Using the Python API

You can also use Argus programmatically in your own Python scripts:

```python
from argus import workflow

# Run Bayesian inference
output_dir = workflow.run_inference(
    config_path="path/to/your_config.ini",
    use_gw=True,
    timestamp="20250930_120000"
)

print(f"Results saved to: {output_dir}")
```

## Next Steps

- Explore the [examples](examples/index.md) for more detailed tutorials
- Learn about the [mathematical background](mathematical_background.md)
- Read about [Bayesian methods](bayesian_methods.md) used in Argus
- Check out the [contributing guide](contributing.md) if you'd like to contribute

:::{tip}
Start with `example_workflow_lite` for testing and development. It runs much faster and is perfect for experimenting with different configurations before committing to a full production run.
:::

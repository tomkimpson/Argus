# Getting Started

This guide will help you install Argus and run your first PTA state-space analysis.

---

## Prerequisites

Argus requires Python 3.11 or 3.12 and uses conda for package management. We strongly recommend using conda to avoid dependency conflicts.

!!! note "Why Conda?"
    Argus uses a conda + pip installation workflow to handle the `enterprise-pulsar` dependency, which has complex system-level dependencies that are difficult to resolve with pip alone. This hybrid approach ensures reliable installation across different systems.

---

## Installation

!!! note
    PyPI distribution is coming soon! For now, please install from source using the workflow below.

### Installation Steps

```console
$ git clone https://github.com/tomkimpson/Argus.git
$ cd Argus
$ conda create -n argus-env python=3.11
$ conda activate argus-env
$ conda install -c conda-forge enterprise-pulsar
$ pip install -e .
```

This workflow:

1. Creates a fresh conda environment with Python 3.11
2. Installs `enterprise-pulsar` via conda to handle its complex dependencies
3. Uses pip to install Argus and all remaining dependencies from `pyproject.toml`

### Development Installation

If you plan to contribute or modify the code, install with development dependencies:

```console
$ git clone https://github.com/tomkimpson/Argus.git
$ cd Argus
$ conda create -n argus-env python=3.11
$ conda activate argus-env
$ conda install -c conda-forge enterprise-pulsar
$ pip install -e ".[dev]"
```

This adds testing tools (pytest), linting (black, ruff), type checking (mypy), and pre-commit hooks.

---

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

---

### Production Analysis

For publication-quality results, use the full workflow:

```console
$ cd workflows/example_workflow
$ python run_analysis.py configs/example_config.ini
```

This uses more MCMC samples and chains for better convergence diagnostics.

---

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

---

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

---

## Further Reading

### Documentation

- [State-Space Methods](state_space.md) - Learn about the theoretical framework underlying Argus
- [Bayesian Inference](bayesian_inference.md) - Understand the statistical methods used for parameter estimation
- [API Reference](api/index.md) - Detailed API documentation for programmatic usage
- [Contributing Guide](contributing.md) - How to contribute to the Argus project

### Academic Papers

- [arXiv:2409.14613](https://arxiv.org/abs/2409.14613) - State-space methods for PTA analysis
- [arXiv:2410.10087](https://arxiv.org/abs/2410.10087) - Bayesian inference techniques for PTAs
- [arXiv:2501.06990](https://arxiv.org/abs/2501.06990) - Advanced implementations and applications
- [JAX Documentation](https://jax.readthedocs.io/) - Learn more about the JAX framework
- [NumPyro Documentation](https://num.pyro.ai/) - NUTS sampling and probabilistic programming

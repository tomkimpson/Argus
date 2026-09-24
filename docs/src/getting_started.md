# Getting Started

This guide will help you install Argus and run your first PTA state-space analysis.

---

## Prerequisites

Argus requires Python 3.11 or 3.12. The package itself installs with pip alone: at runtime it reads per-pulsar `.feather` files and has no dependency on pulsar-timing software.

!!! note "Converting your own .par/.tim files"
    Turning raw `.par`/`.tim` files into feathers is a one-time data-prep step that uses `enterprise-pulsar`, `tempo2` and PINT. These are not pip-installable, so they come from conda via `environment.yml`. See [Preparing data](#preparing-data) below. You do not need them to run Argus on existing feathers.

---

## Installation

!!! note
    PyPI distribution is coming soon! For now, please install from source using the workflow below.

### Installation Steps

```console
$ git clone https://github.com/tomkimpson/Argus.git
$ cd Argus
$ pip install -e .
```

We recommend installing into a fresh virtual environment or conda environment with Python 3.11 or 3.12.

### Preparing data

To convert `.par`/`.tim` files to feathers, create the data-prep environment and run the ingestion script once:

```console
$ conda env create -f environment.yml
$ conda activate argus-dataprep
$ pip install -e .
$ python scripts/ingest_par_tim.py <par_tim_dir> <feather_out_dir>
```

Pass `--timing-package pint` for PINT-format releases such as NANOGrav 15yr. The default (tempo2) is required for the IPTA MDC2 data.

### Development Installation

If you plan to contribute or modify the code, install with development dependencies:

```console
$ git clone https://github.com/tomkimpson/Argus.git
$ cd Argus
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

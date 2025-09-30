# Argus

![Tests](https://github.com/tomkimpson/Argus/actions/workflows/run_test.yml/badge.svg) [![codecov](https://codecov.io/gh/tomkimpson/Argus/graph/badge.svg?token=2PEOHCFV1K)](https://codecov.io/gh/tomkimpson/Argus) [![PyPI version](https://badge.fury.io/py/argus-pta.svg)](https://badge.fury.io/py/argus-pta) [![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://tomkimpson.github.io/Argus/)

**Argus** is a Python package for Bayesian inference on pulsar timing array data using JAX and Kalman filtering techniques. It provides efficient, GPU-accelerated analysis tools for detecting gravitational waves and estimating astrophysical parameters from pulsar timing data.

## Features

- 🚀 **Fast computation** with JAX and GPU acceleration
- 🔬 **Bayesian parameter estimation** using NumPyro
- 📊 **Kalman filtering** for efficient state space modeling
- 🌊 **Gravitational wave detection** in pulsar timing arrays
- 📈 **Comprehensive analysis tools** and visualization
- 🔧 **Example workflows** for quick start and development

## Installation

> **Note:** PyPI distribution coming soon! For now, please install from source.

### Prerequisites

We recommend setting up a fresh virtual environment:

**Using venv:**
```bash
python -m venv argus-env
source argus-env/bin/activate  # On Windows: argus-env\Scripts\activate
```

**Using conda:**
```bash
conda create -n argus-env python=3.11
conda activate argus-env
```

### From Source

```bash
git clone https://github.com/tomkimpson/Argus.git
cd Argus
pip install .
```

### Development Installation

```bash
git clone https://github.com/tomkimpson/Argus.git
cd Argus
pip install -e ".[dev]"
```

## Quick Start

### Example Workflows

Argus includes two example workflows to help you get started:

1. **`example_workflow_lite`** - Lightweight workflow for rapid prototyping and testing
   - Fewer MCMC samples (200 vs 2000+)
   - Faster execution (~minutes vs hours)
   - Perfect for development and testing

2. **`example_workflow`** - Full production workflow
   - Complete MCMC sampling (2000+ samples)
   - Multiple chains for convergence diagnostics
   - Recommended for publication-quality results

### Running an Example

```bash
# Navigate to the example workflow directory
cd workflows/example_workflow_lite

# Run the analysis with the provided config file
python run_analysis.py configs/example_config.ini
```

The workflow will:
- Load pulsar timing data from the IPTA Mock Data Challenge
- Run Bayesian inference using JAX and NumPyro
- Save results including posterior samples and diagnostic plots
- Display a summary of the analysis

### Using the Python API

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

See the `workflows/` directory for complete examples and configuration templates.

## Documentation

Full documentation is available at [https://tomkimpson.github.io/Argus/](https://tomkimpson.github.io/Argus/)

## Requirements

- Python ≥ 3.11, < 3.13
- JAX ≥ 0.4.0  
- NumPyro ≥ 0.15.0
- NumPy, Pandas, Matplotlib
- Enterprise Pulsar (for pulsar timing data handling)

## GPU Support

Argus automatically detects and uses GPU acceleration when available. For optimal performance with large datasets, we recommend using CUDA-compatible GPUs.

## Contributing

We welcome contributions! Please see our [contributing guidelines](docs/src/contributing.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Support

- 📚 [Documentation](https://tomkimpson.github.io/Argus/)
- 🐛 [Issue Tracker](https://github.com/tomkimpson/Argus/issues)
- 💬 [Discussions](https://github.com/tomkimpson/Argus/discussions)
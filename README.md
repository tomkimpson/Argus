# Argus

![Tests](https://github.com/tomkimpson/Argus/actions/workflows/run_test.yml/badge.svg) [![codecov](https://codecov.io/gh/tomkimpson/Argus/graph/badge.svg?token=2PEOHCFV1K)](https://codecov.io/gh/tomkimpson/Argus) [![PyPI version](https://badge.fury.io/py/argus-pta.svg)](https://badge.fury.io/py/argus-pta) [![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://tomkimpson.github.io/Argus/)

**Argus** is a Python package for Bayesian inference on pulsar timing array data using JAX and Kalman filtering techniques. It provides efficient, GPU-accelerated analysis tools for detecting gravitational waves and estimating astrophysical parameters from pulsar timing data.

## Features

- 🚀 **Fast computation** with JAX and GPU acceleration
- 🔬 **Bayesian parameter estimation** using NumPyro
- 📊 **Kalman filtering** for efficient state space modeling
- 🌊 **Gravitational wave detection** in pulsar timing arrays
- 📈 **Comprehensive analysis tools** and visualization
- 🔧 **Easy-to-use command-line interface**

## Installation

### From PyPI (recommended)

```bash
pip install argus-pta
```

### From Source with Pixi (recommended for development)

[Pixi](https://pixi.sh) is a fast, cross-platform package manager that handles complex scientific dependencies better than pip. It's especially recommended for this project due to dependencies like `enterprise-pulsar` and `scikit-sparse`.

```bash
# Install pixi first (see https://pixi.sh for installation instructions)
curl -fsSL https://pixi.sh/install.sh | bash

# Clone and setup the project
git clone https://github.com/tomkimpson/Argus.git
cd Argus
pixi install
pixi run setup-dev
```

### From Source with pip/poetry (alternative)

```bash
git clone https://github.com/tomkimpson/Argus.git
cd Argus
pip install .
```

### Development Installation

**With Pixi (recommended):**
```bash
git clone https://github.com/tomkimpson/Argus.git
cd Argus
pixi shell dev  # Activates development environment
pixi run setup-dev  # Installs package and pre-commit hooks
```

**With pip:**
```bash
git clone https://github.com/tomkimpson/Argus.git
cd Argus
pip install -e ".[dev]"
```

## Quick Start

### Command Line Interface

1. **Create a configuration file template:**
   ```bash
   argus init -o my_analysis.ini
   ```

2. **Edit the configuration file** with your data paths and analysis parameters

3. **Run the analysis:**
   ```bash
   argus run my_analysis.ini
   ```

### Python API

```python
import argus

# Load pulsar timing data
pulsar_data = argus.data_loader.LoadWidebandPulsarData.get_processed_residuals(
    "path/to/data.pkl",
    excluded_psrs=[]
)

# Run Bayesian inference using workflow
output_dir = argus.workflow.run_inference(
    config_path="my_analysis.ini",
    use_gw=True,
    timestamp="20250101_120000"
)

print(f"Results saved to: {output_dir}")
```

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
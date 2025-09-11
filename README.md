# Argus

![Tests](https://github.com/tomkimpson/Argus/actions/workflows/run_test.yml/badge.svg) [![codecov](https://codecov.io/gh/tomkimpson/Argus/graph/badge.svg?token=2PEOHCFV1K)](https://codecov.io/gh/tomkimpson/Argus) [![PyPI version](https://badge.fury.io/py/argus-pta.svg)](https://badge.fury.io/py/argus-pta)

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
data = argus.load_pulsar_data("path/to/data.pkl")

# Set up analysis configuration
config = {
    'nsamples': 2000,
    'nwarmup': 1000, 
    'nchains': 4
}

# Run Bayesian inference
results = argus.run_bayesian_inference(data, config)

# Compare different models
comparison = argus.compare_inference_methods(
    results_paths=["results1.pkl", "results2.pkl"],
    method_names=["With GW", "No GW"]
)
```

## Documentation

Full documentation is available at [https://argus-pta.readthedocs.io/](https://argus-pta.readthedocs.io/)

## Requirements

- Python ≥ 3.11
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

## Citation

If you use Argus in your research, please cite:

```bibtex
@software{argus_pta,
  title = {Argus: Bayesian inference for pulsar timing array data analysis},
  author = {Kimpson, Tom and Hu, J.},
  url = {https://github.com/tomkimpson/Argus},
  year = {2025}
}
```

## Support

- 📚 [Documentation](https://argus-pta.readthedocs.io/)
- 🐛 [Issue Tracker](https://github.com/tomkimpson/Argus/issues)
- 💬 [Discussions](https://github.com/tomkimpson/Argus/discussions)
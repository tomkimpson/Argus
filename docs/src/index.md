# Argus

👋 Welcome to the **Argus** project documentation.

**Argus** is a research project for the [detection of nHz gravitational waves](https://arxiv.org/abs/2105.13270) using Pulsar Timing Arrays (PTAs), leveraging a [state-space representation](https://academic.oup.com/book/16563). It is an ongoing effort to open-source methods developed at the [University of Melbourne](https://github.com/UniMelb-NSGW) and [OzGrav](https://www.ozgrav.org). 

---

## Key Features

- **State-space framework** for PTA data analysis
- **Bayesian parameter estimation** using NUTS sampling with NumPyro
- **Pulsar noise modeling** with flexible priors
- **JAX-based implementation** for GPU acceleration, automatic differentiation

---

## Quick Start

To get started with Argus:

1. **Install the package**: See the [Getting Started](getting_started.md) guide
2. **Run example workflows**: Try `workflows/example_workflow_lite` for rapid prototyping
3. **Explore the API**: Browse the [API Reference](api/index.md) for detailed documentation
4. **Contribute**: Read our [Contributing Guide](contributing.md)

---


## Example Usage

Argus includes two example workflows in the `workflows/` directory:

- **`workflows/example_workflow_lite/`** 
    - Lightweight workflow for rapid prototyping
    - Reduced MCMC samples for faster execution
    - Perfect for testing and development
    - Run with: `python run_analysis.py configs/example_config.ini`

- **`workflows/example_workflow/`** - Full production workflow
    - Complete MCMC sampling for convergence
    - Multiple chains for diagnostics
    - Recommended for publication-quality results

---


## Support

- 🐛 **Bug reports**: [GitHub Issues](https://github.com/ADACS-Australia/tkimpson_2025a/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/ADACS-Australia/tkimpson_2025a/discussions)
- 📧 **Contact**: [University of Melbourne NSGW](https://github.com/UniMelb-NSGW)

---

## Literature

Argus builds off of proof of concept work described in:

- [arXiv:2409.14613](https://arxiv.org/abs/2409.14613)
- [arXiv:2410.10087](https://arxiv.org/abs/2410.10087)
- [arXiv:2501.06990](https://arxiv.org/abs/2501.06990)


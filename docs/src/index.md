# Argus

Welcome to the **Argus** project documentation.

**Argus** is a research project for the [detection of nHz gravitational waves](https://arxiv.org/abs/2105.13270) using Pulsar Timing Arrays (PTAs), leveraging a [state-space representation](https://academic.oup.com/book/16563). It is an ongoing effort to open-source methods developed at the [University of Melbourne](https://github.com/UniMelb-NSGW) and [OzGrav](https://www.ozgrav.org). 

## Key Features

- **State-space framework** for PTA data analysis
- **Bayesian parameter estimation** using NUTS sampling with NumPyro
- **Gravitational wave detection** and characterization
- **Pulsar noise modeling** with flexible priors
- **JAX-based implementation** for high-performance computing



## Quick Start

To get started with Argus:

1. **Install the package**: See the [Getting Started](getting_started.md) guide
2. **Run example workflows**: Try `workflows/example_workflow_lite` for rapid prototyping
3. **Run your first analysis**: Check out the [examples](examples/index.md)
4. **Explore the API**: Browse the workflow and API examples
5. **Contribute**: Read our [Contributing Guide](contributing.md)

## About this Documentation

This documentation provides an overview of the project, installation and usage instructions, API details, and developer notes for contributors.

!!! note
    If you're new to Argus, start with the "Getting Started" section.

## Example Usage

### Example Workflows

Argus includes complete example workflows in the `workflows/` directory:

- **`workflows/example_workflow_lite/`** - Lightweight workflow for rapid prototyping
  - Reduced MCMC samples for faster execution
  - Perfect for testing and development
  - Run with: `python run_analysis.py configs/example_config.ini`

- **`workflows/example_workflow/`** - Full production workflow
  - Complete MCMC sampling for convergence
  - Multiple chains for diagnostics
  - Recommended for publication-quality results

### Tutorials and Examples

- [Getting Started](getting_started.md) - Installation and first analysis
- [Basic parameter estimation with Mock Data Challenge](examples/basic_parameter_estimation.md)
*Developer notes: Nice to have additional analysis examples*
- Multi-parameter estimation with NANOGrav data
- Custom noise models

!!! tip "Interactive Notebooks"
    Interactive Jupyter notebooks are available in the [`notebooks/`](notebooks/index.md) directory of the repository.


---

## Research and Citations

If you use Argus in your research, please cite the relevant papers:

```bibtex
@article{kimpson2024a,
  title={State-space analysis for pulsar timing arrays},
  author={Kimpson, Tom and others},
  journal={arXiv preprint arXiv:2409.14613},
  year={2024}
}
```

## Support

- 🐛 **Bug reports**: [GitHub Issues](https://github.com/ADACS-Australia/tkimpson_2025a/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/ADACS-Australia/tkimpson_2025a/discussions)
- 📧 **Contact**: [University of Melbourne NSGW](https://github.com/UniMelb-NSGW)


For the accompanying papers please see:

- [Kimpson et al. 2024a](https://arxiv.org/abs/2409.14613)
- [Kimpson et al. 2024b](https://arxiv.org/abs/2410.10087)
- [Kimpson et al. 2025](https://arxiv.org/abs/2501.06990)


## Etymology

The project is named **Argus**, after the hundred-eyed giant of Greek mythology, who could see in all directions simultaneously. In a pulsar timing array, multiple pulsars across the sky act as a network of cosmic "eyes". Just as Argus's hundred eyes gave him an omniscient view of his surroundings, a PTA's distributed array of millisecond pulsars creates an all-sky monitoring system capable of detecting gravitational waves through correlated timing residuals.



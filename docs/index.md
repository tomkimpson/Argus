# Argus

Welcome to the **Argus** project documentation.

**Argus** is a research project for the [detection of nHz gravitational waves](https://arxiv.org/abs/2105.13270) using Pulsar Timing Arrays (PTAs), leveraging a [state-space representation](https://en.wikipedia.org/wiki/State-space_representation).

It is an ongoing effort to open-source methods developed at the [University of Melbourne](https://github.com/UniMelb-NSGW) and [OzGrav](https://www.ozgrav.org). For the accompanying papers please see:

- [Kimpson et al. 2024a](https://arxiv.org/abs/2409.14613)
- [Kimpson et al. 2024b](https://arxiv.org/abs/2410.10087)
- [Kimpson et al. 2025](https://arxiv.org/abs/2501.06990)

## About this Documentation

This documentation provides an overview of the project, installation and usage instructions, API details, and developer notes for contributors.

!!! note
    If you're new to Argus, start with the "Getting Started" section.

## Example Usage

You can find examples of standard PTA analysis using various datasets:

- [Basic parameter estimation with Mock Data Challenge](examples/basic_parameter_estimation.md)
- [Multi-parameter estimation with NANOGrav data](examples/nanograv_analysis.md)
- [Custom noise models](examples/custom_noise_models.md)

!!! tip "Interactive Notebooks"
    Interactive Jupyter notebooks are available in the [`notebooks/`](notebooks/index.md) directory of the repository.

## Key Features

- **State-space framework** for PTA data analysis
- **Bayesian parameter estimation** with multiple samplers (NUTS, nested sampling)
- **Gravitational wave detection** and characterization
- **Pulsar noise modeling** with flexible priors
- **JAX-based implementation** for high-performance computing

## Etymology

The project is named **Argus**, after the hundred-eyed giant of Greek mythology.

> In a pulsar timing array, multiple pulsars across the sky act as a network of cosmic "eyes", continuously watching for gravitational waves.

## Quick Start

To get started with Argus:

1. **Install the package**: See the [Getting Started](getting_started.md) guide
2. **Run your first analysis**: Check out the [examples](examples/index.md)
3. **Explore the API**: Browse the [API Reference](api/index.md)
4. **Contribute**: Read our [Contributing Guide](contributing.md)

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
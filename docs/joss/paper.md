---
title: 'Argus: A JAX-based Python package for gravitational wave detection in pulsar timing arrays using state-space methods'
tags:
  - Python
  - astronomy
  - gravitational waves
  - pulsar timing arrays
  - Bayesian inference
  - state-space methods
  - JAX
  - high-performance computing
authors:
  - name: Tom Kimpson
    orcid: 0000-0000-0000-0000
    equal-contrib: true
    affiliation: "1, 2"
    corresponding: true
  - name: J. Hu
    equal-contrib: true
    affiliation: "1, 2"
affiliations:
  - name: University of Melbourne, Australia
    index: 1
    ror: 01ej9dk98
  - name: "OzGrav: Australian Research Council Centre of Excellence for Gravitational Wave Discovery, Australia"
    index: 2
    ror: 05qajvd42
date: 28 August 2025
bibliography: paper.bib
---

# Summary

Pulsar timing arrays (PTAs) are among the most sensitive detectors for nanohertz gravitational waves, monitoring the precise arrival times of radio pulses from millisecond pulsars across the sky. The detection and characterization of gravitational wave signals in PTA data presents significant computational challenges due to the complex correlations between pulsars, the stochastic nature of the signals, and the high-dimensional parameter spaces involved in Bayesian inference. `Argus` addresses these challenges by implementing a state-space framework for PTA analysis, leveraging JAX [@jax2018github] for high-performance computing and automatic differentiation, combined with advanced Kalman filtering techniques for efficient likelihood evaluation.

# Statement of need

Pulsar timing array analysis requires sophisticated statistical methods to extract weak gravitational wave signals from noisy timing residuals. Traditional approaches often struggle with the computational demands of high-dimensional Bayesian inference and the complex correlation structures inherent in gravitational wave signatures, such as the Hellings-Downs correlation pattern [@hellings1983gravitational]. The state-space formulation provides a natural framework for modeling the temporal evolution of stochastic processes in PTAs, enabling efficient recursive estimation through Kalman filtering [@kalman1960new].

`Argus` fills a critical gap in the PTA analysis ecosystem by providing a modern, high-performance implementation of state-space methods specifically designed for gravitational wave detection. The package leverages JAX's just-in-time compilation, automatic differentiation, and GPU acceleration capabilities to handle the computational intensity of PTA Bayesian inference. The mathematical framework separates different physical processes (gravitational waves, pulsar spin noise, and timing model parameters) into manageable blocks, exploiting sparsity for computational efficiency while maintaining full correlations where physically necessary.

The package has already been successfully applied to several research projects [@kimpson2024a; @kimpson2024b; @kimpson2025], demonstrating its utility for both mock data challenges (such as the International Pulsar Timing Array Mock Data Challenge) and ongoing research at major gravitational wave research institutions. `Argus` is designed to serve both researchers conducting cutting-edge PTA science and students learning gravitational wave data analysis techniques.

# Implementation

## State-Space Framework

`Argus` implements a comprehensive state-space formulation for PTA analysis, representing the system through state evolution and observation equations:

$$x_t = F x_{t-1} + w_t$$
$$y_t = H x_t + v_t$$

where $x_t$ represents the hidden state vector containing gravitational wave signals, pulsar spin parameters, and timing model corrections; $y_t$ represents the observed timing residuals; and $w_t$, $v_t$ are process and observation noise terms, respectively.

The state vector is organized into three physically motivated blocks: gravitational wave effects, pulsar spin-down variations, and timing model parameters. This block structure enables efficient computation by exploiting the sparsity pattern in the system matrices while preserving the essential correlations, such as the Hellings-Downs pattern for gravitational waves [@hellings1983gravitational].

## JAX Integration

The core computational engine leverages JAX [@jax2018github] for several critical advantages:

- **Just-in-time compilation**: Kalman filter operations are compiled for optimal performance
- **Automatic differentiation**: Gradients for Bayesian inference are computed automatically
- **Vectorization and parallelization**: Efficient handling of multiple pulsars and parameter dimensions
- **GPU acceleration**: Seamless scaling to GPU hardware for large pulsar arrays

The implementation uses JAX's functional programming paradigm, ensuring numerical stability and enabling advanced optimization techniques through the NumPyro [@phan2019composable] probabilistic programming framework.

## Bayesian Inference

Posterior sampling is performed using the No-U-Turn Sampler (NUTS) [@hoffman2014no] implemented in NumPyro. The combination of efficient likelihood evaluation through Kalman filtering and gradient-based MCMC sampling enables robust parameter estimation even for high-dimensional problems involving dozens of pulsars and hundreds of parameters.

## Software Quality and Testing

`Argus` includes comprehensive unit tests covering core functionality including Kalman filter operations, state-space model construction, and Bayesian inference workflows. The testing suite validates numerical accuracy against analytical solutions where available and ensures compatibility with both CPU and GPU execution environments. Continuous integration workflows maintain code quality and compatibility across different Python versions and hardware configurations.

# Research Impact and Applications

`Argus` has been successfully applied to several research projects, with results published in peer-reviewed literature [@kimpson2024a; @kimpson2024b; @kimpson2025]. These applications demonstrate the package's effectiveness for:

- **Mock data analysis**: Successful application to International Pulsar Timing Array Mock Data Challenges
- **Method validation**: Comparison with traditional PTA analysis approaches
- **Parameter estimation**: Robust inference of gravitational wave and pulsar noise parameters
- **Model comparison**: Bayesian evidence calculation for model selection

The state-space approach implemented in `Argus` offers several advantages over traditional frequency-domain methods, particularly in handling non-stationary processes and incorporating physical prior information about signal evolution.

# Future Directions

Several extensions are planned for `Argus` to expand its capabilities and applications:

- **Real data applications**: Integration with production PTA datasets from NANOGrav, EPTA, PPTA, and other major collaborations
- **Continuous gravitational waves**: Extension of the state-space framework to handle deterministic signals from individual binary systems and rotating neutron stars
- **Advanced noise models**: Implementation of additional stochastic processes including chromatic noise, system-dependent effects, and non-Gaussian noise components
- **Scalability improvements**: Optimization for next-generation PTAs such as the SKA pulsar timing array with hundreds of pulsars
- **Community integration**: Development of interfaces with existing PTA analysis pipelines including ENTERPRISE [@enterprise2020] and libstempo [@vallisneri2020libstempo]

The modular design of `Argus` facilitates these extensions while maintaining backward compatibility and computational efficiency.

# Acknowledgements

We acknowledge support from the Australian Research Council Centre of Excellence for Gravitational Wave Discovery (OzGrav) under grant CE170100004. This research made use of computational resources provided by the University of Melbourne. We thank the International Pulsar Timing Array collaboration for providing mock data challenges that aided in the development and validation of this software. The development of `Argus` builds upon foundational work in state-space methods for pulsar timing analysis, and we acknowledge the broader PTA community for valuable feedback and contributions to the conceptual framework.

# Availability and Documentation

`Argus` is open source software released under the MIT license. The source code is available on GitHub at https://github.com/tomkimpson/Argus, with comprehensive documentation hosted at https://argus-pta.readthedocs.io/. The package can be installed via PyPI and includes extensive examples, tutorials, and API documentation to facilitate adoption by the PTA community.

# References
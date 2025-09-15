---
title: 'Argus: A JAX-based Python package for gravitational wave detection in pulsar timing arrays using state-space filtering methods'
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
  - name: Nicholas O'Neil
    equal-contrib: true
    affiliation: "1, 2"
  - name: Andrew Melatos
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


`Argus` is a high-performance Python package for detecting and characterizing gravitational waves in pulsar timing array (PTA) data. The package provides a complete Bayesian inference framework based on state-space models, using Kalman filtering for efficient likelihood evaluation.

`Argus` leverages the JAX library [@jax2018github] for just-in-time (JIT) compilation, vectorization, parallelization, GPU/TPU acceleration, and end-to-end automatic differentiation of the entire filtering and likelihood pipeline. This enables efficient gradient-based samplers such as the No-U-Turn Sampler (NUTS), facilitating robust analysis of the complex, high-dimensional datasets characteristic of modern PTA experiments.


# Statement of Need

Pulsar timing arrays (PTAs) monitor the precise arrival times of radio pulses from millisecond pulsars distributed across the sky, enabling the detection of nanohertz gravitational waves. The recent discovery of a nanohertz stochastic gravitational-wave background by PTA collaborations [@nanograv2023pta] [@EPTA] [@ParkesPPTA2023] represents a landmark achievement in gravitational wave astronomy. However, the detection and characterization of gravitational wave signals in PTA data presents significant computational challenges due to complex inter-pulsar correlations, stochastic signal characteristics, and high-dimensional parameter spaces in Bayesian inference.

Traditional PTA analysis methods rely primarily on frequency-domain techniques that measure spatial cross-correlation patterns, particularly the Hellings-Downs correlation [@hellings1983gravitational]. These approaches work by estimating power spectral densities (PSDs) of various noise processes under assumptions of stationarity and Gaussianity. While successful, these methods often struggle with the computational demands of high-dimensional Bayesian inference and face limitations when modeling non-stationary processes or when the underlying statistical assumptions are violated.

State-space methods provide a novel and powerful complementary framework for PTA analysis. Instead of operating in the frequency domain with PSDs, they model the temporal evolution of hidden states (such as pulsar spin fluctuations and gravitational wave effects) using Kalman filtering [@kalman1960new] for recursive state estimation. This approach enables the explicit incorporation of physical knowledge about how different processes evolve over time—for example, how gravitational wave signals change amplitude and phase, or how pulsar spin noise behaves—directly into the model structure rather than treating these as generic stochastic processes. Crucially, state-space methods track the actual measured time-ordered realization of intrinsic timing noise in each pulsar, rather than averaging over ensemble realizations. This time-domain approach offers computationally favorable likelihood calculations that scale linearly with observations while naturally handling non-stationary processes. 


Despite their theoretical advantages, state-space methods have seen limited adoption in PTA research, primarily due to the lack of accessible, high-performance implementations. `Argus` fills a critical gap in the PTA analysis ecosystem by providing a modern, high-performance implementation of state-space methods specifically designed for gravitational wave detection. The package leverages JAX's just-in-time compilation, automatic differentiation, and GPU acceleration capabilities to handle the computational intensity of PTA Bayesian inference.  The package matures and formalizes the state-space methodology that was successfully applied in prior research and mock data challenges [@kimpson2024a; @kimpson2024b; @kimpson2025c], transforming proof-of-concept implementations into a robust, production-ready tool. The package serves as an independent analysis pipeline for the stochastic GW background detected by PTAs, providing a vital cross-check for results obtained with standard Hellings-Downs analyses.


![Corner plot showing posterior distributions from Bayesian parameter estimation using Argus on IPTA mock dataset. The plot displays marginal and joint posterior distributions for key noise model parameters including logarithmic amplitudes of various stochastic processes (log₁₀ ρ parameters), timing model error factors (EFAC), and additional white noise terms (EQUAD). The well-constrained posteriors demonstrate the effectiveness of the state-space Kalman filtering approach for robust parameter estimation in pulsar timing array analysis.\label{fig:corner}](images/example_corner_plot.png)


# Future Directions

Several extensions are planned for `Argus` to expand its capabilities and applications:

- **Model selection capabilities**: Implementation of Bayes factor calculations to enable robust model comparison and selection between different gravitational wave models, noise models, and signal hypotheses
- **Dispersion measure corrections**: Integration of time-variable dispersion measure effects and solar wind corrections within the state-space framework for improved timing precision
- **Modular model specification**: Development of a user-friendly interface allowing researchers to easily specify custom state-space models and noise components through configurable modules
- **Real data applications**: Integration with production PTA datasets from NANOGrav, EPTA, PPTA, and other major collaborations
- **Continuous gravitational waves**: Extension of the state-space framework to handle deterministic signals from individual binary systems and rotating neutron stars
- **Advanced noise models**: Implementation of additional stochastic processes including chromatic noise, system-dependent effects, and non-Gaussian noise components
- **Scalability improvements**: Optimization for next-generation PTAs such as the SKA pulsar timing array with hundreds of pulsars
- **Community integration**: Development of interfaces with existing PTA analysis pipelines including ENTERPRISE [@enterprise2020] and libstempo [@vallisneri2020libstempo]
- **Marginalisation over unwanted parameters**: TBD...

While these extensions represent exciting future directions, we release `Argus` in its current form because it successfully addresses the core challenge of Bayesian parameter estimation for pulsar timing array analysis. The package provides a robust, high-performance foundation for gravitational wave detection research and demonstrates significant computational advantages over traditional approaches. We encourage contributions from the broader PTA community to help implement these extensions and further advance the capabilities of state-space methods in gravitational wave astronomy. The modular design of `Argus` facilitates these extensions while maintaining backward compatibility and computational efficiency.

# Acknowledgements

We acknowledge support from the Australian Research Council Centre of Excellence for Gravitational Wave Discovery (OzGrav) under grant CE170100004. This research made use of computational resources provided by the University of Melbourne. We thank the International Pulsar Timing Array collaboration for providing mock data challenges that aided in the development and validation of this software. The development of `Argus` builds upon foundational work in state-space methods for pulsar timing analysis, and we acknowledge the broader PTA community for valuable feedback and contributions to the conceptual framework.

# Availability and Documentation

`Argus` is open source software released under the MIT license. The source code is available on GitHub at https://github.com/tomkimpson/Argus, with comprehensive documentation hosted at https://tomkimpson.github.io/Argus/. The package can be installed via PyPI and includes extensive examples, tutorials, and API documentation to facilitate adoption by the PTA community.

# References
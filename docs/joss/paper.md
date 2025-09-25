---
title: 'Argus: JAX state-space filtering for gravitational wave detection in pulsar timing arrays'
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


`Argus` is a high-performance Python package for detecting and characterizing gravitational waves in pulsar timing array (PTA) data. The package provides a complete Bayesian inference framework based on state-space models, using Kalman filtering for efficient likelihood evaluation. `Argus` leverages the JAX library [@jax2018github] for just-in-time (JIT) compilation, GPU acceleration, and end-to-end automatic differentiation, facilitating rapid Bayesian inference with gradient-based samplers. The state-space approach provides a computationally efficient alternative to traditional frequency-domain methods, offering linear scaling with observations, and natural handling of non-stationary processes in the complex, high-dimensional datasets characteristic of modern PTA experiments.



# Statement of Need


<!-- PTAs in general. -->
The discovery of a nanohertz stochastic gravitational-wave background by pulsar timing array (PTA) collaborations [@nanograv2023pta] [@EPTA] [@ParkesPPTA2023] represents a landmark achievement in gravitational wave astronomy. PTAs monitor the precise arrival times of radio pulses from a collection of millisecond pulsars distributed across the sky. By measuring the spatial correlation of the variations between different pulsars - the characteristic Hellings-Downs curve [@hellings1983gravitational] - PTAs can detect gravitational-waves in a frequency band inaccessible to ground-based interferometers.



<!-- Traditional analyses-->
Traditional PTA data-analysis methods operate in the frequency domain. The various noise processes are treated as Gaussian stationary processes, characterised by their power spectral densities. These noise sources generally fall into two categories; uncorrelated white noise and time-correlated red noise. White noise sources include measurement noise from telescope receivers, while red noise components include pulsar spin noise (intrinsic to the neutron star) and dispersion measure (DM) variations (from electron density fluctuations in the interstellar medium). The GWB signal itself is also modeled as a red noise process with a characteristic power-law PSD, distinguished from the other noise components by its specific spatial correlation, the Hellings-Downs pattern. This frequncy domain modelling is combined with standard Bayesian inference methods [@2009MNRAS.395.1005V], such Markov Chain Monte Carlo (MCMC) algorithms for parameter estimation and model selection.



<!-- State space methods-->
State-space methods provide a novel, powerful and complementary framework for PTA analysis. The approach offers a time-domain realization of the Gaussian processes framework, allowing for a computationally efficient alternative to traditional frequency-domain modeling. Instead of relying on full matrix inversions of the covariance matrix, state-space methods model the temporal evolution of hidden states (such as pulsar spin fluctuations and gravitational-wave effects) using Kalman filtering [@kalman1960new] for recursive state estimation. This approach is computationally favorable, scaling linearly $\mathcal{O}\left(N\right)$ with the number of observations $N$,compared to the $\mathcal{O}\left(N^3\right)$\footnote{qualifier here for referee} cost associated with matrix inversion in traditional methods. State-space methods enable the explicit incorporation of physical knowledge about how different stochastic processes evolve over time directly into the model structure, naturally accommodating non-stationary processes. Additionally,the method tracks the actual, measured, time-ordered realization of intrinsic timing noise in each pulsar, rather than averaging over ensemble realizations, and can readily handle non-Gaussian statistics[@2024arXiv240500058U]



<!-- The gap filled by argus-->
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

We acknowledge support from the Australian Research Council Centre of Excellence for Gravitational Wave Discovery (OzGrav) under grant CE170100004. This work was performed on the OzSTAR national facility at Swinburne University of Technology. The OzSTAR program receives funding in part from the Astronomy National Collaborative Research Infrastructure Strategy (NCRIS) allocation provided by the Australian Government, and from the Victorian Higher Education State Investment Fund (VHESIF) provided by the Victorian Government. This work was supported by software support resources awarded under the Astronomy Data and Computing Services (ADACS) Merit Allocation Program. ADACS is funded from the Astronomy National Collaborative Research Infrastructure Strategy (NCRIS) allocation provided by the Australian Government and managed by Astronomy Australia Limited (AAL). We thank the International Pulsar Timing Array collaboration for providing mock data challenges that aided in the development and validation of this software. The development of `Argus` builds upon foundational work in state-space methods for pulsar timing analysis, and we acknowledge the broader PTA community for valuable feedback and contributions to the conceptual framework.

# Availability and Documentation

`Argus` is open source software released under the MIT license. The source code is available on GitHub at https://github.com/tomkimpson/Argus, with comprehensive documentation hosted at https://tomkimpson.github.io/Argus/. The package can be installed via PyPI and includes extensive examples, tutorials, and API documentation to facilitate adoption by the PTA community.

# References
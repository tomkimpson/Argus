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
    orcid: 0000-0002-6542-6032
    affiliation: "1, 2"
    corresponding: true
  - name: Nicholas O'Neil
    affiliation: "1, 2"
  - name: Andrew Melatos
    affiliation: "1, 2"
  - name: Patrick M. Meyers 
    affiliation: "3"
affiliations:
  - name: School of Physics, University of Melbourne, Parkville, VIC 3010, Australia
    index: 1
    ror: 01ej9dk98
  - name: "Australian Research Council Centre of Excellence for Gravitational Wave Discovery (OzGrav), Australia"
    index: 2
    ror: 02zp3yd51
  - name: "Theoretical Astrophysics Group, California Institute of Technology, Pasadena, CA 91125, USA"
    index: 3
    ror: 05dxps055
date: 29 September 2025
bibliography: paper.bib
---

# Summary
`Argus` is a high-performance Python package for detecting and characterizing gravitational waves in pulsar timing array (PTA) data. The package provides a complete Bayesian inference framework based on state-space models, using Kalman filtering for efficient likelihood evaluation. `Argus` leverages the JAX library [@jax2018github] for just-in-time (JIT) compilation, GPU acceleration, and end-to-end automatic differentiation, facilitating rapid Bayesian inference with gradient-based samplers. The state-space approach provides a computationally efficient alternative to traditional frequency-domain methods, offering linear scaling with observations, and natural handling of non-stationary processes in the complex, high-dimensional datasets characteristic of modern PTA experiments.



# Statement of Need


<!-- PTAs in general. -->
PTAs monitor the precise arrival times of radio pulses from a collection of millisecond pulsars distributed across the sky. By measuring the spatial correlation of the variations between different pulsars - the characteristic Hellings-Downs curve [@hellings1983gravitational] - PTAs can detect gravitational-waves in a frequency band inaccessible to ground-based interferometers. The discovery of a nanohertz stochastic gravitational-wave background by PTA collaborations [@nanograv2023pta; @EPTA; @ParkesPPTA2023] represents a landmark achievement in gravitational wave astronomy.



<!-- Traditional analyses-->
Traditional PTA data-analysis methods operate in the frequency domain. The various noise processes are treated as Gaussian stationary processes, characterised by their power spectral densities (PSDs). These noise sources generally fall into two categories: uncorrelated white noise and time-correlated red noise. White noise sources include measurement noise from telescope receivers, while red noise components include pulsar spin noise (intrinsic to the neutron star) and dispersion measure (DM) variations (from electron density fluctuations in the interstellar medium). The GWB signal itself is also modeled as a red noise process with a characteristic power-law PSD, distinguished from the other noise components by its specific spatial correlation, the Hellings-Downs correlation. This frequency domain modelling is the foundation for widely used packages such as ENTERPRISE [@enterprise2020] and TempoNest [@lentati2013hyper] and is typically combined with standard Bayesian inference methods [@2009MNRAS.395.1005V], such Markov Chain Monte Carlo (MCMC) algorithms for parameter estimation and model selection.


<!-- State space methods-->
State-space methods provide a novel, powerful and complementary framework for PTA data analysis. The approach offers a time-domain realization of the Gaussian processes framework, allowing for a computationally efficient alternative to traditional frequency-domain modeling. Instead of relying on full matrix inversions of the covariance matrix, state-space methods model the temporal evolution of hidden states (such as pulsar spin fluctuations and gravitational-wave effects) using Kalman filtering [@kalman1960new] for recursive state estimation. This approach is computationally favorable, scaling linearly $\mathcal{O}\left(N\right)$ with the number of observations $N$, compared to the $\mathcal{O}\left(N^3\right)$ cost associated with matrix inversion in traditional methods. State-space methods enable the explicit incorporation of physical knowledge about how different stochastic processes evolve over time directly into the model structure, naturally accommodating non-stationary processes. Additionally,the method tracks the actual, measured, time-ordered realization of intrinsic timing noise in each pulsar, rather than averaging over ensemble realizations, and can readily handle non-Gaussian statistics[@2024arXiv240500058U]


<!-- The gap filled by argus-->
Despite their theoretical advantages, state-space methods have seen limited adoption in PTA research, primarily due to the lack of accessible, high-performance implementations. `Argus` provides a modern, production-ready implementation of state-space methods for gravitational-wave detection in PTA data. Argus leverages JAX's just-in-time compilation, automatic differentiation, and GPU acceleration capabilities to handle the computational demands of Bayesian inference at PTA scales. Argus consolidates and formalises the state-space methodology applied in prior work [@kimpson2024a; @kimpson2024b; @kimpson2025c], transforming proof-of-concept implementations into a robust, production-ready tool. 


# Relation to Existing Work
Frequency-domain PTA packages such as ENTERPRISE and TempoNest represent timing noise and GW signals as Gaussian processes specified by power spectra, with inference built on dense covariance operations that in general scale as $\mathcal{O}(N^3)$ (although computational approximations exist that mitigate this naïve cost; @Haasteren2014). In contrast, `Argus` adopts a time-domain, state-space formulation in which latent variables describe the pulsar rotational states and other stochastic processes (e.g., DM variations), and the likelihood is evaluated via a Kalman filter, with $\mathcal{O}\left(N\right)$ complexity [@kalman1960new]. The Kalman filter tracks the actual, measured, time-ordered, random realisation of the intrinsic, achromatic timing noise in every PTA pulsar, and disentangles the GW-induced modulations, rather than averaging over an ensemble of admissible noise realisations through a PSD fit (e.g., @Goncharov2021). Prior PTA state-space demonstrations established feasibility on mock datasets [@kimpson2024a; @kimpson2024b], but remained research prototypes. `Argus` consolidates this methodology into a production-ready JAX implementation with JIT/GPU acceleration and end-to-end autodiff, enabling gradient-based samplers. As such, the package is complementary to ENTERPRISE/TempoNest: it offers an independent cross-check with different numerical/systematic failure modes, while retaining parity in astrophysical content (white/red noise, DM, and a GWB with Hellings–Downs correlations). The same state-space machinery also applies to deterministic sources (e.g., individual SMBHBs), combining Kalman filtering with Bayesian evidence calculations for model selection [@2025MNRAS.536.1489M; @Ashton2022; @vanHaasteren2025], and will be extended in Argus (see Future Directions).



# Functionality
`Argus` is built on JAX, enabling high-throughput computation on CPUs/GPUs/TPUs with JIT compilation and end-to-end automatic differentiation. Its core deliverable is a JAX-jittable log-likelihood for PTA datasets,making it directly suitable for gradient-based Bayesian inference.

**Core functionality**

- **State-space model construction** Stochastic processes like pulsar-intrinsic red noise (modeled as an Ornstein-Uhlenbeck process) and the stochastic gravitational-wave background (GWB) are specified in the time domain as linear stochastic differential equations (SDEs). The package compiles these into a single state-space model, which naturally separates process noise (e.g., physical spin wandering and GWB driving) from measurement noise (e.g., white noise EFAC/EQUAD terms). The characteristic Hellings-Downs spatial correlations for the GWB are implemented by correlating the driving noise processes across all pulsars within the state-space construction. The SDE parameterization yields a Markovian state evolution, naturally accommodating non-stationary or piecewise-stationary behavior in real PTA datasets.

- **Kalman filter likelihood evaluation.** The core of the package is a highly-optimized Kalman filter [@kalman1960new], which evaluates the likelihood of the time-of-arrival data in the time domain. This approach achieves a computational complexity that scales linearly with the number of observations, $\mathcal{O}(N)$.

- **Seamless Sampler Integration.** The log-likelihood and its gradients (provided by JAX's autodiff capabilities)  integrate directly with JAX-native samplers such as those in `numpyro` [@numpyro] or `blackjax`[@blackjax]. This enables the use of efficient gradient-based algorithms like Hamiltonian Monte Carlo (HMC), which can accelerate convergence in high-dimensional parameter spaces.

- **Standardized Data Input**  `Argus`  ingests pulsar timing data through an interface with libstempo via `libstempo` [@vallisneri2020libstempo], ensuring compatibility with standard PTA data formats and analysis workflows.


An example application of `Argus` to a mock dataset from the second IPTA mock data challenge [@2018arXiv181010527H] is shown in Figure \ref{fig:corner}. The reproducible analysis script for this figure is available in the software repository.


![Corner plot showing posterior distributions from Bayesian parameter estimation using Argus on IPTA mock dataset. The plot displays marginal and joint posterior distributions for key noise model parameters including logarithmic amplitudes of various stochastic processes (log₁₀ ρ parameters), timing model error factors (EFAC), and additional white noise terms (EQUAD). The well-constrained posteriors demonstrate the effectiveness of the state-space Kalman filtering approach for robust parameter estimation in pulsar timing array analysis.\label{fig:corner}](images/example_corner_plot.png)





# Future Directions

Several extensions are planned for `Argus` to expand its capabilities and applications:

- **Model selection**: Implementation of Bayes factor calculations to enable robust model comparison and selection between different gravitational wave models, noise models, and signal hypotheses
- **Expanded Physical Models**: Integration of (e.g.) time-variable dispersion measure (DM) effects and deterministic signals from continuous gravitational waves (e.g., from individual supermassive black hole binaries).
- **Advanced Noise Modelling**: Implementation of additional stochastic processes, including chromatic noise, system-dependent effects, and non-Gaussian noise components.
- **Enhanced Modularity**: Development of a user-friendly interface allowing researchers to easily specify custom state-space models and noise components.
- **Performance and Scalability**: Optimization for next-generation PTAs such as the SKA pulsar timing array, which will monitor hundreds of pulsars. 
- **Community integration**: Development of interfaces with other major PTA analysis pipelines such as ENTERPRISE [@enterprise2020] to facilitate cross-verification of results.


While these extensions represent exciting future directions, we release `Argus` in its current form because it successfully addresses the core challenge of Bayesian parameter estimation for pulsar timing array analysis. The package provides a robust, high-performance foundation for gravitational wave detection research and demonstrates significant computational advantages over traditional approaches. We encourage contributions from the broader PTA community to help implement these extensions and further advance the capabilities of state-space methods in gravitational wave astronomy. The modular design of `Argus` facilitates these extensions while maintaining backward compatibility and computational efficiency.

# Acknowledgements

We acknowledge support from the Australian Research Council Centre of Excellence for Gravitational Wave Discovery (OzGrav) under grant CE170100004. This work was performed on the OzSTAR national facility at Swinburne University of Technology. The OzSTAR program receives funding in part from the Astronomy National Collaborative Research Infrastructure Strategy (NCRIS) allocation provided by the Australian Government, and from the Victorian Higher Education State Investment Fund (VHESIF) provided by the Victorian Government. This work was supported by software support resources awarded under the Astronomy Data and Computing Services (ADACS) Merit Allocation Program. ADACS is funded from the Astronomy National Collaborative Research Infrastructure Strategy (NCRIS) allocation provided by the Australian Government and managed by Astronomy Australia Limited (AAL). We thank the International Pulsar Timing Array collaboration for providing mock data challenges that aided in the development and validation of this software. The development of `Argus` builds upon foundational work in state-space methods for pulsar timing analysis, and we acknowledge the broader PTA community for valuable feedback and contributions to the conceptual framework.

# References
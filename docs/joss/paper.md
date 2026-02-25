---
title: 'Argus: JAX state-space filtering for gravitational wave detection with a pulsar timing array'
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
  - name: Nicholas O'Neill
    affiliation: "1, 2"
  - name: Patrick M. Meyers
    affiliation: "3"
  - name: Andrew Melatos
    affiliation: "1, 2"
affiliations:
  - name: School of Physics, University of Melbourne, Parkville, VIC 3010, Australia
    index: 1
    ror: 01ej9dk98
  - name: "Australian Research Council Centre of Excellence for Gravitational Wave Discovery (OzGrav),Parkville, VIC 3010, Australia"
    index: 2
    ror: 02zp3yd51
  - name: "Theoretical Astrophysics Group, California Institute of Technology, Pasadena, CA 91125, USA"
    index: 3
    ror: 05dxps055
date: 13 October 2025
bibliography: paper.bib
draft: false
---

# Summary
`Argus` is a high-performance Python package for detecting and characterizing nanohertz gravitational waves in pulsar timing array (PTA) data. The package provides a complete Bayesian inference framework based on state-space models, using Kalman filtering for efficient likelihood evaluation. `Argus` leverages the JAX library [@jax2018github] for just-in-time (JIT) compilation, GPU acceleration, and end-to-end automatic differentiation, facilitating rapid Bayesian inference with gradient-based samplers. The state-space approach provides a computationally efficient alternative to traditional frequency-domain methods, offering linear scaling with the number of pulse times-of-arrival, and natural handling of non-stationary processes.


# Statement of Need

PTAs monitor the precise arrival times of radio pulses from a collection of millisecond pulsars distributed across the sky. By measuring the correlated variations in these pulse arrival times, PTAs are sensitive to gravitational waves in a frequency band inaccessible to ground-based interferometers. The possible discovery of a nanohertz stochastic gravitational-wave background (GWB) by PTA collaborations [@nanograv2023pta; @EPTA; @ParkesPPTA2023] through measuring the Hellings-Downs spatial correlation [@hellings1983gravitational] represents a landmark achievement in gravitational-wave astronomy.

State-space methods provide a powerful and complementary framework for PTA data analysis, offering an alternative computational structure to traditional frequency-domain modeling. Instead of relying on full covariance matrix inversions, state-space methods model the temporal evolution of hidden states (such as pulsar spin fluctuations and gravitational-wave effects) using Kalman filtering [@kalman1960new] for recursive state estimation. This approach exhibits linear scaling $\mathcal{O}(N)$ with the number of observations $N$, naturally accommodates non-stationary processes, tracks the actual time-ordered realization of intrinsic timing noise in each pulsar rather than averaging over ensemble realizations, and can readily handle non-Gaussian statistics [@2024arXiv240500058U].

Despite these theoretical advantages, state-space methods have seen limited adoption in PTA research, partly due to their recency, and partly due to the lack of accessible, high-performance implementations. `Argus` addresses this gap, providing a modern, science-ready implementation of state-space methods for gravitational-wave detection in PTA data. The package leverages JAX's just-in-time compilation, automatic differentiation, and GPU acceleration to handle the computational demands of Bayesian inference at PTA scales, consolidating and formalising the methodology developed in prior work [@kimpson2024a; @kimpson2024b; @kimpson2025c].


# State of the Field

Traditional PTA data-analysis methods operate in the frequency domain, treating noise processes as Gaussian stationary signals characterized by their power spectral densities. Noise sources generally fall into two categories: uncorrelated white noise (measurement noise from telescope receivers) and time-correlated red noise (pulsar spin noise, dispersion measure variations from electron density fluctuations in the interstellar medium). The GWB signal is modeled as a red noise process with a characteristic power-law spectrum [@Goncharov2021], distinguished by its Hellings-Downs spatial correlation. This frequency-domain framework is the foundation for widely used packages such as ENTERPRISE [@enterprise2020] and TempoNest [@lentati2013hyper], typically combined with Bayesian inference methods such as Markov Chain Monte Carlo for parameter estimation [@2009MNRAS.395.1005V].

While these tools have been central to recent GWB discoveries, they carry inherent limitations. They assume stationary Gaussian processes, require $\mathcal{O}(N^3)$ covariance matrix inversions, and characterize timing noise by fitting a power spectral density -- effectively averaging over an ensemble of admissible noise realizations rather than tracking the specific noise realization in the data (e.g., @Goncharov2021). They also lack built-in support for automatic differentiation and hardware acceleration via GPU/TPU. Prior PTA state-space prototypes established feasibility on mock datasets [@kimpson2024a; @kimpson2024b], but no production-ready implementation was available.

`Argus` fills this gap as the first production-ready, time-domain state-space PTA package built on JAX. It is complementary to ENTERPRISE and TempoNest: the package offers an independent cross-check of GWB analyses with fundamentally different numerical and systematic properties, while retaining parity in astrophysical content (white and red noise, dispersion measure variations, and a GWB with Hellings-Downs correlations).


# Software Design

`Argus` is built on JAX, providing a JAX-jittable log-likelihood for PTA datasets that is directly suitable for gradient-based Bayesian inference on CPUs, GPUs, and TPUs. The software design reflects several deliberate architectural choices.

**JAX as computational backend.** JAX was chosen over alternatives (e.g. PyTorch, pure NumPy) for its functional programming paradigm, which naturally matches the Kalman filter's recursive structure. JAX's composable transformations -- `jit` for compilation, `vmap` for vectorization, and `grad` for automatic differentiation -- enable efficient likelihood evaluation and gradient computation. The XLA compiler provides transparent hardware acceleration across CPU, GPU, and TPU backends.

**Kalman filter as core algorithm.** The Kalman filter [@kalman1960new] evaluates the likelihood with $\mathcal{O}(N)$ complexity, avoiding the $\mathcal{O}(N^3)$ cost of full covariance matrix inversions. It tracks the actual, measured, time-ordered realization of timing noise, enabling separation of gravitational-wave-induced perturbations from intrinsic noise.

**SDE-based process specification.** Stochastic processes -- pulsar-intrinsic red noise (modeled as an Ornstein-Uhlenbeck process), dispersion measure variations, and the GWB -- are specified as linear stochastic differential equations in continuous time, discretized at observation times. This formulation naturally handles irregular sampling cadences and allows new physics to be incorporated by augmenting the state vector, providing a modular framework for extending the model. Hellings-Downs spatial correlations are implemented through the covariance structure coupling processes across pulsars.

**Sampler-agnostic likelihood.** The log-likelihood is a pure JAX function exposing gradients via autodiff, compatible with any JAX-native sampler such as `numpyro` [@phan2019composable] or `blackjax` [@cabezas2024blackjax]. This enables efficient gradient-based algorithms like the No-U-Turn Sampler (NUTS) [@hoffman2014no].

`Argus` ingests pulsar timing data through `libstempo` [@vallisneri2020libstempo], ensuring compatibility with standard PTA data formats.

![Corner plot showing posterior distributions from Bayesian parameter estimation using Argus on the second IPTA mock data challenge [@2018arXiv181010527H]. The first two parameters $h_{\rm a}$, $\gamma_{\rm a}$ describe the amplitude and turnover frequency of the GWB. The middle two parameters $\sigma_{\rm p,0}$, $\gamma_{\rm p,0}$ characterise the red timing noise for an arbitrary pulsar in the array (indexed by 0). The final two parameters, EFAC, EQUAD are the standard white measurement noise parameters for the arbitrary pulsar. The posteriors were obtained using the No-U-Turn Sampler (NUTS) [@hoffman2014no] from `numpyro` [@phan2019composable], leveraging Argus's JAX-native log-likelihood and automatic differentiation for gradient-based sampling. The unimodal marginalised posteriors demonstrate the effectiveness of the state-space Kalman filtering approach for parameter estimation in pulsar timing array analysis.\label{fig:corner}](images/example_corner_plot.png)


# Research Impact Statement

The state-space methodology implemented in `Argus` has been developed and validated through three peer-reviewed publications in Monthly Notices of the Royal Astronomical Society: @kimpson2024a established the Kalman filtering framework for continuous gravitational-wave tracking with a PTA; @kimpson2024b extended this to include pulsar-term contributions; and @kimpson2025c developed the algorithm for detecting the stochastic gravitational-wave background, including Hellings-Downs correlations. `Argus` has been successfully validated on the second International Pulsar Timing Array (IPTA) mock data challenge [@2018arXiv181010527H], an internationally recognized community benchmark.

PTA collaborations worldwide -- NANOGrav, the European PTA (EPTA), the Parkes PTA (PPTA), and the MeerKAT PTA [@2025MNRAS.536.1489M] -- are actively seeking independent analysis pipelines for cross-validation of GWB detection claims. `Argus` provides a fundamentally different methodology (time-domain state-space vs. frequency-domain) for this critical cross-check, with different numerical and systematic failure modes. Next-generation PTA datasets from the Square Kilometre Array will particularly benefit from the $\mathcal{O}(N)$ scaling.

`Argus` has been developed across the University of Melbourne/OzGrav and the California Institute of Technology, with professional software engineering support from the Astronomy Data and Computing Services (ADACS). The package includes comprehensive documentation, an automated test suite, and compatibility with standard PTA data formats, supporting community adoption and contribution.


# AI Usage Disclosure

Generative AI tools were used during the development of `Argus` to assist with software infrastructure tasks, including setting up the documentation website, configuring automated CI/CD workflow testing, and scaffolding boilerplate configuration files. All scientific methodology, algorithm design, core implementation, and numerical validation were carried out by the authors. We affirm that human team members thoroughly reviewed, modified, and validated all AI-generated content while making primary architectural and design decisions.


# Acknowledgements

We acknowledge support from the Australian Research Council Centre of Excellence for Gravitational Wave Discovery (OzGrav) under grant CE170100004. NJO is the recipient of a Melbourne Research Scholarship. This work was performed on the OzSTAR national facility at Swinburne University of Technology. The OzSTAR program receives funding in part from the Astronomy National Collaborative Research Infrastructure Strategy (NCRIS) allocation provided by the Australian Government, and from the Victorian Higher Education State Investment Fund (VHESIF) provided by the Victorian Government. This work was supported by software support resources awarded under the Astronomy Data and Computing Services (ADACS) Merit Allocation Program. ADACS is funded from the Astronomy National Collaborative Research Infrastructure Strategy (NCRIS) allocation provided by the Australian Government and managed by Astronomy Australia Limited (AAL). We thank the International Pulsar Timing Array collaboration for providing mock data challenges that aided in the development and validation of this software.

# References

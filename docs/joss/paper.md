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

Pulsar timing arrays (PTAs) are among the most sensitive detectors for nanohertz gravitational waves, monitoring the precise arrival times of radio pulses from millisecond pulsars across the sky. The recent discovery of a nanohertz stochastic gravitational-wave background by PTA collaborations [@2023ApJ...951L...8A; @2023arXiv230616214A] represents a landmark achievement in gravitational wave astronomy. However, the detection and characterization of gravitational wave signals in PTA data presents significant computational challenges due to complex inter-pulsar correlations, stochastic signal characteristics, and high-dimensional parameter spaces in Bayesian inference.

Traditional PTA analysis methods rely primarily on frequency-domain techniques that measure spatial cross-correlation patterns, particularly the Hellings-Downs correlation [@hellings1983gravitational]. These approaches work by estimating power spectral densities (PSDs) of various noise processes and averaging over ensemble realizations of timing noise. While successful, these methods often struggle with the computational demands of high-dimensional Bayesian inference and face limitations when modeling non-stationary processes.

State-space methods provide a powerful complementary framework for PTA analysis. Instead of operating in the frequency domain with PSDs, they model the temporal evolution of hidden states (such as pulsar spin fluctuations and gravitational wave effects) using Kalman filtering [@kalman1960new] for recursive state estimation. This approach enables the explicit incorporation of physical knowledge about how different processes evolve over time—for example, how gravitational wave signals change amplitude and phase, or how pulsar spin noise behaves—directly into the model structure rather than treating these as generic stochastic processes. Crucially, state-space methods track the actual measured time-ordered realization of intrinsic timing noise in each pulsar, rather than averaging over ensemble realizations. This time-domain approach offers computationally favorable likelihood calculations that scale linearly with observations while naturally handling non-stationary processes.

Despite their theoretical advantages, state-space methods have seen limited adoption in PTA research, primarily due to the lack of accessible, high-performance implementations. `Argus` fills a critical gap in the PTA analysis ecosystem by providing a modern, high-performance implementation of state-space methods specifically designed for gravitational wave detection. The package matures and formalizes the state-space methodology that was successfully applied in prior research and mock data challenges [@kimpson2024a; @kimpson2024b; @kimpson2025], transforming proof-of-concept implementations into a robust, production-ready tool. 






# Statement of need

Pulsar timing arrays (PTAs) are among the most sensitive detectors for nanohertz gravitational waves, monitoring the precise arrival times of radio pulses from millisecond pulsars across the sky. The detection and characterization of gravitational wave signals in PTA data presents significant computational challenges due to the complex correlations between pulsars, the stochastic nature of the signals, and the high-dimensional parameter spaces involved in Bayesian inference. 

The recent discovery of a nanohertz stochastic gravitational-wave background (GWB) by PTA collaborations [@2023ApJ...951L...8A; @2023arXiv230616214A] is a landmark result. 

The primary analysis technique relies on measuring the Hellings-Downs spatial cross-correlation pattern in timing data [@hellings1983gravitational]. 


As with any major discovery, independent verification through alternative and complementary methods is crucial.

State-space methods provide a powerful alternative framework. Instead of operating in the frequency domain, they model the temporal evolution of the system's hidden states (e.g., pulsar spin fluctuations and GW effects) and use a Kalman filter [@kalman1960new] for recursive state estimation. This approach offers a favourable time-domain likelihood calculation that scales linearly with the number of observations.



Pulsar timing array analysis requires sophisticated statistical methods to extract weak gravitational wave signals from noisy timing residuals. Traditional approaches often struggle with the computational demands of high-dimensional Bayesian inference and the complex correlation structures inherent in gravitational wave signatures, such as the Hellings-Downs correlation pattern [@hellings1983gravitational]. 



`Argus` fills a critical gap in the PTA analysis ecosystem by providing a modern, high-performance implementation of state-space methods specifically designed for gravitational wave detection. The package leverages JAX's just-in-time compilation, automatic differentiation, and GPU acceleration capabilities to handle the computational intensity of PTA Bayesian inference. The mathematical framework separates different physical processes (gravitational waves, pulsar spin noise, and timing model parameters) into manageable blocks, exploiting sparsity for computational efficiency while maintaining full correlations where physically necessary.


`Argus` fills a critical gap in the PTA analysis ecosystem by providing a modern, high-performance, and open-source implementation of these methods. By leveraging JAX, it overcomes the significant computational challenges of high-dimensional Bayesian inference in PTA data analysis. The package has already been successfully applied to mock data challenges and research projects [@kimpson2024a; @kimpson2024b; @kimpson2025], demonstrating its readiness for cutting-edge gravitational wave science.



The package has already been successfully applied to several research projects [@kimpson2024a; @kimpson2024b; @kimpson2025], demonstrating its utility for both mock data challenges (such as the International Pulsar Timing Array Mock Data Challenge) and ongoing research at major gravitational wave research institutions. `Argus` is designed to serve both researchers conducting cutting-edge PTA science and students learning gravitational wave data analysis techniques.




The state-space formulation provides a natural framework for modeling the temporal evolution of stochastic processes in PTAs, enabling efficient recursive estimation through Kalman filtering [@kalman1960new].



SMBHBs that emit GWs with sufficiently large amplitudes may be resolvable individually with PTAs \citep{Jenet2004,Sesana2010,Yardley2010,Babak2012,2013CQGra..30v4004E,Zhu2014PPTA,Zhu10, Babak2016,Zhupulsarterms,2023arXiv230616226A,Arzoumanian2023}. State-space algorithms are a promising and complementary approach to standard PTA cross-correlation analyses for GWs from individual SMBHBs \citep{KimpsonPTA1,KimpsonPTA2}. PTA state-space algorithms use a Kalman filter \citep{Kalman1,Meyers2021,Melatos2023} to track the intrinsic rotational state of the pulsars in the array. The optimal estimate of the state-space evolution provided by the Kalman filter is combined with a Bayesian nested sampler \citep{Skilling,Ashton2022} to infer the time invariant parameters of a single-source GW and calculate the Bayesian evidence for models with and without the GW present. State-space algorithms disentangle GW-induced modulations in every PTA pulsar's spin frequency from other fluctuations by tracking the actual, measured, time-ordered, random realisation of the intrinsic, achromatic timing noise in every PTA pulsar \citep[e.g.][]{Shannon2010,Lasky2015,Caballero2016,Goncharov2021} by harnessing the adaptive gain of the Kalman filter \citep{Kalman1,zarchan2000fundamentals}. In contrast, traditional PTA algorithms average over the ensemble of admissible timing noise realisations when inferring the noise power spectral density (PSD). \newline 


Argus matures and formalizes the state-space methodology that was successfully applied in prior research and mock data challenges [@kimpson2024a; @kimpson2024b; @kimpson2025], demonstrating its readiness for cutting-edge gravitational wave science.


`Argus` has been successfully applied to several research projects, with results published in peer-reviewed literature [@kimpson2024a; @kimpson2024b; @kimpson2025]. These applications demonstrate the package's effectiveness for: T



he state-space approach implemented in `Argus` offers several advantages over traditional frequency-domain methods, particularly in handling non-stationary processes and incorporating physical prior information about signal evolution.


- **Mock data analysis**: Successful application to International Pulsar Timing Array Mock Data Challenges
- **Method validation**: Comparison with traditional PTA analysis approaches
- **Parameter estimation**: Robust inference of gravitational wave and pulsar noise parameters
- **Model comparison**: Bayesian evidence calculation for model selection

`Argus` includes comprehensive unit tests covering core functionality including Kalman filter operations, state-space model construction, and Bayesian inference workflows. The testing suite validates numerical accuracy against analytical solutions where available and ensures compatibility with both CPU and GPU execution environments. Continuous integration workflows maintain code quality and compatibility across different Python versions and hardware configurations.


## State-Space Framework

`Argus` implements a comprehensive state-space formulation for PTA analysis, representing the system through state evolution and observation equations:

$$x_t = F x_{t-1} + w_t$$
$$y_t = H x_t + v_t$$

where $x_t$ represents the hidden state vector containing gravitational wave signals, pulsar spin parameters, and timing model corrections; $y_t$ represents the observed timing residuals; and $w_t$, $v_t$ are process and observation noise terms, respectively.

The state vector is organized into three physically motivated blocks: gravitational wave effects, pulsar spin-down variations, and timing model parameters. This block structure enables efficient computation by exploiting the sparsity pattern in the system matrices while preserving the essential correlations, such as the Hellings-Downs pattern for gravitational waves [@hellings1983gravitational].



## Pulsar Spin Evolution

The intrinsic spin frequency of the $n$-th pulsar, $f_{\rm p}^{(n)}(t)$, is a hidden state variable. Its evolution is modeled as a mean-reverting Ornstein-Uhlenbeck process, a type of Gauss-Markov process described by the stochastic differential equation:
$$
df_{\rm p}^{(n)}(t) = -\frac{1}{\tau_n} \left( f_{\rm p}^{(n)}(t) - \mu_n \right) dt + \sqrt{\frac{2\sigma_n^2}{\tau_n}} dW_n(t)
$$
where $\mu_n$, $\tau_n$, and $\sigma_n^2$ are the mean, relaxation time, and variance of the process, respectively, and $dW_n(t)$ is a Wiener process.

## Measurement Equation and GW Background

The frequency measured at Earth, $f_{\rm m}^{(n)}(t)$, is related to the intrinsic pulsar frequency through a measurement equation that includes contributions from the GW background. The key innovation is to also model the time-varying amplitude of the stochastic GW background at the $n$-th pulsar, $a^{(n)}(t)$, as a Gauss-Markov process. The measurement equation is:
$$
f_{\rm m}^{(n)}(t) = f_{\rm p}^{(n)}(t) + a^{(n)}(t) + \epsilon_k
$$
where $\epsilon_k$ represents instrumental white noise. The evolution of $a^{(n)}(t)$ is described by another stochastic differential equation, derived from GW theory and the Hellings-Downs correlation, which allows the Kalman filter to track its evolution as a hidden state.

The Kalman filter recursively estimates the hidden states ($f_{\rm p}^{(n)}(t_k)$ and $a^{(n)}(t_k)$) from the sequence of measurements $f_{\rm m}^{(n)}(t_k)$. This state estimation is combined with a Bayesian nested sampler to infer static parameters, such as the amplitude of the GW background ($A_{\rm GWB}$) and its spectral index ($\gamma_{\rm GWB}$).





# Research Impact and Applications

The primary application of this software is to serve as an independent analysis pipeline for the stochastic GW background detected by PTAs.

* **Verification:** It provides a vital cross-check for results obtained with standard Hellings-Downs analyses.
* **Computational Efficiency:** The Kalman filter is computationally fast, making this method attractive for rapid analysis, parameter space exploration, and processing the increasingly large datasets from PTAs.
* **Validation:** The algorithm has been successfully validated against astrophysically representative synthetic data, demonstrating its ability to accurately recover the injected parameters of a GW background.
* **Generalizability:** The state-space framework is flexible and can be extended to simultaneously search for multiple individual GW sources in addition to the stochastic background.




![Corner plot showing posterior distributions from Bayesian parameter estimation using Argus on a 2-pulsar mock dataset. The plot displays marginal and joint posterior distributions for key noise model parameters including logarithmic amplitudes of various stochastic processes (log₁₀ ρ parameters), timing model error factors (EFAC), and additional white noise terms (EQUAD). The well-constrained posteriors demonstrate the effectiveness of the state-space Kalman filtering approach for robust parameter estimation in pulsar timing array analysis.](images/corner_plot_run_example_run_2pulsars_smooth0.1_efac_equad.png)




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

- **Marginalisation over unwanted parameter**

While these extensions represent exciting future directions, we release `Argus` in its current form because it successfully addresses the core challenge of Bayesian parameter estimation for pulsar timing array analysis. The package provides a robust, high-performance foundation for gravitational wave detection research and demonstrates significant computational advantages over traditional approaches. We encourage contributions from the broader PTA community to help implement these extensions and further advance the capabilities of state-space methods in gravitational wave astronomy. The modular design of `Argus` facilitates these extensions while maintaining backward compatibility and computational efficiency.

# Acknowledgements

We acknowledge support from the Australian Research Council Centre of Excellence for Gravitational Wave Discovery (OzGrav) under grant CE170100004. This research made use of computational resources provided by the University of Melbourne. We thank the International Pulsar Timing Array collaboration for providing mock data challenges that aided in the development and validation of this software. The development of `Argus` builds upon foundational work in state-space methods for pulsar timing analysis, and we acknowledge the broader PTA community for valuable feedback and contributions to the conceptual framework.

# Availability and Documentation

`Argus` is open source software released under the MIT license. The source code is available on GitHub at https://github.com/tomkimpson/Argus, with comprehensive documentation hosted at https://argus-pta.readthedocs.io/. The package can be installed via PyPI and includes extensive examples, tutorials, and API documentation to facilitate adoption by the PTA community.

# References
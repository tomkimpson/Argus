# Bayesian Inference with the Kalman Filter

A key output of the Kalman filter is the **likelihood** of the observed data given the model parameters. This likelihood can be computed efficiently as a byproduct of the filtering process through the prediction-error decomposition. This makes the Kalman filter a natural building block for Bayesian parameter estimation.

By combining the Kalman filter likelihood with prior distributions on model parameters (e.g., gravitational wave background amplitude, noise parameters), we can use standard Bayesian inference methods to:

- Estimate posterior distributions for physical parameters of interest
- Compute Bayes factors for model comparison
- Quantify uncertainties in a principled way

---

## Implementation in JAX

Argus is written in [JAX](https://github.com/google/jax), a high-performance numerical computing library with automatic differentiation. This provides several key advantages:

- **GPU/TPU Acceleration:** JAX code can run on modern accelerators with minimal code changes, dramatically speeding up likelihood evaluations
- **Automatic Differentiation:** JAX provides automatic computation of gradients, enabling the use of gradient-based sampling methods like Hamiltonian Monte Carlo (HMC)
- **Advanced Sampling:** Argus integrates seamlessly with [NumPyro](https://num.pyro.ai/), which implements the No-U-Turn Sampler (NUTS), an efficient variant of HMC that can explore complex posterior distributions much more efficiently than traditional methods like Metropolis-Hastings
- **JIT Compilation:** JAX's just-in-time compilation optimizes the Kalman filter operations for maximum performance

This combination of efficient likelihood computation and gradient-based sampling makes it practical to perform full Bayesian inference on PTA datasets that would be computationally intractable with traditional methods.

---

## Further Reading

### Documentation

- [State-Space Methods](state_space.md) - Overview of the state-space framework and its advantages
- [Kalman Filter Mathematics](kalman_mathematics.md) - Detailed mathematical derivations for the Kalman filter
- [Bayesian Implementation](bayesian_implementation.md) - Advanced parameterization techniques for high-dimensional spaces
- [Getting Started](getting_started.md) - Run your first Bayesian PTA analysis

### Academic Papers

#### JAX and NumPyro

- [JAX Documentation](https://jax.readthedocs.io/) - High-performance numerical computing with automatic differentiation
- [NumPyro Documentation](https://num.pyro.ai/) - Probabilistic programming with NUTS sampling
- [Phan et al. (2019)](https://arxiv.org/abs/1912.11554) - "Composable Effects for Flexible and Accelerated Probabilistic Programming in NumPyro"

#### Hamiltonian Monte Carlo and NUTS

- [Betancourt (2017)](https://arxiv.org/abs/1701.02434) - "A Conceptual Introduction to Hamiltonian Monte Carlo"
- [Hoffman & Gelman (2014)](https://jmlr.org/papers/v15/hoffman14a.html) - "The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo"
- [Neal (2011)](https://arxiv.org/abs/1206.1901) - "MCMC Using Hamiltonian Dynamics"

#### PTA Bayesian Inference

- [arXiv:2409.14613](https://arxiv.org/abs/2409.14613) - State-space methods for PTA analysis
- [van Haasteren et al. (2009)](https://doi.org/10.1111/j.1365-2966.2009.14590.x) - "Placing limits on the stochastic gravitational-wave background using European Pulsar Timing Array data"
- [Lentati et al. (2013)](https://doi.org/10.1103/PhysRevD.87.104021) - "TEMPONEST: A Bayesian approach to pulsar timing analysis"

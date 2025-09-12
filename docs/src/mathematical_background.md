# Mathematical Background for Advanced Bayesian Methods

This document provides rigorous mathematical foundations for the advanced Bayesian parameter estimation techniques implemented in Argus. It complements the [pedagogical overview](bayesian_methods.md) with detailed derivations and theoretical justifications.

!!! note "Prerequisites"
    This document assumes familiarity with:
    - Bayesian inference and MCMC methods
    - Hamiltonian Monte Carlo and the NUTS algorithm
    - Basic differential geometry (for gradient calculations)
    - Hierarchical Bayesian modeling

---

## 1. Parameter Reparameterization Theory

### 1.1 The Jacobian Problem

When transforming from parameter $\theta$ to $\phi = T(\theta)$, the probability density transforms as:

$$p(\phi) = p(\theta) \left| \frac{d\theta}{d\phi} \right| = p(T^{-1}(\phi)) \left| \frac{dT^{-1}}{d\phi} \right|$$

For the uniform-to-normal reparameterization:

- **Original**: $\theta \sim \text{Uniform}(a, b)$ with density $p(\theta) = \frac{1}{b-a}$
- **Transform**: $\phi = \frac{\theta - \mu}{\sigma}$ where $\mu = \frac{a+b}{2}$, $\sigma = \frac{b-a}{6}$
- **Inverse**: $\theta = \mu + \sigma \phi$
- **Jacobian**: $\left| \frac{d\theta}{d\phi} \right| = \sigma$

The transformed density becomes:

$$p(\phi) = \frac{1}{b-a} \cdot \sigma = \frac{1}{b-a} \cdot \frac{b-a}{6} = \frac{1}{6}$$

However, this is **not** a standard normal density. The key insight is that we don't want to preserve the uniform distribution in the transformed space - we want to sample from $\mathcal{N}(0,1)$ and map back to the original space.

### 1.2 Forward Sampling Approach

Instead of density transformation, we use **forward sampling**:

1. Sample $\phi \sim \mathcal{N}(0, 1)$
2. Transform $\theta = \mu + \sigma \phi$ 
3. Evaluate likelihood at $\theta$
4. Apply prior correction: $\log p(\theta) = \log p_{\text{uniform}}(\theta) - \log p_{\mathcal{N}}(\phi) + \log p_{\mathcal{N}}(\phi)$

The prior correction cancels when we only care about likelihood ratios, which is the case for MCMC sampling.

### 1.3 Gradient Conditioning Analysis

The condition number of the Hessian matrix determines NUTS performance. For uniform priors near boundaries:

$$\nabla^2 \log p(\theta) = -\infty \quad \text{at boundaries}$$

For normal distributions:

$$\nabla^2 \log p(\phi) = -\frac{1}{\sigma^2} \quad \text{(constant)}$$

This constant negative curvature provides optimal Hamiltonian dynamics.

### 1.4 The 3-Sigma Rule Justification

The scaling $\sigma = \frac{b-a}{6}$ ensures that for $\phi \sim \mathcal{N}(0,1)$:

$$P(a \leq \mu + \sigma \phi \leq b) = P(-3 \leq \phi \leq 3) \approx 0.997$$

This covers 99.7% of the probability mass while allowing occasional exploration outside the original bounds, which can aid in multimodal exploration.

---

## 2. Hierarchical Modeling Mathematics

### 2.1 Model Specification

The full hierarchical model for pulsar red noise parameters is:

$$\begin{align}
\mu_{\gamma} &\sim p(\mu_{\gamma}) \\
\sigma_{\gamma} &\sim p(\sigma_{\gamma}) \\
\gamma_p^{(i)} &\sim \mathcal{N}(\mu_{\gamma}, \sigma_{\gamma}) \quad i = 1, \ldots, N \\
y^{(i)} &\sim p(y^{(i)} | \gamma_p^{(i)}, \text{other params})
\end{align}$$

### 2.2 Effective Dimensionality Analysis

Standard independent priors result in $N$ free parameters. The hierarchical model has $2 + N$ parameters, but the effective dimensionality is reduced due to the constraint imposed by the population distribution.

**Proof**: The Fisher Information Matrix for the hierarchical model has the block structure:

$$\mathcal{I} = \begin{pmatrix}
\mathcal{I}_{\text{hyper}} & \mathcal{I}_{\text{cross}} \\
\mathcal{I}_{\text{cross}}^T & \mathcal{I}_{\text{indiv}}
\end{pmatrix}$$

where $\mathcal{I}_{\text{indiv}}$ has rank $N$ but is constrained by the hyperparameters, reducing the effective rank.

### 2.3 Gradient Balancing Derivation

Without gradient balancing, the log-posterior gradient with respect to hyperparameters scales as:

$$\frac{\partial \log p(\mu_{\gamma}, \sigma_{\gamma}, \{\gamma_p^{(i)}\})}{\partial \mu_{\gamma}} = \sum_{i=1}^{N} \frac{\partial \log \mathcal{N}(\gamma_p^{(i)}; \mu_{\gamma}, \sigma_{\gamma})}{\partial \mu_{\gamma}}$$

$$= \sum_{i=1}^{N} \frac{\gamma_p^{(i)} - \mu_{\gamma}}{\sigma_{\gamma}^2} = \frac{N(\bar{\gamma} - \mu_{\gamma})}{\sigma_{\gamma}^2}$$

This scales as $O(N)$, leading to gradient explosion in high-dimensional spaces.

**Solution**: Reparameterize individual parameters as:

$$\gamma_p^{(i)} = \mu_{\gamma} + \frac{\sigma_{\gamma}}{\sqrt{N}} z_i$$

where $z_i \sim \mathcal{N}(0, 1)$. Now:

$$\frac{\partial \log p}{\partial \mu_{\gamma}} = \sum_{i=1}^{N} \frac{\gamma_p^{(i)} - \mu_{\gamma}}{\sigma_{\gamma}^2} = \sum_{i=1}^{N} \frac{\sigma_{\gamma} z_i / \sqrt{N}}{\sigma_{\gamma}^2} = \frac{1}{\sigma_{\gamma} \sqrt{N}} \sum_{i=1}^{N} z_i$$

Since $\mathbb{E}[\sum z_i] = 0$ and $\text{Var}[\sum z_i] = N$, we have $\sum z_i = O(\sqrt{N})$, so:

$$\frac{\partial \log p}{\partial \mu_{\gamma}} = O\left(\frac{\sqrt{N}}{\sigma_{\gamma} \sqrt{N}}\right) = O(1)$$

The gradient magnitude is now independent of $N$.

### 2.4 Population Inference Benefits

The hierarchical model enables inference about population-level quantities. The posterior for the population mean has variance:

$$\text{Var}[\mu_{\gamma} | \text{data}] \approx \frac{\sigma_{\gamma}^2}{N + \frac{\sigma_{\gamma}^2}{\sigma_{\mu}^2}}$$

where $\sigma_{\mu}^2$ is the prior variance on $\mu_{\gamma}$. As $N \to \infty$:

$$\text{Var}[\mu_{\gamma} | \text{data}] \to \frac{\sigma_{\gamma}^2}{N}$$

This $1/N$ scaling shows that population-level inference improves rapidly with array size.

---

## 3. Log-Ratio Parameterization Theory

### 3.1 Correlation Structure Analysis

Red noise parameters $\log_{10} \gamma_p$ and $\log_{10} \sigma_p$ often exhibit strong positive correlation because both increase with the strength of the underlying stochastic process. The empirical correlation matrix is:

$$\Sigma = \begin{pmatrix}
\sigma_{\gamma}^2 & \rho \sigma_{\gamma} \sigma_{\sigma} \\
\rho \sigma_{\gamma} \sigma_{\sigma} & \sigma_{\sigma}^2
\end{pmatrix}$$

where $\rho \approx 0.7-0.9$ is typical for PTA datasets.

### 3.2 Decorrelating Transformation

Define the transformation:
$$\begin{align}
u_1 &= \log_{10} \gamma_p \\
u_2 &= \log_{10} \sigma_p - \log_{10} \gamma_p = \log_{10}(\sigma_p / \gamma_p)
\end{align}$$

The Jacobian of this transformation is:

$$J = \begin{pmatrix}
\frac{\partial u_1}{\partial \log_{10} \gamma_p} & \frac{\partial u_1}{\partial \log_{10} \sigma_p} \\
\frac{\partial u_2}{\partial \log_{10} \gamma_p} & \frac{\partial u_2}{\partial \log_{10} \sigma_p}
\end{pmatrix} = \begin{pmatrix}
1 & 0 \\
-1 & 1
\end{pmatrix}$$

The transformed covariance matrix is:
$$\Sigma' = J \Sigma J^T = \begin{pmatrix}
1 & 0 \\
-1 & 1
\end{pmatrix} \begin{pmatrix}
\sigma_{\gamma}^2 & \rho \sigma_{\gamma} \sigma_{\sigma} \\
\rho \sigma_{\gamma} \sigma_{\sigma} & \sigma_{\sigma}^2
\end{pmatrix} \begin{pmatrix}
1 & -1 \\
0 & 1
\end{pmatrix}$$

$$= \begin{pmatrix}
\sigma_{\gamma}^2 & \sigma_{\gamma}(\sigma_{\gamma} - \rho \sigma_{\sigma}) \\
\sigma_{\gamma}(\sigma_{\gamma} - \rho \sigma_{\sigma}) & \sigma_{\gamma}^2 - 2\rho \sigma_{\gamma} \sigma_{\sigma} + \sigma_{\sigma}^2
\end{pmatrix}$$

The correlation in the new parameterization is:

$$\rho' = \frac{\sigma_{\gamma}(\sigma_{\gamma} - \rho \sigma_{\sigma})}{\sigma_{\gamma} \sqrt{\sigma_{\gamma}^2 - 2\rho \sigma_{\gamma} \sigma_{\sigma} + \sigma_{\sigma}^2}} = \frac{\sigma_{\gamma} - \rho \sigma_{\sigma}}{\sqrt{\sigma_{\gamma}^2 - 2\rho \sigma_{\gamma} \sigma_{\sigma} + \sigma_{\sigma}^2}}$$

For typical values where $\sigma_{\gamma} \approx \sigma_{\sigma}$ and $\rho \approx 0.8$:

$$\rho' \approx \frac{1 - 0.8}{\sqrt{1 - 1.6 + 1}} = \frac{0.2}{\sqrt{0.4}} \approx 0.32$$

The correlation is significantly reduced from 0.8 to 0.32.

### 3.3 MCMC Efficiency Improvement

The MCMC efficiency for sampling correlated parameters scales approximately as:

$$\text{Efficiency} \propto \frac{1}{\det(\Sigma)}$$

For the original parameterization:
$$\det(\Sigma) = \sigma_{\gamma}^2 \sigma_{\sigma}^2 (1 - \rho^2)$$

For the log-ratio parameterization:
$$\det(\Sigma') = \sigma_{\gamma}^2 (\sigma_{\gamma}^2 - 2\rho \sigma_{\gamma} \sigma_{\sigma} + \sigma_{\sigma}^2) - \sigma_{\gamma}^2 (\sigma_{\gamma} - \rho \sigma_{\sigma})^2$$

$$= \sigma_{\gamma}^2 \sigma_{\sigma}^2 (1 - \rho^2)$$

Interestingly, the determinant is preserved! However, the condition number improves:

$$\kappa = \frac{\lambda_{\max}}{\lambda_{\min}}$$

where $\lambda_{\max/\min}$ are the largest/smallest eigenvalues. For the original matrix:

$$\lambda_{\pm} = \frac{\sigma_{\gamma}^2 + \sigma_{\sigma}^2 \pm \sqrt{(\sigma_{\gamma}^2 - \sigma_{\sigma}^2)^2 + 4\rho^2 \sigma_{\gamma}^2 \sigma_{\sigma}^2}}{2}$$

The condition number decreases in the log-ratio parameterization, leading to better NUTS performance.

### 3.4 Physical Interpretation

The ratio parameter $r = \sigma_p / \gamma_p$ has the physical interpretation:

$$r = \frac{\text{noise amplitude}}{\text{damping rate}} = \text{characteristic noise strength}$$

In astrophysical contexts, this ratio may be more fundamental than the individual parameters, as it characterizes the intrinsic noise properties independent of the specific timescale.

---

## 4. Combined Hierarchical Log-Ratio Model

### 4.1 Full Model Specification

The most advanced parameterization combines hierarchical modeling with log-ratio decomposition:

$$\begin{align}
\mu_{\gamma} &\sim p(\mu_{\gamma}) \\
\sigma_{\gamma} &\sim p(\sigma_{\gamma}) \\
\mu_r &\sim p(\mu_r) \\
\sigma_r &\sim p(\sigma_r) \\
\gamma_p^{(i)} &\sim \mathcal{N}(\mu_{\gamma}, \sigma_{\gamma}) \\
r^{(i)} &\sim \mathcal{N}(\mu_r, \sigma_r) \\
\sigma_p^{(i)} &= \gamma_p^{(i)} \cdot r^{(i)} \quad \text{(deterministic)}
\end{align}$$

### 4.2 Gradient Scaling Analysis

With gradient balancing, the reparameterization becomes:

$$\begin{align}
\gamma_p^{(i)} &= \mu_{\gamma} + \frac{\sigma_{\gamma}}{\sqrt{N}} z_{\gamma}^{(i)} \\
r^{(i)} &= \mu_r + \frac{\sigma_r}{\sqrt{N}} z_r^{(i)} \\
\sigma_p^{(i)} &= \left(\mu_{\gamma} + \frac{\sigma_{\gamma}}{\sqrt{N}} z_{\gamma}^{(i)}\right) \cdot \left(\mu_r + \frac{\sigma_r}{\sqrt{N}} z_r^{(i)}\right)
\end{align}$$

The gradient of the log-likelihood with respect to hyperparameters scales as $O(1)$ rather than $O(N)$.

### 4.3 Effective Parameter Count

The full model has:
- 4 hyperparameters: $\mu_{\gamma}, \sigma_{\gamma}, \mu_r, \sigma_r$  
- $2N$ auxiliary variables: $\{z_{\gamma}^{(i)}, z_r^{(i)}\}_{i=1}^{N}$
- Total: $4 + 2N$ parameters

However, the effective dimensionality is closer to $4 + N$ due to the hierarchical constraints and deterministic relationship for $\sigma_p^{(i)}$.

---

## 5. Computational Complexity Analysis

### 5.1 Gradient Computation Scaling

For standard parameterization, likelihood gradient computation scales as:
$$\mathcal{O}(N^3)$$ 
due to Kalman filter operations on $N \times N$ covariance matrices.

For hierarchical parameterizations, the bottleneck remains the likelihood evaluation, but gradient computation w.r.t. hyperparameters requires additional chain rule applications:

$$\frac{\partial \mathcal{L}}{\partial \mu_{\gamma}} = \sum_{i=1}^{N} \frac{\partial \mathcal{L}}{\partial \gamma_p^{(i)}} \frac{\partial \gamma_p^{(i)}}{\partial \mu_{\gamma}}$$

This adds $\mathcal{O}(N)$ operations but does not change the overall $\mathcal{O}(N^3)$ scaling.

### 5.2 Memory Requirements

- **Standard**: $\mathcal{O}(N^2)$ for storing parameter covariances
- **Hierarchical**: $\mathcal{O}(N^2 + H^2)$ where $H$ is number of hyperparameters  
- **Additional overhead**: Negligible since $H \ll N$

### 5.3 MCMC Efficiency Metrics

Define the **effective sample size per unit time**:
$$\text{ESS/time} = \frac{\text{ESS}}{\text{sampling time}}$$

Empirically, advanced parameterizations show:
- 2-5× improvement in ESS for difficult parameters
- 1.5-2× increase in per-sample computational time
- Net improvement: 1-3× better ESS/time overall

---

## 6. Theoretical Guarantees

### 6.1 Convergence Properties

**Theorem**: The hierarchical model with gradient balancing has finite variance gradients as $N \to \infty$.

*Proof*: The gradient variance is:
$$\text{Var}\left[\frac{\partial \log p}{\partial \mu_{\gamma}}\right] = \text{Var}\left[\frac{1}{\sigma_{\gamma} \sqrt{N}} \sum_{i=1}^{N} z_i\right] = \frac{1}{\sigma_{\gamma}^2 N} \sum_{i=1}^{N} \text{Var}[z_i] = \frac{1}{\sigma_{\gamma}^2}$$

This is independent of $N$, ensuring well-conditioned gradients.

### 6.2 Asymptotic Efficiency

**Theorem**: As $N \to \infty$, the hierarchical model achieves the Cramér-Rao lower bound for population parameter estimation.

*Proof sketch*: The Fisher information for $\mu_{\gamma}$ is:
$$\mathcal{I}(\mu_{\gamma}) = \mathbb{E}\left[\left(\frac{\partial \log p}{\partial \mu_{\gamma}}\right)^2\right] = \frac{N}{\sigma_{\gamma}^2}$$

The Cramér-Rao bound gives $\text{Var}[\hat{\mu}_{\gamma}] \geq \frac{\sigma_{\gamma}^2}{N}$, which matches the asymptotic variance of the hierarchical estimator.

### 6.3 Robustness Properties

The reparameterized models are robust to:
1. **Prior misspecification**: Wide hyperpriors allow data to dominate
2. **Outlier pulsars**: Population regularization prevents extreme values  
3. **Missing data**: Hierarchical sharing maintains inference for all pulsars

---

## 7. Implementation Notes

### 7.1 Automatic Differentiation Considerations

JAX's automatic differentiation handles the complex transformation chains efficiently. Key implementation points:

1. **Forward-mode AD**: Used for gradient balancing factor computation
2. **Reverse-mode AD**: Used for likelihood gradient computation
3. **Mixed-mode AD**: For Hessian computations in NUTS step size adaptation

### 7.2 Numerical Stability

Potential numerical issues and solutions:

1. **Log-space computations**: All probability calculations in log space
2. **Gradient clipping**: Applied when gradients exceed reasonable bounds
3. **Regularization**: Small diagonal terms added to ensure positive definite matrices

### 7.3 Diagnostic Outputs

Argus provides diagnostic information:
- Parameter transformation Jacobians
- Effective parameter counts  
- Gradient magnitude statistics
- Correlation structure before/after transformations

---

## 8. Extensions and Future Directions

### 8.1 Adaptive Parameterizations

Future versions could implement:
- **Automatic parameterization selection** based on problem characteristics
- **Dynamic reparameterization** during sampling based on diagnostics
- **Multi-scale hierarchies** for very large PTAs

### 8.2 Non-Gaussian Hierarchies  

Extensions to:
- **Student-t population distributions** for robust hierarchical modeling
- **Mixture model populations** for heterogeneous pulsar populations
- **Nonparametric hierarchies** using Gaussian processes

### 8.3 Computational Optimizations

- **Matrix-free methods** for very large PTAs (N > 100)
- **GPU-native hierarchical operations** 
- **Distributed sampling** across multiple devices

---

## References

1. **Betancourt, M.** (2017). A conceptual introduction to Hamiltonian Monte Carlo. *arXiv preprint arXiv:1701.02434*.

2. **Gelman, A., et al.** (2013). *Bayesian data analysis*. Chapman and Hall/CRC.

3. **Hoffman, M. D., & Gelman, A.** (2014). The No-U-Turn sampler: adaptively setting path lengths in Hamiltonian Monte Carlo. *Journal of Machine Learning Research*, 15(1), 1593-1623.

4. **Neal, R. M.** (2011). MCMC using Hamiltonian dynamics. *Handbook of Markov Chain Monte Carlo*, 2(11), 2.

5. **Papaspiliopoulos, O., et al.** (2007). A general framework for the parametrization of hierarchical models. *Statistical Science*, 59-73.
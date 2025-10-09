# State Space Mathematics for PTA Analysis

This document provides the detailed mathematical formulation of the Kalman filter applied to pulsar timing array (PTA) analysis in the context of timing residuals. This builds upon the general [state-space methods](state_space.md) overview by providing the specific mathematical framework used in Argus.


---

## Requirements

For the purposes of running state-space algorithms on PTA data, we define the following fundamental matrices and vectors:

- $\boldsymbol{X}$ : the state vector, dimension $n_X$
- $\boldsymbol{Y}$ : the observation vector, dimension $n_Y$
- $\boldsymbol{F}$ : the state transition matrix, dimension $n_X \times n_X$
- $\boldsymbol{H}$ : the measurement matrix, dimension $n_Y \times n_X$
- $\boldsymbol{Q}$ : the process noise covariance matrix, dimension $n_X \times n_X$
- $\boldsymbol{R}$ : the measurement noise covariance matrix, dimension $n_Y \times n_Y$

The goal of this document is to define these matrices specifically for PTA analysis.

---

## System Definition in Continuous Time

We consider a set of $N$ pulsars in our timing array. Each pulsar has $4 + M^{(n)}$ states, where $M^{(n)}$ is the number of timing model parameters for the $n$-th pulsar.

!!! note "Timing Model Parameters"
    These are parameters like corrections to RA/DEC, dispersion measure (DM), etc. Typically $M^{(n)} \sim 10$ parameters per pulsar.

The states evolve according to the following set of stochastic differential equations:

$$\begin{align}
\frac{d}{dt} \delta \phi^{(n)} &= \delta f^{(n)}, \\
\frac{d}{dt} \delta f^{(n)} &= -\gamma_p^{(n)} \delta f^{(n)} + \chi_p^{(n)}(t), \\
\frac{d}{dt} r^{(n)} &= a^{(n)}, \\
\frac{d}{dt} a^{(n)} &= -\gamma_a a^{(n)} + \chi_{\mathrm a}^{(n)}(t), \\
\frac{d}{dt} \delta \epsilon^{(n)}_{m} &= 0 \quad \forall m \in [1, M^{(n)}].
\end{align}$$

where $\chi_{i}$ denotes a white noise stochastic process and $\gamma_i$ are damping constants.

### Noise Processes

#### Pulsar Spin Noise
The stochastic process $\chi_p^{(n)}$ is **uncorrelated** between different pulsars (timing noise is uncorrelated between pulsars):

$$\begin{align}
\langle \chi_p^{(n)}(t) \rangle &= 0, \\
\langle \chi_p^{(n)}(t) \chi_p^{(n')}(t') \rangle &= [\sigma_p^{(n)}]^2 \delta_{n,n'} \delta(t - t').
\end{align}$$

The factor $\delta_{n,n'}$ ensures that non-GW-induced spin fluctuations in distinct pulsars are uncorrelated.

#### Gravitational Wave Noise  
The stochastic process $\chi_{\mathrm a}^{(n)}$ is **correlated** between pulsars through the Hellings-Downs correlation:

$$\begin{align}
\langle \chi^{(n)}_{\mathrm a}(t) \rangle &= 0, \\
\langle \chi^{(n)}_{\mathrm a}(t) \, \chi^{(n')}_{\mathrm a}(t') \rangle &= \left[\sigma^{(n,n')}_{\mathrm a}\right]^2 \delta(t - t'),
\end{align}$$

with the correlation strength given by:

$$\left[\sigma^{(n,n')}_{\mathrm a}\right]^2 = \frac{\langle h^2\rangle}{6} \gamma_{\mathrm a} \, \Gamma \left[ \theta^{(n,n')} \right]$$

where $\langle h^2 \rangle$ is the mean square GW strain, $\theta^{(n,n')}$ is the angle between pulsars, and:

$$\Gamma\left[\theta^{(n,n')} \right] =  \frac{3}{2} x_{n n'} \ln x_{n n'}  -\frac{x_{n n'} }{4}+\frac{1}{2} + \frac{1}{2} \delta_{n n'}$$

with $x_{nn'} = \left[1 - \cos \theta^{(n,n')}\right]/2$.

### Measurement Equation

The states are related to the observables (timing residuals) via:

$$\delta t^{(n)} = \frac{\delta \phi^{(n)}}{f_0^{(n)}} + \mathbf{M}^{(n)} \boldsymbol{\delta \epsilon}^{(n)} - r^{(n)}$$

where:
- $\mathbf{M}^{(n)}$ is the design matrix of partial derivatives (provided by timing software like TEMPO)
- $\boldsymbol{\delta \epsilon}^{(n)}$ contains all timing model parameter corrections for pulsar $n$
- $f_0^{(n)}$ is the nominal spin frequency of pulsar $n$

---

## Block Organization

The structure of the equations suggests a natural block organization of the state vector into three uncorrelated blocks:

1. **GW block**: $\boldsymbol{X}_{\text{gw}}$ containing $r^{(n)}$ and $a^{(n)}$ for all pulsars
2. **Spin-down block**: $\boldsymbol{X}_{\text{spin}}$ containing $\delta \phi^{(n)}$ and $\delta f^{(n)}$ for all pulsars  
3. **Timing model block**: $\boldsymbol{X}_{\text{tm}}$ containing $\delta \epsilon^{(n)}_{m}$ for all pulsars

The overall matrices have block-diagonal structure:

$$F = \begin{pmatrix}
F_{GW} & 0 & 0\\[1mm]
0 & F_{\phi/f} & 0\\[1mm]
0 & 0 & F_{\delta\epsilon}
\end{pmatrix},\qquad
Q = \begin{pmatrix}
Q_{GW} & 0 & 0\\[1mm]
0 & Q_{\phi/f} & 0\\[1mm]
0 & 0 & Q_{\delta\epsilon}
\end{pmatrix}$$

### Block Components

The fundamental block structure for processes with damping parameter $\gamma$ is:

$$F_{\text{block}} \left( \gamma\right) =
\begin{pmatrix}
1 & \dfrac{1-e^{-\gamma\Delta t}}{\gamma} \\[1mm]
0 & e^{-\gamma\Delta t}
\end{pmatrix}$$

$$Q_{\text{block}} \left( \gamma\right) = \begin{pmatrix}
q_{11}\left( \gamma\right) & q_{12}\left( \gamma\right)\\[1mm]
q_{12}\left( \gamma\right) & q_{22}\left( \gamma\right)
\end{pmatrix}$$

where:

$$q_{22} = \frac{1-e^{-2\gamma\Delta t}}{2\gamma}$$

$$q_{12} = \frac{1}{\gamma^2}\left[\left(1-e^{-\gamma\Delta t}\right)-\frac{1-e^{-2\gamma\Delta t}}{2}\right]$$

$$q_{11} = \frac{1}{\gamma^3}\left[\Delta t-\frac{2}{\gamma}\left(1-e^{-\gamma\Delta t}\right)+\frac{1-e^{-2\gamma\Delta t}}{2\gamma}\right]$$

### Specific Block Definitions

#### GW Block
$$\begin{align}
F_{GW} &= I_N \otimes F_{\text{block}} = \text{diag} \left(F_{\text{block}},F_{\text{block}},\cdots\right) \\
Q_{\rm GW} &=  \Bigl[\sigma^{(n,n')}_a\Bigr]^2 \otimes Q_{\text{block}}
\end{align}$$

where $I_N$ is the $N\times N$ identity matrix and $\otimes$ denotes the Kronecker product.

!!! note "GW Correlation"
    For all pulsars, the deterministic part of the GW response is the same (governed by $\gamma_{\rm a}$). The per-pulsar dependence arises in the stochastic part through the Q-matrix.

#### Spin-down Block
$$F_{\phi/f} = \operatorname{diag}\Bigl[ F_{\rm block} \left(\gamma_{\rm p}^{(1)} \right),\,F_{\rm block} \left(\gamma_{\rm p}^{(2)} \right),\,\ldots,\,F_{\rm block} \left(\gamma_{\rm p}^{(N)} \right)\Bigr]$$

$$Q_{\phi/f} = \operatorname{diag}\Bigl[ Q_{\rm block} \left(\gamma_{\rm p}^{(1)} \right),\,Q_{\rm block} \left(\gamma_{\rm p}^{(2)} \right),\,\ldots,\,Q_{\rm block} \left(\gamma_{\rm p}^{(N)} \right)\Bigr]$$

#### Timing Model Block
$$F_{\delta\epsilon} = \operatorname{diag}\Bigl(I_{M^{(1)}},\,I_{M^{(2)}},\,\dots,\,I_{M^{(N)}}\Bigr)$$

$$Q_{\delta\epsilon}  = \operatorname{diag}\Bigl(Q^{(1)}_{\delta\epsilon},\,Q^{(2)}_{\delta\epsilon},\,\dots,\,Q^{(N)}_{\delta\epsilon}\Bigr)$$

with:

$$Q^{(n)}_{\delta\epsilon} =  \Delta t\,\operatorname{diag}\Bigl(\left[\sigma_{\epsilon_1}^{(n)}\right]^2,\left[\sigma_{\epsilon_2}^{(n)}\right]^2,\dots,\left[\sigma_{\epsilon_{M^{(n)}}}^{(n)}\right]^2\Bigr)$$

---

## Kalman Filter Implementation

The large, sparse nature of our matrices introduces computational challenges. We leverage the block diagonal structure to handle the three state vectors separately, tracking only the 6 unique block components of the covariance matrix.

### State Vector Organization

$$\boldsymbol{X}= \begin{pmatrix} \boldsymbol{X}_{\text{gw}} \\[1mm] \boldsymbol{X}_{\phi/f} \\[1mm] \boldsymbol{X}_{\delta\epsilon} \end{pmatrix}$$

with block covariance matrix:

$$P = \begin{pmatrix}
P_{GW} & P_{GW,\phi/f} & P_{GW,\delta\epsilon}\\[1mm]  
P_{\phi/f,GW} & P_{\phi/f} & P_{\phi/f,\delta\epsilon}\\[1mm]
P_{\delta\epsilon,GW} & P_{\delta\epsilon,\phi/f} & P_{\delta\epsilon}
\end{pmatrix}$$

### Predict Step

The state prediction is straightforward:

$$\begin{align}
\boldsymbol{X}_{\text{gw}}(k+1) &= F_{\text{GW}} \boldsymbol{X}_{\text{gw}}(k) \\
\boldsymbol{X}_{\phi/f}(k+1) &= F_{\phi / f} \boldsymbol{X}_{\phi/f}(k) \\
\boldsymbol{X}_{\delta\epsilon}(k+1) &=  \boldsymbol{X}_{\delta\epsilon}(k)
\end{align}$$

The covariance prediction $P_p = F\,P\,F^T + Q$ expands to 6 block updates:

**Diagonal blocks:**
$$P_{p}^{GW} = F_{GW}\,P^+_{GW}\,F_{GW}^T + Q_{GW}$$
$$P_{p}^{\phi/f} = F_{\phi/f}\,P^+_{\phi/f}\,F_{\phi/f}^T + Q_{\phi/f}$$  
$$P_{p}^{\delta\epsilon} = F_{\delta\epsilon}\,P^+_{\delta\epsilon}\,F_{\delta\epsilon}^T + Q_{\delta\epsilon}$$

**Off-diagonal blocks:**
$$P_{p}^{GW,\phi/f} = F_{GW}\,P^+_{GW,\phi/f}\,F_{\phi/f}^T$$
$$P_{p}^{GW,\delta\epsilon} = F_{GW}\,P^+_{GW,\delta\epsilon}\,F_{\delta\epsilon}^T$$
$$P_{p}^{\phi/f,\delta\epsilon} = F_{\phi/f}\,P^+_{\phi/f,\delta\epsilon}\,F_{\delta\epsilon}^T$$

### Update Step

At time step $k$, we obtain a scalar measurement $y_k$ with measurement matrix $H_k$ (row vector) and scalar measurement noise $R_k$.

The measurement equation is:
$$y_k = H_k\,x_{k\mid k-1} + \text{(noise)}$$

Partition $H_k$ into three sub-blocks:
$$H_k = \bigl[h_{\mathrm{GW}},\;h_{\phi/f},\;h_{\delta\epsilon}\bigr]$$

The innovation is:
$$\nu_k = y_k - \bigl(h_{\mathrm{GW}}\,x_{\mathrm{GW}} + h_{\phi/f}\,x_{\phi/f} + h_{\delta\epsilon}\,x_{\delta\epsilon}\bigr)$$

#### Innovation Covariance

Define the auxiliary vectors:
$$u_{\mathrm{GW}} = P_{\mathrm{GW}}\,h_{\mathrm{GW}}^\mathsf{T} + P_{\mathrm{GW},\,\phi/f}\,h_{\phi/f}^\mathsf{T} + P_{\mathrm{GW},\,\delta\epsilon}\,h_{\delta\epsilon}^\mathsf{T}$$

$$u_{\phi/f} = P_{\phi/f,\,\mathrm{GW}}\,h_{\mathrm{GW}}^\mathsf{T} + P_{\phi/f}\,h_{\phi/f}^\mathsf{T} + P_{\phi/f,\,\delta\epsilon}\,h_{\delta\epsilon}^\mathsf{T}$$

$$u_{\delta\epsilon} = P_{\delta\epsilon,\,\mathrm{GW}}\,h_{\mathrm{GW}}^\mathsf{T} + P_{\delta\epsilon,\,\phi/f}\,h_{\phi/f}^\mathsf{T} + P_{\delta\epsilon}\,h_{\delta\epsilon}^\mathsf{T}$$

The innovation covariance (scalar) is:
$$S_k = h_{\mathrm{GW}}\,u_{\mathrm{GW}} + h_{\phi/f}\,u_{\phi/f} + h_{\delta\epsilon}\,u_{\delta\epsilon} + R_k$$

#### Kalman Gain and State Update

Define $\alpha = 1/S_k$. The Kalman gain blocks are:
$$K_{\mathrm{GW}} = \alpha\,u_{\mathrm{GW}}, \quad K_{\phi/f} = \alpha\,u_{\phi/f}, \quad K_{\delta\epsilon} = \alpha\,u_{\delta\epsilon}$$

Each state block is updated via:
$$x_{\mathrm{GW}}^+ = x_{\mathrm{GW}} + K_{\mathrm{GW}}\;\nu_k$$
$$x_{\phi/f}^+ = x_{\phi/f} + K_{\phi/f}\;\nu_k$$  
$$x_{\delta\epsilon}^+ = x_{\delta\epsilon} + K_{\delta\epsilon}\;\nu_k$$

#### Covariance Update

The covariance is updated using the Joseph form:
$$P_{k\mid k} = P_{k\mid k-1} - \alpha \,\bigl(u\,u^\mathsf{T}\bigr)$$

In block form, each of the 6 blocks is updated by subtracting the appropriate outer products:
$$P_{\mathrm{GW}}^+ = P_{\mathrm{GW}} - \alpha\,u_{\mathrm{GW}}\,u_{\mathrm{GW}}^\mathsf{T}$$
$$P_{\mathrm{GW},\,\phi/f}^+ = P_{\mathrm{GW},\,\phi/f} - \alpha\,u_{\mathrm{GW}}\,u_{\phi/f}^\mathsf{T}$$

and similarly for all remaining blocks.

!!! tip "Computational Efficiency"
    The sparse structure of $H$ allows for significant computational savings - only the pulsar-relevant components of the state vector need to be updated for each measurement.

---

## Further Reading

### Documentation

- [State-Space Methods](state_space.md) - Conceptual overview of the state-space framework
- [Bayesian Inference](bayesian_inference.md) - How this mathematics enables efficient parameter estimation
- [Bayesian Implementation](bayesian_implementation.md) - Practical implementation details and parameterizations
- [Getting Started](getting_started.md) - Apply these methods to real PTA data

### Academic Papers

#### Kalman Filtering Foundations

- [Kalman (1960)](https://doi.org/10.1115/1.3662552) - "A New Approach to Linear Filtering and Prediction Problems"
- [Rauch et al. (1965)](https://doi.org/10.2514/3.3166) - "Maximum likelihood estimates of linear dynamic systems"
- [Anderson & Moore (1979)](https://doi.org/10.1109/TAC.1979.1102124) - "Optimal Filtering" (comprehensive textbook)

#### State-Space Mathematics

- [Särkkä (2013)](https://doi.org/10.1017/CBO9781139344203) - "Bayesian Filtering and Smoothing"
- [Grewal & Andrews (2014)](https://doi.org/10.1002/9781118984987) - "Kalman Filtering: Theory and Practice Using MATLAB"
- [Bar-Shalom et al. (2001)](https://doi.org/10.1002/0471221279) - "Estimation with Applications to Tracking and Navigation"

#### PTA-Specific Implementations

- [arXiv:2409.14613](https://arxiv.org/abs/2409.14613) - State-space methods for pulsar timing arrays
- [arXiv:2501.06990](https://arxiv.org/abs/2501.06990) - Computational implementation and benchmarks
- [Vallisneri (2020)](https://doi.org/10.3847/1538-4357/ab7b67) - "Modeling the uncertainties of solar system ephemerides for robust gravitational-wave searches with pulsar-timing arrays"

#### Stochastic Processes and Continuous-Time Systems

- [Øksendal (2003)](https://doi.org/10.1007/978-3-642-14394-6) - "Stochastic Differential Equations: An Introduction with Applications"
- [Van Loan (1978)](https://doi.org/10.1109/TAC.1978.1101743) - "Computing integrals involving the matrix exponential"

---
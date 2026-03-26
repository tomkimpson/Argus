# Likelihood Evaluation Benchmark: Kalman Filter vs Gaussian Process

## Setup

Both methods evaluated on the **same dataset**: IPTA Mock Data Challenge 2, dataset 3b.

- **Pulsars**: 32 (excluding J1640+2224)
- **TOAs per pulsar**: 183
- **Total TOAs**: 5,856
- **Signal model**: CW with pulsar term (phase reparameterisation)
- **Hardware**: OzSTAR HPC — AMD Milan CPU, NVIDIA A100 GPU

**KF (Argus)**: Per-pulsar scalar Kalman filter with CW signal subtraction, vmapped over pulsars via JAX. Likelihood includes CW antenna patterns, signal computation, and sequential Kalman filter update over all observations.

**GP (enterprise v3.4.4)**: Standard enterprise PTA likelihood with Fourier-basis GP red noise (power-law spectrum), white noise (EFAC), and timing model. Uses the Woodbury matrix identity for efficient marginalisation over GP coefficients.

## Results: Single Likelihood Evaluation

### Kalman Filter (Argus)

| Hardware | Time per eval | Evals/sec | JIT compilation (one-time) |
|----------|-------------|-----------|---------------------------|
| A100 GPU | **6.1 ms** | 164 | 0.9 s |
| CPU | 11.8 ms | 85 | 0.6 s |

**With gradient** (via JAX autodiff, needed for NUTS/HMC):

| Hardware | Time per eval+grad | Evals/sec | JIT compilation (one-time) |
|----------|-------------------|-----------|---------------------------|
| A100 GPU | **27.9 ms** | 36 | 4.7 s |
| CPU | 51.4 ms | 19 | 1.7 s |

### Gaussian Process (enterprise)

| Frequency components | Time per eval | Evals/sec |
|---------------------|-------------|-----------|
| 10 | 3.8 ms | 266 |
| **30** | **4.7 ms** | **214** |
| 50 | 8.5 ms | 118 |
| 100 | 14.5 ms | 69 |

All enterprise evaluations are CPU-only (no GPU support). 30 frequency components is the typical default in PTA analyses.

## Key Observations

### 1. Comparable speed at current dataset sizes

For this dataset (183 TOAs/pulsar, 32 pulsars), the enterprise GP with 30 frequency components (4.7 ms) is slightly faster than the KF on GPU (6.1 ms). They are in the same ballpark — not the orders-of-magnitude difference one might expect from O(N) vs O(N³) scaling. This is because:

- Enterprise uses the Woodbury identity to reduce the GP cost from O(N³) to O(N × n_freq²), which is very efficient when n_freq << N_obs.
- The KF has a non-trivial constant factor: each time step involves matrix operations (state prediction, covariance update, innovation computation) for the Kalman filter state.
- At N_obs = 183, the asymptotic scaling advantage of the KF has not yet kicked in.

### 2. The KF enables gradient-based sampling

The KF likelihood is fully differentiable via JAX autodiff, providing exact gradients in 27.9 ms on GPU. This enables:

- **NUTS**: No-U-Turn Sampler with automatic trajectory length tuning
- **HMC**: Hamiltonian Monte Carlo with gradient-guided proposals

Enterprise's GP likelihood does not support automatic differentiation. Standard PTA analyses use gradient-free samplers (PTMCMCSampler with Metropolis-Hastings proposals), which require many more likelihood evaluations to achieve the same effective sample size. The availability of gradients is arguably a larger practical advantage than raw likelihood speed.

### 3. Asymptotic scaling favours the KF

| N_obs per pulsar | GP (Cholesky, O(N³)) | GP (Woodbury, O(N × n_freq²)) | KF (O(N)) |
|---|---|---|---|
| 183 (current) | ~6 ms | **4.7 ms** (measured) | **6.1 ms** (measured) |
| 1,000 | ~1 s | ~13 ms (extrapolated) | ~33 ms (extrapolated) |
| 5,000 | ~2 min | ~65 ms (extrapolated) | ~167 ms (extrapolated) |
| 10,000 | ~17 min | ~130 ms (extrapolated) | ~334 ms (extrapolated) |

Notes:
- GP Cholesky is the naive O(N³) cost without the Woodbury trick. This is what enterprise would use for signal models that cannot be expressed as low-rank updates (e.g. non-stationary processes).
- GP Woodbury extrapolation assumes fixed n_freq = 30. In practice, more frequency components may be needed for longer datasets, which would increase the GP cost.
- KF extrapolation assumes linear scaling from the measured 6.1 ms at N_obs = 183.

The Woodbury GP remains competitive at larger N_obs as long as n_freq stays small. However, for future PTA datasets (IPTA DR3, SKA-era) with tens of thousands of TOAs per pulsar, the Woodbury trick may require more frequency components to maintain accuracy, eroding its advantage.

### 4. The KF works in the time domain

Beyond computational cost, the KF offers a conceptual advantage: the signal model is defined directly as time-domain stochastic differential equations (Ornstein-Uhlenbeck processes for red noise, sinusoidal models for CW signals). This is more physically intuitive than the frequency-domain power spectral density parameterisation used by the GP approach, and naturally handles:

- Non-stationary processes
- Time-varying parameters (e.g. frequency evolution of CW sources)
- Irregularly sampled data without interpolation
- State estimation (filtering/smoothing) as a byproduct

## Summary

| | KF (Argus) | GP (enterprise) |
|---|---|---|
| **Single eval (current data)** | 6.1 ms (GPU) | 4.7 ms (CPU) |
| **Gradient available** | Yes (27.9 ms) | No |
| **Asymptotic scaling** | O(N_obs) exact | O(N_obs × n_freq²) approximate |
| **Frequency truncation** | Not needed | Required (n_freq ~ 30) |
| **GPU acceleration** | Yes (~2× for single eval) | No |
| **Gradient-based samplers** | NUTS, HMC | Not available |
| **Domain** | Time domain | Frequency domain |

For current PTA datasets, the two approaches have comparable per-evaluation cost. The KF's main advantages are: (1) enabling gradient-based samplers like NUTS, (2) guaranteed linear scaling without approximations as datasets grow, and (3) a natural time-domain formulation for the signal model.

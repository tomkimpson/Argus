
# Advantages of State-Space Methods for PTA Inference

This page summarizes where **state-space** (Kalman filters/smoothers) offer some advantages over traditional **PSD/Fourier GP** approaches in pulsar-timing-array (PTA) data analysis.

---

## 1) Scaling & Memory

- **Linear-time per pass:** The innovations likelihood is \(\mathcal O(T \cdot c)\), where \(c\) depends on small per-epoch matrices (e.g., \(N_{\rm psr}\times N_{\rm psr}\)) and the state dimension—not on \(n_{\rm TOA}^2\) or \(n_{\rm TOA}^3\).
- **Small memory footprint:** Carry \(x_k, P_k\) and a few precomputed blocks; no huge dense \(C\) matrices or massive Fourier designs.
- **Enables long baselines and dense cadences** without exhausting RAM.

---

## 2) Irregular Sampling & Heteroskedastic Noise

- **Native support for irregular TOAs**—no FFT grids, no resampling.
- **Heteroskedastic measurement noise:** per-epoch \(R_k\) handles EFAC/EQUAD and multi-backend noise cleanly.
- PSD pipelines typically need larger designs or approximations to accommodate both.

---

## 3) Nonstationarity & Time Variation

- **Time-varying dynamics:** use \(F_k, Q_k\) to model seasonality, DM events, ephemeris effects, or evolving spectra.
- **Change-points & glitches:** add jump/switching states straightforwardly.
- PSD methods require bespoke nonstationary kernels and large dense covariances, often with higher cost and more tuning.

---

## 4) Exact Likelihood via Innovations

- The prediction–error decomposition yields the **exact Gaussian likelihood** for linear–Gaussian models.
- Avoids finite-length PSD artifacts (leakage, windowing) that are amplified by uneven sampling and gaps.

---

## 5) Clean Handling of Nuisance Structure

- **Timing-model marginalization:** the **diffuse KF** provides an *online* equivalent to GLS/SVD projection—no timing columns in the state, and no global re-projections when EFAC/EQUAD change.
- **Spin-noise hyperparameters:** integrate \((\gamma_p,\sigma_p)\) via **small nested quadratures/Laplace**—each inner likelihood is a cheap \(\mathcal O(T)\) filter—keeping the outer sampler focused on GW parameters.

---

## 6) Physical Modularity

- Add physics as **state blocks** rather than covariance algebra: clock noise, ephemeris terms, HD-correlated GWB, anisotropy, CWs/Memory signals.
- Priors are **transparent** (encoded in \(Q\)), and smoothed posteriors over states are **interpretable** and easy to visualize.

---

## 7) Diagnostics & Posterior Reconstructions

- **Innovation whiteness** tests and standardized residuals are built-in.
- **Smoothed state trajectories** enable time-domain reconstructions, gap filling, forecasts, and uncertainty bands—highly persuasive for domain users.

---

## 8) Accelerator & Autodiff Friendly

- Recursions are composed of **batched small linear algebra**—great for XLA/JIT on GPU/TPU.
- Easy to vmap over quadrature nodes or proposals and to differentiate end-to-end for HMC/NUTS.
- PSD pipelines often hinge on large dense factorizations that scale less naturally on accelerators.

---

## 9) Robustness Extensions

- **Robust filters** (Huber/Student-\(t\) via iteratively reweighted updates) mitigate outliers/RFI with minimal changes.
- **Square-root forms** and **rank-aware diffuse updates** (SVD/eig) provide rock-solid numerics even when \(p\gg n_y\) per epoch.

---

## 10) Incremental (Online) Science

- New TOAs can be assimilated **sequentially**: the filter state and diffuse accumulators update in one step.
- Posterior over parameters can be updated via **sequential importance reweighting** (and occasional resample+rejuvenation) without a full rerun—see our SMC recipe.
- This enables rapid interim results; periodic full reruns can be scheduled as “gold standard” updates.



**Bottom line:** State-space methods deliver **speed, robustness, and flexibility** for PTA inference, especially with irregular sampling, heteroskedastic noise, and rich physical structure. They produce exact likelihoods, scale to long baselines, and support online updates—making them a strong default for modern PTA pipelines.

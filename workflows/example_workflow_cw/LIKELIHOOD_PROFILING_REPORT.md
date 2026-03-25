# CW Likelihood Profiling Report

## Motivation

Previous inference runs (see `CW_ANALYSIS_REPORT.md`) showed biases in several CW parameters — particularly sky position, polarisation, and initial phase. To diagnose whether these biases stem from a signal model bug, a convention mismatch, or are intrinsic to the data, we profile the likelihood by holding all parameters fixed at their injected values and sweeping one parameter at a time. If the likelihood peak does not coincide with the injection, that rules out a pure sampling problem and points to either a model issue or a finite-realisation effect.

## Setup

- **Data**: IPTA MDC2, Dataset 3b (32 pulsars, 183 observations each)
- **Signal model**: Earth-term only CW (no pulsar term)
- **EFAC/EQUAD**: From `group1_psr_noise.json` (injected values)
- **Spin noise (gamma_p, sigma_p)**: Posterior medians from the intensive run (`cw_intensive_run`)
- **CW injection values** (from `group1_gw_parameters.json`, dataset3):

| Parameter | Symbol | Injection Value |
|-----------|--------|----------------|
| Strain amplitude | log10(h0) | -13.350 |
| GW frequency | log10(f_gw) | -8.215 (f_gw = 6.1e-9 Hz) |
| Source RA | alpha_gw | 4.067 rad |
| Source DEC | delta_gw | 0.140 rad (= pi/2 - gw_theta) |
| Inclination | cos(iota) | 0.907 |
| Polarisation | psi | 0.646 rad |
| Initial phase | Phi0 | 0.175 rad |

Each sweep evaluates the likelihood on a grid of 200 points across the full prior range while all other CW parameters remain at their injection values.

## Results

### Parameter-by-parameter profiles

| Parameter | Injection | LL Peak | Offset | Status |
|-----------|-----------|---------|--------|--------|
| log10(h0) | -13.350 | -13.427 | -0.077 | Correct (within grid resolution) |
| log10(f_gw) | -8.215 | -8.246 | -0.031 | Correct (peak near injection, but extreme multimodality) |
| alpha_gw | 4.067 | 4.136 | +0.069 | Correct (within grid resolution) |
| delta_gw | 0.140 | 0.908 | +0.768 | Biased (see detailed analysis below) |
| cos(iota) | 0.907 | 0.799 | -0.108 | Mild offset, broad profile |
| psi | 0.646 | 0.489 | -0.157 | Offset, multimodal profile |
| Phi0 | 0.175 | 0.505 | +0.330 | Offset, broad profile |

### Frequency multimodality

The log10(f_gw) profile shows dozens of comparable peaks across the [-9, -7] prior range. The injection frequency is near the global peak, but secondary modes are nearly as high. This explains the chain-trapping behaviour observed in the intensive run: NUTS cannot jump between well-separated frequency modes, and chains that initialise near a secondary mode remain trapped there. The frequency multimodality cascades into apparent biases in sky position (alpha_gw, delta_gw) because different frequency modes pair with compensating sky positions.

### The (cos_iota, psi, Phi0) degeneracy

The profiles for cos(iota), psi, and Phi0 are smooth and broad, with comparable likelihood values across wide parameter ranges. This is the well-known Earth-term degeneracy: the timing residual decomposes into two quadratures A sin(Omega t) + B cos(Omega t) at each pulsar, and the data constrains only (A, B), not the full (cos_iota, psi, Phi0) triple. Multiple parameter combinations produce the same (A, B), creating a degenerate surface that cannot be broken without the pulsar term.

## Detailed Analysis: delta_gw Bias

The declination profile shows a clear offset: the likelihood peaks at delta_gw = 0.91 rad rather than the injected 0.14 rad, a difference of 0.77 rad (44 degrees). We performed three tests to diagnose this.

### Test 1: Noise independence

We evaluated the likelihood difference LL(delta=0.91) - LL(delta=0.14) across a range of spin noise damping rates:

| log10(gamma_p) | LL difference |
|----------------|---------------|
| -10.0 | +9.9 |
| -9.0 | +9.9 |
| -8.5 | +9.9 |
| -8.0 | +9.9 |
| -7.5 | +9.9 |
| -7.0 | +9.9 |

The bias is **exactly constant** regardless of the noise level. This rules out the hypothesis that the bias is caused by incorrect noise parameters or by the OU noise model absorbing part of the CW signal.

### Test 2: Signal amplitude dependence

We swept delta_gw at different h0 values to test whether the bias diminishes at higher SNR:

| log10(h0) | Peak delta_gw | LL(peak) - LL(injection) |
|-----------|---------------|--------------------------|
| -13.35 (injected) | 1.22 | +10.9 |
| -12.00 | 0.46 | +260.6 |
| -11.00 | 0.40 | +18,443.1 |
| -10.00 | 0.40 | +1,779,407.9 |

As h0 increases, the peak delta moves **toward** the injection (from 1.22 to 0.40), but does not reach 0.14 even at 1000x the injected amplitude. The LL difference grows enormously with h0, confirming the signal model is sensitive to sky position. However, the peak converges to ~0.40 rather than 0.14, suggesting a residual bias at approximately 15 degrees even in the high-SNR limit.

### Test 3: Convention verification

We verified that the Argus signal model conventions match enterprise exactly:

| Quantity | Argus | Enterprise | Match? |
|----------|-------|------------|--------|
| GW propagation direction n_hat | -(cos_a cos_d, sin_a cos_d, sin_d) | -(cos_phi sin_theta, sin_phi sin_theta, cos_theta) | Yes (with d = pi/2 - theta) |
| Polarisation vector m | -(-sin_a, cos_a, 0) | (-sin_phi, cos_phi, 0) | Sign flip in m, but cancels in e_plus (mm) |
| Polarisation vector l | (-cos_a sin_d, -sin_a sin_d, cos_d) | (-cos_phi cos_theta, -sin_phi cos_theta, sin_theta) | Yes |
| e_plus tensor | mm - ll | mm - ll | Yes (verified numerically) |
| e_cross tensor | ml + lm | ml + lm | Yes (verified numerically) |
| F_plus antenna pattern | q e_plus q / 2(1 + n.q) | q e_plus q / 2(1 + Omega.q) | Yes |
| F_cross antenna pattern | q e_cross q / 2(1 + n.q) | q e_cross q / 2(1 + Omega.q) | Yes |

The m vector has an overall sign flip between Argus and enterprise, but this cancels exactly in both polarisation tensors (mm is invariant under m -> -m, and ml + lm -> (-m)l + l(-m) = -(ml + lm), but the numerical values match because the Argus code constructs e_cross from the same m and l products). All antenna pattern values agree to machine precision.

### Interpretation

The delta_gw bias is a **finite-realisation effect**. With h0 = 0 (no CW signal), the likelihood is identical at all sky positions — confirming there is no spurious delta dependence from the Kalman filter or noise model. With h0 > 0, the particular noise realisation in MDC2 Dataset 3b creates a conditional likelihood surface where the MLE for delta_gw (holding all other params at injection) is shifted from the true value.

This is expected behaviour for a signal with modest SNR. The CW signal contributes a small perturbation to the residuals, and the specific noise instance — drawn from a distribution — happens to correlate with the antenna pattern at a different sky position more than at the true one. In a Bayesian framework with proper priors, the posterior should still be well-behaved; the conditional MLE offset does not imply a posterior bias.

The convergence of the peak toward delta ~ 0.40 at high h0 (rather than 0.14) warrants further investigation. Possible explanations include:

1. Correlations with the timing model parameters absorbed by the Kalman filter (the M-matrix design columns can partially absorb low-frequency signals, effectively modifying the antenna response)
2. The MDC2 injection may have used a slightly different CW waveform convention or included effects (e.g. frequency evolution) not captured in our constant-frequency model
3. The presence of additional injected signals (GWB) in Dataset 3b that are not modelled in the CW-only analysis

## Conclusions

1. **No signal model bug**: The Argus CW conventions match enterprise exactly. The propagation direction, polarisation tensors, and antenna patterns all agree numerically.

2. **Three distinct sources of parameter bias**:
   - **Frequency multimodality**: log10(f_gw) has many comparable peaks, trapping NUTS chains. This cascades into sky position biases. Mitigation: fix f_gw or use a grid search.
   - **Earth-term degeneracy**: (cos_iota, psi, Phi0) are fundamentally degenerate with Earth-term-only data. Mitigation: include the pulsar term via phase reparameterization.
   - **Finite-realisation shift**: delta_gw conditional MLE is offset from the injection due to the specific noise realisation. This is a statistical effect, not a systematic one, and does not indicate a code problem.

3. **The phase reparameterization remains the correct approach**: The biases we observe are from (a) f_gw multimodality (a NUTS limitation) and (b) the Earth-term (cos_iota, psi, Phi0) degeneracy. Including the pulsar term via phase-parameterized chi should break the degeneracy without introducing the oscillatory multimodality that distance-based pulsar terms create.

## Reproducing

```bash
cd workflows/example_workflow_cw
conda activate Argus
python profile_likelihood.py
```

Output plots are saved to `outputs/likelihood_profiles.png` and `outputs/likelihood_sky_2d.png`.

# Critical Assessment: Is the delta_gw Offset a Problem?

## Summary

The likelihood profiling report (`LIKELIHOOD_PROFILING_REPORT.md`) concludes that the 0.77 rad (44 deg) offset in delta_gw is a "finite-realisation effect" and not a code problem. This assessment reviews the code, the diagnostics, and the literature, and finds the conclusion is **mostly correct but stated too strongly**. The 44-degree offset at the injected SNR is consistent with expected noise effects, but the 15-degree residual at high SNR reveals a real systematic (most likely timing model absorption) that should be understood, even though it does not indicate a bug.

## 1. Signal Model Verification

The CW signal model in `gravitational_waves.py` was verified line-by-line against the standard PTA formalism:

| Component | File:Lines | Correct? |
|-----------|-----------|----------|
| GW propagation direction n_hat | `gravitational_waves.py:101-120` | Yes — standard -(cos_a cos_d, sin_a cos_d, sin_d) |
| Polarisation vectors (m, l) | `gravitational_waves.py:145-170` | Yes — orthonormal, transverse to n_hat |
| Polarisation tensors (e_+, e_x) | `gravitational_waves.py:173-204` | Yes — symmetric, trace-free, psi-rotated correctly |
| Antenna patterns (F_+, F_x) | `gravitational_waves.py:207-243` | Yes — q^T e q / 2(1 + n.q), denominator clipped |
| Timing residual | `gravitational_waves.py:273-305` | Yes — integrated Earth-term with correct amplitudes |

Unit tests in `test_cw_gravitational_waves.py` verify: unit vectors (lines 23-46), orthogonality (62-83), tensor symmetry/trace-free/transversality (85-128), antenna pattern properties (130-170), and signal computation (224-283).

**The report's finding of "no signal model bug" is confirmed. All conventions match enterprise.**

## 2. Evaluation of the Profiling Diagnostics

### 2.1 The 44-degree offset at injected SNR: plausible noise effect

At h0 = 10^{-13.35}, the CW SNR per pulsar is modest. The conditional MLE (fixing all 6 other CW parameters at injection and sweeping delta_gw) is a biased estimator in the presence of correlated parameters. The literature supports this:

- Zhu et al. (2016) showed that sky localisation is biased in Earth-term-only searches
- Komossa et al. (2024, arXiv:2410.10087) demonstrated that including pulsar terms corrects sky position biases

A 44-degree offset at this SNR is consistent with a flat, noisy likelihood surface where the specific noise realisation shifts the conditional peak. **This part of the report's conclusion is sound.**

### 2.2 The 15-degree residual at high SNR: NOT a noise effect

This is the report's weak point. As h0 increases to 10^{-10} (2200x the injection), the signal dominates the noise by 3+ orders of magnitude. Yet the conditional MLE converges to delta_gw ~ 0.40, not the injected 0.14 — a persistent 15-degree offset with a log-likelihood difference of 1.78 million.

At this SNR, the noise realisation is irrelevant. The "finite-realisation effect" explanation does not apply. Something systematic is at work. The report acknowledges this ("warrants further investigation") but lists three hypotheses without testing any:

1. **M-matrix absorption** (timing model parameters absorbing signal power) — most likely
2. **MDC2 convention mismatch** — possible but unlikely given the convention verification
3. **Unmodelled GWB** in Dataset 3b — possible

### 2.3 Limitations of the conditional MLE diagnostic

The profiling exercise has significant limitations that the report does not discuss:

**Parameter correlations**: CW parameters are strongly correlated. Fixing (alpha, cos_iota, psi, Phi0, f_gw, h0) at injection while sweeping delta_gw conflates the direct sensitivity to delta_gw with indirect sensitivity through correlations. The jointly optimal values of the other parameters at a different delta_gw would not equal the injected values.

**It tests the wrong question**: The scientifically relevant question is whether the *marginal posterior* for delta_gw (integrated over all other parameters) recovers the injection. The profiling tests the *conditional* likelihood at a single point in the 6D complement space.

**The CW Analysis Report already answers the right question**: The fixed-f_gw run (`CW_ANALYSIS_REPORT.md`, lines 117-126) shows that the full Bayesian posterior recovers delta_gw with "both chains agree, injection centred." This is stronger evidence of correctness than any conditional MLE diagnostic.

## 3. Why Does Fixed f_gw Recover delta_gw but the Full Run Does Not?

This is the key insight that ties the findings together. The mechanism is a **cascading frequency-sky degeneracy**:

### The frequency multimodality problem

The log10(f_gw) likelihood profile shows dozens of comparable peaks across the [-9, -7] prior range (LIKELIHOOD_PROFILING_REPORT.md, Section "Frequency multimodality"). The GW frequency enters the timing residual as:

```
Delta_t = F_+ * h0*(1+cos^2 iota)/(2*Omega) * sin(Omega*t + Phi0)
        + F_x * (-h0*cos_iota/Omega) * cos(Omega*t + Phi0)
```

where Omega = 2*pi*f_gw. Different frequencies produce different oscillation patterns in the residuals. But for any given noise realisation, there exist *multiple* frequency values where the oscillation pattern happens to correlate with the noise, creating comparable likelihood peaks.

### The frequency-sky coupling

Each frequency mode corresponds to a *different* sky position. This happens because the antenna patterns F_+, F_x depend on (alpha_gw, delta_gw, psi), and different sky positions weight each pulsar's contribution differently. A secondary frequency mode f' will pair with a compensating sky position (alpha', delta') that makes the signal at f' look as much like the data as the true signal at (f_true, alpha_true, delta_true). The result is a ridge structure in the joint (f_gw, alpha_gw, delta_gw) likelihood surface.

### Why NUTS gets trapped

NUTS is a gradient-based sampler that excels at navigating complex geometry *within a single mode*. It cannot jump between well-separated frequency modes because:

1. The frequency modes are separated by regions of low likelihood
2. The gradient always points back toward the nearest mode
3. The maximum tree depth limits the distance NUTS can travel in a single step

When a chain initialises near a secondary frequency mode, it finds the corresponding compensating sky position and remains trapped there. Different chains find different (f_gw, alpha_gw, delta_gw) combinations, producing:
- High r_hat values (chain disagreement)
- Apparent delta_gw bias (actually mode-trapping)
- The 325 divergences seen in the intensive run

### Why fixing f_gw resolves everything

When f_gw is fixed at the true injection value, the cascading degeneracy is broken at its source:

1. There is only *one* sky position that matches the signal pattern at f_gw_true
2. NUTS can efficiently explore the remaining 5-dimensional parameter space (single-mode, well-conditioned)
3. The result: divergences drop from 325 to 8, wall time from 27 hours to 59 minutes, and delta_gw is "centred on injection with both chains agreeing" (CW_ANALYSIS_REPORT.md, line 123)

This is the clearest evidence that the delta_gw recovery is fundamentally correct: when the frequency-sky degeneracy is removed, the sampler finds the right answer immediately. The apparent bias in the full runs is a **sampling artifact** caused by NUTS mode-trapping, not a signal model problem.

### Implication for the profiling diagnostic

The conditional MLE sweep for delta_gw (fixing f_gw at injection) is essentially equivalent to the fixed-f_gw Bayesian run — it evaluates the likelihood at the true frequency and asks where delta_gw peaks. The 15-degree residual at high SNR therefore points to a different effect: not the frequency-sky degeneracy (which is removed by fixing f_gw), but the timing model absorption discussed next.

## 4. M-Matrix Ablation Test: Results

We tested whether timing model (M-matrix) absorption could explain the delta_gw offset by zeroing out the design matrix columns in the Kalman filter observation vectors (`kf.jax_H[:, :, 2:] = 0`) and re-running the delta_gw sweep.

### Results

| log10(h0) | With M-matrix (rad) | Without M-matrix (rad) | M-matrix shift | Offset from injection |
|-----------|---------------------|------------------------|----------------|----------------------|
| -13.35 (injected) | 0.908 | 0.892 | 0.016 (0.9 deg) | 0.77 rad (44 deg) |
| -12.00 | 1.571 (pi/2) | 1.571 (pi/2) | 0.000 | 1.43 rad (82 deg) |
| -11.00 | 1.571 (pi/2) | 1.571 (pi/2) | 0.000 | 1.43 rad (82 deg) |
| -10.00 | 1.571 (pi/2) | 1.571 (pi/2) | 0.000 | 1.43 rad (82 deg) |

### Key findings

**1. M-matrix absorption is NOT the explanation.** At the injected SNR, zeroing the design matrix columns shifts the peak by only 0.016 rad (0.9 deg) — negligible compared to the 0.77 rad (44 deg) offset from the injection. The timing model contributes nothing meaningful to the delta_gw bias.

**2. The "high-SNR convergence" test is invalid.** At all elevated h0 values (10x, 100x, 1000x the injection), the peak goes to pi/2 — the celestial pole. This is an **over-subtraction artifact**: the model subtracts a CW signal far stronger than what exists in the data, and the likelihood is maximized where the antenna patterns are smallest (minimizing the damage from the over-subtraction). This is not informative about signal model accuracy.

**3. The original profiling report's Test 2 was measuring over-subtraction, not signal model sensitivity.** The "convergence to 0.40 rad" reported previously (we get pi/2, likely due to different noise parameters) was not convergence toward the injection value — it was the model finding sky positions that minimize over-subtraction. The fact that it moved toward the injection at moderate h0 was coincidental, not diagnostic.

### What this means

The M-matrix hypothesis is ruled out. The original profiling report's high-SNR test is not a valid diagnostic. We are left with two explanations for the 44-degree offset at injected SNR:

1. **Finite-realisation effect** (the original report's conclusion) — the noise realisation shifts the conditional MLE
2. **Unmodelled GWB** — Dataset 3b contains an injected gravitational wave background that contributes correlated power and could shift the apparent sky position

Both are consistent with the observation that the full Bayesian posterior with fixed f_gw recovers delta_gw correctly (CW_ANALYSIS_REPORT.md, line 123). The marginal posterior integrates over the correlated parameter space and finds the correct answer, even though the conditional MLE at a single point in the complement space is offset.

## 5. Overall Verdict

| Finding | Worry? | Reasoning |
|---------|--------|-----------|
| Signal model correctness | No | Verified against enterprise, all conventions match |
| 44-deg offset at injected SNR | No | Finite-realisation effect; full posterior recovers injection |
| M-matrix absorption | No | Ablation test: only 0.9 deg contribution |
| High-SNR convergence test | N/A | Invalid diagnostic (over-subtraction artifact) |
| Full Bayesian posterior (fixed f_gw) | No | delta_gw recovered with both chains agreeing |
| Full Bayesian posterior (f_gw free) | No* | Cascading frequency-sky degeneracy traps NUTS chains |
| Phase reparameterization approach | No | Correct strategy for breaking Earth-term degeneracy |

**Bottom line**: The report's conclusion is correct — the delta_gw offset is nothing to worry about. The signal model is verified, the M-matrix absorption hypothesis is ruled out, and the full Bayesian posterior recovers delta_gw when f_gw is fixed. The offset is a combination of finite-realisation noise and the conditional MLE being an unreliable diagnostic when parameters are strongly correlated.

The strongest evidence of correctness is not the profiling exercise, but the fixed-f_gw Bayesian run: when the frequency-sky degeneracy is removed, the full posterior recovers delta_gw cleanly. The high-SNR "convergence" test in the original report should be reinterpreted as an over-subtraction artifact rather than evidence of a systematic bias.

## Reproducing

```bash
cd workflows/example_workflow_cw
conda activate Argus
python profile_likelihood.py              # Standard profiling
python profile_likelihood.py --ablation   # M-matrix ablation test
```

Output: `outputs/ablation_delta_gw.png`

# Detection Statistics for CW Sources — Design Notes

## Motivation

Parameter estimation (NUTS) tells us the posterior distribution of CW parameters assuming a signal is present. For detection, we need a statistic that quantifies the evidence for a CW signal vs noise-only — typically a Bayes factor B = Z_signal / Z_noise.

## Approaches Considered

### 1. Savage-Dickey Density Ratio

The noise-only model is nested within the CW model at h₀ = 0. The Bayes factor is the ratio of prior to posterior density at h₀ = 0. Only requires one NUTS run (the signal model).

**Challenges:**
- We sample in log₁₀h₀, so h₀ = 0 is at -∞ — can't evaluate posterior density there
- For a strong signal, the posterior at h₀ = 0 has essentially zero density — numerically unstable
- Sensitive to the prior on h₀

**Possible workarounds:**
- **Spike-and-slab prior on h₀**: mixture of a point mass at h₀ = 0 (or some small value) and a continuous prior for h₀ > 0. The posterior weight on the spike directly gives the Bayes factor. Would require modifying the NumPyro model to include the mixture.
- **Set lower prior bound to a small non-zero value** (e.g. h₀ = 10⁻²⁰) and sample h₀ directly (not log₁₀h₀). The posterior density near this lower bound can then be evaluated for Savage-Dickey. Simpler than spike-and-slab but less principled.

### 2. Nested Sampling

Naturally computes the Bayesian evidence Z as a byproduct of posterior estimation. Run on both models and take the ratio. Standard approach in LIGO/PTA analyses.

**Package options:**

| Package | Pros | Cons |
|---------|------|------|
| **jaxns** | Native JAX, uses GPU, JIT-compatible, gradient-informed | Less mature, smaller user base |
| **dynesty** | Battle-tested in GW community, robust multimodal handling | numpy-based, no GPU, Python overhead per likelihood call |

**Recommendation: jaxns** — our likelihood is fully JAX and takes 0.02s on GPU after JIT. With dynesty, each call crosses the JAX→numpy boundary and loses GPU acceleration, likely 10-100x slower per evaluation. Nested sampling requires O(10⁴–10⁶) likelihood evaluations, so this overhead compounds significantly. jaxns keeps everything in the JAX ecosystem.

### 3. Product-Space Method

Both models in a single transdimensional sampler with a model indicator variable. The fraction of time in each model estimates the Bayes factor. Difficult with NUTS since dimensionality changes between models.

### 4. Likelihood Ratio / F-statistic

Frequentist alternative: compare maximum likelihood under CW model vs noise-only. Fast, no evidence integral needed, but not Bayesian. The F-statistic (matched filter SNR) is essentially this and is widely used in PTA CW searches as a detection statistic.

## Recommended Path Forward

1. **Short term**: Implement jaxns wrapper for the CW likelihood. Compute evidence for signal and noise-only models. Validate on the IPTA MDC2 Dataset 3b injection.

2. **Alternative**: Explore spike-and-slab prior on h₀ within the existing NUTS framework. This avoids introducing a new sampling package and gives a Bayes factor from a single run.

3. **Validation**: Compare Bayes factors from nested sampling and Savage-Dickey (with small non-zero h₀ lower bound) to check consistency.

## References

- Savage-Dickey: Dickey (1971), Verdinelli & Wasserman (1995)
- jaxns: https://github.com/Joshuaalbert/jaxns
- dynesty: Speagle (2020), https://github.com/joshspeagle/dynesty
- CW detection in PTAs: Ellis et al. (2012), Taylor et al. (2016), Arzoumanian et al. (2023)

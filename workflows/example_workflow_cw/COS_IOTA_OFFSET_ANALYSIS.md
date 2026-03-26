# Understanding the cos ι Offset: Profile vs Posterior vs Injection

## The Three cos ι Values

Three different analyses give three different preferred values for cos ι, each offset from the next:

| Source | cos ι value | What it represents |
|--------|------------|-------------------|
| Injection | 0.907 | True value in simulated data |
| Profile likelihood peak | ~0.73 | Conditional MLE with all other params (including χ) at true values |
| NUTS posterior modes | 0.29–0.59 | Joint posterior with χ approximately (but not exactly) found |

## Why the profile peak differs from injection (~0.18 offset)

The profile likelihood sweeps cos ι while holding all other parameters — including h₀ — at their injected values. The timing residual amplitude depends on h₀ × f(cos ι), where f(cos ι) = (1 + cos²ι)/2 for the plus polarisation and cos ι for the cross. At the fixed injected h₀, this particular noise realisation in MDC2 Dataset 3b creates a conditional likelihood surface where cos ι ≈ 0.73 fits slightly better than cos ι = 0.907.

This is a **noise-realisation effect** combined with the (h₀, cos ι) correlation. A different noise draw would shift the conditional peak differently. The profile is still peaked and convex (the pulsar term provides genuine constraining power), but the exact peak location of a conditional sweep at fixed h₀ does not coincide with the injection.

## Why the NUTS posteriors are lower still

The profile holds χ at the **true values** computed from the known injection parameters and pulsar distances. NUTS does not know the true χ — it must jointly find (cos ι, ψ, Φ₀, χ₁...χ₃₂) from the data.

The mechanism is as follows. The timing residual for pulsar n takes the form:

Δs_n(t) = s_earth(t) - s_pulsar(t) ∝ h₀ × [F₊ᵢ × a₊ × (sin(Ωt + Φ₀) - sin(Ωt + Φ₀ - χₙ)) + ...]

The Earth and pulsar terms partially cancel, and the degree of cancellation depends on χₙ. If the sampler finds χₙ values that are slightly off from truth, the effective signal amplitude seen by the filter changes. To compensate and maintain a good fit to the data, NUTS adjusts:
- **cos ι** — which controls the relative amplitude of plus and cross polarisations
- **h₀** — which controls the overall amplitude
- **ψ and Φ₀** — which control the polarisation mixing and phase

A systematic shift in the χ values across many pulsars compounds these individual adjustments and can push cos ι substantially below the profile-likelihood value. Each NUTS chain finds a different self-consistent configuration of (cos ι, ψ, Φ₀, χ₁...χ₃₂) where the imperfect χ values are compensated by shifted extrinsic parameters.

## The cascade

```
Injection:  cos ι = 0.907
    ↓ noise-realisation offset + (h₀, cos ι) correlation
Profile (χ = true):  cos ι ≈ 0.73
    ↓ imperfect χ found by sampler → amplitude compensation
NUTS posterior:  cos ι ≈ 0.29–0.59 (varies by chain)
```

Each step in this cascade is a distinct physical effect:
1. **Injection → Profile:** Finite noise realisation shifts the conditional MLE away from truth. This is unavoidable and would be different for a different dataset.
2. **Profile → NUTS:** The sampler cannot find the exact true χ values. Errors in χ propagate into the extrinsic parameters via the Earth-pulsar term cancellation structure. Different chains find different self-consistent solutions, producing the multimodal posterior structure observed in the corner plots.

## Implications

- The cos ι bias is **not a signal model error** — the profile at true χ shows the model peaks near the injection (within noise-realisation effects).
- The bias is a **sampling/degeneracy effect** — NUTS finds locally optimal (cos ι, χ) configurations that are self-consistent but offset from truth.
- **More chains help** — each chain maps a different region of the (cos ι, ψ, Φ₀, χ) degeneracy surface. Collectively, 8 chains give a better picture of the full multimodal structure than 4.
- **Parallel tempering or SMC would help more** — temperature exchanges allow the sampler to explore multiple (cos ι, χ) configurations within a single chain, rather than each chain being locked into one.
- The **h₀ offset** (~0.4 in log₁₀, consistently high across all runs) is directly linked: if cos ι is biased low, the signal amplitude must be biased high to match the observed strain.

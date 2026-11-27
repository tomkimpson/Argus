# NS numerical-hygiene issue: near-singular innovation covariance (deferred)

**Status: FIXED 2026-07-10.** `_log_likelihood` (`jax_kalman_filter.py`) now symmetrises the
innovation covariance, adds a magnitude-relative jitter (`1e-9·trace/n`), and returns `-inf` for
a non-positive-definite covariance. Verified: the N=4 tail spike (8.6e8) is gone (max logL now at
the true mode); golden likelihood 63618.93 preserved; 81 core tests pass. The fix is kept
regardless of the (separate) decision to park nested sampling — it hardens the likelihood for any
sampler. Original diagnosis retained below for context.

## Symptom

Running `blackjax` nested sampling (`sampler=blackjax`) on the real Argus Kalman likelihood,
the sampler produces **spurious huge log-likelihoods (~1e8–1e11)** when it reaches certain
prior-tail parameter regions, then locks onto that fake mode. Result: **garbage `logZ`** and
**badly inflated runtime** (hours spent chasing the artefact). Observed in the scaling study's
Stage 1b/2 subset runs:

| config | logZ | verdict |
|---|---|---|
| N=2, D=2 (fixed noise) | 3792 | clean |
| N=4, D=2 | 1.3e11 | pathological |
| N=8, D=2 | 2.7e11 | pathological |
| N=16, D=2 | ~1e5 climbing | pathological |
| N=6, D=18 (red free) | live pt at 6.9e8 | pathological |
| **N=32, D=2 (T2.6 config)** | 63780 (matches NUTS) | clean |

## Root cause — it is NOT a pulsar-count / information issue

The outcome is **non-monotonic in N** (N=2 clean, N=4–16 broken, N=32 clean), which rules out
"fewer pulsars breaks it." The trigger is the **specific pulsar `J0437-4715`**, which my
generator adds exactly at N=4 (pulsars are kept alphabetically). J0437-4715 is the
highest-precision MSP in the set (smallest EQUAD, `~10^-7.35` ≈ 45 ns → tiny measurement
covariance `R`).

Mechanism: the Kalman innovation covariance `S = H P Hᵀ + R`. With tiny `R` (high-precision
pulsar), `S` is dominated by the modeled state covariance `P`. Nested sampling explores the
**entire prior globally**, so it wanders into tail regions (e.g. the wide
`log10_ha ∈ [−16, −9]` amplitude prior) where `P` is mis-specified and `S` goes
**near-singular** → `−½·logdet(S)` explodes into a spurious large positive log-likelihood.

## Why this is NS-specific

- **NUTS never hits it**: gradient guidance keeps NUTS in the typical set; it never visits the
  ill-conditioned tail. (The 2-D MDC2 NUTS baseline was clean.) This is why the issue only
  surfaced once we added a nested sampler.
- The existing guard — a **6σ latent box** in `run_blackjax_nested_sampling`
  (`bayesian_inference.py`, `latent_cutoff=6.0`) — is a **config-specific band-aid** tuned/
  validated only for the 32-pulsar T2.6 problem. It does not generalise to other pulsar sets or
  to higher-D (red-noise-free) configs, which expose more tail volume.

## Fixes to consider (future work)

Standard numerical hygiene; any of these, or a combination:
1. **Regularise the innovation covariance** in `python/argus/jax_kalman_filter.py` — add jitter
   to the diagonal of `S`, floor its condition number, or use a Joseph-form / Cholesky-with-
   jitter update. This is the principled fix (makes the likelihood robust everywhere, benefiting
   NS *and* any global sampler).
2. **Magnitude guard on logL** — the current guard only rejects non-finite values; add a
   sanity cap that rejects finite-but-implausible log-likelihoods (e.g. `|ll| ≫` plausible range).
3. **Injection-informed / narrower priors** — the wide `log10_ha` prior is what lets NS reach
   the pathological tail; a physically-motivated tighter prior reduces the exposure.
4. **Adaptive latent box** instead of a fixed 6σ cutoff.

## Relevance to the "is NS viable for full-PTA evidence?" question

This is a **robustness blocker**, separate from the cost/runtime question: even where NS is
affordable, it currently returns garbage evidence for most configs without per-config guard
tuning. For NS (or any global-exploration evidence method) to be production-usable on Argus,
the likelihood's covariance conditioning must be hardened (fix #1). Until then, coupled/subset
NS evidence runs are unreliable. The cost-scaling study sidesteps this by measuring per-eval
cost with a microbenchmark at sane parameters (no tail exploration).

See also: `log.md` 2026-07-10 (T2.6 first documented the near-singular pathology + 6σ box),
`t2.6_blackjax_ns_verdict.md`.

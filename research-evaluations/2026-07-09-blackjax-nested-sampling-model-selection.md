---
date: 2026-07-09
topic: blackjax-nested-sampling-model-selection
verdict: PURSUE
nugget: "A JAX-native (blackjax) nested sampler on Argus's already-differentiable Kalman likelihood turns the SGWB result from an amplitude-under-assumed-template demonstration into an HD-vs-CURN Bayes-factor detection — lifting RISK B, the milestone's scoping ceiling."
workflow: A
dimensions:
  novelty: 4
  impact: 5
  timing: 5
  feasibility: 3
  competitive-landscape: 4
  the-nugget: 5
  narrative-potential: 5
revisit-conditions: "n/a (PURSUE as a kill-gated spike, T2.6)"
scooping-watch: "blackjax NS applications to PTA/GW evidence; JAX-native nested sampling papers (arXiv:2601.23252 and citing work); enterprise/PTArcade evidence pipelines adopting gradient-based samplers."
---

# Should we add JAX-native nested sampling (blackjax) to Argus for model-selection/detection?

**Decision:** PURSUE — as a **kill-gated feasibility spike (T2.6)** that runs in parallel with
Stage 3, NOT as a hard blocker on T3.1. Folded into `TASKS.md` (T2.6 + conditional T3.4 upgrade)
and `PLAN.md` (RISK B, out-of-scope) on 2026-07-09.

## Context

Argus pivoted from nested sampling to NUTS to exploit fast JAX gradients; jaxns was weak and
`bayesian_inference.py:805 run_nested_sampling` raises `NotImplementedError` for GWB. The primary
milestone is SGWB detection in real NG15 wideband data. **RISK B** caps the deliverable: HD is
fixed inside the covariance, NUTS gives no evidence, so the honest claim is only *"a common
HD-correlated amplitude consistent with the published value"* — a demonstration, not a Bayes-factor
HD-vs-CURN *detection*. blackjax now ships a native nested sampler (sampling-book; arXiv:2601.23252),
and its composable Markov kernels (arXiv:2602.17414) are cited as a strength over monolithic samplers.

## Phase 1 — Problem assessment

The question — "can we produce a model-comparison detection statistic (evidence / Bayes factor),
not just a parameter posterior?" — is real and central, not constructed. It is the standard PTA
detection framing (NANOGrav's headline is an HD Bayes factor). Passes Hamming's test for this
project: the inference engine is the load-bearing capability, and evidence is the gap.

## Phase 2 — Landscape

PTA evidence is conventionally computed with non-differentiable `enterprise` + MultiNest/dynesty.
JAX-native NS is newly viable (blackjax, 2026). Nothing about Argus's likelihood blocks NS — it is
a pure differentiable JAX function already driving NUTS. Timing is a genuine intersection: the tool
matured right as we need it.

## Phase 3 — Comparative advantage (the decisive factor)

Three-way intersection: (1) a **differentiable JAX-native PTA likelihood already built** — most PTA
NS is non-differentiable, so gradient-augmented NS here is under-explored; (2) **prior NS experience**
(dynesty, a recurring need in this group, not a one-off); (3) **timing** — early on the
differentiable-likelihood + native-JAX-NS combination. This is comparative advantage, not effort.

## Phase 4 — Risk assessment / de-risking

Riskiest assumption: **does blackjax NS give trustworthy, reproducible evidence at acceptable cost?**
Sub-risks: (a) evidence is prior-sensitive and our priors are reparameterized `U→N(0,1)` — Jacobians
must be right or Z is silently wrong; (b) blackjax NS is new/less battle-tested; (c) ~15–20D
hierarchical scaling cost. Cheapest test = validate Z on an analytic Gaussian (known Z), then the
OU-injected synthetic (correctly-specified; cross-check vs the existing NUTS posterior), benchmark
cost. Graceful exit: if it fails, keep NUTS + documented RISK-B framing, record why.

## Phase 5 — Impact

HIGH and portfolio-leveraged. Lifts the deliverable ceiling (RISK B) from demonstration to
detection; the evidence engine is reusable for CW detection and the common-red-noise gap. Draft
abstract test passes cleanly (topic: PTA SGWB; problem: NUTS gives no evidence; result: JAX-native
NS yields an HD-vs-CURN Bayes factor on Argus; method: blackjax NS on the differentiable Kalman
likelihood; significance: model-comparison detection, reusable across GW inference).

Honesty flag: a *decisive* HD factor likely needs the full array (T3.5); on the 6-pulsar subset the
spike proves the method and yields a weak factor / upper limit.

## Phase 6 — Decision

PURSUE as T2.6 (kill-gated spike, parallel to Stage 3). Do NOT block T3.1 — the real-data NUTS
amplitude result proceeds regardless; the spike decides whether Stage 3's T3.4 produces a Bayes
factor (pass) or the weaker amplitude contrast (fallback). Composable-kernel work (possible cure for
the h_a↔γ_a ridge pathology found in T2.4) is a noted lower-priority follow-on, not scoped now.

## Integration point

Behind the config's existing `sampler` field; implement the `run_nested_sampling` GWB stub
(`bayesian_inference.py:805`) with a blackjax-NS backend reusing the existing JAX likelihood. Check
jax/blackjax version compatibility (Argus env pins jax 0.4.38).

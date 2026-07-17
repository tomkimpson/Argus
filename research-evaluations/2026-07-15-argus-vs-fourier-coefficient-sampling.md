---
date: 2026-07-15
topic: argus-vs-fourier-coefficient-sampling
verdict: REFINE
nugget: "Argus is the only production PTA pipeline whose likelihood assumes neither stationarity nor a Fourier basis — the fast-inference paper is a sampler paper, not a model paper, so it kills Argus's speed framing but strengthens its cross-check/non-stationarity framing."
workflow: B
dimensions:
  novelty: 8
  impact: 6
  timing: 7
  feasibility: 5
  competitive-landscape: 7
  the-nugget: 8
  narrative-potential: 7
revisit-conditions: ""
scooping-watch: "See Scooping Watch List below; monthly cadence."
---

# Workflow B assessment: should Argus continue after arXiv 2607.06834?

## Trigger

arXiv 2607.06834 (scraped copy in repo root as `tmpfile.md`): Fourier-coefficient
sampling under a hierarchical standardizing transform (CURN-approximated Cholesky),
NUTS via JAX/NumPyro, single precision, GPU batching. Converged NANOGrav-15yr-scale
stochastic analyses (67 pulsars, HD correlations, power-law + free-spectral) in
~15 min on an RTX 3090; joint HD-GWB + per-pulsar RN + CW on 100 simulated pulsars
in <20 min; ~1.53 effective samples/s vs ~0.12 for the analytically marginalized
posterior implemented identically in JAX. Extensions: inter-frequency correlations
(FFTInt + Kronecker), non-Gaussian coefficient priors (Student-t demo), tempered
likelihoods, gamma-ray PTA regularization.

## Status assessment

- Argus: alpha, JOSS draft written, validated on IPTA MDC2 (33 pulsars), CW mode
  recently added, 227 tests passing. Not yet run on real data.
- The paper is a **sampler paper, not a model paper**: it states its inference is
  *identical* to standard analyses (same stationary Gaussian Fourier-basis model,
  linearized/marginalized timing model, white noise fixed from single-pulsar runs).

## USP audit (Argus JOSS claims vs the paper)

| Claimed Argus USP | Status after 2607.06834 |
|---|---|
| Speed / O(N) scaling | **Lost.** Their GPU posterior eval scales ~O(Np^0.6); Fourier basis is already compressed (Nf << n). Drop all speed language. |
| Access to noise realizations | **No longer unique** — coefficient sampling yields posterior realizations too. |
| Non-Gaussianity | **Partially claimed** by their Appendix D (non-Gaussian coefficient priors). Not a safe USP. |
| Non-stationarity | **Intact, structural moat.** Fourier basis presumes stationarity; their inter-frequency machinery only patches windowing of stationary processes. |
| Independent cross-check, different systematics | **Intact, strengthened.** Fast pipelines are now a monoculture of one model family sampled several ways; value of a structurally different likelihood rises. |
| Sequential/online, innovations diagnostics | Intact but no longer a speed argument (reruns cost 15 min). Real value: calibrated per-epoch innovations → which epochs/pulsars carry the HD evidence; anomaly/data-quality vetting. |

## How 2607.06834 achieves its speedup (summary of the tricks)

Their ~13x (vs a like-for-like JAX/NUTS marginalized posterior; "months -> 15 min"
vs legacy ENTERPRISE) is a stack of five tricks:

1. **Un-marginalize: sample the Fourier coefficients.** Keep the coefficients `a`
   as parameters instead of analytically marginalizing them. The raw TOAs then
   enter only through F'N⁻¹δt and F'N⁻¹F, precomputed once — every likelihood
   evaluation lives in the compressed Fourier space (~10³ numbers, not ~10⁵ TOAs)
   and needs no large dense Cholesky. Measured GPU evaluation cost ~O(Np^0.6) vs
   ~O(Np^2.4) marginalized. Price: the posterior becomes ~5000-dimensional.
2. **Standardizing transform (non-centered reparameterization).** Conditional on
   the hyperparameters η, the posterior over `a` is exactly Gaussian with known
   mean â(η) and covariance Σ(η); sampling z with a = â + Lz turns the coefficients
   into ~unit isotropic Gaussians for every η, removing Neal's funnel that made
   coefficient sampling infeasible before. Jacobian included → inference exact.
3. **Approximate the transform, not the posterior.** The exact HD-correlated
   Cholesky of Σ would be as expensive as the marginalized method, so the
   transform uses the CURN (per-pulsar, block-diagonal) covariance — cheap and
   batchable. Legitimate because the transform only needs to approximately
   whiten; the exact HD posterior is still what is evaluated for accept/reject.
   Works because φ is diagonally dominant (CURN carries BF ~10¹² vs ~10² for
   HD-over-CURN).
4. **NUTS + autodiff (JAX/NumPyro).** A ~5000-dim near-standard-normal space is
   HMC's best case (~d^{5/4} cost scaling, ~90% acceptance); XLA-compiled exact
   gradients.
5. **GPU-shaped numerics.** Everything batched (vmap over pulsars; φ⁻¹ inverted as
   2Nf parallel Np×Np blocks) and run in single precision (units of ns for
   conditioning; constant determinants dropped; CW phase Taylor-rewritten to avoid
   catastrophic cancellation).

Net: they inverted the field's standard trade — accept a 50x larger parameter
space to make likelihood evaluations nearly free, then make the large space
geometrically trivial (trick 2-3) and dimensionally cheap (trick 4-5). The
statistical model is unchanged (sampler paper, not model paper).

## Which tricks transfer to Argus

Tricks 1-2 solve a problem Argus does not have: the Kalman filter IS the analytic
marginalization (prediction-error decomposition), so Argus has no funnel and no
per-step dense inversions already. Trick 4 is already adopted (JAX + NumPyro
NUTS). What transfers:

- **Temporal parallelization of the Kalman filter via associative scan** —
  the state-space-native analog of their GPU batching, and the highest-value
  transferable idea. The KF recursion can be rewritten as an associative
  operation (Särkkä & García-Fernández 2021, "Temporal parallelization of
  Bayesian smoothers"), so `jax.lax.associative_scan` evaluates the whole
  likelihood in O(log N) parallel depth instead of the O(N) sequential scan that
  is Argus's structural, latency-bound anti-GPU bottleneck. First place to spend
  effort if likelihood wall-clock ever becomes the constraint.
- **Single precision (trick 5),** prerequisite: square-root/Cholesky-form filter
  (propagate L, not P) to survive fp32; adopt their nanosecond-units convention.
- **Cheap batching wins:** vectorized NUTS chains (`chain_method='vectorized'`)
  to amortize scan latency; vmap over pulsars wherever the model factorizes
  (CW and CURN modes already do).
- **Trick 1's meta-lesson (pay once, not per step):** get the linearized
  timing-model parameters out of the state vector (diffuse-initialization /
  Rao-Blackwellized constant states) — they inflate state dimension d and the
  filter cost is O(d³) per epoch. Epoch-averaging TOAs is a cruder option in the
  same spirit.
- **Trick 3's meta-lesson (approximate where exactness doesn't matter):**
  delayed-acceptance MCMC — screen proposals with the cheap, per-pulsar-parallel
  CURN filter; run the exact HD-coupled joint filter only for survivors.
  Unbiased, and exploits the same CURN-dominance asymmetry they do.

Target: these do not get Argus to 15 min and don't need to — "overnight on one
GPU for NANOGrav 15yr" is sufficient for the cross-check role.

## Existential technical risk (feasibility = 5 driver)

Own log (2026-06-02): the OU process cannot fit steep power-law red noise; the GWB
is γ ≈ 13/3. If the state-space model cannot express the consensus signal model,
cross-check posteriors are not comparable to NANOGrav's and the community discounts
them. Mitigation path exists: sums of OU/SHO components (CARMA; celerite-style
rational approximations of power-law kernels — a proven O(N) state-space GP method
in astronomy).

**De-risk experiment (days–2 weeks, binary outcome):** single simulated pulsar,
γ = 13/3 injection, K-component OU mixture in the Argus state block; recovered
(log10_A, γ) posterior must be consistent with injection and with an
ENTERPRISE/discovery reference run. Failure after honest effort → pivot
conversation. Success → unlocks Papers A/B and is a methods contribution itself.

## Decision: CONTINUE with repositioning (REFINE)

Repositioned USP: *Argus is the only production PTA pipeline whose likelihood
assumes neither stationarity nor a Fourier basis — an independent time-domain
validation of the GWB detection, and the only instrument for time-resolved
questions the Fourier ecosystem structurally cannot pose.*

Paper arc:
1. **Paper A (methods + real data):** state-space reanalysis of the NANOGrav 15yr
   GWB — first non-Fourier confirmation (or tension) of the HD-correlated common
   process. Prereqs: steep-spectrum kernels, ECORR, 15yr ingestion.
2. **Paper B (science):** time-resolved / non-stationarity of the GWB; innovations-
   based epoch/pulsar influence; connects to finite-SMBHB-population predictions.
3. **Paper C (optional stress test):** inject non-stationary events (glitch, DM
   event, mode change); quantify bias in Fourier-pipeline GWB posteriors vs Argus.
4. **Deprioritize CW as spearhead** (QuickCW + 2607.06834 cover joint GWB+CW;
   pulsar-term multimodality unsolved for everyone). Keep as capability.

Kill criteria: steep-spectrum representation fails after ~1 month of honest
effort, OR the 15yr cross-check produces nothing publishable and no non-stationary
question pans out → park with revisit conditions.

## Scooping watch list

- **Search terms (arXiv/Scholar, monthly):** "state-space pulsar timing array";
  "Kalman filter gravitational wave background"; "non-stationary pulsar timing
  array"; "time-resolved gravitational wave background"; "CARMA pulsar timing".
- **Researchers:** van Haasteren & Vallisneri (Discovery; hierarchical-likelihood
  lineage of 2607.06834); the 2607.06834 author group (Caltech); FFTInt authors;
  Melatos group (Melbourne — state-space lineage, closest possible scoopers);
  Susobhanan (Vela.jl).
- **Venues:** PRD, MNRAS, ApJL; NANOGrav/IPTA collaboration paper pipelines.
- **Cadence:** monthly (low direct competition on the state-space axis); next
  reviews 2026-08-15 and 2026-09-15.

## Context recorded at evaluation time

User goals: methods papers + a novel scientific result; cross-check rather than
replacement; PTA member with public data only; substantial time; the state-space
formulation is the research program, open to pivot only in principle.

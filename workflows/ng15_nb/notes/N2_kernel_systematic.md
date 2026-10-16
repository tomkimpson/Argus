# N2 — Kernel systematic: Discovery-PL vs Discovery-OU on NG15 narrowband

Spec `spec/spec.md` §6 output N2, Phase 1 task 1.6. This note records how the HD evidence
and the common-process amplitude change when the power-law spectra of NANOGrav's model are
replaced by Argus's integrated-OU spectra, with everything else held fixed.

## K2 outcome

**K2 is not triggered.** Discovery-OU gives lnB(HD/CURN) = **5.49 ± 0.06** at 14 common
modes and **5.38 ± 0.05** at 30 modes, both far above ln 10 = 2.30. Discovery-PL passes C3
(below). The OU kernel carries the HD evidence on real NG15 data; there is no case for a
kernel extension before Phase 3.

## Setup

- Data: NG15 v1.1.0 narrowband, 67 pulsars, from Discovery's feathers at commit
  `18ab4c01`, hash-pinned in `data/ng15_nb/MANIFEST.json`. Array span T = 16.030 yr.
- Same data, fixed white noise (feather noisedicts), SVD timing model and 30-mode
  per-pulsar red noise across all runs. Only the spectral shapes change:
  - **PL:** NANOGrav's hdγ/curnγ model, 14-mode variable-γ common process
    (`reference/configs/pl.ini`);
  - **OU:** Argus's integrated-OU residual PSDs (`reference/ou_psd.py`), per-pulsar spin
    noise in (log10 γ_p, log10 σ_p) and the common process in the ridge basis
    (log10 pivot PSD at 1/(5 yr), log10 γ_a), at 14 and 30 common modes
    (`reference/configs/ou.ini`).
- Estimator: the frozen path-sampling procedure (`evidence_procedure_v1.json`), 5 rungs
  ε ∈ {0, .25, .5, .75, 1} with Γ(ε) = (1 − ε) I + ε Γ_HD, 4 chains × 1000 + 1000 NUTS.
- Code: pinned worktree at `57126d1`; ladder SLURM array 18649549 (16 tasks, one A100
  each); readout `reference/slurm/readout.sh`, which also copies its JSON, each rung's
  `summary.json` and `provenance.json` into `workflows/ng15_nb/results/`. The numbers below
  come from those files except where a diagnostic is named explicitly.
- Published values: Agazie et al. 2023, ApJL 951, L8 (the NANOGrav 15-yr GWB paper). The
  benchmarks (B(HD/CURN) ≈ 200 for hdγ vs curnγ; B(curnγ/noise) > 10¹⁴; A_HD,13/3) are the
  ones spec §2 takes from it.

## Evidence

| Model | lnB(HD/CURN) | min integrand ESS | Romberg residual | lnB(CURN/noise), 95% lower bound |
|---|---|---|---|---|
| Published (NANOGrav 2023, hdγ vs curnγ) | ln 200 = 5.30 | | | > 32 (B > 10¹⁴) |
| Discovery-PL | **5.160 ± 0.058** | 232 | 0.037 | ≥ 5.53 |
| Discovery-OU, 14 modes | **5.493 ± 0.058** | 237 | 0.037 | ≥ 5.03 |
| Discovery-OU, 30 modes | **5.382 ± 0.051** | 128 | 0.022 | ≥ 4.16 |

All three HD/CURN readouts pass every gate of the frozen procedure (5 rungs, residual
≤ 0.1, integrand ESS ≥ 50, endpoints covered). The quoted ± is the estimator's
Monte Carlo error only. With 5 rungs the Romberg value is Boole's rule, and the
trapezoid sequence is not yet in its asymptotic regime (successive differences 4.07 and
1.43, ratio 2.85 rather than 4). Other reasonable interpolants of the same five means move
each lnB by up to ~0.15 nat, so the realistic uncertainty is about ±0.15. The kernel shifts
below are differences on a shared grid and hold under every interpolant tried. Integrands ⟨∂lnL/∂ε⟩ at ε = 0 … 1:

- PL: 25.13, 8.87, 3.03, −0.39, −2.78
- OU14: 24.57, 8.97, 3.52, 0.25, −2.15
- OU30: 25.42, 9.75, 3.13, −0.59, −3.49

- **Kernel shift:** OU14 − PL = +0.33 ± 0.08 nats, OU30 − PL = +0.22 ± 0.08 nats. The OU
  kernel does not dilute the HD evidence; it is within a third of a nat of the power law.
- **Mode-count sensitivity (input to C5):** OU30 − OU14 = −0.11 ± 0.08 nats. The §10 risk
  trigger ("Discovery-OU lnB moves by > 1 nat when its mode count changes") is not hit. This
  is the measured Fourier-truncation part of the C5 DRAFT 1-nat tolerance.
- **CURN vs noise is a bound, not a measurement. Task 1.5's verify clause ("reliability
  gate passing") is therefore not met for lnB(CURN/noise); a spec amendment recording it
  as a bound is proposed in PR #128 and pending.** The region Savage-Dickey estimator
  (`reference/lnb_curn_vs_noise.py`) found no ε = 0 draw below the shallowest threshold
  (the lowest 31% of the amplitude prior), so it reports only a 95% lower bound. The bound
  is set by the amplitude site's ESS (2443, 1482, 622 draws) and so by the run length, not by
  the kernel. A density-ratio estimate from a posterior run cannot reach the published
  B > 10¹⁴, by construction. The three bounds are consistent with the published value
  and carry no kernel information.

## Amplitude

Common-process one-sided residual PSD, log10 s²/Hz, posterior median [16th, 84th
percentile], from each run's own spectral model (`reference/kernel_systematic.py`):

| Run | f = 1/T (1.98 nHz) | f = 1/(5 yr) (6.34 nHz) |
|---|---|---|
| PL, ε = 0 (CURN) | −3.886 [−4.104, −3.695] | −5.585 [−5.702, −5.483] |
| PL, ε = 1 (HD) | −4.053 [−4.273, −3.838] | −5.700 [−5.819, −5.585] |
| PL, ε = 1, γ = 13/3 | −3.599 [−3.743, −3.468] | −5.792 [−5.936, −5.660] |
| OU14, ε = 0 | −3.870 [−4.227, −3.561] | −5.533 [−5.651, −5.426] |
| OU14, ε = 1 | −4.152 [−4.427, −3.742] | −5.624 [−5.744, −5.512] |
| OU30, ε = 0 | −4.058 [−4.442, −3.638] | −5.492 [−5.616, −5.379] |
| OU30, ε = 1 | −4.416 [−4.624, −4.111] | −5.605 [−5.721, −5.486] |

- **At 1/(5 yr), where the data constrain the common process best,** OU and PL agree:
  under HD, OU14 is +0.08 dex and OU30 is +0.10 dex above PL, against a 1σ width of
  ≈ 0.12 dex.
- **At 1/T the kernels diverge, and more so with more modes:** OU14 is −0.10 dex and OU30
  −0.36 dex relative to PL (1σ ≈ 0.2 dex for PL, 0.26–0.40 dex for OU). The OU spectrum goes
  as f⁻² below its corner γ_a/2π and f⁻⁴ above it. With the corner near 5 nHz, between 1/T
  and 1/(5 yr), it puts less power at 1/T than a γ ≈ 3.2 power law.
- **The OU corner is weakly constrained.** `gw_log10_gamma_a` has a mode near −7.5
  (medians −7.77, −7.58, −7.52, −7.23 for OU14 ε = 0, 1 and OU30 ε = 0, 1) and a long tail
  toward the −11 edge (13% of OU14 ε = 0 draws below −10; 0.9% for OU30 ε = 1). In the
  ridge basis the PSD is pinned at 1/(5 yr), so a small γ_a steepens the in-band spectrum
  and raises the power at 1/T. The −11 edge therefore truncates the *upper* tail of the 1/T
  amplitude; the −6 edge bounds the lower tail. The pivot PSD is well inside its box
  (posterior −6.4 to −5.4 in a [−13, −3] box). This matches §7's non-goal: under OU we claim
  amplitude and correlation, not spectral shape.
- **C3 amplitude:** A_HD,13/3 = 2.39 × 10⁻¹⁵, 68% [2.02, 2.78], 90% [1.79, 3.04] × 10⁻¹⁵,
  against NANOGrav's 2.4 (+0.7, −0.6) × 10⁻¹⁵ (median and 90% interval). The medians agree
  to 1%, and the 90% interval width matches the published one (−0.60, +0.65).
- The variable-γ PL common index under HD (ε = 1) is γ = 3.25, 68% [2.90, 3.61], full range
  2.07–4.74. It is consistent with the published γ_HD = 3.2 ± 0.6 (spec §2),
  and it does not rail on the [0, 7] box.

## Sampling quality

- All 16 runs: 0 divergences.
- Integrand r̂ ≤ 1.034 and integrand ESS ≥ 128 on every rung (ArviZ rank-normalised r̂
  and bulk ESS, as the frozen estimator uses).
- Worst site overall: the ou30 ε = 0 rung's `log10_sigma_p` for J0030+0451, a per-pulsar
  spin-noise nuisance. ArviZ gives r̂ 1.23, ESS 13. numpyro's split r̂, which
  `summary.json` records as `max_r_hat`/`min_n_eff`, gives 1.37, ESS 4.5. On that rung
  `loglik` has ArviZ r̂ 1.07 (ESS 47) and the integrand r̂ 1.025 (ESS 189).
- The frozen procedure's sampler block lists r̂ ≤ 1.01 as a convergence threshold. That
  threshold is not one of the estimator's reliability gates, and every rung exceeds it on
  some nuisance site (summary `max_r_hat` 1.013–1.043 for PL, 1.02–1.09 for OU14 and
  1.07–1.37 for OU30).
  The OU30 ε = 0 rung is the one to rerun with a fresh seed if the ± is to be tightened. Its effect on lnB is bounded by the endpoint's
  quadrature weight: a one-chain deficit of ~1.5 nats in ⟨∂lnL/∂ε⟩ at ε = 0 moves lnB by
  ≲ 0.1 nat, and it cannot move it anywhere near ln 10.
- Cost: 73.4 A100-h (SLURM `Elapsed`), against §5's ~50 A100-h estimate. The ou30 rungs
  took ~9.6 h each, ~4× an ou14 rung, and account for 48 of the 73 h. `summary.json`'s
  `sample_wall_s` reads higher than `Elapsed` (4.2 h against 2.3 h for some ou14 and pl
  rungs, 11.9 h against 10.0 h for ou30 ε = 1) because it includes time the job spent
  suspended.

## Findings from the reference build (tasks 1.2–1.3)

- **Sign-flipped red-noise γ in the feather noisedicts.** Discovery's v1.1.0 feathers store
  `<psr>_red_noise_gamma` with the wrong sign (all 25 entries are negative). With the stored
  sign the J1909−3744 lnL is 468373.81; with the sign corrected it is 469069.57, about 696
  nats higher and close to the published MAP. The ladders sample red noise, so they are
  unaffected; `reference/check_env.py` asserts the sign and corrects it.
- **`argus.model.get_Q_block` q11 cancels catastrophically at γ·dt ≪ 1.** Relative error
  ≈ ε/(γ dt)², e.g. ~3–4 × 10⁻⁴ at γ = 10⁻¹¹ s⁻¹, dt = 1 d (4.0 × 10⁻⁴ measured). Pinned in
  `test/test_ng15_nb_ou_psd.py`, not fixed here. It matters for Phase 3/4, where the γ_a
  posterior reaches 10⁻¹¹.

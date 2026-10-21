# N3 — Per-pulsar validation: Argus vs Discovery-OU, one pulsar at a time

Spec `spec/spec.md` §11 Phase 3 (tasks 3.1–3.3), check C4, figure F4. This note records
whether each NG15 pulsar's Argus red-noise posterior agrees with the kernel-matched
Discovery-OU reference before any 67-pulsar Argus run.

## Outcome

**C4 passes: 65/67 pulsars (97.0%) are within 1σ at both f = 1/T and f = 1/(5 yr)**
(97.0% at each frequency separately; threshold 90%). The median over pulsars of the larger
z is 0.02. 39 pulsars pile up at the σp floor in both pipelines, so their posteriors are
prior-dominated and agree trivially; among the other 28, C4 holds for 26 (92.9%), one
outlier short of failing. Both outliers point to Discovery-OU's 30-mode Fourier
truncation, confirmed for J1012+5307 and likely for J0437−4715:

| Pulsar | z at 1/T | z at 1/(5 yr) | Shallow | Diagnosis |
|---|---|---|---|---|
| J0437−4715 | 1.59 | 1.74 | no | Likely Discovery's 30-mode truncation, from the likelihood only: Discovery's own lnL at Argus's point is −3.0 nat at 30 modes and +8.1 at 100. |
| J1012+5307 | 1.21 | 1.21 | yes | Discovery's 30-mode truncation, confirmed: a 100-mode Discovery-OU posterior agrees with Argus (z = 0.03). |

Neither is an Argus compression error: both pulsars' compressed likelihoods match the
dense oracle at posterior draws (below).

Tasks 3.1 and 3.2 pass for **all 67 pulsars in both pipelines** (r̂ ≤ 1.01, bulk ESS ≥ 400,
0 divergences, no σp rail), after 7 first-pass failures were rerun at a stricter NUTS
target (below).

## Setup

- Data: NG15 v1.1.0 narrowband, 67 pulsars, as in N1/N2. T = 16.030 yr (the 67-pulsar
  array span), so the C4 frequencies are 1/T = 1.98 nHz and 1/(5 yr) = 6.34 nHz.
- Model per pulsar: spin red noise only (integrated OU), fixed white noise and ECORR, the
  timing model marginalised; no common process.
  - **Argus** (`single_psr/run_argus.py`): the pulsar's D2 compression at the adopted
    layout (two scalars per DMX window; epoch cells for J1012+5307, J1643−1224,
    J1705−1903, J1903+0327), the sparse marginal filter, diffuse timing prior. The GW
    block is switched off at h_a = 1e-30; raising it ×1e5 changes lnL by 0.0 at every
    checked point.
  - **Discovery-OU** (`single_psr/run_discovery.py`): `reference/models.py`'s per-pulsar
    pieces with Argus's integrated-OU PSD on 30 Fourier modes over T.
- Priors: exact uniform on the `reference/configs/ou.ini` boxes, log10 γp ∈ [−12, −6],
  log10 σp ∈ [−22, −11.5], identical in both pipelines.
- NUTS: 4 chains × (1000 warmup + 4000 draws), target 0.9, tree depth 8,
  seed 3100 + manifest index. The 4000 draws (the frozen procedure has 1000) are what make
  ESS ≥ 400 reachable along the slowly mixing γp direction.
- "Within 1σ" (spec C4, defined in the 2026-10-20 amendment):
  z = |m_A − m_D| / √(σ_A² + σ_D²) ≤ 1 on log10 PSD, σ = (q84 − q16)/2, at both
  frequencies. `test/test_ng15_nb_c4.py` holds the known-answer tests.
- Code and SLURM jobs (one A100 per pulsar):
  - pinned worktree `9dbdc75`: first-pass arrays 18812223 (Argus) and 18812224
    (Discovery); 18859905, 18862036 and 18862645 re-ran the 17 Argus pulsars that hit
    the original 4 h limit, at 24 h; 18812217 ran the prior-box oracle checks;
  - `d0fa1e7`: the stricter-target reruns 18896014, 18896015, 18915817 (Argus) and
    18896016 (Discovery), and the J0437−4715 oracle check 18915823;
  - `58407b8` and `f882918`: the first γp spot check and the other posterior-point oracle
    checks. The B1855+09 prior-box check records `58407b8`, but ran from a tree holding the
    origin-shift code before it was committed;
  - `5d6ca54` and `143ee87`: the γp spot check with both ceilings raised (18955930), the
    J1012+5307 mode test (same job) and its 100-mode posterior (18957720).

  Readout `single_psr/c4_compare.py` → `results/single_psr/c4.json` and
  `f4_psd_scatter.png`. c4.json is authoritative for tasks 3.1/3.2: it re-judges every
  record under the 2026-10-20 rule, while 26 per-pulsar JSONs written before that rule
  still carry `passes_3_1: false` for a γp-ceiling "rail". F4's hollow marker uses Argus's
  σp-floor flag, so J0437−4715 plots as a detection though Discovery piles up there.
- Reruns write under `<pipeline>_ta<target>` whatever `--num-samples` is, so the 8000-draw
  J0437−4715 and J0645+5158 runs share `argus_ta0.99/` with the 4000-draw ones; J0645's
  failed 4000-draw rerun was moved to `argus_ta0.99_n4000/` by hand. Rerunning with the
  default draws would overwrite them.

## Tasks 3.1 and 3.2: sampler diagnostics

| | Argus | Discovery-OU |
|---|---|---|
| Pass, first pass | 62/67 | 65/67 |
| Pass, final | **67/67** | **67/67** |
| Max r̂ (final) | 1.0046 | 1.0064 |
| Min bulk ESS (final) | 871 | 918 |
| Divergences (final) | 0 | 0 |
| σp rails | 0 | 0 |
| No detection (lower σp edge pile-up, reported) | 39 | 40 |
| Detected, at OU's f⁻² limit (γp ceiling > 5%, reported) | 14 | 12 |

First-pass failures and their reruns (`--target-accept 0.99`, under `<pipeline>_ta0.99/`):

| Pipeline | Pulsar | First pass | Rerun |
|---|---|---|---|
| Argus | B1937+21 | 12 divergences | 0 div., r̂ 1.001, ESS 3639 |
| Argus | J0437−4715 | ESS 366 | 8000 draws: 0 div., r̂ 1.001, ESS 1315 |
| Argus | J0645+5158 | 103 divergences, r̂ 1.015 | 4000 draws: 0 div. but r̂ 1.0103 (fails, kept in `argus_ta0.99_n4000/`); 8000 draws: r̂ 1.001, ESS 1836 |
| Argus | J1909−3744 | 6 divergences | 0 div., r̂ 1.002, ESS 1907 |
| Argus | J1911+1347 | 2 divergences | 0 div., r̂ 1.001, ESS 3087 |
| Discovery | B1937+21 | 18 divergences | 0 div., r̂ 1.002, ESS 3633 |
| Discovery | J1909−3744 | 70 divergences | 0 div., r̂ 1.003, ESS 2370 |

B1937+21 and J1909−3744 diverged in both pipelines at target 0.9, which points at the
posterior's geometry (sharp, high-S/N red noise), not either code. C4 uses the rerun
wherever one exists; a rerun changes no pulsar's C4 verdict (J0437−4715's z moves 1.58 → 1.59).

Detected pulsars at the f⁻² limit. Argus: J0437−4715, J0645+5158, J1012+5307, J1600−3053,
J1643−1224, J1705−1903, J1713+0747, J1745+1017, J1747−4036, J1802−2124, J1853+1303,
J1903+0327, J1946+3417, J2145−0750. Discovery has the same list less J0437−4715 and
J1747−4036. No pulsar has > 5% of its mass on the γp floor.

### The γp ceiling (spec §12, 2026-10-20 and 2026-10-22)

Task 3.1 as first written failed a detected pulsar with mass at any prior edge. Mass at the
γp ceiling is OU's shallowest (f⁻²) spectrum, which shallow-spectrum pulsars want (A3), so
the amendment reports it and gates only the σp edges. A spot check on three of them
raised Discovery's γp ceiling to −4, and then both ceilings (γp −4, σp −9):

| Pulsar | γp median: −6 → γp −4 → both | Ceiling mass at γp −4 | σp ceiling mass at γp −4 | γp ceiling mass, both | log10 S(1/T) median: −6 → γp −4 → both |
|---|---|---|---|---|---|
| J1745+1017 | −7.07 → −6.66 → −5.74 | 0.0 | 0.12 | 0.05 | −0.01 → −0.41 → −0.64 |
| J2145−0750 | −6.32 → −5.29 → −5.27 | 0.059 | 0.0 | 0.06 | −3.40 → −3.42 → −3.42 |
| J1903+0327 | −6.28 → −5.92 → −5.31 | 0.0 | 0.17 | 0.06 | −1.55 → −1.61 → −1.62 |

Raising γp's ceiling alone moved two pulsars onto the σp ceiling, since along the f⁻² ridge
σp rises with γp. With both raised, all three run on to the new γp ceiling: the likelihood
is near-flat along the ridge, so no finite ceiling holds the mass (an earlier reading of
the first check, that the posteriors close off below −4, was wrong). The in-band PSD
moves by ≤ 0.07 dex for two pulsars and 0.6 dex, within a posterior 1.6 dex wide, for
J1745+1017. Whether production (Phase 5) should change the box is a §9 question.

## Likelihood checks: Argus vs the dense oracle

The compressed single-pulsar likelihood against `oracle.oracle_loglike` on the raw TOAs.
The oracle's own precision is the spread of its lnL when its time origin moves by
1000 days (exact in theory).

| Pulsar | Points | Argus − oracle spread [nat] | Oracle floor [nat] | Verdict |
|---|---|---|---|---|
| J1909−3744 | 20 posterior | 1.9e-4 | 1.6e-4 | pass |
| B1937+21 | 20 posterior | 1.4e-2 | 2.4e-2 | pass (C1's oracle-precision clause) |
| J1012+5307 (epoch cells) | 20 posterior | 3.2e-8 | — | pass |
| J0437−4715 | 20 posterior | 3.7e-3 | 8e-9 | pass (< 0.01) |
| B1855+09 | 3 prior box | 2.8e-3 | 4.4 | pass |

At the prior-box corners the oracle itself loses precision (B1855+09's floor there is
4.4 nat). J1909−3744 (16.6 nat) and B1937+21 (0.95 nat) differ from the oracle there too;
their prior-box floors were not measured, so these are set aside by analogy with
B1855+09, and the posterior-point checks are the ones that count.

## The outliers and Discovery-OU's 30-mode truncation

### J1012+5307: confirmed by a 100-mode posterior

Both pipelines put J1012+5307 at the γp ceiling, but with 84% (Argus) and 48% (Discovery)
of the mass there, and the in-band PSDs sit 0.23 dex apart with σ ≈ 0.13 dex. Both run on
the same box, so the box edge cannot explain the gap. Discovery's lnL at
log10 γp = −6.05 (`single_psr/j0437_modes.py --pulsar J1012+5307 --gamma-p -6.05`),
relative to no red noise:

| Modes | log10 σp = −14 | −13.4 | −13 |
|---|---|---|---|
| 30 | +34.0 | +77.2 | +84.9 |
| 100 | +34.3 | +80.4 | +96.3 |
| 300 | +34.3 | +80.6 | +96.9 |

More modes favour the higher σp that Argus finds, converging by 100. The 100-mode
Discovery-OU posterior (`run_discovery.py --pulsar J1012+5307 --rn-components 100`, under
`discovery_m100/`; r̂ 1.001, ESS 3332, 0 divergences) gives log10 γp −6.051, log10 σp
−12.997, 82% of the mass at the ceiling, and log10 S(1/T) −2.313 against Argus's −2.307:
z = 0.03 at both frequencies.

### J0437−4715: the likelihood only

Argus detects red noise in J0437−4715 (log10 σp ≈ −13.4, log10 γp ≈ −6.2 with 33% of
its mass at the ceiling); 30-mode Discovery-OU gives only an upper limit
(log10 σp ≈ −18.5 ± 2.3). C4's z is only 1.6–1.7 because Discovery's posterior is so wide.
Discovery's own lnL at Argus's point, relative to no red noise
(`single_psr/j0437_modes.py`):

| Modes | log10 σp = −16 | −14 | −13.4 | −13 |
|---|---|---|---|---|
| 30 | 0.0 | −1.2 | −3.0 | −3.6 |
| 100 | 0.0 | +2.7 | +8.1 | +5.3 |
| 300 | 0.0 | +2.7 | +8.3 | +5.4 |

The power lies above 30/T ≈ 1.9 yr⁻¹, outside the 30-mode basis; Argus's continuous-time
kernel has no truncation. This is a kernel-matching gap in the reference, not an Argus
error. It is recorded as an observation, not a claim. Open caveats, parked for now:

- only the likelihood at one point was checked, not a 100-mode Discovery posterior;
- the power may be mis-modelled white or chromatic noise absorbed by OU's f⁻² edge,
  rather than astrophysical red noise (J0437−4715 is the array's most precise pulsar);
- it sits far above the GW band, so its bearing on the GWB is likely small;
- whether other pulsars show the same 30-vs-100-mode gap is unchecked (a §9 question for
  C5's reference).

## Cost

Argus single-pulsar runs took a median 2.4 h per pulsar on one A100 (5000 NUTS iterations
of the latency-bound sequential filter), up to 11.7 h at target 0.9 and 17.5 h for
J1909−3744 at 0.99, ≈ 256 A100-h of sampling in all (§5 budgeted ~70). Discovery-OU took
about a minute per pulsar of sampling at 30 modes and about 2 minutes at 100. This bears on §5's budget
question for Phase 5, not on C4.

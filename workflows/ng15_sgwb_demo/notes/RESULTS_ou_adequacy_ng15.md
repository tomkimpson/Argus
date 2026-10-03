# Is Argus's OU per-pulsar red noise adequate for real NG15? (2026-10-02)

Branch `feat/ng15-ou-adequacy`. Motivation: MDC2 2b's +1.54 dex pivot bias was blamed on
OU-vs-power-law misspecification, but 2b's red noise was *generated* as power laws, so it
cannot say whether real pulsars need one. This tests OU against what NANOGrav actually sees.

Detail tables: `notes/ng15_red_noise_budget.md` (steps 1-2), `notes/ng15_single_psr_ou.md`
(step 3). Plots: `outputs/ng15_ou_adequacy/step{2,3}_spectra.png` (gitignored).

## Step 0 — the repo's OU PSD is two-sided (0.30 dex)

`ou_psd` (`inject_powerlaw_gwb.py`) and every copy of it (`check_mdc2_truth.py`,
`compare_ou_recovery.py`, the spike's pivot-matched sigma_p) are **two-sided**, but are
compared with the **one-sided** enterprise power law. `scripts/check_psd_sidedness.py`
simulates with the repo's own generators and takes one-sided Welch PSDs of the
first-differenced residual:

| series | median log10(Welch / model) |
|---|---|
| white-noise control (one-sided 2s²dt) | −0.005 |
| per-pulsar OU red noise vs `S_OU` | +0.307 |
| GW OU vs `ou_psd` | +0.293 |

So every OU-vs-power-law pivot comparison so far read the OU 0.30 dex too LOW.
OU-vs-OU comparisons (the 6-seed calibration, the self-gen spike) are unaffected.

**Fixed 2026-10-03.** Script-side only. `ou_psd`, `check_mdc2_truth.ou_residual_psd` and
`compare_ou_recovery.ou_residual_psd` are now one-sided (factor 2). `check_psd_sidedness.py`
is now a regression check (offsets +0.006 / −0.008 dex). `test/test_injection_psd.py` pins the
convention analytically. Truth sidecars carry a `psd_convention` key, and the old OU ones are
marked two-sided. The library's ridge parameter `log10_pivot_psd` is deliberately unchanged
(the frozen evidence procedure and every ridge config depend on it). It stays the TWO-sided
density, now documented: one-sided = `log10_pivot_psd + log10 2`.

Corrected readouts (`check_mdc2_truth.py`, pivot 1/(5 yr)):

| run | recovered (one-sided) | injected | bias |
|---|---|---|---|
| 2b flat ε=0 (`mdc2_flat_eps000_psym`) | −4.475 ± 0.064 | −6.319 | **+1.84 dex** (was quoted +1.54) |
| 1b flat ε=0 (`mdc2_d1_flat_psym_eps000`) | −6.243 ± 0.708 | −6.908 | +0.94σ, covered |
| M1 Stage C path-sampled | −6.541 ± 1.80 | −6.319 | −0.12σ, covered |

The task-5.1 "matched" OU injection (`log10_ha = −12.919767`) is 0.30 dex louder than its
power-law partner; the matched value is −13.070282. Errata are prepended to the affected notes.

## Step 1 — which pulsars have red noise above the white floor?

Published NG15 wideband power-law red-noise posteriors (`{PSR}_red_noise_{log10_A,gamma}`
in the release chains) vs the one-sided white floor of the 30-day-binned data with the
collapsed scalar EFAC/EQUAD. Selected = 5th-percentile red PSD at f_1 = 1/T above the
floor; in-band = f_k (k ≤ 30, NANOGrav's red-noise basis) with median red PSD above the floor.

**25 of 68 selected.** The other 43 have no red noise the data can see, so the kernel
choice cannot matter for them. Sanity: B1937+21 and J1909-3744 selected, J0437-4715
(prior-dominated) not.

## Step 2 — can one OU spectrum sit inside the published 90% band? (CPU)

Minimax fit of a one-sided OU spectrum to the band at the in-band frequencies; PASS =
inside the 5-95% band at every one. Fitter self-test: exact OU recovered to 0.000 dex,
gamma=6 power law fails.

**20 / 25 pass.** The 5 failures are J1012+5307, J1643-1224, J1903+0327, J1705-1903,
J2234+0944 — all **shallow**, gamma_PL ≈ 0.2-1.1, and every best fit sits at the
gamma_p ceiling. An OU residual spectrum is never shallower than f^-2 (corner above the
band), so it cannot follow them. The failure mode is shallow spectra, **not** steep ones.

## Step 3 — direct single-pulsar Argus OU fits on real data (GPU)

Each of the 25 pulsars on its own 30-day binned epochs, DMX dropped, scalar EFAC/EQUAD
fixed, GW fixed negligible, flat priors log10 gamma_p ∈ [−12, −4], log10 sigma_p ∈
[−20, −9]. All 25 healthy (r_hat ≤ 1.013, ESS ≥ 254, ≤ 0.1% divergences), 7-47 min each
on one A100. PASS = Argus 90% band overlaps the NANOGrav 90% band at every in-band frequency.

**16 / 25 pass.** The 9 failures split into three distinct causes:

| class | pulsars | what happens | cause |
|---|---|---|---|
| shallow spectrum | J1012+5307, J1643-1224, J1903+0327, J2234+0944, J1705-1903 (13-16 of 21-30 bins overlap) | OU forced steeper: ~+1 dex high at f_1, low at high f | **kernel** — predicted by step 2 |
| no red noise seen | J0645+5158, J0610-2100, B1953+29 | Argus posterior is an upper limit 4-6 dex below NANOGrav | **data treatment** — step 2 passed |
| amplitude offset | B1937+21 | same slope, Argus ~+0.8 dex high mid-band | unexplained |

**"No red noise seen" is not the kernel.** For these three, the binned residuals scatter
*less* than their own white errors (chi²/N = 0.19-0.55 even with correctly-binned white
noise). Degrees of freedom were removed upstream: the residuals are NANOGrav post-fit
residuals with one DMX per epoch subtracted, and with few receivers per epoch a DMX offset
is nearly degenerate with an achromatic epoch offset, so it absorbs red power. NANOGrav keeps
DMX in the marginalised design matrix; Argus drops the DMX columns (`build_aligned_feathers`,
necessary after binning) but keeps the DMX-subtracted residuals, so it never sees that power.
Hypothesis consistent with the chi² deficit, not proven.

**Secondary data-treatment effect:** Argus applies the per-TOA EQUAD *after* binning, where it
should shrink as 1/sqrt(TOAs per bin). This inflates the white level by up to 0.33 dex in rms
(J1713+0747, B1937+21, J1909-3744). It is small next to the classes above but biases red
noise low wherever EQUAD dominates.

## Bottom line

- **OU is adequate for most of NG15:**
  - 43 / 68 pulsars have no visible red noise.
  - Of the 25 that do, OU matches 16 directly and 20 in shape.
  - These include the GWB-sensitive steep-noise pulsars (J1909-3744, J1713+0747, J0613-0200,
    J1744-1134, B1855+09, J2145-0750, J1918-0642).
  - A universal switch to a power-law kernel is not justified by the data.
- **OU genuinely fails on 5 shallow-spectrum pulsars** (gamma_PL ≲ 1). This is the same
  failure mode as MDC2 2b (gamma 2.3-2.7, shape not steepness), and it is the one most likely
  to push power into a common term. The targeted fix is a kernel that reaches slopes below
  f^-2. For example, add a second OU on the *phase* state, whose residual spectrum runs from
  flat to f^-2, so the mixture spans 0-4. Alternatively exclude these pulsars, at a stated
  cost.
- **The bigger M3 risk found here is data treatment, not the kernel.** DMX-subtracted
  residuals combined with a DMX-free design matrix hide red power (3 pulsars here, and
  plausibly partly elsewhere), and post-binning EQUAD inflates white noise. Both bias red
  noise, and therefore potentially the GWB, LOW. They need resolving before an M3 run.
- **Sidedness:** fixed in the scripts on 2026-10-03 (see step 0). Older quoted OU-vs-power-law
  numbers carry errata.

Caveat: the NANOGrav reference is itself a posterior *under a power-law model* with its own
data treatment (unbinned, per-backend white noise, DMX marginalised). Disagreement says the
two pipelines disagree; the attributions above are the most direct reading of the
diagnostics, not proofs.

## Reproduce

```
python scripts/check_psd_sidedness.py                      # step 0, ~15 s
python scripts/ng15_red_noise_budget.py --self-test        # fitter check
python scripts/ng15_red_noise_budget.py                    # steps 1-2, ~1 min (SLURM CPU)
sbatch slurm_scripts/ng15_single_prep.sh                   # step 3 data, ~25 min CPU
sbatch --array=0-24 slurm_scripts/ng15_single_psr.sh       # step 3 fits, 25 x 1 A100
python scripts/compare_single_psr_ou.py                    # step 3 readout
```

# Is Argus's OU per-pulsar red noise adequate for real NG15? (2026-10-02)

Branch `feat/ng15-ou-adequacy`. Motivation: MDC2 2b's pivot bias (+1.84 dex one-sided; quoted
+1.54 before the step-0 fix) was blamed on
OU-vs-power-law misspecification, but 2b's red noise was *generated* as power laws, so it
cannot say whether real pulsars need one. This tests OU against what NANOGrav actually sees.

Detail tables: `notes/ng15_red_noise_budget.md` (steps 1-2), `notes/ng15_single_psr_ou.md`
(step 3), `notes/ng15_white_noise_diagnostics.md` (step-3 white-noise diagnostics). Plots:
`outputs/ng15_ou_adequacy/step{2,3}_spectra.png` (gitignored).

> **Prior caveat (found in review 2026-10-03).** `red_noise_prior = flat` is NOT uniform on the
> NUTS path. `parameter_sampling.sample_reparameterized_parameters` draws N(0, 1/sqrt N) and maps
> it to `mid + z (hi-lo)/6`, with no Uniform term, so the prior is an unbounded Gaussian
> N(mid, (hi-lo)/(6 sqrt N)). This is a pre-existing library bug, not fixed here. Step 3 (N = 1)
> therefore ran with log10 gamma_p ~ N(−8, 1.33) and log10 sigma_p ~ N(−14.5, 1.83), not flat
> boxes. It also affects every earlier 33-pulsar "flat prior" MDC2 result, where the prior sd is
> 0.17 / 0.23 dex.

## Step 0 — the repo's OU PSD was two-sided (0.30 dex); fixed

Before the fix, `ou_psd` (`inject_powerlaw_gwb.py`) and every copy of it (`check_mdc2_truth.py`,
`compare_ou_recovery.py`, the spike's pivot-matched sigma_p) were **two-sided**, but were
compared with the **one-sided** enterprise power law. `scripts/check_psd_sidedness.py`
simulates with the repo's own generators and takes one-sided Welch PSDs of the
first-differenced residual. Values below are from BEFORE the fix; after it the two OU rows read
+0.006 / −0.008:

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
fixed, GW fixed negligible. `red_noise_prior = flat` with boxes log10 gamma_p ∈ [−12, −4],
log10 sigma_p ∈ [−20, −9]. What actually ran is Gaussian, N(−8, 1.33) × N(−14.5, 1.83), unbounded
(see the caveat above). All 25 healthy (r_hat ≤ 1.013, ESS ≥ 254, ≤ 0.1% divergences), 7-47 min each
on one A100. PASS = Argus 90% band overlaps the NANOGrav 90% band at every in-band frequency.
That is lenient: two equal-width bands keep overlapping until their medians are ~2.3σ apart.
So PASS means "not inconsistent", not "matched".

**16 / 25 pass.** The 9 failures split into three distinct causes:

| class | pulsars | what happens | cause |
|---|---|---|---|
| shallow spectrum | J1012+5307, J1643-1224, J1903+0327, J2234+0944, J1705-1903 (13-16 of 21-30 bins overlap) | OU forced steeper: ~+1 dex high at f_1, low at high f | **kernel** — predicted by step 2 |
| no red noise seen | J0645+5158, J0610-2100, B1953+29 | Argus posterior is an upper limit well below NANOGrav (how far below, quoted as 4-6 dex, is set by the Gaussian prior tail) | **data treatment** — step 2 passed |
| amplitude offset | B1937+21 | same slope, Argus ~+0.8 dex high mid-band | unexplained |

**"No red noise seen" is not the kernel.** For these three, the binned residuals scatter
*less* than their own white errors. chi²/N is 0.20 / 0.27 / 0.55 (J0645 / J0610 / B1953) against
the correctly binned white noise (`scripts/ng15_white_noise_diagnostics.py`). Degrees of freedom were removed upstream: the residuals are NANOGrav post-fit
residuals with one DMX per epoch subtracted, and with few receivers per epoch a DMX offset
is nearly degenerate with an achromatic epoch offset, so it absorbs red power. NANOGrav keeps
DMX in the marginalised design matrix; Argus drops the DMX columns (`build_aligned_feathers`,
necessary after binning) but keeps the DMX-subtracted residuals, so it never sees that power.
Hypothesis consistent with the chi² deficit, not proven.

**Secondary data-treatment effect:** Argus applies the per-TOA EQUAD *after* binning, where it
should shrink as 1/sqrt(TOAs per bin). Using the white-floor summary 1/mean(1/var), this inflates
the white level by up to **0.41 dex** in rms: J1713+0747 +0.41, B1937+21 +0.29, J1909-3744 +0.26
(`scripts/ng15_white_noise_diagnostics.py`). It is small next to the classes above but biases
red noise low wherever EQUAD dominates. Step 1's white floor uses the same post-binning EQUAD,
so its 25/68 selection is conservative.

## Bottom line

- **OU is adequate for most of NG15:**
  - 43 / 68 pulsars have no visible red noise.
  - Of the 25 that do, OU is not inconsistent with 16 directly and fits 20 in shape (one of
    the 20, J2234+0611, has only 2 in-band bins, so its pass is uninformative).
  - These include the GWB-sensitive steep-noise pulsars (J1909-3744, J1713+0747, J0613-0200,
    J1744-1134, B1855+09, J2145-0750, J1918-0642).
  - A universal switch to a power-law kernel is not justified by the data.
- **OU genuinely fails on 5 shallow-spectrum pulsars** (gamma_PL ≲ 1). That is shallower than
  OU's f^-2 limit. It is a shape failure, like MDC2 2b's, but not the same case: 2b's
  gamma 2.3-2.7 lies inside OU's 2-4 slope range. It is the failure most likely to push power
  into a common term. The targeted fix is a kernel that reaches slopes below
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
python scripts/ng15_red_noise_budget.py                    # steps 1-2, ~30 s CPU
sbatch slurm_scripts/ng15_single_prep.sh                   # step 3 data (PINT ingest, cold ~25 min)
sbatch --array=0-24 slurm_scripts/ng15_single_psr.sh       # step 3 fits, 25 x 1 A100
python scripts/compare_single_psr_ou.py                    # step 3 readout
python scripts/ng15_white_noise_diagnostics.py             # chi2/N and EQUAD inflation
# step-0 corrected readouts (1b needs its own amplitude, else 2b's is used silently):
python scripts/check_mdc2_truth.py --run mdc2_flat_eps000_psym
python scripts/check_mdc2_truth.py --run mdc2_d1_flat_psym_eps000 --log10-a -15.18045606445813
python scripts/check_mdc2_truth.py --run mdc2_stageC_path_sampled
```

Inputs default to the OzSTAR NG15 release path (`NG15_ROOT` in `reduce_ng15_white_noise.py`;
`ng15_red_noise_budget.py --ng15-wideband` overrides it).

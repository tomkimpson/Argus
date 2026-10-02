# Task 4.5 — the null ensemble: size, resolution, cost, and the decision to defer

**Date:** 2026-09-08 · **Task:** `sgwb-detection-route` 4.5 · **Status:** decided

`sgwb/null-calibration` requires that the ensemble be sized *before* it is launched:

> **Requirement: The calibration ensemble size is planned and its cost bounded.** The number of
> null realisations SHALL be chosen and recorded in advance from the resolution needed for the
> intended claim, and the calibration SHALL use a per-realisation procedure cheap enough to
> make that ensemble feasible on the available compute.

This note is that record. **The decision is to defer the ensemble**, and to keep the reported
bound at `p < 1` until it is run.

## The re-cost that was owed

`design.md` costed the null calibration on the assumption that estimator A would win the
bake-off, and wrote the contingency down explicitly:

> Estimator A (one run per realisation) is what makes this feasible at all; **if the bake-off
> forces estimator B, the null calibration is re-costed before the full-array stage, and may be
> reported at 33 pulsars only.**

The bake-off (task 1.8) *did* force estimator B: A agreed on value (0.105 ± 0.095 against
0.175 ± 0.017, 0.73σ) but failed its own `fold_agreement` gate, i.e. it declines to certify in
the regime this project is in. The re-cost has been outstanding since. It is below.

## The cost, measured rather than estimated

One realisation is a full 5-rung path-sampling ladder plus a CPU readout. Measured wall times
on the three ladders run so far, all on `gpu:4`:

| ladder | per-rung wall time |
|---|---|
| `mdc2_d1_ladder` (empirical priors, 1b) | 4.1–5.0 h |
| `mdc2_d1_flat` (flat priors, 1b) | 1.5–2.4 h |
| `mdc2_d1_null` (flat priors, 1b, scrambled) | 1.6–2.3 h |

So a flat-prior realisation is **5 rungs × ~2 h × 4 GPUs ≈ 40 A100-hours**, plus ~30 min of
CPU readout at ~19 GB. Rungs are array tasks and run concurrently, so the *wall* clock is
~2.5 h; the *cost* is 40 GPU-hours regardless.

| N | p-value floor when the observed statistic exceeds every realisation | cost at 33 pulsars |
|---|---|---|
| 1 | `p < 1` | 40 A100-h (**spent — task 4.2b**) |
| 10 | `p < 0.1` | ~400 A100-h |
| 20 | `p < 0.05` | ~800 A100-h |
| 100 | `p < 0.01` | ~4000 A100-h |

At 68 pulsars every row is worse, and the array run itself is the other dominant cost.

## The decision: defer 4.3, 4.4, 4.6 and 4.7

A false-alarm probability is only worth what it is spent on. At 33 pulsars on MDC2 the answer
is already known — the signal is injected — so an ensemble here buys a rehearsal of the
machinery, not a claim. The claim happens on real NG15 data, and the spec has already pre-baked
a degraded null exactly there: task 6.7 runs the null calibration on the subset "at whatever
ensemble size the compute supports", and 7.6 at "the affordable ensemble size". Spending
~400–4000 A100-hours now, on the dataset where the answer is not in question, would take that
budget away from the stage where it is.

So:

- **4.3** (warm-start vs full-warmup agreement) — deferred. It exists to halve the per-realisation
  cost, which only matters once an ensemble is actually being run. Note that if the eventual
  ensemble is run cold, 4.3 is not needed at all; it is a cost lever, not a correctness gate.
- **4.4** (statistic variance vs chain length) — deferred. It sizes the chains for an ensemble
  that is not being launched, and it needs pilot scrambles to measure.
- **4.6** (the MDC2 ensemble and its false-alarm probability) — deferred to 6.7 / 7.6.
- **4.7** (the ensemble on the matched no-injection control) — deferred with 4.6. The control
  dataset it needs now exists (`data/mdc2_d1_nogwb`, built for task 1.9), so this becomes cheap
  to pick up if the decision is revisited.

## Why this is compliant rather than evasive

The spec anticipates precisely this situation and says what to do:

> Scenario: Ensemble not affordable
> - **WHEN** the ensemble needed for the intended claim is not affordable on the available compute
> - **THEN** the claim is downgraded to the resolution the affordable ensemble supports, and the
>   limitation is stated with the result

and, for the reporting side:

> Scenario: Observed value beyond every null realisation
> - **THEN** the result is reported as an upper bound `p < 1/N` for the ensemble size `N`, and is
>   not reported as a smaller number than the ensemble can resolve

Deferring downgrades the claim to `N = 1`, i.e. `p < 1`, which is *no significance at all* — and
that is what will be stated. What the single realisation from 4.2b does buy is a falsification
check, and it passed decisively: `lnB = -0.766 ± 0.010` under a scrambled sky against
`+3.043 ± 0.015` with the true ORF, with the integrand negative and monotonic at every rung. The
estimator responds to the correlation pattern rather than to prior width. That is a statement
about the method, not a probability, and it must not be presented as one.

## The standing obligation this creates

**No detection claim may quote a false-alarm probability until an ensemble exists.** Any result
artefact produced before then states `p < 1` with `N = 1`, names the single scramble as a
falsification check, and says the ensemble was deferred on cost. Task 7.8 assembles the result
artefact and is where this is most likely to be forgotten.

## What would reopen this

- Warm starts validated (4.2) and shown to reproduce the parent posterior halve the cost to
  ~20 A100-h per realisation, putting `p < 0.1` within ~200 A100-h.
- A measured statistic-variance-vs-length curve (4.4) that shows short chains suffice would cut
  it further; the scatter is currently unmeasured, so this is a real unknown rather than an
  assumed saving.
- Any change that makes estimator A usable would return the cost to one run per realisation,
  which is what made the ensemble look affordable in the original design.

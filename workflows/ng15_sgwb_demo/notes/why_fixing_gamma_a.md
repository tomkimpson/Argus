# Why fixing `log10_gamma_a` kills the power-law→OU posterior pathology

**Status:** reasoning note (T2.4 follow-up). §1–§7 were written *before* the fixed-γₐ validation
sweep returned and lay out the a-priori argument. **⚠️ OUTCOME (2026-07-09): the sweep REFUTED
the key safety claim** — fixing γₐ cures the sampling pathology but the recovered band amplitude
is **not** invariant to the fixed value (it is stolen by the free per-pulsar red noise), so
γₐ-fixing is **not** a safe standalone cure here. Read §4's invariance argument together with
[Empirical validation](#empirical-validation) at the bottom, which shows *why* it fails.

**One-line summary:** the OU band power depends on the product `h_a²·γ_a`, so `(h_a, γ_a)`
trade along a curved ridge that NUTS cannot traverse cleanly. Fixing `γ_a` collapses the
ridge to a single well-conditioned scale (`h_a`). It is safe because the *observable* — the
band-referenced amplitude — is invariant to the (unidentifiable) corner as long as the corner
sits at or below the observation band.

---

## 1. The problem

Argus models the SGWB as a single-corner Ornstein–Uhlenbeck (OU) process under a fixed
Hellings–Downs template. On the **power-law-injected** data (the realistic proxy for real
NG15, `log10_A_gw=-14.6`, `γ=13/3`) the recovery posterior is **pathological**, and more
compute barely helps:

| run | max r̂ | min ESS | divergences |
|---|---|---|---|
| power-law→OU, lite (200/100/2) | 1.050 | 34 | 0 (but stuck) |
| power-law→OU, hi-res (1000/1000/4, dense_mass, tree 10) | 1.051 | 51 | **50** |
| OU→OU control, hi-res (same settings) | 1.004 | 2496 | 0 |

The pathology is **specific to model mis-specification** (power-law injection). The
OU-injected control converges cleanly with `log10_gamma_a` identified near its injected value
(−8.30±0.70 vs injected −8.5). See §5.

---

## 2. The OU residual PSD and its two regimes

The injected/recovered OU residual power spectral density (matches the injector and the
`injection_truth.json` OU note) is

```
S_r(f) = σ_a² / [ (2πf)² ( γ_a² + (2πf)² ) ] ,   with   σ_a² = (h_a²/12) · γ_a
```

Write `ω = 2πf` and the corner `f_c = γ_a / 2π`. Two limits:

- **Above the corner** (`ω ≫ γ_a`, i.e. `f ≫ f_c`):
  `γ_a² + ω² ≈ ω²`, so
  ```
  S_r(f) ≈ (h_a²/12) · γ_a / ω⁴   ∝   h_a² · γ_a · f⁻⁴
  ```
- **Below the corner** (`ω ≪ γ_a`, i.e. `f ≪ f_c`):
  `γ_a² + ω² ≈ γ_a²`, so
  ```
  S_r(f) ≈ (h_a²/12) / (γ_a · ω²)   ∝   (h_a² / γ_a) · f⁻²
  ```

So `f⁻⁴` is the *steepest* slope the OU can produce (well above its corner) — the closest it
can get to the true `f⁻¹³ᐟ³ ≈ f⁻⁴·³³`.

---

## 3. Why the posterior is a ridge (the degeneracy)

A PTA constrains the amplitude only in its sensitive band (~`1/T` to a few/`T`). Put the
corner at or below that band and the whole band lives in the `f⁻⁴` regime, where

```
band power   S_r(f*)  ∝  h_a² · γ_a       ⇔     2·log10_h_a + log10_γ_a = const.
```

That single combination is what the data pin down. The **orthogonal** direction —
`h_a² / γ_a`, i.e. `2·log10_h_a − log10_γ_a` — is essentially unconstrained: you can raise
`γ_a` and lower `h_a` together and keep the band power (hence the likelihood) almost
unchanged. In `(log10_h_a, log10_γ_a)` space that is a long, curved ridge. NUTS has to crawl
along a curving, poorly-conditioned valley → the divergences, the stuck ESS, the elevated r̂.

`dense_mass` learns a *global* linear correlation, which is why it helped a little, but the
ridge is *curved* (the corner also reshapes the spectrum), so a single mass matrix can't
straighten it.

---

## 4. Why fixing `γ_a` removes it — and why that's safe

**It removes the pathology.** Hold `γ_a` at a constant and the degenerate direction is gone:
the only free GW scale is `h_a`, and in the `f⁻⁴` regime `S_r(f*) ∝ h_a²` maps monotonically
and 1:1 to the band amplitude. A single well-behaved scale parameter → NUTS should return 0
divergences and high ESS. This is the textbook cure for ridge-induced pathology: fix (or
reparameterize away) the unconstrained degenerate direction.

**It does not bias the observable — the invariance argument.** Our reported quantity is the
band-referenced amplitude `S_r(f*)` at `f* ≈ 1/(5 yr)`. Suppose we fix `γ_a` to `c₁` vs `c₂`
(both below the band). In each case the fit drives `h_a` to reproduce the *same*
data-determined band power:

```
h_a²(c) · c  =  (band power)   ⇒   h_a(c) ∝ 1/√c ,   but   S_r(f*)  unchanged.
```

So the choice of fixed `γ_a` shifts only the *nuisance* value of `h_a`; the **band amplitude
we actually claim is invariant**. We lose nothing measurable, because (§5) the corner is not
identifiable for a power-law signal in the first place — there is no "true" corner to find.

**Precedent.** This is exactly the simplification used for OU models of pulsar spin-down /
timing noise: the mean-reversion timescale is weakly constrained, so it is fixed to a
physical value and the amplitude is read off. Same logic, same justification.

---

## 5. Scope: this only matters for (mis-specified) power-law-like data

- **OU-injected data → correctly specified → fix nothing.** There is a genuine `(h_a, γ_a)`
  that generated the data, the data really has a corner, and `γ_a` is identifiable. The
  control posterior is a proper interior blob (−8.30±0.70), 0 divergences, ESS 2496. Fixing
  `γ_a` would be unnecessary (though harmless — the band amplitude still comes back).
- **Power-law-injected data → mis-specified → fixing `γ_a` is the right move.** No true
  corner exists; many `(h_a, γ_a)` fit equally imperfectly → ridge → pathology.
- **Real NG15 data is believed to be a power law**, so on real data we are in the
  mis-specified regime and would hit the same pathology. That is *why* this matters for
  Stage 3, not merely for a synthetic test.

---

## 6. What value to fix it at

Choose the corner **at or below the lowest sampled frequency** so the entire band is in the
`f⁻⁴` regime (where §4's invariance holds and the shape best approximates the steep power
law). For this dataset (`T ≈ 12 yr`):

```
f_min = 1/T ≈ 2.6e-9 Hz     ⇒   corner at band bottom ⇔ log10_γ_a ≈ log10(2π/T) ≈ −7.8
```

- `log10_γ_a ≲ −7.8` → corner at/below the band → **safe** (invariance holds).
- Recommended: **`log10_γ_a ≈ −8.5`** — τ_c = 1/γ_a ≈ 10 yr ≈ baseline; matches the injector
  default *and* what the OU control's data actually preferred; corner comfortably below band.
- Fixing the corner **inside** the band (e.g. ≳ −7.5) is *not* safe: part of the band flips
  to the shallow `f⁻²` regime, the shape changes, and the recovered amplitude then depends on
  the choice.

---

## 7. Caveats

1. **Report the band amplitude / strain, not `log10_h_a` or `A_gw`.** With `γ_a` fixed, `h_a`
   is conditional on that choice (§4); its raw value is only meaningful through the fixed
   `γ_a`. The band-referenced PSD/strain (what `scripts/compare_ou_recovery.py` computes) is
   the invariant, publishable number.
2. **A second degeneracy may remain.** On power-law data the GW competes with the **free
   per-pulsar red noise** to absorb the steep low-frequency power. Fixing `γ_a` does not touch
   that GW↔red-noise trade-off. The free fit pushing the corner *up* into the band (the
   "wrong" direction for steepness, −7.3) is a hint that more than the simple ridge is in
   play. So fixing `γ_a` should *greatly* improve conditioning but might not give a perfectly
   clean posterior — verify, don't assume.
3. **Fixing `γ_a` bakes in one OU shape.** Since Argus can't match `f⁻¹³ᐟ³` anyway, this only
   chooses *which* imperfect OU shape; combined with caveat 1 it is not a loss, but the result
   is explicitly "amplitude under an assumed OU corner," consistent with the RISK-B honest
   framing in `PLAN.md` §2/§6.

---

## Empirical validation

**Prediction (to confirm/refute):** fixing `log10_gamma_a` at each of {−8.5, −8.0, −7.8} on
the power-law-injected data should give (a) divergences ≈ 0, ESS ≫ 51, r̂ ≈ 1 (pathology
gone), **and** (b) a recovered band amplitude at `f=1/(5 yr)` that is *stable across all three*
and ≈ −0.2 dex vs the injected truth (invariance). The three values span "well below band"
(−8.5) to "at band bottom" (−7.8), so any drift at −7.8 would map the safe range.

Test harness: `configs/ng15_config_fixg.ini` + `slurm_scripts/ng15_fixg_sweep.sh`
(`target_accept=0.90`, identical to the pathological confirm run, so a 50→0 divergence drop
is cleanly attributable to fixing `γ_a`). Analyse with
`scripts/compare_ou_recovery.py --mode powerlaw --run-prefix ng15_fixg_<tag>`.

**Results (job 14031438, 2026-07-09):** prediction **half-confirmed, half-REFUTED.** The
pathology is cured, but the band amplitude is **NOT invariant** — it tracks the fixed `γ_a`.

| fixed log10_γ_a | divergences | min ESS | max r̂ | recovered log10_h_a | band bias @1/(5yr) [amp dex / σ] |
|---|---|---|---|---|---|
| −8.5 | **0** | 551 | 1.005 | −14.83 ± 0.65 | **−1.95 / −3.0σ** |
| −8.0 | **0** | 268 | 1.013 | −14.42 ± 0.83 | **−1.30 / −1.6σ** |
| −7.8 | _(job timed out at 3 h before this run finished — not obtained)_ ||||| 
| *free-γ_a ref (confirm)* | 50 | 51 | 1.051 | −13.45 | −0.20 / −0.46σ |

**Verdict: fixing `γ_a` removes the pathology but is NOT safe as a standalone cure here.**
- **(a) Pathology cured — confirmed.** Both fixed runs give 0 divergences and ESS 268–551 (vs
  the free run's 50 / 51). Removing the `h_a↔γ_a` ridge does clean up the sampler.
- **(b) Invariance REFUTED.** The recovered band amplitude is *not* invariant to the fixed `γ_a`
  — it drifts monotonically: bias −1.95 dex (−8.5) → −1.30 dex (−8.0) → [−0.20 dex at the
  free-preferred −7.3]. Fixing the corner *below* the band badly **under-recovers** the GW
  amplitude (100× low at −8.5).

**Why the invariance argument (§4) failed — caveat 2 was the dominant effect.** §4 assumed the GW
OU is the *only* red component carrying the band power. It is not: the **free per-pulsar red
noise competes for it.** When `γ_a` is pinned low (corner below band), the OU is a poorer in-band
shape match, so signal power leaks into the (uncorrelated) per-pulsar red noise and the GW
amplitude collapses. The free fit recovered the right amplitude precisely *because* it could float
`γ_a` up to ≈−7.3 (corner in-band), where the GW OU wins the partition against red noise. So the
free-`γ_a` "pathology" was doing real work — it is not a mere nuisance direction.

**Consequence for the plan.** The real lever is the **GW↔red-noise degeneracy**, not the
`h_a↔γ_a` ridge. Do **not** adopt γ_a-fixing with free red noise. Options:
1. **Keep γ_a free** (accept the pathology; push `target_accept≥0.95`, longer warmup) — it does
   recover the band amplitude, just with poor mixing.
2. **Fix/constrain the per-pulsar red noise** (e.g. from single-pulsar noise runs) so it cannot
   steal GW power — the more principled fix, and standard in NANOGrav-style analyses. *Then*
   fixing `γ_a` may become safe (test: fixed red noise × {fixed, free} γ_a).
3. Best candidate to try next: **fixed red noise + fixed γ_a** → likely clean *and* unbiased.

(The −7.8 point was lost to the 3 h wall clock; the −8.5→−8.0 trend is already monotonic and the
conclusion does not depend on it. If we want the full curve, rerun a shorter single-value job.)

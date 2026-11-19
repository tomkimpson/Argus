# Probe 2 (config v2): why log10 γa sits on its −6 ceiling

**Date:** 2026-11-13. **Spec:** Phase 5, task 5.2; §12 entry 2026-11-13.
**Run:** `ng15_nb_v2_probe_eps000`, job 19908499, 4 × A100, cancelled 2026-11-13 at 6 d 7 h
(≈ 18 h before its 7-day limit; segment 2, at 45/100 draws, would not have finished).
**Data:** the segment-1 checkpoint, 200 warmup + 100 draws per chain
(`Argus-ng15nb-p5/workflows/ng15_nb/outputs/ng15_nb_v2_probe_eps000/*_partial.nc`), and
the SLURM log `workflows/ng15_nb/outputs/logfiles/pilot_19908499.out`.
**Numbers:** `probe2_v2_gamma_a_diagnosis.json`, produced by
`workflows/ng15_nb/array/probe_gamma_a_diagnosis.py` (h5py read; arviz on the login node
times out). The readout proper is `probe2_v2_partial_eps000.json` (`pilot_readout.py`).

## Summary

The handoff of 2026-11-12 read probe 2 as "γa split into two non-crossing modes (two chains
at ≈ −6.1, two at ≈ −7.5; r̂ 1.24)". The traces do not support that. γa is a **ceiling
spike with a tail to the prior floor**, and every chain visits both. In the two chains that
move between states, γa tracks **J1713+0747's red noise draw by draw**: when J1713's own OU
term is on and flat in band (γp ≈ −5, σp ≈ −13.7), γa sits near −7.9 and is almost never at
the ceiling; when that term is off, γa returns to −6 and the common process, at its
shallowest, carries the high-frequency excess instead. The other two chains stay in the
"off" state for 84–94 % of their draws and so never test the coupling. The likelihood is
flat in γa throughout: the data do not say which process owns the excess. This is the §10
risk ("white-like excess … leaks into the common process") realised in the form the
2026-11-03 amendment set as the trigger for a model change.

## Findings

1. **One spike, not two modes.** Per chain, γa spans −6.0 down to −9.4, −11.0, −10.9 and
   −9.6; 57 % of all draws lie within 0.5 dex of the ceiling (per chain 0.80, 0.39, 0.27,
   0.82); pooled quantiles (5, 25, 50, 75, 95 %) −10.4, −7.62, −6.29, −6.09, −6.02. Lag-1
   autocorrelation 0.21–0.69. The density is not quite "spike plus flat tail": per 0.5 dex
   (pooled) it runs 15, 14, 7, 7, 10, 10 from −11 to −8, then **51** in [−8, −7.5), 39 in
   [−7.5, −7), a dip of 19 in [−7, −6.5), and the spike of 228 in [−6.5, −6). The bump at
   ≈ −7.75 is in chains 1 and 2 (19 and 29 draws) and is where J1713 is on; chains 0 and 3
   have almost nothing below −7.5 (7 and 1 draws; 10 and 12 below −7). Chain 1 (median −7.36) spends its last ten draws at −6.0 to
   −6.4; chain 3 (median −6.16) visits −7.2 in its last ten. The "two modes" were medians of
   100 autocorrelated draws of a one-sided, heavy-tailed trace.

2. **The likelihood does not prefer the ceiling.** Per-chain lnL medians 7786910.2 to
   7786913.9 (sd ≈ 6.5); mean lnL by γa bin: [−11, −9) 7786910.9, [−9, −7.5) 7786912.5,
   [−7.5, −6.5) 7786909.9, [−6.5, −6) 7786912.1; corr(γa, lnL) per chain −0.02, −0.23,
   0.26, 0.15. The handoff's next step 1 ("does the −6 mode fit better?") is answered: no.

3. **J1713+0747 co-moves with γa, draw by draw, in the chains that test it.** Define a
   pulsar's OU term as *white-like on* when σp > −16 and γp > −6 (on, and flat across the
   band). Per chain, for J1713+0747:

   | chain | white-like on (draws) | γa median, % at ceiling when on | when off | corr(γa, state) |
   |---|---|---|---|---|
   | 0 | 16 | −6.22, 81 % | −6.13, 80 % | +0.08 |
   | 1 | 62 | −7.87, 6 % | −6.13, 92 % | −0.70 |
   | 2 | 56 | −7.95, 7 % | −6.45, 52 % | −0.43 |
   | 3 | 6 | −6.15, 83 % | −6.16, 82 % | −0.01 |

   Chains 1 and 2 switch state 8 and 20 times in 100 draws and show the coupling at the
   draw level. Chains 0 and 3 are in the off state 84 % and 94 % of the time; their on-draws
   are single-draw flickers (run lengths 1–2, 13 and 6 runs) with γa unmoved, so they do not
   test the coupling, and their occupancy of the off basin *is* the mixing failure (γa r̂
   1.24). Binning J1713's on-fraction by γa (pooled): ≥ 0.78 below −7.5 except 0.60 in [−8.5, −8) (n = 10), 0.54 in
   [−7.5, −7.2), 0.27 in [−7.2, −7.0), ≤ 0.26 above −7.0. The flip sits at γa ≈ −7.2 to −7.5. When on, J1713 sits at
   γp ≈ −4.9…−5.0 and σp ≈ −13.5…−14.0, the shallowest spectrum its OU allows (f⁻² across
   the band), i.e. the >30/T excess that the 2026-11-03 amendment found in nine pulsars at
   100–200 Discovery modes, now in the array's most precise pulsar.
   J0645+5158 shows the same more weakly (chain 1: on → −7.80, 23 %; off → −6.23, 57 %;
   corr −0.41). J1600−3053 and J0023+0923 reach |corr| 0.31. J0509+0856 ranks high on the
   {0, 3}-vs-{1, 2} median split (σp 3.5 dex) but shows no within-chain coupling (|corr|
   ≤ 0.17); that ranking mixes chain identity with the effect and is not evidence.
   B1855+09 is a separate 1-vs-3 split between two on states (finding 5), not part of this.

4. **Both probes already ran two of K4's three levers.** `ng15_nb_v1.ini` and
   `ng15_nb_v2.ini` both set `max_tree_depth = 8` and `dense_mass_blocks = log10_pivot_psd,
   log10_gamma_a` (probe 1's log also prints "Max tree depth: 8"); the probe-2 checkpoint's
   adapted inverse mass has a (2, 2) block for (pivot, γa) and a diagonal of 134 for
   (γp, σp). Probe 1's 127 steps was therefore a depth-7 tree chosen by NUTS under a depth-8
   cap, not a cap. The `.nc` does not store steps, but the checkpoint's final sampler state
   does: `num_steps` = 255, 127, 255, 255 (chains 0–3), i.e. depth 8 saturated in three of
   four chains at the last iteration, with running mean acceptance 0.955, 0.88, 0.948,
   0.953 against a 0.9 target; the sampling-phase rate (segment 2, ≈ 1,700 s per draw)
   against probe 1's 6.6 s per step agrees. Only `target_accept 0.95` is untried.

5. **γa is not the only blocker.** The readout's max r̂ is 1.33 on γp (bulk ESS 10). By a
   classic split r̂ on 50-draw halves: σp > 1.1 in 8 pulsars (B1855+09 1.77, J0030+0451
   1.25, J1713+0747 1.23, J0613−0200 1.16, J1944+0907 1.11), γp > 1.1 in 1 (B1855+09 1.47,
   one chain at −5.3 and three at ≈ −9.3). B1855+09 is a 1-vs-3 split of the J1738+0333
   kind seen in probe 1, but between two *on* states (σp −13.07 vs −15.1, both above −16),
   and it is uncoupled from γa (restricting to γa ≤ −7.2 leaves it unchanged). Neither lever
   below is argued to fix it; it is an open blocker without a lever.

6. **Cost, like for like.** Segment 1 (200 warmup + 100 draws) took 124–130 h per chain,
   1,490–1,560 s per iteration; segment 2's sampling-only rate was ≈ 1,700 s per draw. A
   1000 + 1000 rung is then ≈ 1000 × 1,500 + 1000 × 1,700 s ≈ 37 d and ≈ 3,500 A100-h,
   3.5 × §5's budget. Bulk ESS 10 from 400 draws is ≈ 40 × short of 400. Longer runs and
   tuning cannot meet the Phase 5 Done-when as configured.

7. **Checks that passed.** 0 divergences, no frozen chain, no repeated draws; EFAC = 1 and
   EQUAD = 0 exactly in every draw (the white noise was fixed as intended; the MDC2
   broken-symlink problem does not recur here). Pivot PSD stable at −5.96 ± 0.03 across
   chains (Discovery −5.86), r̂ 1.04.

## Reading

Units first. Argus's red noise is an integrated OU process: residual PSD
∝ γ / (ω² (γ² + ω²)) with ω = 2πf, so f⁻² below the corner ω = γ and f⁻⁴ above it
(`parameter_sampling.py`, spec A3). γ is an angular rate in s⁻¹. (The 2026-11-03 entry
calls the high-γp OU "flat across the band"; f⁻² is this note's reading of the same
limit, from the pivot inversion in `parameter_sampling.py`.) The −6 ceiling puts the
common process's corner at a 73-day period: f⁻² over the whole band, the shallowest
spectrum the model allows. The band's low edge is log10 (2π/16 yr) ≈ −7.9; below about
−8.5 the corner is out of band and γa is unidentifiable, which is the flat tail to −11 (the
pivot parameterisation makes the in-band PSD independent of γa there).

A pulsar with a >30/T excess has two owners available: its own OU at γp ≈ −5 (on, flat),
or the common process at its shallowest (γa → −6). The likelihood does not separate them
(finding 2), so NUTS drifts between the states, pulsar by pulsar. The per-pulsar "on" state
is a thin ridge in (γp, σp) against a large "off" volume, which is why the chains spend
most of their time off with γa at the ceiling (findings 1, 3). (B1855+09's 1-vs-3 split in
finding 5 is between two on states and is uncoupled from γa, so this argument does not
cover it.) Finding 3's chains 1 and 2 also show what breaks the tie: when
J1713 owns its excess, γa leaves the ceiling (6–7 % at ceiling instead of 80–92 %). So the
indifference is between owners, not within either; a change that fixes the owner resolves
γa. Two levers do that:

- **A. Lower γa's ceiling** so the common process cannot be f⁻² across the band. No code
  change; a prior choice. The γa prior is uniform and nothing else would change, so v3's
  posterior is v2's restricted to γa below the ceiling, which the 400 draws preview
  (`lever_a_preview` in the JSON, at −7.0). Where to cut: finding 1's density has a dip in
  [−7, −6.5) between the J1713-on bump at ≈ −7.75 and the spike, and finding 3's flip in
  J1713's on-fraction is at −7.5 to −7.2, so **−7.0** cuts at the dip (25 % of the kept
  mass within 0.5 dex of the new ceiling, against 45 % for a cut at −7.5, which would go
  through the bump). A ceiling at −7.0 puts the corner at a 2-year period, the common
  process f⁻⁴ above 1/(2 yr). Discovery's OU-14 v2 reference at ε = 0 (γa 5–95 % −10.74 to
  −7.43, median −8.11, max −6.45) has 0.1 % of its mass above −7.0 (9.5 % above −7.5).
  Restricting v2's draws to γa ≤ −7.0 keeps 153 of 400 (chains 0–3: 10, 59, 72, 12; 86 %
  from chains 1 and 2, so the preview rests mostly on two chains). Per chain: γa medians
  −7.88, −7.89, −7.86, −7.21 and 5th percentiles −8.99, −10.68, −10.69, −8.41; J1713 on in
  0 %, 97 %, 71 %, 8 % of the kept draws. Within each chain that has both states below
  the ceiling, the on-draws fit better by +11.4 (chain 1, 2 off-draws), +8.8 (chain 2,
  21 off) and +8.9 nats (chain 3, 11 off), so below the ceiling the J1713-off state is not
  an equal-lnL third owner; it is the poorer fit that chains 0 and 3 are nevertheless
  sitting in. The preview shows the per-pulsar on state as the likely owner below the
  ceiling; it cannot show that NUTS will *hold* it (the thin-ridge volume ratio is
  unchanged), which is what the probe tests.
- **B. A sampled per-pulsar white-like term** beside the OU: a flat residual PSD
  (EQUAD-like, per-TOA variance), which is a different shape from the OU's f⁻² floor. A
  one-parameter owner for the excess, without the (γp, σp) ridge geometry. Needs the
  compression to carry per-cell statistics for a θ-dependent variance. Its single-pulsar
  gate (below) is necessary but cannot by itself test the owner indifference, which only
  an array run sees.

Neither is the remaining K4 lever (target_accept 0.95), which addresses step geometry, not
the owner trade.

## Recommended next steps

In order; nothing below spends A100 time until 1 is in.

1. **Discovery single-pulsar check** (hours): J1713+0747, J0645+5158 and the nine ceiling
   pulsars of 2026-11-03, at 100–200 modes, OU red noise plus a free flat-PSD (EQUAD-like)
   term. Does γp close off its ceiling and the term carry the excess? This tells whether
   B's term is the right shape (flat, as against the OU's f⁻²). It does not test the owner
   indifference.
2. **Lever A as the array-level test**: freeze v3 = v2 with γa in [−11, −7.0] (floor kept,
   see below), rerun the Discovery OU-14 reference on it (C5 needs the same box), and run a
   shorter probe (150 + 100, ≈ 4 d, ≈ 400 A100-h). Pass criteria are set by the preview,
   not by "off the ceiling" (25 % of the restricted mass is within 0.5 dex of −7.0 by
   construction): every chain's γa median within ≈ 0.3 dex of the others and near −7.9,
   J1713 and J0645 on in ≥ 70 % of every chain's draws (preview: 71–97 % in the chains that
   move), γa and the pivot r̂ ≤ 1.05, γp/σp r̂ well under probe 2's, step size not below
   probe 2's. Fail: a chain holds the J1713-off state below the ceiling (as chains 0 and 3
   do in the preview, at ≈ 9 nats worse), i.e. the on-ridge is still too thin to hold, and
   B is needed. B1855+09's two-on-state split is outside A's criteria; watch it, but no
   lever targets it.
3. **Lever B** only if 2 fails or 1 says the excess is not OU-shaped: scope the sampled
   term in `array/sparse_filter.py` and the compression, then v4 and a probe.
4. **Re-budget** the pilot and the Phase 6 ladder from the passing probe. The current rate
   is 3.5 × budget before the ladder.
5. γa's floor: −11 is 3 dex below identifiability (≈ −8.5) and the tail wastes draws, but
   moving it changes the prior volume that GW-vs-noise evidence sees. Leave it for now;
   decide when the owner question is settled.
6. Not recommended first: `target_accept 0.95` alone (≈ 2 × cost, geometry only); a γa
   reparameterisation alone (same); a 66-pulsar probe without J1713 (≈ 500 A100-h for one
   bit, and `run_array.py` refuses `excluded_psrs`).

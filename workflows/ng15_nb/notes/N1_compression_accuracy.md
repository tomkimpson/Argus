# N1 — Compression accuracy and cost on NG15 narrowband

Spec `spec/spec.md` §6 output N1, Phase 2 tasks 2.5 and 2.6. This note records how well
the epoch compression (D2) reproduces the dense likelihood on the raw TOAs (C1), as a
function of the grid width Δ, and what the 67-pulsar likelihood plus gradient costs (K3).

## Outcome

- **Layout chosen:** two scalars (position and velocity) per DMX window, stamped at each
  window's weighted mean time, except for four pulsars compressed into epoch cells:
  J1012+5307, J1643−1224, J1705−1903 and J1903+0327. That gives 15,372 observations and
  M_sum = 1,607 timing columns over 67 pulsars.
- **Why not a coarser Δ.** On window cells J1012+5307 misses the spread bound at every Δ,
  so it takes epoch cells whatever Δ is. J1909−3744 passes up to 14 d and B1937+21 up to
  7 d. Coarsening is not adopted because it does not cut the sparse filter's cost (7 d:
  9.2 s vs 9.6 s for one point), not because of C1. A 7 d layout would also need its own
  epoch-rule pass over the 67 pulsars, which was not run.
- **C1 passes for all three C1 pulsars** at that layout: J1909−3744 5.3e-4, B1937+21
  8.6e-3 and J1012+5307 3.9e-8 nat. C1 gates on the spread; gradient errors are reported,
  not gated (amended 2026-10-18). On window cells J1909−3744 and B1937+21 have gradient
  errors of 3.5e-4 and 1.1e-2 against the oracle.
- **Cost (K3):** 0.84 s per value-plus-gradient evaluation, with 32 points vmapped on one
  A100. That is between K3's 0.5 s target and its 1 s stop, and Phase 2 closes on it (§12,
  2026-10-18). It is batched throughput: one chain's latency is 6.7 s for one point, or
  26.8 s per 32-point step (see Open).

## Setup

- Data: the Phase 2 ingest caches (task 2.1) of the NG15 v1.1.0 narrowband feathers pinned
  in `data/ng15_nb/MANIFEST.json`, with fixed white noise.
- Compression: `workflows/ng15_nb/compression/compress.py` (task 2.2).
  - Window cells: the DMX projection, then one position scalar at t_w and one velocity
    scalar along the projected (t − t_w) direction (amended 2026-10-18).
  - Grid cells (Δ = 7, 14, 30 d): stamped at the grid centre, with the position row
    shifted onto the rate states.
  - Epoch cells: one position scalar per ECORR epoch, with that pulsar's DMX columns kept
    in β.
- Filter: the Argus marginal filter with initial accumulators (task 2.3), via
  `oracle.filter_loglike`, diffuse timing prior.
- Oracle: dense continuous-time integrated-OU covariance at every raw TOA, with white
  noise, ECORR and the full M after the rank rule, diffuse (`oracle.py`, task 2.4).
- Points: 20 draws from the Phase 1 Discovery-OU ε = 0 posterior (OU14) for each pulsar
  (the C1 gate), plus 20 points uniform in the `ou.ini` prior box (a stress test that does
  not gate).
- Metrics:
  - spread is max − min over the points of lnL_filter + const − lnL_oracle;
  - gradient error is max over the points of ‖g_f − g_o‖∞ / ‖g_o‖∞;
  - C1 requires a spread ≤ 0.01 nat; the gradient error is reported, not gated (amended
    2026-10-18). `passes_c1` in the result files follows that rule.
- Drivers:
  - `scripts/c1_compression.py` and `compression/slurm/c1.sh` write `results/c1/<pulsar>[_epoch].json`;
  - `scripts/epoch_cell_rule.py` writes `results/c1/epoch_rule*.json`;
  - `scripts/bench_sparse.py` writes `results/k3/sparse_*.json`.

## C1 against the dense oracle, posterior points

Spread (nat) / gradient error. B1855+09 is a small control pulsar, not one of the three
C1 pulsars.

| Pulsar | DMX window | 7 d | 14 d | 30 d | epoch cells |
|---|---|---|---|---|---|
| J1909−3744 | 5.3e-4 / 3.5e-4 | 4.6e-3 / 3.7e-3 | 2.1e-3 / 1.8e-3 | 7.1e-2 / 1.6e-2 | 1.4e-5 / 1.9e-6 |
| B1937+21 | 8.6e-3 / 1.1e-2 | 5.7e-3 / 7.1e-3 | 2.0e-2 / 3.1e-2 | 0.22 / 0.34 | 4.2e-3 / 2.3e-3 |
| J1012+5307 | 5.4e-2 / 3.8e-2 | 9.6e-2 / 2.2e-2 | 0.37 / 0.11 | 2.7 / 0.35 | 3.9e-8 / 4.2e-9 |
| B1855+09 | 1.5e-6 / 2.5e-7 | 1.3e-3 / 1.5e-3 | 2.8e-2 / 3.0e-2 | 9.3e-2 / 0.10 | 1.5e-6 / 2.5e-7 |

- **Single-scalar layout.** One scalar per window (the original D2) failed at every Δ; it
  is superseded (§12, 2026-10-18).
- **Oracle precision.** Moving the oracle's time origin by 1000 d is exact in theory, but
  it changes the oracle alone by 1.5e-2 nat / 2.3e-3 on B1937+21 and by 8e-6 / 1.1e-6 on
  B1855+09. No pass relies on this: B1937+21's epoch-cell residual (4.2e-3) and every
  adopted-layout residual sit inside 0.01 nat outright. The origin-shift run was a scratch
  check and is not committed, so treat these figures as indicative.
- **Prior stress set.** Spreads there are large: up to 27 nat for J1909−3744 at the DMX
  window, and still 4.1 nat (J1909−3744) and 9.2 nat (B1937+21) in epoch cells.
  - B1855+09's 0.10 nat is oracle-side: its two filter layouts agree there to 3e-5.
  - For the others it has not been diagnosed. The prior box reaches γ and amplitude
    combinations far outside the posterior, where the 23k–35k-point dense Cholesky is the
    likely limit.
  - These points do not gate C1.

## The epoch-cell rule over all 67 pulsars

Each pulsar's window likelihood is compared with its own epoch-cell likelihood (the
near-exact reference) at its 20 posterior draws. A pulsar takes epoch cells when the
spread exceeds 0.01 nat (amended 2026-10-18: spread only; gradient differences are
reported).

| Rule | Pulsars on epoch cells | M_sum | Observations |
|---|---|---|---|
| none (all windows) | 0 | 1,080 | 14,811 |
| spread > 0.01 nat (adopted) | 4 | 1,607 | 15,372 |
| spread > 0.01 or gradient > 1e-3 | 19 | 3,596 | — |
| spread > 0.01 or gradient > 1e-4 (C1's two bounds) | 33 | 5,072 | 18,957 |

The four adopted pulsars have spreads of 0.11 (J1705−1903), 0.11 (J1643−1224), 0.054
(J1012+5307) and 0.043 (J1903+0327) nat. The other 63 sum to ≤ 0.05 nat (if they add
worst-case).

## Cost (K3)

All timings are value plus gradient over all 136 parameters on one A100.

| Filter, layout | Steps | One point | Per evaluation, 32 vmapped |
|---|---|---|---|
| dense union grid, DMX window | 14,820 | 28.6 s | — |
| dense union grid, 7 d | 1,150 | 2.3 s | — |
| dense union grid, 30 d | 376 | 0.74 s | — |
| sparse filter, all windows | 14,811 | 6.2 s | 0.87 s |
| sparse filter, epoch rule (at fc1228e) | 15,372 | 10.6 s | 1.27 s |
| sparse + chunked Ξ + checkpointed Q, epoch rule | 15,372 | 6.7 s | **0.84 s** |

- The dense rows come from §12 (2026-10-18). The dense filter needs a checkpointed copy at
  fine Δ, because its stock reverse mode needs 220 GB.
- Files (`results/k3/README.md` lists them all): the "all windows" row is
  `sparse_dmx_window_v2.json`; the "epoch rule" row was committed at fc1228e and then
  overwritten by the chunked run in `sparse_dmx_window_epoch_rule.json`.
- The dense union path is 0.097 nat below the sparse one at the DMX window. That is a
  known jitter bias of the dense reference: its 1e-9·s̄ jitter also lands on masked unit
  slots (`union.py` docstring). The sparse filter has no masked slots, and the t_w-stamped
  dense run before whitening (`k3/dmx_window_tw_stamped.json`) agrees with it to 2.4e-6 nat.
- The sparse filter's cost grows with M_sum. A GPU profile traced it to bandwidth-bound
  per-step passes over Ξ (4N × M_sum) and to per-step residual writes from
  `get_Q_block`'s series. Chunking Ξ and checkpointing Q remove both, and leave lnL
  unchanged to 1e-8 nat and the gradient to 2e-9.
- Peak memory at 32 points is 42 GB.
- A100 nodes vary by about 10% between jobs, so variants were compared within one job.

## Open

- **Budget (unresolved).** §5's 800 A100-h assumes 0.5 s of per-chain latency. 0.84 s is
  batched throughput, and warmup cannot be split across lockstep chains. The ladder is
  ≈ 1,340 A100-h if the draws split across many short chains with little warmup, and
  ≈ 10,700 A100-h if each chain needs §5's length. To be settled before Phase 5 (spec §5,
  §12 2026-10-19).

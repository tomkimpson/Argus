# N1 — Compression accuracy and cost on NG15 narrowband

Spec `spec/spec.md` §6 output N1, Phase 2 tasks 2.5 and 2.6. This note records how well
the epoch compression (D2) reproduces the dense likelihood on the raw TOAs (C1), as a
function of the grid width Δ, and what the 67-pulsar likelihood plus gradient costs (K3).

## Outcome

- **Layout chosen:** two scalars (position and velocity) per DMX window, stamped at each
  window's weighted mean time, except for four pulsars compressed into epoch cells:
  J1012+5307, J1643−1224, J1705−1903 and J1903+0327. That gives 15,372 observations and
  M_sum = 1,607 timing columns over 67 pulsars.
- **No coarser Δ passes C1.** Every Δ ≥ 7 d misses the spread bound on at least one of the
  four pulsars tested.
- **Cost (K3):** 0.84 s per value-plus-gradient evaluation, with 32 points vmapped on one
  A100. That is between K3's 0.5 s target and its 1 s stop, and Phase 2 closes on it (§12,
  2026-10-18).

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
  - C1 requires a spread ≤ 0.01 nat and a gradient error ≤ 1e-4.
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
  B1855+09. B1937+21's epoch-cell residual is therefore at the oracle's precision, which
  counts as a pass (amended C1).
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
| sparse filter, epoch rule | 15,372 | 10.6 s | 1.27 s |
| sparse + chunked Ξ + checkpointed Q, epoch rule | 15,372 | 6.7 s | **0.84 s** |

- The dense rows come from §12 (2026-10-18). The dense filter needs a checkpointed copy at
  fine Δ, because its stock reverse mode needs 220 GB.
- The sparse filter's cost grows with M_sum. A GPU profile traced it to bandwidth-bound
  per-step passes over Ξ (4N × M_sum) and to per-step residual writes from
  `get_Q_block`'s series. Chunking Ξ and checkpointing Q remove both, and leave lnL
  unchanged to 1e-8 nat and the gradient to 2e-9.
- Peak memory at 32 points is 42 GB.
- A100 nodes vary by about 10% between jobs, so variants were compared within one job.

## Open

- **J1909−3744 and B1937+21 against the oracle gradient bound** (decision pending). Under
  the adopted rule both stay on window cells. There they pass the oracle spread bound
  (5.3e-4 and 8.6e-3 nat) but not the gradient bound (3.5e-4 and 1.1e-2). Both pass in
  epoch cells.
- **Budget.** At 0.84 s the 67-psr ladder scales to about 1,340 A100-h, against the 800
  assumed in §5 (§12, 2026-10-18).

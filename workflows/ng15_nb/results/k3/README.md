# K3 benchmark results (task 2.6)

Value plus gradient over all 136 parameters, 67 pulsars, one A100. Files written before
2026-10-19 carry no git SHA; the commit that added each is given instead.

| File | Filter | Layout | Headline | Commit |
|---|---|---|---|---|
| `dmx_window.json`, `7d.json`, `14d.json`, `30d.json` | dense union grid, whitened, checkpointed scan copy | centre-stamped grid (DMX window: t_w) | 28.6 / 2.3 / 1.3 / 0.74 s for one point | 3f26b3f |
| `dmx_window_tw_stamped.json` | dense union grid before whitening | DMX window, t_w-stamped | 28.6 s; agrees with sparse to 2.4e-6 nat | 51f9851 |
| `sparse_dmx_window.json`, `sparse_7d.json` | sparse filter, **first version** (superseded) | all windows / 7 d | 9.6 / 9.2 s for one point; 4.6 s/eval at 32 points | ec91067 |
| `sparse_dmx_window_v2.json` | sparse filter, elementwise predict | all windows | 6.2 s; 0.87 s/eval at 32 points | ec91067 |
| `sparse_dmx_window_epoch_rule.json` | sparse + chunked Ξ + checkpointed Q (current) | epoch rule (adopted) | 6.7 s; **0.84 s/eval** at 32 points | 6edc98e |

- The epoch-rule layout's pre-chunking run (10.6 s; 1.27 s/eval) was committed at fc1228e
  under the same file name, and then overwritten.
- The whitened dense path is 0.097 nat below the sparse filter. That is the dense
  reference's jitter on masked slots (`compression/union.py` docstring), not a sparse error.
- Only the epoch-rule file needs `--epoch-rule --chunks 8 1`. For the others, pass
  `--delta-days` and `--out`. The SLURM scripts are under `compression/slurm/`.

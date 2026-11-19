"""The 67-pulsar NG15 narrowband likelihood behind the harness's Kalman-filter seam.

`bayesian_inference.run_nuts_sampling` asks its filter for one thing: `get_likelihood(θ)`.
`SparseArrayFilter` answers it with the production likelihood of spec Phases 3-6: every
pulsar's D2 compression at the adopted layout (two scalars per DMX window, or epoch cells
for the pulsars `results/c1/epoch_rule.json` selects), fed to the sparse marginal filter
under the diffuse timing prior. It is `union.array_loglike_sparse` plus the constant, with
θ.orf_epsilon applied to the ORF the way the stock filter applies it.

White noise and ECORR are fixed inside the compression (A2): each scalar's σ already
carries the per-backend EFAC/EQUAD and ECORR. The sparse filter would still rescale it by
θ.EFAC and θ.EQUAD, so `get_likelihood` pins those to 1 and 0, whatever θ carries. The
driver (`run_array.py`) also fixes them in the harness so they are not sampled.
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "compression"))
import compress  # noqa: E402
import ingest  # noqa: E402
import union  # noqa: E402

import jax.numpy as jnp  # noqa: E402
from argus import jax_kalman_filter as jk  # noqa: E402


class SparseArrayFilter:
    """`get_likelihood(θ)` over the compressed array, constant included.

    Parameters
    ----------
    ingest_dir : path
        Directory of the per-pulsar ingest caches (`<name>.npz`, `scripts/ingest_nb.py`).
    names : list of str
        Pulsars in the order the sampler's per-pulsar vectors use.
    epoch_cells : collection of str
        Pulsars compressed per ECORR epoch rather than per DMX window (D2's rule).
    segment, chunk : int
        The sparse filter's checkpoint segment and Ξ chunk lengths (K3's benchmark
        settings, `results/k3/sparse_dmx_window_epoch_rule.json`).
    """

    def __init__(self, ingest_dir, names, epoch_cells, segment=128, chunk=8):
        caches = [ingest.load(Path(ingest_dir) / f"{n}.npz") for n in names]
        comps = [
            compress.compress_pulsar(c, per_epoch=n in epoch_cells)
            for n, c in zip(names, caches)
        ]
        self.inp = union.build_sparse_inputs(comps, union.hd_matrix(caches))
        self.names = list(names)
        self.Npsr = len(names)
        self.M_sum = int(self.inp["obs"]["m"].shape[1])
        self.n_obs = self.inp["n_obs"]
        self.epoch_cells = sorted(set(epoch_cells) & set(names))
        self.segment = segment
        self.chunk = chunk
        self._P_eps_inv = jnp.zeros((self.M_sum, self.M_sum))
        self._efac = jnp.ones(self.Npsr)
        self._equad = jnp.zeros(self.Npsr)
        # Read by the harness's resume fingerprint (`bayesian_inference._data_fingerprint`):
        # the base ORF, one row per observation of everything the filter consumes, and
        # the timing prior.
        self.hellings_downs_matrix = self.inp["orf"]
        obs = self.inp["obs"]
        self.jax_data = np.column_stack(
            [
                np.asarray(obs[k], dtype=np.float64)
                for k in ("dt", "psr", "h", "y", "err")
            ]
            + [np.asarray(obs["m"], dtype=np.float64).reshape(self.n_obs, -1)]
        )
        self.timing_prior = "diffuse"

    def get_likelihood(self, θ):
        θ = θ.replace(EFAC=self._efac, EQUAD=self._equad)
        return (
            jk._run_kalman_filter_marginal_sparse(
                θ=θ,
                obs=self.inp["obs"],
                Npsr=self.Npsr,
                M_sum=self.M_sum,
                hellings_downs_matrix=self.inp["orf"],
                P_eps_inv=self._P_eps_inv,
                diffuse=True,
                accum0=self.inp["accum0"],
                segment=self.segment,
                chunk=self.chunk,
            )
            + self.inp["const"]
        )

    def summary(self):
        return {
            "n_pulsars": self.Npsr,
            "n_obs": self.n_obs,
            "M_sum": self.M_sum,
            "epoch_cells": self.epoch_cells,
            "segment": self.segment,
            "chunk": self.chunk,
            "const": float(self.inp["const"]),
            "orf_offdiag_absmax": float(
                np.max(np.abs(np.asarray(self.inp["orf"]) - np.eye(self.Npsr)))
            ),
        }

"""Union-grid inputs for the array filter from per-pulsar compressions (spec task 2.6).

Each pulsar's compression (two scalars per cell, `compress.compress_pulsar`) is placed on
the union of every pulsar's cell times. A union time carries two filter steps, dt = 0
apart: the position step and the velocity step. A pulsar without a cell at that time, or
without a velocity scalar in that cell, is masked there. The β-only remainders combine
block-diagonally into one (A₀, b₀, c₀).

State layout follows the library (`model.compute_H_matrix_for_step`): GW [r_n, ṙ_n] at
columns 2n, 2n+1; spin [φ_n, φ̇_n] at 2N + 2n, 2N + 2n + 1; timing β'_n from 4N on.

`offset` undoes the log 2π each masked (pulsar, step) adds to the filter's log-det
stream, and `const` sums the compressions' constants, so

    array_loglike(p, inp) + inp["const"]

is the dense flat-prior likelihood of the raw TOAs up to A1 (state linear per cell).
"""

import jax.numpy as jnp
import numpy as np
from scipy.linalg import block_diag

from argus import bayesian_inference
from argus import gravitational_waves
from argus import jax_kalman_filter as jk

LN2PI = np.log(2.0 * np.pi)


def hd_matrix(caches):
    """Hellings–Downs matrix (unit diagonal) from the caches' sky positions."""
    th = np.array([c["theta"] for c in caches])
    ph = np.array([c["phi"] for c in caches])
    u = np.column_stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
    ang = np.arccos(np.clip(u @ u.T, -1.0, 1.0))
    np.fill_diagonal(ang, 0.0)
    return gravitational_waves.hellings_downs(ang)


def build_union_inputs(comps, orf):
    """Masked union-grid filter inputs for a list of two-scalar compressions."""
    N = len(comps)
    Ks = [c["m"].shape[1] for c in comps]
    starts = 4 * N + np.concatenate([[0], np.cumsum(Ks)[:-1]])
    n_states = 4 * N + int(np.sum(Ks))
    cell_t = [c["times"][c["has_scalar"]] for c in comps]
    times, inverse = np.unique(np.concatenate(cell_t), return_inverse=True)
    T = 2 * times.size

    y = np.zeros((T, N))
    err = np.ones((T, N))
    mask = np.zeros((T, N))
    H = np.zeros((T, N, n_states))
    split = np.cumsum([t.size for t in cell_t])[:-1]
    for n, (c, u) in enumerate(zip(comps, np.split(inverse, split))):
        has = c["has_scalar"]
        hv = c["has_velocity"][has]
        pos, vel = 2 * u, 2 * u + 1
        cols = slice(starts[n], starts[n] + Ks[n])
        y[pos, n], err[pos, n], mask[pos, n] = c["s_hat"][has], np.sqrt(c["sigma2"][has]), 1.0
        H[pos, n, 2 * n] = -1.0
        H[pos, n, 2 * N + 2 * n] = 1.0 / c["f0"]
        H[pos, n, cols] = c["m"][has]
        vel = vel[hv]
        y[vel, n] = c["v_hat"][has][hv]
        err[vel, n] = np.sqrt(c["sigma2_v"][has][hv])
        mask[vel, n] = 1.0
        H[vel, n, 2 * n + 1] = -1.0
        H[vel, n, 2 * N + 2 * n + 1] = 1.0 / c["f0"]
        H[vel, n, cols] = c["m_v"][has][hv]

    A0 = block_diag(*[c["accum0"][0] for c in comps])
    b0 = np.concatenate([c["accum0"][1] for c in comps])
    c0 = float(sum(c["accum0"][2] for c in comps))
    return {
        "data": jnp.asarray(y),
        "data_errors": jnp.asarray(err),
        "H_matrices": jnp.asarray(H),
        "dt_array": jnp.asarray(np.diff(np.repeat(times, 2))),
        "mask_matrices": jnp.asarray(mask),
        "accum0": (jnp.asarray(A0), jnp.asarray(b0), jnp.asarray(c0)),
        "orf": jnp.asarray(orf),
        "Npsr": N,
        "M_sum": int(np.sum(Ks)),
        "offset": 0.5 * LN2PI * float(np.sum(mask == 0)),
        "const": float(sum(c["const"] for c in comps)),
        "n_union_times": int(times.size),
    }


def array_loglike(p, inp):
    """Diffuse marginal filter on the union grid.

    p = (log10 ha, log10 γa, log10 σp[N], log10 γp[N]).
    """
    # Sizes from shapes, so `inp` can be a jit argument.
    N = inp["data"].shape[1]
    M_sum = inp["H_matrices"].shape[-1] - 4 * N
    θ = bayesian_inference.Parameters(
        log10_gamma_a=p[1],
        γa=10.0 ** p[1],
        ha=10.0 ** p[0],
        σp=10.0 ** p[2 : 2 + N],
        γp=10.0 ** p[2 + N : 2 + 2 * N],
        EFAC=jnp.ones(N),
        EQUAD=jnp.zeros(N),
    )
    return inp["offset"] + jk._run_kalman_filter_marginal(
        θ=θ,
        data=inp["data"],
        data_errors=inp["data_errors"],
        H_matrices=inp["H_matrices"],
        Npsr=N,
        M_sum=M_sum,
        hellings_downs_matrix=inp["orf"],
        dt_array=inp["dt_array"],
        dim_x=2 * N,
        n_states=4 * N + M_sum,
        P_eps_inv=jnp.zeros((M_sum, M_sum)),
        mask_matrices=inp["mask_matrices"],
        diffuse=True,
        accum0=inp["accum0"],
    )

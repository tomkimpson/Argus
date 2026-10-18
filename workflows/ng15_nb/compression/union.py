"""Union-grid inputs for the array filter from per-pulsar compressions (spec task 2.6).

Each pulsar's compression (two scalars per cell, `compress.compress_pulsar`) is placed on
the union of every pulsar's cell times. A union time carries two filter steps, dt = 0
apart: the position step and the velocity step. Cells are stamped at their grid centres
(or at t_w for DMX-window cells), so with a grid Δ pulsars share steps; the position row
carries the stamp-to-t_w shift on the rate states (see `oracle.filter_inputs`). A pulsar
without a cell at that time, or
without a velocity scalar in that cell, is masked there. The β-only remainders combine
block-diagonally into one (A₀, b₀, c₀).

State layout follows the library (`model.compute_H_matrix_for_step`): GW [r_n, ṙ_n] at
columns 2n, 2n+1; spin [φ_n, φ̇_n] at 2N + 2n, 2N + 2n + 1; timing β'_n from 4N on.

Every observation is whitened to unit noise (y/σ, H/σ, σ → 1). The library's
innovation jitter is 1e-9 × the mean S over the pulsars observed at a step, so once
pulsars share steps an observation with a huge σ² (a velocity scalar from a cell spanning
seconds) would otherwise set the jitter on its well-measured neighbours. On the synthetic
pair at Δ = 30 d that cost 3.8 nat; whitening makes every S ≥ 1 and comparable.

`offset` undoes the log 2π each masked (pulsar, step) adds to the filter's log-det
stream and the whitening's Σ log σ, and `const` sums the compressions' constants, so

    array_loglike(p, inp) + inp["const"]

is the dense flat-prior likelihood of the raw TOAs up to A1 (state linear per cell),
less a small jitter bias. The jitter is added as 1e-9·s̄·I over all n pulsars, so the
masked unit slots get it too, where s̄ is the mean S of the observed pulsars at that step
(≫ 1 after whitening). Each such slot loses ½ log(1 + 1e-9·s̄), which depends on the
parameters. On the 67-pulsar DMX-window grid this totals 0.097 nat; the sparse filter
(`jk._run_kalman_filter_marginal_sparse`), which Phases 3-6 use, has no masked slots and
no such bias. This path is kept as the dense reference for tests and K3.
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
        y[pos, n], err[pos, n], mask[pos, n] = (
            c["s_hat"][has],
            np.sqrt(c["sigma2"][has]),
            1.0,
        )
        H[pos, n, 2 * n] = -1.0
        H[pos, n, 2 * N + 2 * n] = 1.0 / c["f0"]
        H[pos, n, cols] = c["m"][has]
        shift = (c["t_weighted"] - c["times"])[has]
        H[pos, n, 2 * n + 1] = -shift
        H[pos, n, 2 * N + 2 * n + 1] = shift / c["f0"]
        vel = vel[hv]
        y[vel, n] = c["v_hat"][has][hv]
        err[vel, n] = np.sqrt(c["sigma2_v"][has][hv])
        mask[vel, n] = 1.0
        H[vel, n, 2 * n + 1] = -1.0
        H[vel, n, 2 * N + 2 * n + 1] = 1.0 / c["f0"]
        H[vel, n, cols] = c["m_v"][has][hv]

    # Whiten (exact: the scaling's Jacobian Σ log σ goes into the offset).
    log_sigma = float(np.sum(np.log(err[mask > 0])))
    y, H, err = y / err, H / err[:, :, None], np.ones_like(err)

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
        "offset": 0.5 * LN2PI * float(np.sum(mask == 0)) - log_sigma,
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


# --------------------------------------------------------------------------- #
# Sparse path: one scalar observation per filter step
# --------------------------------------------------------------------------- #
def build_sparse_inputs(comps, orf):
    """Per-observation inputs for `jk._run_kalman_filter_marginal_sparse`.

    Every position and velocity scalar of every pulsar, sorted by time (position
    before velocity within a cell). No masking, so no masked-step offset, and no
    whitening: the sparse path's innovations are scalars and carry no PD jitter.
    `const` is the same as in `build_union_inputs`.
    """
    N = len(comps)
    Ks = [c["m"].shape[1] for c in comps]
    starts = np.concatenate([[0], np.cumsum(Ks)[:-1]])
    M_sum = int(np.sum(Ks))
    t, kind, psr, h, y, err, m = [], [], [], [], [], [], []
    for n, c in enumerate(comps):
        has = c["has_scalar"]
        hv = c["has_velocity"][has]
        times = c["times"][has]
        shift = (c["t_weighted"] - c["times"])[has]
        f0 = c["f0"]
        rows = np.zeros((has.sum(), M_sum))
        rows[:, starts[n] : starts[n] + Ks[n]] = c["m"][has]
        t.append(times)
        kind.append(np.zeros(has.sum()))
        psr.append(np.full(has.sum(), n))
        h.append(
            np.column_stack(
                [-np.ones_like(shift), -shift, np.full_like(shift, 1 / f0), shift / f0]
            )
        )
        y.append(c["s_hat"][has])
        err.append(np.sqrt(c["sigma2"][has]))
        m.append(rows)
        rows_v = np.zeros((hv.sum(), M_sum))
        rows_v[:, starts[n] : starts[n] + Ks[n]] = c["m_v"][has][hv]
        t.append(times[hv])
        kind.append(np.ones(hv.sum()))
        psr.append(np.full(hv.sum(), n))
        h.append(np.tile([0.0, -1.0, 0.0, 1.0 / f0], (hv.sum(), 1)))
        y.append(c["v_hat"][has][hv])
        err.append(np.sqrt(c["sigma2_v"][has][hv]))
        m.append(rows_v)
    t, kind, psr = np.concatenate(t), np.concatenate(kind), np.concatenate(psr)
    order = np.lexsort((kind, psr, t))
    t = t[order]
    A0 = block_diag(*[c["accum0"][0] for c in comps])
    b0 = np.concatenate([c["accum0"][1] for c in comps])
    c0 = float(sum(c["accum0"][2] for c in comps))
    return {
        "obs": {
            "dt": jnp.asarray(np.concatenate([[0.0], np.diff(t)])),
            "psr": jnp.asarray(psr[order].astype(np.int32)),
            "h": jnp.asarray(np.concatenate(h)[order]),
            "y": jnp.asarray(np.concatenate(y)[order]),
            "err": jnp.asarray(np.concatenate(err)[order]),
            "m": jnp.asarray(np.concatenate(m)[order]),
        },
        "accum0": (jnp.asarray(A0), jnp.asarray(b0), jnp.asarray(c0)),
        "orf": jnp.asarray(orf),
        "const": float(sum(c["const"] for c in comps)),
        "n_obs": int(t.size),
    }


def array_loglike_sparse(p, inp, segment=128, chunk=8):
    """`array_loglike` on the sparse path (same p, same likelihood up to rounding)."""
    obs = inp["obs"]
    N = inp["orf"].shape[0]
    M_sum = obs["m"].shape[1]
    θ = bayesian_inference.Parameters(
        log10_gamma_a=p[1],
        γa=10.0 ** p[1],
        ha=10.0 ** p[0],
        σp=10.0 ** p[2 : 2 + N],
        γp=10.0 ** p[2 + N : 2 + 2 * N],
        EFAC=jnp.ones(N),
        EQUAD=jnp.zeros(N),
    )
    return jk._run_kalman_filter_marginal_sparse(
        θ=θ,
        obs=obs,
        Npsr=N,
        M_sum=M_sum,
        hellings_downs_matrix=inp["orf"],
        P_eps_inv=jnp.zeros((M_sum, M_sum)),
        diffuse=True,
        accum0=inp["accum0"],
        segment=segment,
        chunk=chunk,
    )

"""Dense brute-force oracle for compression accuracy (spec C1, task 2.4).

The oracle evaluates the single-pulsar Argus model on the raw TOAs with no compression
and no time gridding:

- the **continuous-time** covariance of the integrated OU processes at every TOA time
  (the analytic kernel, not the filter's gridded F/Q), so the oracle does not share A1;
- the fixed per-backend white noise plus ECORR blocks;
- the DMX columns and the timing model in the compression's own β basis (the rank rule's
  `basis`), all marginalised under the flat prior.

Model. The filter measures −r(t) + φ(t)/f0, where r (GW, corner γa, driving variance
σa² = ha² γa / 12 for a single pulsar, Γ = 1) and φ (spin, corner γp, driving σp²) are
integrated OU processes, dx = v dt, dv = −γ v dt + σ dW, whose velocity starts at its
stationary variance σ²/(2γ) and whose position is pinned at the first epoch (prior
variance 1e-40). With the position measured from an origin t_o,

    Cov[x(s), x(t)] = σ²/(2γ³) [2γ min(s,t) + E(s) + E(t) − E(|t − s|)],  E(x) = expm1(−γx).

That form cancels catastrophically at small γt (the bracket is O((γt)²) out of O(γt)
terms). With a = γ min(s,t), u = γ|t − s| and h(a) = e^{−a} − 1 + a it is rewritten
exactly as −a·expm1(−u) + h(a)(1 + e^{−u}), with h from its series below a = 0.1.

The choice of origin only adds a random constant to the residual, which lies in the span
of the timing model's offset column and so cancels exactly under the flat prior.

Both likelihoods take p = (log10 ha, log10 γa, log10 σp, log10 γp) and are written in
JAX so their gradients come from autodiff.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from argus import bayesian_inference  # noqa: E402
from argus import jax_kalman_filter as jk  # noqa: E402

LN2PI = np.log(2.0 * np.pi)


_H_SERIES = [(-1) ** k / math.factorial(k) for k in range(2, 20)]


def _h(a):
    """h(a) = e^{−a} − 1 + a without cancellation: its Taylor series below a = 0.1."""
    small = a < 0.1
    s = jnp.where(small, a, 0.05)
    big = jnp.where(small, 1.0, a)
    poly = jnp.zeros_like(s)
    for c in reversed(_H_SERIES):
        poly = poly * s + c
    return jnp.where(small, s**2 * poly, jnp.expm1(-big) + big)


def _iou_cov(t, gamma, sigma2):
    """Continuous-time integrated-OU position covariance at times t (from the origin)."""
    # h is increasing, so h(γ min(s,t)) = min(h(γs), h(γt)): the series runs on the 1-D
    # time vector only, keeping memory at a few n×n arrays.
    hv = _h(gamma * t)
    a = gamma * jnp.minimum(t[:, None], t[None, :])
    u = gamma * jnp.abs(t[:, None] - t[None, :])
    g = -a * jnp.expm1(-u) + jnp.minimum(hv[:, None], hv[None, :]) * (1.0 + jnp.exp(-u))
    return sigma2 / (2.0 * gamma**3) * g


def physical(p):
    """(log10 ha, log10 γa, log10 σp, log10 γp) -> (ha, γa, σp, γp)."""
    return 10.0 ** p[0], 10.0 ** p[1], 10.0 ** p[2], 10.0 ** p[3]


def oracle_inputs(cache, comp):
    """Arrays the dense oracle needs: raw TOAs, fixed noise, [D, timing @ basis]."""
    t = cache["toas"] - cache["toas"].min()
    D = cache["M"][:, cache["is_dmx"]]
    timing = cache["M"][:, ~cache["is_dmx"]] @ comp["basis"]
    return {
        "t": jnp.asarray(t),
        "z": jnp.asarray(cache["residuals"]),
        "white_var": jnp.asarray(cache["white_var"]),
        "epoch": jnp.asarray(cache["ecorr_epoch"]),
        "ecorr_var": jnp.asarray(cache["ecorr_var"][cache["ecorr_epoch"]]),
        "X": jnp.asarray(np.column_stack([D, timing])),
        "f0": float(cache["f0"]),
    }


def oracle_loglike(p, inp):
    """Dense flat-prior lnL, −½[zᵀC⁻¹z − bᵀA⁻¹b + logdet(2πC) + logdet A]."""
    ha, ga, sp, gp = physical(p)
    t = inp["t"]
    C = _iou_cov(t, ga, ha**2 * ga / 12.0) + _iou_cov(t, gp, sp**2) / inp["f0"] ** 2
    same = inp["epoch"][:, None] == inp["epoch"][None, :]
    C = C + jnp.where(same, inp["ecorr_var"][:, None], 0.0)
    C = C + jnp.diag(inp["white_var"])
    L = jnp.linalg.cholesky(C)
    tri = jax.scipy.linalg.solve_triangular
    zt = tri(L, inp["z"], lower=True)
    Xt = tri(L, inp["X"], lower=True)
    Q, R = jnp.linalg.qr(Xt)
    proj = zt - Q @ (Q.T @ zt)
    return -0.5 * (
        proj @ proj
        + t.size * LN2PI
        + 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
        + 2.0 * jnp.sum(jnp.log(jnp.abs(jnp.diag(R))))
    )


def filter_inputs(comp):
    """Single-pulsar inputs for `_run_kalman_filter_marginal` from a compression.

    A one-scalar compression gives one step per cell with a scalar. A two-scalar one gives
    two steps per such cell at t_w, dt = 0 apart: the position scalar (H on r and φ) then
    the velocity scalar (H on ṙ and φ̇), the latter masked where the cell has none. Each
    masked step adds log 2π to the filter's log-det stream; `offset` undoes that, so
    filter_loglike + comp["const"] is the dense likelihood.
    """
    has = comp["has_scalar"]
    two = bool(comp["has_velocity"].any())
    f0 = comp["f0"]
    K = comp["m"].shape[1]
    if not two:
        rows = [(comp["s_hat"][has], comp["sigma2"][has], comp["m"][has], 0, np.ones(has.sum()))]
        times = comp["times"][has]
    else:
        hv = comp["has_velocity"][has]
        sig_v = np.where(hv, comp["sigma2_v"][has], 1.0)
        rows = [
            (comp["s_hat"][has], comp["sigma2"][has], comp["m"][has], 0, np.ones(has.sum())),
            (comp["v_hat"][has], sig_v, comp["m_v"][has], 1, hv.astype(float)),
        ]
        times = np.repeat(comp["times"][has], 2)
    T = times.size
    stride = len(rows)
    y, err, mask = np.zeros((T, 1)), np.ones((T, 1)), np.ones((T, 1))
    H = np.zeros((T, 1, 4 + K))
    for j, (val, var, m, col, msk) in enumerate(rows):
        sl = slice(j, None, stride)
        y[sl, 0], err[sl, 0], mask[sl, 0] = val, np.sqrt(var), msk
        H[sl, 0, 0 + col] = -1.0  # GW redshift r (or ṙ)
        H[sl, 0, 2 + col] = 1.0 / f0  # spin phase φ (or φ̇)
        H[sl, 0, 4:] = m
    A0, b0, c0 = comp["accum0"]
    return {
        "data": jnp.asarray(y),
        "data_errors": jnp.asarray(err),
        "H_matrices": jnp.asarray(H),
        "dt_array": jnp.asarray(np.diff(times)),
        "mask_matrices": jnp.asarray(mask),
        "accum0": (jnp.asarray(A0), jnp.asarray(b0), jnp.asarray(c0)),
        "offset": 0.5 * LN2PI * float(np.sum(mask == 0)),
    }


def filter_loglike(p, inp):
    """Argus marginal filter (diffuse) on the compressed cells, with accum0."""
    ha, ga, sp, gp = physical(p)
    θ = bayesian_inference.Parameters(
        log10_gamma_a=p[1],
        γa=ga,
        ha=ha,
        γp=jnp.atleast_1d(gp),
        σp=jnp.atleast_1d(sp),
        EFAC=jnp.ones(1),
        EQUAD=jnp.zeros(1),
    )
    K = inp["H_matrices"].shape[-1] - 4
    return inp["offset"] + jk._run_kalman_filter_marginal(
        θ=θ,
        data=inp["data"],
        data_errors=inp["data_errors"],
        H_matrices=inp["H_matrices"],
        Npsr=1,
        M_sum=K,
        hellings_downs_matrix=jnp.ones((1, 1)),
        dt_array=inp["dt_array"],
        dim_x=2,
        n_states=4 + K,
        P_eps_inv=jnp.zeros((K, K)),
        mask_matrices=inp["mask_matrices"],
        diffuse=True,
        accum0=inp["accum0"],
    )

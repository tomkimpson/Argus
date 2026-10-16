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

The choice of origin only adds a random constant to the residual, which lies in the span
of the timing model's offset column and so cancels exactly under the flat prior.

Both likelihoods take p = (log10 ha, log10 γa, log10 σp, log10 γp) and are written in
JAX so their gradients come from autodiff.
"""

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from argus import bayesian_inference  # noqa: E402
from argus import jax_kalman_filter as jk  # noqa: E402

LN2PI = np.log(2.0 * np.pi)


def _iou_cov(t, gamma, sigma2):
    """Continuous-time integrated-OU position covariance at times t (from the origin)."""
    E = jnp.expm1(-gamma * t)
    dt = jnp.abs(t[:, None] - t[None, :])
    g = 2.0 * gamma * jnp.minimum(t[:, None], t[None, :]) + E[:, None] + E[None, :]
    g = g - jnp.expm1(-gamma * dt)
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
    """Single-pulsar inputs for `_run_kalman_filter_marginal` from a compression."""
    has = comp["has_scalar"]
    times = comp["times"][has]
    m = comp["m"][has]
    T, K = m.shape
    H = np.zeros((T, 1, 4 + K))
    H[:, 0, 0] = -1.0  # GW redshift state r
    H[:, 0, 2] = 1.0 / comp["f0"]  # spin phase state
    H[:, 0, 4:] = m
    A0, b0, c0 = comp["accum0"]
    return {
        "data": jnp.asarray(comp["s_hat"][has][:, None]),
        "data_errors": jnp.asarray(np.sqrt(comp["sigma2"][has])[:, None]),
        "H_matrices": jnp.asarray(H),
        "dt_array": jnp.asarray(np.diff(times)),
        "mask_matrices": jnp.ones((T, 1)),
        "accum0": (jnp.asarray(A0), jnp.asarray(b0), jnp.asarray(c0)),
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
    return jk._run_kalman_filter_marginal(
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

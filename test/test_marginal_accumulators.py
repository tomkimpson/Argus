"""Initial accumulators (A₀, b₀, c₀) on the marginalized timing-model filter.

Epoch compression (spec D2) splits each window's data into one scalar that sees the
dynamic state and an orthogonal remainder that depends on the timing parameters β alone.
The remainder never touches the state, so its contribution to the filter's (A, b, c)
accumulators is a constant that can be precomputed once and passed in as `accum0`.

These tests pin two properties (spec C2 and task 2.3):

- zero accumulators leave the likelihood bit-for-bit unchanged (the golden tests use
  atol = 1.0, so they alone cannot show this);
- a nonzero A₀ equals filtering the equivalent β-only rows as extra epochs.
"""

import math
from unittest.mock import Mock, patch

import jax.numpy as jnp
import numpy as np
import pytest

from argus import bayesian_inference
from argus import jax_kalman_filter as jk

from test.test_jax_kalman_filter import _realistic_pulsar_data

LN2PI = math.log(2.0 * math.pi)


def _params():
    return bayesian_inference.Parameters(
        log10_gamma_a=-9.0,
        γa=1e-9,
        ha=1e-15,
        γp=jnp.full(2, 1e-8),
        σp=jnp.full(2, 1e-15),
        EFAC=jnp.ones(2),
        EQUAD=jnp.zeros(2),
    )


def _run(kf, θ, diffuse, **overrides):
    """Call the marginal filter with the arrays a constructed filter would pass."""
    kwargs = dict(
        θ=θ,
        data=kf.jax_data,
        data_errors=kf.jax_data_errors,
        H_matrices=kf.jax_H_matrices,
        Npsr=kf.Npsr,
        M_sum=int(kf.M_sum),
        hellings_downs_matrix=kf.hellings_downs_matrix,
        dt_array=kf.jax_t_diffs,
        dim_x=2 * kf.Npsr,
        n_states=int(kf.nx),
        P_eps_inv=kf.P_eps_inv,
        mask_matrices=kf.jax_mask_matrices,
        diffuse=diffuse,
    )
    kwargs.update(overrides)
    return jk._run_kalman_filter_marginal(**kwargs)


@pytest.fixture
def kf():
    with patch("argus.io_manager.get_argus_logger", return_value=Mock()):
        return jk.JaxKalmanFilter(data=_realistic_pulsar_data(), timing_prior="diffuse")


@pytest.mark.parametrize("diffuse", [True, False])
def test_zero_accumulators_are_bit_identical(kf, diffuse):
    """C2: zero (A₀, b₀, c₀) reproduces the call without the argument exactly."""
    θ = _params()
    M = int(kf.M_sum)
    zeros = (jnp.zeros((M, M)), jnp.zeros((M, 1)), jnp.zeros(()))
    ll_plain = _run(kf, θ, diffuse)
    ll_zero = _run(kf, θ, diffuse, accum0=zeros)
    assert np.isfinite(float(ll_plain))
    assert float(ll_zero) == float(ll_plain)


@pytest.mark.parametrize("diffuse", [True, False])
def test_nonzero_accumulators_equal_appended_rows(kf, diffuse):
    """A₀, b₀, c₀ from β-only rows equal filtering those rows as extra epochs.

    An appended epoch whose H has zeros on the dynamic columns has S = R and innovation
    sensitivity Ψ = H_eps, so it adds Ψ'R⁻¹Ψ, Ψ'R⁻¹z and z²/R to (A, b, c) and log(2πR)
    to L. The accum0 run omits that last, parameter-independent log-det term, so the two
    likelihoods differ by exactly 0.5 Σ log(2πR).
    """
    θ = _params()
    rng = np.random.default_rng(1)
    n_extra, n_dyn, M = 3, 4 * kf.Npsr, int(kf.M_sum)
    sigma = 1e-6

    H_extra = np.zeros((n_extra, kf.Npsr, n_dyn + M))
    for n in range(kf.Npsr):
        lo = int(kf.M_start_indices[n]) - n_dyn
        hi = int(kf.M_start_indices[n + 1]) - n_dyn
        H_extra[:, n, n_dyn + lo : n_dyn + hi] = rng.standard_normal((n_extra, hi - lo))
    z_extra = rng.standard_normal((n_extra, kf.Npsr)) * sigma
    err_extra = np.full((n_extra, kf.Npsr), sigma)

    # The equivalent accumulators: stack the rows and sum their contributions.
    Psi = H_extra[:, :, n_dyn:].reshape(-1, M)
    z = z_extra.reshape(-1, 1)
    w = 1.0 / err_extra.reshape(-1, 1) ** 2
    A0 = Psi.T @ (w * Psi)
    b0 = Psi.T @ (w * z)
    c0 = float((z.T @ (w * z))[0, 0])
    logdet_extra = float(np.sum(LN2PI + np.log(err_extra**2)))

    ll_accum = _run(
        kf, θ, diffuse, accum0=(jnp.asarray(A0), jnp.asarray(b0), jnp.asarray(c0))
    )
    ll_append = _run(
        kf,
        θ,
        diffuse,
        data=jnp.concatenate([kf.jax_data, jnp.asarray(z_extra)]),
        data_errors=jnp.concatenate([kf.jax_data_errors, jnp.asarray(err_extra)]),
        H_matrices=jnp.concatenate([kf.jax_H_matrices, jnp.asarray(H_extra)]),
        dt_array=jnp.concatenate([kf.jax_t_diffs, jnp.full(n_extra, 86400.0)]),
        mask_matrices=jnp.concatenate(
            [kf.jax_mask_matrices, jnp.ones((n_extra, kf.Npsr))]
        ),
    )
    # The extra rows must actually move the likelihood, or the test proves nothing.
    assert abs(float(ll_accum) - float(_run(kf, θ, diffuse))) > 1.0
    np.testing.assert_allclose(
        float(ll_accum) - 0.5 * logdet_extra, float(ll_append), rtol=1e-12, atol=1e-6
    )

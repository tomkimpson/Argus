"""The filtered covariance must stay exactly symmetric through the Joseph update.

The Joseph form P = (I-KH) Pp (I-KH)' + K R K' is symmetric in exact arithmetic but not
in floating point. Left unsymmetrised, the asymmetric roundoff grows geometrically
across epochs (~1e-19 -> 1e-10 relative over 150 MDC2 epochs) and eventually drives the
variance of one pulsar's measured combination phi/f0 - r negative. The innovation
covariance inherits the negative eigenvalue, the slogdet sign guard fires, and the log
likelihood is -inf. NUTS chains that stepped into those points froze in warmup (step size
~1e-14) on MDC2 2b and on the OU self-generated spike dataset. Before the holes appear,
the same drift makes the two mathematically identical backends disagree by several nats.

These tests pin both symptoms at parameter vectors taken from the frozen chains.
"""

import json
import math
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from argus import bayesian_inference
from argus import jax_kalman_filter as jk

from test.test_masked_marginal_filter import mdc2  # noqa: F401  (fixture)

HERE = os.path.dirname(os.path.abspath(__file__))


def _ill_scaled_problem(seed=0, n_psr=3):
    """A predicted covariance and observation shaped like the PTA problem: per pulsar,
    spin phase variance ~1e-5 observed through 1/f0 alongside a GW redshift variance
    ~1e-10, measured to ~1e-13. That spread is what makes the Joseph product asymmetric.
    """
    rng = np.random.default_rng(seed)
    n = 4 * n_psr
    scales = np.concatenate(
        [np.tile([1e-10, 1e-27], n_psr), np.tile([1e-5, 1e-20], n_psr)]
    )
    A = rng.standard_normal((n, n))
    C = A @ A.T / n + np.eye(n)
    d = np.sqrt(scales)
    Pp = jnp.asarray(C * np.outer(d, d))
    H = np.zeros((n_psr, n))
    for i in range(n_psr):
        H[i, 2 * i] = -1.0
        H[i, 2 * n_psr + 2 * i] = 1.0 / (200.0 + 100.0 * i)
    R = jnp.asarray(np.diag(rng.uniform(0.5, 2.0, n_psr) * 1e-13))
    z = jnp.asarray(rng.standard_normal(n_psr) * 1e-6)
    return jnp.zeros((n, 1)), Pp, jnp.asarray(H), R, z


@pytest.mark.parametrize("seed", range(5))
def test_update_returns_an_exactly_symmetric_covariance(seed):
    xp, Pp, H, R, z = _ill_scaled_problem(seed)
    _, P, _, _ = jk._update(xp, Pp, H, R, z)
    np.testing.assert_array_equal(P, P.T)


@pytest.mark.parametrize("seed", range(5))
def test_marginal_update_returns_an_exactly_symmetric_covariance(seed):
    xp, Pp, H, R, z = _ill_scaled_problem(seed)
    H_eps = jnp.ones((H.shape[0], 2))
    Xi = jnp.zeros((Pp.shape[0], 2))
    _, P, *_ = jk._update_marginal(xp, Pp, Xi, H, H_eps, R, z)
    np.testing.assert_array_equal(P, P.T)


def _frozen_params(label, base, names):
    with open(os.path.join(HERE, "data/frozen_nuts_points.json")) as f:
        p = json.load(f)[label]
    log10_ga = p["log10_gamma_a"]
    return bayesian_inference.Parameters(
        γa=10.0**log10_ga,
        ha=10.0 ** p["log10_ha"],
        log10_gamma_a=log10_ga,
        γp=10.0 ** jnp.array([p["log10_gamma_p"][n] for n in names]),
        σp=10.0 ** jnp.array([p["log10_sigma_p"][n] for n in names]),
        EFAC=base.EFAC,
        EQUAD=base.EQUAD,
        orf_epsilon=p["orf_epsilon"],
    )


@pytest.fixture(scope="module")
def filters(mdc2):  # noqa: F811
    pulsar_data, base = mdc2
    names = [str(n) for n in pulsar_data["metadata"]["name"]]
    marginal = jk.JaxKalmanFilter(
        data=pulsar_data, use_gw=True, use_marginal=True, timing_prior="informative"
    )
    sequential = jk.JaxKalmanFilter(
        data=pulsar_data, use_gw=True, use_marginal=False, timing_prior="informative"
    )
    return marginal, sequential, base, names


FROZEN = ["mdc2_2b_eps0_chain1", "ou_selfgen_eps0_chain2", "ou_selfgen_eps0_chain3"]


@pytest.mark.parametrize("label", FROZEN)
def test_frozen_nuts_points_have_a_finite_likelihood(filters, label):
    marginal, sequential, base, names = filters
    params = _frozen_params(label, base, names)
    for kf in (marginal, sequential):
        assert math.isfinite(float(kf.get_likelihood(params)))


@pytest.mark.parametrize("label", FROZEN)
def test_backends_agree_at_frozen_nuts_points(filters, label):
    """On this real-data set the two backends agree only to ~0.1-0.3 nats even at healthy
    points (0.07 at the golden), a floor set by the augmented-state filter carrying 427
    timing states. The unsymmetrised filter missed by 2.8 and 6.6 nats at the self-gen
    points, so 0.5 separates the bug from that floor."""
    marginal, sequential, base, names = filters
    params = _frozen_params(label, base, names)
    np.testing.assert_allclose(
        float(marginal.get_likelihood(params)),
        float(sequential.get_likelihood(params)),
        rtol=0,
        atol=0.5,
    )


@pytest.mark.parametrize("label", FROZEN)
def test_gradient_is_finite_at_frozen_nuts_points(filters, label):
    """NUTS needs the gradient as well as the value; the fix must not introduce NaNs."""
    marginal, _, base, names = filters
    params = _frozen_params(label, base, names)
    g = jax.grad(lambda s: marginal.get_likelihood(params.replace(σp=s)))(params.σp)
    assert bool(jnp.all(jnp.isfinite(g)))

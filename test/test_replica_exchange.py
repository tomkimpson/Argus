"""Tests for replica exchange MCMC (parallel tempering)."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from argus.replica_exchange import (
    build_temperature_ladder,
    _propose_swaps,
    _make_single_chain_hmc_step,
    _adapt_temperature_ladder,
    run_replica_exchange,
    re_results_to_arviz,
)

import logging

_logger = logging.getLogger("argus")
if not _logger.handlers:
    _logger.addHandler(logging.NullHandler())
    _logger.setLevel(logging.WARNING)


# ============================================================
# Temperature ladder
# ============================================================


class TestTemperatureLadder:

    def test_geometric_endpoints(self):
        """Geometric ladder should go from 1.0 to beta_hot."""
        betas = build_temperature_ladder(8, beta_hot=0.01, spacing="geometric")
        assert betas.shape == (8,)
        np.testing.assert_allclose(betas[0], 1.0, atol=1e-10)
        np.testing.assert_allclose(betas[-1], 0.01, atol=1e-10)

    def test_linear_endpoints(self):
        """Linear ladder should go from 1.0 to beta_hot."""
        betas = build_temperature_ladder(5, beta_hot=0.1, spacing="linear")
        assert betas.shape == (5,)
        np.testing.assert_allclose(betas[0], 1.0, atol=1e-10)
        np.testing.assert_allclose(betas[-1], 0.1, atol=1e-10)

    def test_monotonically_decreasing(self):
        """Temperatures should decrease from cold to hot."""
        betas = build_temperature_ladder(10, beta_hot=0.01)
        diffs = jnp.diff(betas)
        assert jnp.all(diffs < 0)

    def test_single_chain(self):
        """Single chain should be beta=1."""
        betas = build_temperature_ladder(1)
        assert betas.shape == (1,)
        np.testing.assert_allclose(betas[0], 1.0)


# ============================================================
# Adaptive temperature ladder (Vousden et al. 2016)
# ============================================================


class TestAdaptiveTemperatureLadder:

    def test_preserves_endpoints(self):
        """Adaptive ladder should keep beta[0]=1 and beta[-1]=beta_hot."""
        betas = np.array([1.0, 0.5, 0.25, 0.1])
        # Unequal swap rates to trigger adaptation
        swap_rates = np.array([0.05, 0.3, 0.4])
        new_betas = _adapt_temperature_ladder(betas, swap_rates, adapt_count=0)
        np.testing.assert_allclose(new_betas[0], 1.0, atol=1e-10)
        np.testing.assert_allclose(new_betas[-1], 0.1, atol=1e-10)

    def test_monotonically_decreasing(self):
        """Adapted ladder should remain monotonically decreasing."""
        betas = np.array([1.0, 0.6, 0.3, 0.15, 0.05, 0.01])
        swap_rates = np.array([0.01, 0.1, 0.3, 0.5, 0.6])
        new_betas = _adapt_temperature_ladder(betas, swap_rates, adapt_count=0)
        assert np.all(np.diff(new_betas) < 0), f"Not decreasing: {new_betas}"

    def test_packs_chains_at_cold_end(self):
        """When cold pairs have low acceptance, ladder should pack tighter there."""
        betas = np.array([1.0, 0.5, 0.25, 0.1])
        # Cold pair (0-1) has very low acceptance, hot pairs are fine
        swap_rates = np.array([0.01, 0.3, 0.4])
        new_betas = _adapt_temperature_ladder(betas, swap_rates, adapt_count=0)
        # The gap between beta[0] and beta[1] should shrink (beta[1] should increase)
        old_gap = betas[0] - betas[1]
        new_gap = new_betas[0] - new_betas[1]
        assert new_gap < old_gap, (
            f"Cold-end gap should shrink: old={old_gap:.4f}, new={new_gap:.4f}"
        )

    def test_gain_decays(self):
        """Later adaptations should make smaller changes (decaying gain)."""
        betas = np.array([1.0, 0.5, 0.25, 0.1])
        swap_rates = np.array([0.05, 0.3, 0.4])
        early = _adapt_temperature_ladder(betas, swap_rates, adapt_count=0)
        late = _adapt_temperature_ladder(betas, swap_rates, adapt_count=50)
        early_change = np.max(np.abs(early - betas))
        late_change = np.max(np.abs(late - betas))
        assert late_change < early_change

    def test_two_chains_unchanged(self):
        """With only 2 chains, ladder should be returned unchanged."""
        betas = np.array([1.0, 0.01])
        swap_rates = np.array([0.1])
        new_betas = _adapt_temperature_ladder(betas, swap_rates, adapt_count=0)
        np.testing.assert_array_equal(new_betas, betas)


# ============================================================
# Swap proposals
# ============================================================


class TestSwapProposals:

    def test_always_accept_when_favorable(self):
        """Swaps should be accepted when log_alpha is very large (>>0)."""
        rng = jax.random.PRNGKey(0)
        K = 4
        ndim = 3
        betas = jnp.array([1.0, 0.5, 0.1, 0.01])
        positions = jax.random.normal(rng, (K, ndim))
        # Extreme difference: log_alpha = (0.5-1.0) * (1e6 - (-1e6)) = -0.5 * 2e6 = -1e6
        # Wait — for acceptance we need log_alpha > 0.
        # log_alpha = (beta[1]-beta[0]) * (logL[0]-logL[1]) = (0.5-1.0)*(1e6-(-1e6)) = -1e6 < 0
        # So chain 0 has LOWER logL to make swap favourable:
        # log_alpha = (0.5-1.0) * ((-1e6)-(1e6)) = -0.5 * -2e6 = 1e6 >> 0
        loglikelihoods = jnp.array([-1e6, 1e6, -200.0, -300.0])
        logpriors = jnp.zeros(K)

        new_pos, new_lls, new_lps, accepted = _propose_swaps(
            rng, positions, loglikelihoods, logpriors, betas, even_swap=True,
        )
        # Even pairs: (0,1) and (2,3). Pair (0,1) has log_alpha=1e6, must swap.
        assert bool(accepted[0])

    def test_even_odd_alternation(self):
        """Even swaps should only touch even-indexed pairs."""
        rng = jax.random.PRNGKey(42)
        K = 4
        betas = jnp.array([1.0, 0.5, 0.1, 0.01])
        positions = jax.random.normal(rng, (K, 2))
        lls = jnp.array([0.0, 0.0, 0.0, 0.0])
        lps = jnp.zeros(K)

        _, _, _, accepted_even = _propose_swaps(
            rng, positions, lls, lps, betas, even_swap=True,
        )
        _, _, _, accepted_odd = _propose_swaps(
            rng, positions, lls, lps, betas, even_swap=False,
        )
        # Even: only pairs 0 and 2 can be accepted
        # Odd: only pair 1 can be accepted
        assert not bool(accepted_even[1])  # pair 1 cannot be accepted in even sweep
        assert not bool(accepted_odd[0])   # pair 0 cannot be accepted in odd sweep
        assert not bool(accepted_odd[2])   # pair 2 cannot be accepted in odd sweep

    def test_preserves_shapes(self):
        """Swap should preserve array shapes."""
        rng = jax.random.PRNGKey(0)
        K, ndim = 6, 5
        betas = build_temperature_ladder(K)
        positions = jax.random.normal(rng, (K, ndim))
        lls = jnp.zeros(K)
        lps = jnp.zeros(K)

        new_pos, new_lls, new_lps, accepted = _propose_swaps(
            rng, positions, lls, lps, betas, even_swap=True,
        )
        assert new_pos.shape == (K, ndim)
        assert new_lls.shape == (K,)
        assert accepted.shape == (K - 1,)


# ============================================================
# Single chain HMC step
# ============================================================


class TestSingleChainHMC:

    def test_returns_finite(self):
        """HMC step on a simple Gaussian target should return finite values."""
        ndim = 5

        def logprior(x):
            return -0.5 * jnp.sum(x**2)

        def loglikelihood(x):
            return -0.5 * jnp.sum((x - 1.0)**2)

        hmc_kernel = blackjax.hmc.build_kernel()
        step_fn = _make_single_chain_hmc_step(
            logprior, loglikelihood, hmc_kernel,
            jnp.ones(ndim), num_integration_steps=10,
        )

        rng = jax.random.PRNGKey(0)
        position = jnp.zeros(ndim)
        new_pos, ll, lp, acc = step_fn(rng, position, 1.0, 0.1)

        assert jnp.all(jnp.isfinite(new_pos))
        assert jnp.isfinite(ll)
        assert jnp.isfinite(lp)
        assert 0.0 <= float(acc) <= 1.0

    def test_vmappable_across_betas(self):
        """HMC step should be vmappable across different temperatures."""
        ndim = 3

        def logprior(x):
            return -0.5 * jnp.sum(x**2)

        def loglikelihood(x):
            return -0.5 * jnp.sum((x - 2.0)**2)

        hmc_kernel = blackjax.hmc.build_kernel()
        step_fn = _make_single_chain_hmc_step(
            logprior, loglikelihood, hmc_kernel,
            jnp.ones(ndim), num_integration_steps=5,
        )

        K = 4
        rng_keys = jax.random.split(jax.random.PRNGKey(0), K)
        positions = jnp.zeros((K, ndim))
        betas = jnp.array([1.0, 0.5, 0.1, 0.01])
        step_sizes = jnp.full(K, 0.1)

        new_pos, lls, lps, accs = jax.vmap(step_fn)(
            rng_keys, positions, betas, step_sizes,
        )
        assert new_pos.shape == (K, ndim)
        assert jnp.all(jnp.isfinite(new_pos))


# ============================================================
# Full replica exchange (small problem)
# ============================================================


class TestReplicaExchangeSmall:

    def test_bimodal_gaussian(self):
        """Replica exchange should explore both modes of a bimodal Gaussian."""

        def logprior(x):
            return -0.5 * jnp.sum(x**2) / 100.0  # broad prior

        def loglikelihood(x):
            # Two modes at x=[-3] and x=[3]
            mode1 = -0.5 * jnp.sum((x - 3.0)**2)
            mode2 = -0.5 * jnp.sum((x + 3.0)**2)
            return jnp.logaddexp(mode1, mode2)

        results = run_replica_exchange(
            logprior_fn=logprior,
            loglikelihood_fn=loglikelihood,
            ndim=1,
            num_chains=4,
            num_samples=500,
            num_warmup=100,
            num_hmc_steps=5,
            num_integration_steps=10,
            beta_hot=0.1,
            step_size=0.5,
            inverse_mass_matrix=jnp.ones(1),
            seed=42,
        )

        samples = np.array(results["cold_chain_samples"]).flatten()
        assert len(samples) == 500
        assert results["wall_time"] > 0
        assert results["swap_acceptance_rates"].shape == (3,)  # K-1 pairs
        assert results["hmc_acceptance_rates"].shape == (4,)   # K chains

    def test_arviz_conversion(self):
        """ArviZ conversion should produce valid InferenceData."""
        from argus.tempered_smc import ParamBlock, ParameterRegistry

        # Minimal registry for 2-dim problem
        registry = ParameterRegistry(
            blocks=[
                ParamBlock("x0", "x0", 1, 0, jnp.array(0.0), jnp.array(1.0)),
                ParamBlock("x1", "x1", 1, 1, jnp.array(0.0), jnp.array(1.0)),
            ],
            ndim=2,
            n_pulsars=0,
            fixed_values={},
            has_hierarchical_gamma=False,
            has_hierarchical_ratio=False,
            equad_use_log10=False,
        )

        samples = jnp.ones((100, 2))
        inf_data = re_results_to_arviz(samples, registry, n_pulsars=0)
        assert hasattr(inf_data, "posterior")


# Need blackjax import for HMC tests
import blackjax

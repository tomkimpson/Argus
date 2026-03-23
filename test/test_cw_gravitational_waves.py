"""Tests for the continuous wave signal model functions in gravitational_waves.py."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from argus.gravitational_waves import (
    gw_propagation_direction,
    pulsar_direction,
    polarization_vectors,
    polarization_tensors,
    antenna_pattern_single,
    compute_antenna_patterns,
    cw_timing_residual,
    compute_cw_signal_single_pulsar,
)


class TestGWPropagationDirection:
    """Tests for the GW propagation direction vector."""

    def test_unit_vector(self):
        """n_hat should be a unit vector."""
        n = gw_propagation_direction(1.0, 0.5)
        assert jnp.allclose(jnp.linalg.norm(n), 1.0, atol=1e-14)

    def test_north_pole(self):
        """Source at north pole (delta=pi/2): n_hat should be (0, 0, -1)."""
        n = gw_propagation_direction(0.0, jnp.pi / 2)
        assert jnp.allclose(n, jnp.array([0.0, 0.0, -1.0]), atol=1e-14)

    def test_equator_zero_ra(self):
        """Source at (RA=0, DEC=0): n_hat should be (-1, 0, 0)."""
        n = gw_propagation_direction(0.0, 0.0)
        assert jnp.allclose(n, jnp.array([-1.0, 0.0, 0.0]), atol=1e-14)

    def test_negative_sign(self):
        """n_hat should point opposite to the source direction."""
        alpha, delta = 1.2, 0.3
        n = gw_propagation_direction(alpha, delta)
        q = pulsar_direction(alpha, delta)
        assert jnp.allclose(n, -q, atol=1e-14)


class TestPulsarDirection:
    """Tests for the pulsar direction vector."""

    def test_unit_vector(self):
        q = pulsar_direction(2.0, -0.3)
        assert jnp.allclose(jnp.linalg.norm(q), 1.0, atol=1e-14)

    def test_equator_ra_pi_half(self):
        """Pulsar at (RA=pi/2, DEC=0): q_hat should be (0, 1, 0)."""
        q = pulsar_direction(jnp.pi / 2, 0.0)
        assert jnp.allclose(q, jnp.array([0.0, 1.0, 0.0]), atol=1e-14)


class TestPolarizationVectors:
    """Tests for polarization basis vectors."""

    def test_orthogonality(self):
        """m_hat and l_hat should be orthogonal."""
        m, l = polarization_vectors(1.5, 0.7)
        assert jnp.allclose(jnp.dot(m, l), 0.0, atol=1e-14)

    def test_unit_norm(self):
        """m_hat and l_hat should be unit vectors."""
        m, l = polarization_vectors(1.5, 0.7)
        assert jnp.allclose(jnp.linalg.norm(m), 1.0, atol=1e-14)
        assert jnp.allclose(jnp.linalg.norm(l), 1.0, atol=1e-14)

    def test_orthogonal_to_propagation(self):
        """m_hat and l_hat should be orthogonal to the propagation direction."""
        alpha, delta = 2.0, 0.5
        n = gw_propagation_direction(alpha, delta)
        m, l = polarization_vectors(alpha, delta)
        assert jnp.allclose(jnp.dot(m, n), 0.0, atol=1e-14)
        assert jnp.allclose(jnp.dot(l, n), 0.0, atol=1e-14)


class TestPolarizationTensors:
    """Tests for polarization tensors."""

    def test_symmetry(self):
        """Polarization tensors should be symmetric."""
        e_plus, e_cross = polarization_tensors(1.0, 0.5, 0.3)
        assert jnp.allclose(e_plus, e_plus.T, atol=1e-14)
        assert jnp.allclose(e_cross, e_cross.T, atol=1e-14)

    def test_trace_free(self):
        """Polarization tensors should be trace-free."""
        e_plus, e_cross = polarization_tensors(1.0, 0.5, 0.3)
        assert jnp.allclose(jnp.trace(e_plus), 0.0, atol=1e-14)
        assert jnp.allclose(jnp.trace(e_cross), 0.0, atol=1e-14)

    def test_transverse(self):
        """Polarization tensors should be transverse to the propagation direction."""
        alpha, delta, psi = 1.0, 0.5, 0.3
        n = gw_propagation_direction(alpha, delta)
        e_plus, e_cross = polarization_tensors(alpha, delta, psi)
        # e_ij * n_j should be zero for all i
        assert jnp.allclose(e_plus @ (-n), jnp.zeros(3), atol=1e-14)
        assert jnp.allclose(e_cross @ (-n), jnp.zeros(3), atol=1e-14)

    def test_psi_zero_reduces_to_unrotated(self):
        """With psi=0, tensors should match unrotated form."""
        alpha, delta = 1.0, 0.5
        m, l = polarization_vectors(alpha, delta)
        e_plus, e_cross = polarization_tensors(alpha, delta, 0.0)

        expected_plus = jnp.outer(m, m) - jnp.outer(l, l)
        expected_cross = jnp.outer(m, l) + jnp.outer(l, m)

        assert jnp.allclose(e_plus, expected_plus, atol=1e-14)
        assert jnp.allclose(e_cross, -expected_cross, atol=1e-14)

    def test_psi_rotation_periodicity(self):
        """Polarization tensors should be periodic in psi with period pi."""
        alpha, delta, psi = 1.0, 0.5, 0.7
        e_plus_1, e_cross_1 = polarization_tensors(alpha, delta, psi)
        e_plus_2, e_cross_2 = polarization_tensors(alpha, delta, psi + jnp.pi)
        assert jnp.allclose(e_plus_1, e_plus_2, atol=1e-14)
        assert jnp.allclose(e_cross_1, e_cross_2, atol=1e-14)


class TestAntennaPatternFunctions:
    """Tests for antenna pattern functions."""

    def test_finite_values(self):
        """Antenna patterns should be finite for generic sky positions."""
        F_plus, F_cross = antenna_pattern_single(0.5, 0.3, 2.0, -0.5, 0.7)
        assert jnp.isfinite(F_plus)
        assert jnp.isfinite(F_cross)

    def test_source_behind_pulsar_clipping(self):
        """When source is behind pulsar, denominator clipping should prevent divergence."""
        # Source direction opposite to pulsar direction
        alpha, delta = 1.0, 0.5
        # Pulsar in the same direction as the source (n_hat = -q_hat, so n·q = -1)
        F_plus, F_cross = antenna_pattern_single(alpha, delta, alpha, delta, 0.0)
        assert jnp.isfinite(F_plus)
        assert jnp.isfinite(F_cross)

    def test_vectorized_matches_single(self):
        """Vectorized computation should match single-pulsar results."""
        alpha_gw, delta_gw, psi = 2.0, -0.5, 0.7
        ra_arr = jnp.array([0.5, 1.5, 3.0])
        dec_arr = jnp.array([0.3, -0.2, 0.8])

        F_plus_vec, F_cross_vec = compute_antenna_patterns(
            ra_arr, dec_arr, alpha_gw, delta_gw, psi
        )

        for i in range(3):
            F_plus_i, F_cross_i = antenna_pattern_single(
                ra_arr[i], dec_arr[i], alpha_gw, delta_gw, psi
            )
            assert jnp.allclose(F_plus_vec[i], F_plus_i, atol=1e-14)
            assert jnp.allclose(F_cross_vec[i], F_cross_i, atol=1e-14)

    def test_jit_compatible(self):
        """Antenna pattern computation should be JIT-compilable."""
        jitted_fn = jax.jit(antenna_pattern_single)
        F_plus, F_cross = jitted_fn(0.5, 0.3, 2.0, -0.5, 0.7)
        assert jnp.isfinite(F_plus)
        assert jnp.isfinite(F_cross)


class TestCWTimingResidual:
    """Tests for CW timing residual computation."""

    def test_zero_amplitude(self):
        """Zero strain amplitude should give zero residual."""
        result = cw_timing_residual(
            t=1e8, f_gw=1e-8, h0=0.0, cos_iota=0.5, Phi0=0.3,
            F_plus=0.1, F_cross=-0.2,
        )
        assert jnp.allclose(result, 0.0, atol=1e-30)

    def test_zero_antenna_patterns(self):
        """Zero antenna patterns should give zero residual."""
        result = cw_timing_residual(
            t=1e8, f_gw=1e-8, h0=1e-14, cos_iota=0.5, Phi0=0.3,
            F_plus=0.0, F_cross=0.0,
        )
        assert jnp.allclose(result, 0.0, atol=1e-30)

    def test_face_on_source(self):
        """Face-on source (cos_iota=0): only cross term contributes to sin, plus to cos."""
        # For cos_iota = 0: (1+cos^2 iota)/2 = 1/2, and cos_iota = 0
        # So Delta_s_cross = 0, only Delta_s_plus contributes
        result = cw_timing_residual(
            t=1e8, f_gw=1e-8, h0=1e-14, cos_iota=0.0, Phi0=0.0,
            F_plus=1.0, F_cross=1.0,
        )
        # Delta_s_plus = h0 * 0.5 / Omega * sin(Omega*t)
        # Delta_s_cross = 0
        Omega = 2 * np.pi * 1e-8
        expected = 1.0 * 1e-14 * 0.5 / Omega * np.sin(Omega * 1e8)
        assert jnp.allclose(result, expected, rtol=1e-10)

    def test_residual_scales_with_h0(self):
        """Residual should scale linearly with h0."""
        kwargs = dict(t=1e8, f_gw=1e-8, cos_iota=0.5, Phi0=0.3, F_plus=0.1, F_cross=-0.2)
        r1 = cw_timing_residual(h0=1e-14, **kwargs)
        r2 = cw_timing_residual(h0=2e-14, **kwargs)
        assert jnp.allclose(r2, 2.0 * r1, rtol=1e-10)

    def test_differentiable(self):
        """CW timing residual should be differentiable w.r.t. all CW parameters."""
        def residual_fn(h0, f_gw, cos_iota, Phi0):
            return cw_timing_residual(1e8, f_gw, h0, cos_iota, Phi0, 0.1, -0.2)

        grad_fn = jax.grad(residual_fn, argnums=(0, 1, 2, 3))
        grads = grad_fn(1e-14, 1e-8, 0.5, 0.3)
        for g in grads:
            assert jnp.isfinite(g)


class TestComputeCWSignalSinglePulsar:
    """Tests for vectorized single-pulsar CW signal computation."""

    def test_matches_scalar_loop(self):
        """Vectorized result should match computing each time step individually."""
        toas = jnp.array([1e8, 2e8, 3e8, 4e8])
        f_gw, h0, cos_iota, Phi0 = 1e-8, 1e-14, 0.5, 0.3
        F_plus, F_cross = 0.1, -0.2

        vec_result = compute_cw_signal_single_pulsar(
            toas, f_gw, h0, cos_iota, Phi0, F_plus, F_cross
        )

        for i, t in enumerate(toas):
            scalar_result = cw_timing_residual(t, f_gw, h0, cos_iota, Phi0, F_plus, F_cross)
            assert jnp.allclose(vec_result[i], scalar_result, atol=1e-20)

    def test_output_shape(self):
        """Output should have same shape as input TOAs."""
        toas = jnp.linspace(0, 1e9, 100)
        result = compute_cw_signal_single_pulsar(
            toas, 1e-8, 1e-14, 0.5, 0.3, 0.1, -0.2
        )
        assert result.shape == toas.shape

    def test_jit_compatible(self):
        """Single-pulsar CW signal should be JIT-compilable."""
        jitted_fn = jax.jit(compute_cw_signal_single_pulsar)
        toas = jnp.array([1e8, 2e8, 3e8])
        result = jitted_fn(toas, 1e-8, 1e-14, 0.5, 0.3, 0.1, -0.2)
        assert jnp.all(jnp.isfinite(result))

    def test_pulsar_term_differs_from_earth_only(self):
        """With nonzero pulsar distance, signal should differ from Earth-term only."""
        toas = jnp.array([1e8, 2e8, 3e8, 4e8])
        f_gw, h0, cos_iota, Phi0 = 1e-8, 1e-14, 0.5, 0.3
        F_plus, F_cross = 0.1, -0.2

        earth_only = compute_cw_signal_single_pulsar(
            toas, f_gw, h0, cos_iota, Phi0, F_plus, F_cross,
            pulsar_distance=0.0, geometric_factor=0.5,
        )
        with_pulsar = compute_cw_signal_single_pulsar(
            toas, f_gw, h0, cos_iota, Phi0, F_plus, F_cross,
            pulsar_distance=1e10, geometric_factor=0.5,
        )
        assert not jnp.allclose(earth_only, with_pulsar)

    def test_pulsar_term_zero_distance_matches_earth(self):
        """With zero distance, result should match Earth-term only."""
        toas = jnp.array([1e8, 2e8, 3e8])
        earth = compute_cw_signal_single_pulsar(
            toas, 1e-8, 1e-14, 0.5, 0.3, 0.1, -0.2,
        )
        zero_dist = compute_cw_signal_single_pulsar(
            toas, 1e-8, 1e-14, 0.5, 0.3, 0.1, -0.2,
            pulsar_distance=0.0, geometric_factor=1.5,
        )
        assert jnp.allclose(earth, zero_dist, atol=1e-20)

"""Unit tests for model module (Kalman filter math operations)."""

import pytest
import numpy as np
import jax.numpy as jnp
from argus import model


class TestFBlock:
    """Tests for get_F_block function."""

    def test_basic_transition(self):
        """Test basic state transition matrix computation."""
        γ = 1.0
        dt = 0.1
        F = model.get_F_block(γ, dt)

        assert F.shape == (2, 2)
        assert F[0, 0] == 1.0  # First element should be 1
        assert F[1, 0] == 0.0  # Bottom left should be 0
        assert jnp.isclose(F[1, 1], jnp.exp(-γ * dt))

    def test_f12_element(self):
        """Test F[0,1] element calculation."""
        γ = 2.0
        dt = 0.5
        F = model.get_F_block(γ, dt)

        # F[0,1] = (1 - exp(-γ*dt)) / γ
        expected_f12 = (1 - jnp.exp(-γ * dt)) / γ
        assert jnp.isclose(F[0, 1], expected_f12)

    def test_small_dt(self):
        """Test with small time step (numerical stability)."""
        γ = 1.0
        dt = 1e-6
        F = model.get_F_block(γ, dt)

        # Should approximate identity + dt * generator
        assert jnp.isclose(F[0, 0], 1.0)
        assert jnp.isclose(F[1, 1], 1.0, atol=1e-5)

    def test_large_gamma(self):
        """Test with large gamma values."""
        γ = 1000.0
        dt = 0.01
        F = model.get_F_block(γ, dt)

        # exp(-γ*dt) should be very small
        assert jnp.isclose(F[1, 1], jnp.exp(-γ * dt))
        assert F[1, 1] < 0.01


class TestQBlock:
    """Tests for get_Q_block function."""

    def test_basic_process_noise(self):
        """Test basic process noise matrix computation."""
        γ = 1.0
        dt = 0.1
        Q = model.get_Q_block(γ, dt)

        assert Q.shape == (2, 2)
        # Q should be symmetric
        assert jnp.isclose(Q[0, 1], Q[1, 0])
        # Q should be positive definite (diagonal elements > 0)
        assert Q[0, 0] > 0
        assert Q[1, 1] > 0

    def test_symmetry(self):
        """Test that Q matrix is symmetric."""
        γ = 2.0
        dt = 0.5
        Q = model.get_Q_block(γ, dt)

        assert jnp.allclose(Q, Q.T)

    def test_positive_definite(self):
        """Test that Q is positive semi-definite."""
        γ = 1.5
        dt = 0.2
        Q = model.get_Q_block(γ, dt)

        # Check eigenvalues are non-negative (within numerical tolerance)
        # Small negative values (~1e-5) can occur due to floating point arithmetic
        eigenvalues = jnp.linalg.eigvalsh(Q)
        assert jnp.all(eigenvalues >= -1e-4)

    def test_small_dt_limit(self):
        """Test Q matrix for small time steps."""
        γ = 1.0
        dt = 1e-6
        Q = model.get_Q_block(γ, dt)

        # All elements should be very small for small dt
        assert Q[0, 0] < 1e-10
        assert Q[1, 1] < 1e-5

    def test_q11_matches_exact_integrated_ou(self):
        """q11 must equal the exact integrated-OU position variance (regression).

        For the state (x, v) with dx = v dt, dv = -γ v dt + dW (unit-PSD noise),
        q11 = ∫_0^dt [(1-e^{-γτ})/γ]² dτ. This pins the γ**2 normalization: a γ**3
        typo inflates q11 by a factor 1/γ, which for PTA-scale γ ~ 1e-9 is a ~1e9
        error in the GW/red-noise process noise. Compared against an independent
        numerical quadrature (robust to the closed form's cancellation at small γdt).
        """
        for γ, dt in [(1e-8, 30 * 86400.0), (1e-9, 30 * 86400.0), (2.0, 0.5)]:
            q11 = float(model.get_Q_block(γ, dt)[0, 0])
            τ = np.linspace(0.0, dt, 400_001)
            quad = float(np.trapz(((1 - np.exp(-γ * τ)) / γ) ** 2, τ))

            assert np.isclose(q11, quad, rtol=1e-4)
            # The γ**3 typo would put q11 a factor 1/γ above the true value.
            assert not np.isclose(q11, quad / γ, rtol=1e-2)

    def test_q11_small_dt_cubic_limit(self):
        """For γ·dt << 1 the position variance approaches the dt³/3 white-noise result."""
        γ, dt = 1e-3, 1.0  # γ·dt = 1e-3: small enough for dt³/3, mild cancellation
        q11 = float(model.get_Q_block(γ, dt)[0, 0])
        assert np.isclose(q11, dt**3 / 3.0, rtol=2e-3)


class TestFSpin:
    """Tests for get_F_spin function."""

    def test_block_diagonal_structure(self):
        """Test that F_spin creates block diagonal matrix."""
        gamma = jnp.array([1.0, 2.0, 3.0])
        dt = 0.1
        F_spin = model.get_F_spin(gamma, dt)

        # Should be 6x6 for 3 pulsars (2x2 blocks)
        assert F_spin.shape == (6, 6)

        # Check block diagonal structure - off-block-diagonal should be zero
        assert jnp.allclose(F_spin[0:2, 2:4], 0.0)
        assert jnp.allclose(F_spin[2:4, 4:6], 0.0)
        assert jnp.allclose(F_spin[0:2, 4:6], 0.0)

    def test_individual_blocks(self):
        """Test that individual blocks match get_F_block."""
        gamma = jnp.array([1.0, 2.0])
        dt = 0.1
        F_spin = model.get_F_spin(gamma, dt)

        # First block should match get_F_block(gamma[0], dt)
        F_block_0 = model.get_F_block(gamma[0], dt)
        assert jnp.allclose(F_spin[0:2, 0:2], F_block_0)

        # Second block should match get_F_block(gamma[1], dt)
        F_block_1 = model.get_F_block(gamma[1], dt)
        assert jnp.allclose(F_spin[2:4, 2:4], F_block_1)

    def test_single_pulsar(self):
        """Test with a single pulsar."""
        gamma = jnp.array([1.5])
        dt = 0.2
        F_spin = model.get_F_spin(gamma, dt)

        assert F_spin.shape == (2, 2)
        F_expected = model.get_F_block(gamma[0], dt)
        assert jnp.allclose(F_spin, F_expected)


class TestQSpin:
    """Tests for get_Q_spin function."""

    def test_block_diagonal_structure(self):
        """Test that Q_spin creates block diagonal matrix."""
        gamma = jnp.array([1.0, 2.0])
        dt = 0.1
        sigma_p = jnp.array([1e-15, 2e-15])
        Q_spin = model.get_Q_spin(gamma, dt, sigma_p)

        # Should be 4x4 for 2 pulsars
        assert Q_spin.shape == (4, 4)

        # Check block diagonal structure
        assert jnp.allclose(Q_spin[0:2, 2:4], 0.0)
        assert jnp.allclose(Q_spin[2:4, 0:2], 0.0)

    def test_scaling_with_sigma(self):
        """Test that Q scales with sigma_p."""
        gamma = jnp.array([1.0])
        dt = 0.1
        sigma_p1 = jnp.array([1e-15])
        sigma_p2 = jnp.array([2e-15])

        Q1 = model.get_Q_spin(gamma, dt, sigma_p1)
        Q2 = model.get_Q_spin(gamma, dt, sigma_p2)

        # Q2 should be 2x larger than Q1
        assert jnp.allclose(Q2, Q1 * 2.0, rtol=1e-10)

    def test_symmetry(self):
        """Test that Q_spin blocks are symmetric."""
        gamma = jnp.array([1.0, 2.0])
        dt = 0.1
        sigma_p = jnp.array([1e-15, 2e-15])
        Q_spin = model.get_Q_spin(gamma, dt, sigma_p)

        # Overall matrix should be symmetric
        assert jnp.allclose(Q_spin, Q_spin.T)


class TestPrecomputeRMatrices:
    """Tests for precompute_R_matrices function."""

    def test_basic_r_matrix(self):
        """Test basic R matrix computation."""
        σ = jnp.array([[1e-7, 2e-7], [1.5e-7, 1.8e-7]])  # 2 epochs, 2 pulsars
        EFAC = jnp.array([1.0, 1.2])
        EQUAD = jnp.array([1e-8, 1.5e-8])

        R = model.precompute_R_matrices(σ, EFAC, EQUAD)

        # Should have shape (2, 2, 2) - 2 epochs, each with 2x2 diagonal matrix
        assert R.shape == (2, 2, 2)

    def test_diagonal_structure(self):
        """Test that R matrices are diagonal."""
        σ = jnp.array([[1e-7, 2e-7]])  # 1 epoch, 2 pulsars
        EFAC = jnp.array([1.0, 1.0])
        EQUAD = jnp.array([0.0, 0.0])

        R = model.precompute_R_matrices(σ, EFAC, EQUAD)

        # Check off-diagonal elements are zero
        assert jnp.isclose(R[0, 0, 1], 0.0)
        assert jnp.isclose(R[0, 1, 0], 0.0)

    def test_efac_equad_calculation(self):
        """Test EFAC and EQUAD are applied correctly."""
        σ = jnp.array([[1e-7, 2e-7]])
        EFAC = jnp.array([2.0, 1.5])
        EQUAD = jnp.array([1e-8, 0.0])

        R = model.precompute_R_matrices(σ, EFAC, EQUAD)

        # R[i,i] = (EFAC[i] * σ[i])^2 + EQUAD[i]^2
        expected_r00 = (EFAC[0] * σ[0, 0]) ** 2 + EQUAD[0] ** 2
        expected_r11 = (EFAC[1] * σ[0, 1]) ** 2 + EQUAD[1] ** 2

        assert jnp.isclose(R[0, 0, 0], expected_r00)
        assert jnp.isclose(R[0, 1, 1], expected_r11)


class TestComputeHMatrixForStep:
    """Tests for compute_H_matrix_for_step function."""

    def test_basic_h_matrix_shape(self):
        """Test H matrix has correct shape."""
        Npsr = 2
        nx = 20  # Total state dimension
        M_start_indices = np.array([8, 13, 20])  # Start indices for timing params
        design_matrices = [np.random.randn(10, 5), np.random.randn(10, 7)]
        f0 = np.array([200.0, 150.0])

        H = model.compute_H_matrix_for_step(
            time_step_index=0,
            Npsr=Npsr,
            nx=nx,
            M_start_indices=M_start_indices,
            pulsar_design_matrices=design_matrices,
            use_gw=True,
            f0=f0,
        )

        assert H.shape == (Npsr, nx)

    def test_gw_term_inclusion(self):
        """Test that GW terms are included when use_gw=True."""
        Npsr = 1
        nx = 10
        M_start_indices = np.array([4, 10])
        design_matrices = [np.ones((5, 6))]
        f0 = np.array([200.0])

        H_with_gw = model.compute_H_matrix_for_step(
            time_step_index=0,
            Npsr=Npsr,
            nx=nx,
            M_start_indices=M_start_indices,
            pulsar_design_matrices=design_matrices,
            use_gw=True,
            f0=f0,
        )

        # Redshift coefficient should be -1.0
        assert H_with_gw[0, 0] == -1.0

    def test_gw_term_exclusion(self):
        """Test that GW terms are excluded when use_gw=False."""
        Npsr = 1
        nx = 10
        M_start_indices = np.array([4, 10])
        design_matrices = [np.ones((5, 6))]
        f0 = np.array([200.0])

        H_without_gw = model.compute_H_matrix_for_step(
            time_step_index=0,
            Npsr=Npsr,
            nx=nx,
            M_start_indices=M_start_indices,
            pulsar_design_matrices=design_matrices,
            use_gw=False,
            f0=f0,
        )

        # Redshift coefficient should be 0.0
        assert H_without_gw[0, 0] == 0.0

    def test_spin_term_coefficient(self):
        """Test spin noise term coefficient."""
        Npsr = 1
        nx = 10
        M_start_indices = np.array([4, 10])
        design_matrices = [np.ones((5, 6))]
        f0 = np.array([200.0])

        H = model.compute_H_matrix_for_step(
            time_step_index=0,
            Npsr=Npsr,
            nx=nx,
            M_start_indices=M_start_indices,
            pulsar_design_matrices=design_matrices,
            use_gw=True,
            f0=f0,
        )

        # Spin coefficient should be 1/f0
        spin_idx = 2  # For first pulsar, spin index is 2
        assert np.isclose(H[0, spin_idx], 1.0 / f0[0])


class TestPrecomputeHMatrix:
    """Tests for precompute_H_matrix function."""

    def test_precompute_all_steps(self):
        """Test precomputing H for all time steps."""
        Npsr = 2
        nx = 20
        M_start_indices = np.array([8, 13, 20])
        n_epochs = 5
        design_matrices = [np.random.randn(n_epochs, 5), np.random.randn(n_epochs, 7)]
        f0 = np.array([200.0, 150.0])

        H_all = model.precompute_H_matrix(
            Npsr=Npsr,
            nx=nx,
            M_start_indices=M_start_indices,
            pulsar_design_matrices=design_matrices,
            use_gw=True,
            f0=f0,
        )

        # Should have H matrix for each time step
        assert H_all.shape == (n_epochs, Npsr, nx)

    def test_consistency_across_steps(self):
        """Test that precomputed matrices match individual computations."""
        Npsr = 1
        nx = 10
        M_start_indices = np.array([4, 10])
        n_epochs = 3
        design_matrices = [np.random.randn(n_epochs, 6)]
        f0 = np.array([200.0])

        H_all = model.precompute_H_matrix(
            Npsr=Npsr,
            nx=nx,
            M_start_indices=M_start_indices,
            pulsar_design_matrices=design_matrices,
            use_gw=True,
            f0=f0,
        )

        # Check first step matches
        H_0 = model.compute_H_matrix_for_step(
            time_step_index=0,
            Npsr=Npsr,
            nx=nx,
            M_start_indices=M_start_indices,
            pulsar_design_matrices=design_matrices,
            use_gw=True,
            f0=f0,
        )

        assert np.allclose(H_all[0], H_0)

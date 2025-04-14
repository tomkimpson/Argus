import unittest
import jax
import jax.numpy as jnp
from jax.scipy.linalg import block_diag
from jax import random
import numpy as np

# Import the module to be tested
from argus import jmath  # Assuming the module is named jmath

class TestJMath(unittest.TestCase):

    def test_get_F_block(self):
        """Test get_F_block function.

        This test verifies the shape and basic element values of the 
        state transition block matrix computed by `get_F_block`. 
        It also tests a specific edge case where gamma is zero.
        """

        gamma = 0.5
        dt = 0.1
        F_block = jmath.get_F_block(gamma, dt)
        
        self.assertEqual(F_block.shape, (2, 2))
        self.assertTrue(jnp.allclose(F_block[0, 0], 1.0))
        self.assertTrue(jnp.allclose(F_block[1, 0], 0.0))

        # Test with gamma = 0 (edge case)
        F_block_zero_gamma = jmath.get_F_block(0.0, dt)
        expected_F_zero_gamma = jnp.array([[1.0, dt], [0.0, 1.0]])
        self.assertTrue(jnp.allclose(F_block_zero_gamma, expected_F_zero_gamma))

    def test_get_Q_block(self):
        """Test get_Q_block function.

        This test checks the shape and symmetry of the process noise 
        covariance block matrix computed by `get_Q_block`. It also 
        performs a qualitative check for numerical stability with small input values.
        """

        gamma = 0.5
        dt = 0.1
        Q_block = jmath.get_Q_block(gamma, dt)
        self.assertEqual(Q_block.shape, (2, 2))
        self.assertTrue(jnp.allclose(Q_block[0, 1], Q_block[1, 0])) #symmetry

        # Test with small gamma and dt (check for numerical stability - qualitatively)
        small_gamma = 1e-6
        small_dt = 1e-6
        Q_block_small = jmath.get_Q_block(small_gamma, small_dt)
        self.assertTrue(jnp.all(jnp.isfinite(Q_block_small)))

    def test_get_F_spin(self):
        """Test get_F_spin function.

        This test verifies the shape and block diagonal structure of the 
        state transition matrix for spin noise, computed by `get_F_spin`.
        """

        gamma = jnp.array([0.1, 0.2, 0.3])
        dt = 0.1
        F_spin = jmath.get_F_spin(gamma, dt)
        self.assertEqual(F_spin.shape, (6, 6))
        self.assertEqual(F_spin.diagonal().shape, (6,))

        # Check block diagonal structure
        self.assertTrue(jnp.allclose(F_spin[:2, :2], jmath.get_F_block(gamma[0], dt)))
        self.assertTrue(jnp.allclose(F_spin[2:4, 2:4], jmath.get_F_block(gamma[1], dt)))
        self.assertTrue(jnp.allclose(F_spin[4:, 4:], jmath.get_F_block(gamma[2], dt)))

    def test_get_F(self):
        """Test get_F function.

        This test checks the shapes of the GW and spin transition matrices 
        computed by `get_F`. It also validates the Kronecker product 
        structure of the GW transition matrix.
        """

        gamma_gw = 0.5
        gamma_spin = jnp.array([0.1, 0.2])
        dt = 0.1
        Npsr = 2
        M_sum = 4
        F_gw, F_spin = jmath.get_F(gamma_gw, gamma_spin, dt, Npsr, M_sum)

        self.assertEqual(F_gw.shape, (4, 4))
        self.assertEqual(F_spin.shape, (4, 4))

        # Check F_gw structure (Kronecker product)
        expected_F_gw_block = jmath.get_F_block(gamma_gw, dt)
        expected_F_gw = jnp.kron(jnp.eye(Npsr), expected_F_gw_block)
        self.assertTrue(jnp.allclose(F_gw, expected_F_gw))

    def test_get_Q_spin(self):
        """Test get_Q_spin function.

        This test verifies the shape and block diagonal structure of the 
        process noise covariance matrix for spin noise, computed by `get_Q_spin`. 
        It also checks the scaling of the blocks by sigma_p.
        """

        gamma = jnp.array([0.1, 0.2, 0.3])
        dt = 0.1
        sigma_p = jnp.array([1.0, 2.0, 3.0])
        Q_spin = jmath.get_Q_spin(gamma, dt, sigma_p)

        self.assertEqual(Q_spin.shape, (6, 6))

        # Check block diagonal structure and scaling by sigma_p
        self.assertTrue(jnp.allclose(Q_spin[:2, :2], jmath.get_Q_block(gamma[0], dt) * sigma_p[0]))
        self.assertTrue(jnp.allclose(Q_spin[2:4, 2:4], jmath.get_Q_block(gamma[1], dt) * sigma_p[1]))
        self.assertTrue(jnp.allclose(Q_spin[4:, 4:], jmath.get_Q_block(gamma[2], dt) * sigma_p[2]))

    def test_get_Q(self):
        """Test get_Q function.

        This test checks the shapes of the GW, spin, and timing process noise 
        covariance matrices computed by `get_Q`. It also validates the Kronecker 
        product structure of the GW covariance matrix and the identity-based 
        structure of the timing covariance matrix.
        """

        gamma_gw = 0.5
        sigma_a2 = jnp.array([[1.0, 0.5], [0.5, 2.0]])  # Example covariance
        gamma_spin = jnp.array([0.1, 0.2])
        sigma_p2 = jnp.array([1.0, 2.0])
        dt = 0.1
        Npsr = 2
        M_sum = 4
        eps = 0.01
        Q_gw, Q_spin, Q_timing = jmath.get_Q(gamma_gw, sigma_a2, gamma_spin, sigma_p2, dt, Npsr, M_sum, eps)

        self.assertEqual(Q_gw.shape, (4, 4))
        self.assertEqual(Q_spin.shape, (4, 4))
        self.assertEqual(Q_timing.shape, (4, 4))

        # Check Q_gw structure (Kronecker product)
        expected_Q_gw_block = jmath.get_Q_block(gamma_gw, dt)
        expected_Q_gw = jnp.kron(sigma_a2, expected_Q_gw_block)
        self.assertTrue(jnp.allclose(Q_gw, expected_Q_gw))

        # Check Q_timing structure
        expected_Q_timing = jnp.eye(M_sum) * eps**2
        self.assertTrue(jnp.allclose(Q_timing, expected_Q_timing))

    def test_compute_predicted_state(self):
        """Test compute_predicted_state function.

        This test verifies the shape of the predicted state vector and 
        checks that the GW and spin components are transformed correctly 
        by their respective transition matrices, while the timing components 
        remain unchanged.
        """

        key = random.PRNGKey(0)
        gw_size = 4
        spin_size = 4
        timing_size = 2
        total_size = gw_size + spin_size + timing_size

        # Create random state vector
        x = random.normal(key, (total_size,))

        # Create dummy F matrices
        F_gw = jnp.eye(gw_size) * 0.9
        F_spin = jnp.eye(spin_size) * 0.8
        F_list = (F_gw, F_spin)

        x_pred = jmath.compute_predicted_state(F_list, x, gw_size, spin_size)

        self.assertEqual(x_pred.shape, (total_size, 1))

        # Check that GW and spin states are transformed, timing states are unchanged
        self.assertTrue(jnp.allclose(x_pred[:gw_size].flatten(), F_gw @ x[:gw_size]))
        self.assertTrue(jnp.allclose(x_pred[gw_size:gw_size+spin_size].flatten(), F_spin @ x[gw_size:gw_size+spin_size]))
        self.assertTrue(jnp.allclose(x_pred[gw_size+spin_size:].flatten(), x[gw_size+spin_size:]))

    def test_compute_predicted_covariance(self):
        """Test compute_predicted_covariance function.

        This test verifies the shape and symmetry of the predicted covariance 
        matrix. It also performs basic sanity checks on the diagonal elements 
        to ensure they are at least as large as the corresponding process noise.
        """

        key = random.PRNGKey(0)
        gw_size = 4
        spin_size = 4
        timing_size = 2
        total_size = gw_size + spin_size + timing_size

        # Create random covariance matrix (positive semi-definite)
        P = random.normal(key, (total_size, total_size))
        P = P @ P.T + jnp.eye(total_size) # Ensure PSD

        # Create dummy F and Q matrices
        F_gw = jnp.eye(gw_size) * 0.9
        F_spin = jnp.eye(spin_size) * 0.8
        F_list = (F_gw, F_spin)
        Q_gw = jnp.eye(gw_size) * 0.1
        Q_spin = jnp.eye(spin_size) * 0.2
        Q_timing = jnp.eye(timing_size) * 0.05
        Q_list = (Q_gw, Q_spin, Q_timing)

        P_pred = jmath.compute_predicted_covariance(P, F_list, Q_list, gw_size, spin_size)

        self.assertEqual(P_pred.shape, (total_size, total_size))

        #Check symmetry (covariance matrices should be symmetric)
        self.assertTrue(jnp.allclose(P_pred, P_pred.T))

        # Basic sanity checks on block diagonal elements (should be >= Q)
        self.assertTrue(jnp.all(jnp.diag(P_pred[:gw_size, :gw_size]) >= jnp.diag(Q_gw)))
        self.assertTrue(jnp.all(jnp.diag(P_pred[gw_size:gw_size+spin_size, gw_size:gw_size+spin_size]) >= jnp.diag(Q_spin)))
        self.assertTrue(jnp.all(jnp.diag(P_pred[gw_size+spin_size:, gw_size+spin_size:]) >= jnp.diag(Q_timing)))

    def test_precompute_F_matrices(self):
        """Test precompute_F_matrices function.

        This test verifies the shape of the precomputed F matrices and 
        checks that the matrices at each timestep are computed correctly 
        by comparing them to the output of get_F.
        """

        gamma_a = 0.5
        gamma_p = jnp.array([0.1, 0.2])
        dt_array = jnp.array([0.1, 0.2, 0.3])
        Npsr = 2
        M_sum = 4

        F_gw_matrices, F_spin_matrices = jmath.precompute_F_matrices(gamma_a, gamma_p, dt_array, Npsr, M_sum)

        self.assertEqual(F_gw_matrices.shape, (3, 4, 4)) # (n_timesteps, n_gw, n_gw)
        self.assertEqual(F_spin_matrices.shape, (3, 4, 4)) # (n_timesteps, n_spin, n_spin)

        # Check that the matrices at each timestep are computed correctly
        for i, dt in enumerate(dt_array):
            F_gw_expected, F_spin_expected = jmath.get_F(gamma_a, gamma_p, dt, Npsr, M_sum)
            self.assertTrue(jnp.allclose(F_gw_matrices[i], F_gw_expected))
            self.assertTrue(jnp.allclose(F_spin_matrices[i], F_spin_expected))

    def test_precompute_Q_matrices(self):
        """Test precompute_Q_matrices function.

        This test verifies the shape of the precomputed Q matrices and 
        checks that the matrices at each timestep are computed correctly 
        by comparing them to the output of get_Q.
        """

        gamma_a = 0.5
        sigma_a2 = jnp.array([[1.0, 0.5], [0.5, 2.0]])
        gamma_p = jnp.array([0.1, 0.2])
        sigma_p2 = jnp.array([1.0, 2.0])
        dt_array = jnp.array([0.1, 0.2, 0.3])
        Npsr = 2
        M_sum = 4
        eps = 0.01

        Q_gw_matrices, Q_spin_matrices, Q_timing_matrices = jmath.precompute_Q_matrices(
            gamma_a, sigma_a2, gamma_p, sigma_p2, dt_array, Npsr, M_sum, eps
        )

        self.assertEqual(Q_gw_matrices.shape, (3, 4, 4))
        self.assertEqual(Q_spin_matrices.shape, (3, 4, 4))
        self.assertEqual(Q_timing_matrices.shape, (3, 4, 4))

        # Check that the matrices at each timestep are computed correctly
        for i, dt in enumerate(dt_array):
            Q_gw_expected, Q_spin_expected, Q_timing_expected = jmath.get_Q(
                gamma_a, sigma_a2, gamma_p, sigma_p2, dt, Npsr, M_sum, eps
            )
            self.assertTrue(jnp.allclose(Q_gw_matrices[i], Q_gw_expected))
            self.assertTrue(jnp.allclose(Q_spin_matrices[i], Q_spin_expected))
            self.assertTrue(jnp.allclose(Q_timing_matrices[i], Q_timing_expected))

    def test_precompute_R_matrices(self):
        """Test precompute_R_matrices function.

        This test verifies that the measurement noise covariance matrix R 
        is computed correctly from the given noise parameters.
        """

        sigma = jnp.array([1.0, 2.0, 3.0])
        EFAC = jnp.array([1.1, 1.2, 1.3])
        EQUAD = jnp.array([0.1, 0.2, 0.3])
        psr_indices = jnp.array([0, 1, 2])

        R_matrices = jmath.precompute_R_matrices(sigma, EFAC, EQUAD, psr_indices)

        expected_R = jnp.square(sigma * EFAC) + jnp.square(EQUAD)
        self.assertTrue(jnp.allclose(R_matrices, expected_R))


if __name__ == '__main__':
    unittest.main()
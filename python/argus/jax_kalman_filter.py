"""Module which implements JAX-based Kalman filter algorithm."""

import numpy as np
from tqdm import tqdm
from argus.jmath import get_xp, get_P_blocks, get_Pp_blocks, precompute_F_matrices, precompute_Q_matrices
from line_profiler import profile

from functools import partial
import jax
import jax.numpy as jnp
from jax import lax 

@profile
def _log_likelihood(y, cov):
    """Given the innovation and innovation covariance, get the likelihood."""
    log_likelihood = -0.5 * (jnp.log(2.0 * jnp.pi * cov) + (y * y) / cov)
    return log_likelihood

@profile
def _predict(x, P, F_list, Q_list):
    """Predict the next state and covariance."""
    xp = get_xp(F_list, x, 72, 72)
    P_list = get_P_blocks(P, 72, 72)
    Pp = get_Pp_blocks(F_list, P_list, Q_list)
    return xp, Pp

@profile    
def _update(xp, Pp, H, R, z):
    """Update the state and covariance with a new observation."""
    # Now run through the update algorithm
    y = z - H @ xp                                  # innovation. For this class, this is just a scalar
    S = H @ Pp @ H.T + R                           # innovation covariance, a scalar  
    K = Pp @ H.T / S                               # Kalman gain, dimension (n_x, 1)
    x = xp + K * y                                 # Updated state, dimension (n_x, 1)
    P = (jnp.eye(len(xp)) - K @ H) @ Pp           # Updated covariance, dimension (n_x, n_x)
    return x, P, y, S

@partial(jax.jit, static_argnames=('Npsr', 'M_sum'))
def _run_kalman_filter_scan(θ, data, data_errors, psr_indices, H_matrices, Npsr, M_sum, dt_array, x0, P0):
    """Run the Kalman filter algorithm over all observations and return a log likelihood."""
    # Precompute all F matrices for this parameter set, for different dt values
    F_matrices = precompute_F_matrices(θ.γa, θ.γp, dt_array, Npsr, M_sum)
    Q_matrices = precompute_Q_matrices(θ.γa, θ.γp, dt_array, Npsr, M_sum, θ.σeps)
    
    # First update
    H = H_matrices[0]
    x, P, y, S = _update(xp=x0, Pp=P0, H=H, R=data_errors[0], z=data[0])
    ll0 = _log_likelihood(y, S)

    def step(carry, inputs):
        x, P = carry
        dt_idx, z, R, H = inputs

        # Get precomputed matrices for this timestep
        F_gw_at_timestep = F_matrices[0][dt_idx]    # First element   [0] of tuple, indexed by timestep
        F_spin_at_timestep = F_matrices[1][dt_idx]  # Second element  [1] of tuple, indexed by timestep
        F = (F_gw_at_timestep, F_spin_at_timestep)

        Q_gw_at_timestep = Q_matrices[0][dt_idx]      # First element  [0] of tuple, indexed by timestep
        Q_spin_at_timestep = Q_matrices[1][dt_idx]    # Second element [1] of tuple, indexed by timestep
        Q_timing_at_timestep = Q_matrices[2][dt_idx]  # Third element  [2] of tuple, indexed by timestep
        Q = (Q_gw_at_timestep, Q_spin_at_timestep, Q_timing_at_timestep)

        x_predict, P_predict = _predict(x, P, F, Q)
        x, P, y, S = _update(x_predict, P_predict, H, R, z)
        ll = _log_likelihood(y, S)
        return (x, P), ll

    # Pack inputs for scan
    inputs = (jnp.arange(len(dt_array)), data[1:], data_errors[1:], H_matrices[1:])

    # Run scan loop
    (xf, Pf), ll_arr = lax.scan(step, (x, P), inputs)
    total_ll = ll0 + jnp.sum(ll_arr)
    return total_ll

class JaxScalarKalmanFilter:
    """A class to implement the linear Kalman filter on scalar inputs using JAX.

    Args:
        model: Class which defines all the Kalman machinery e.g. state transition models, covariance matrices etc.
        observations: 2D array which holds the noisy observations recorded at the detector
        x0: A 1D array which holds the initial guess of the initial states
        P0: The uncertainty in the guess of P0
    """

    def __init__(self, model, observations, x0, P0, **kwargs):
        """Initialize the class."""
        self.model = model
        self.observations = observations
        self.x0 = x0
        self.P0 = P0

        # Extract the observations into separate arrays
        self.toa = self.observations[:, 0]
        self.data = self.observations[:, 1]
        self.data_errors = self.observations[:, 2]
        self.psr_indices = self.observations[:, 3].astype(int)
        self.N_timesteps = len(self.observations)
        self.t_diffs = np.diff(self.toa)

        assert np.isscalar(self.data[0])

        # Precompute the observation matrices and assign them to model.H_matrix_list
        self.model.precompute_H_matrix(self.psr_indices)
    
        # Convert to JAX arrays for faster processing
        self._prepare_jax_arrays()

    def _prepare_jax_arrays(self):
        """Convert numpy arrays to JAX arrays"""
        # Convert observations and related data
        self.jax_data = jnp.array(self.data)
        self.jax_data_errors = jnp.array(self.data_errors)
        self.jax_psr_indices = jnp.array(self.psr_indices)
        self.jax_t_diffs = jnp.array(self.t_diffs)
        
        # Convert initial state and covariance
        self.jax_x0 = jnp.array(self.x0.reshape(-1, 1))
        self.jax_P0 = jnp.array(self.P0)
        
        # Convert H matrices
        self.jax_H_matrices = jnp.array([h for h in self.model.H_matrix_list])

    def get_likelihood(self, θ):
        """Run the Kalman filter algorithm over all observations and return a log likelihood."""
        return _run_kalman_filter_scan(
            θ=θ,
            data=self.jax_data,
            data_errors=self.jax_data_errors,
            psr_indices=self.jax_psr_indices,
            H_matrices=self.jax_H_matrices,
            Npsr=self.model.Npsr,
            M_sum=self.model.M_sum,
            dt_array=self.jax_t_diffs,
            x0=self.jax_x0,
            P0=self.jax_P0
        ) 
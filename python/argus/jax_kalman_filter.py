"""Module which implements JAX-based Kalman filter algorithm."""

import numpy as np

from argus.jmath import precompute_F_matrices, precompute_Q_matrices,compute_predicted_covariance,compute_predicted_state
from functools import partial
import jax
import jax.numpy as jnp
from jax import lax 

@jax.named_call
def _log_likelihood(y: jax.Array, cov: jax.Array) -> jax.Array:
    """Calculate the log likelihood given innovation and innovation covariance.
    
    Args:
        y: Innovation term (measurement residual), scalar
        cov: Innovation covariance, scalar
        
    Returns
    -------
        float: Log likelihood value
    """
    log_likelihood = -0.5 * (jnp.log(2.0 * jnp.pi * cov) + (y * y) / cov)
    return log_likelihood

@jax.named_call
def _predict(x: jax.Array, P: jax.Array, F_list: tuple, Q_list: tuple) -> tuple[jax.Array, jax.Array]:
    """Predict the next state and covariance.
    
    Args:
        x: Current state vector
        P: Current covariance matrix
        F_list: Tuple of state transition matrices
        Q_list: Tuple of process noise matrices
        
    Returns
    -------
        tuple: (predicted state, predicted covariance)
        
    Note:
        TODO: Hard-coded dimensions (72,72) should be passed as parameters
    """
    xp = compute_predicted_state(F_list, x, 72, 72)
    Pp = compute_predicted_covariance(P,F_list,Q_list,72,72)
    return xp, Pp

@jax.named_call
def _update(xp: jax.Array, Pp: jax.Array, H: jax.Array, R: jax.Array, z: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Update the state and covariance with a new observation.
    
    Args:
        xp: Predicted state vector
        Pp: Predicted covariance matrix
        H: Observation matrix
        R: Observation noise (scalar)
        z: Observation (scalar)
        
    Returns
    -------
        tuple: (updated state, updated covariance, innovation, innovation covariance)
    """
    y = z - H @ xp                                  # innovation (scalar)
    S = H @ Pp @ H.T + R                           # innovation covariance (scalar)
    K = Pp @ H.T / S                               # Kalman gain
    x = xp + K * y                                 # Updated state
    P = (jnp.eye(len(xp)) - K @ H) @ Pp           # Updated covariance
    return x, P, y, S

@jax.named_call
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

    def __init__(self, model, observations: np.ndarray, x0: np.ndarray, P0: np.ndarray, **kwargs):
        """Initialize the class."""
        if observations.ndim != 2:
            raise ValueError("observations must be a 2D array")
        
        if observations.shape[1] != 4:
            raise ValueError("observations must have 4 columns: time, data, errors, psr_indices")
            
        if x0.shape[0] != P0.shape[0] or P0.shape[0] != P0.shape[1]:
            raise ValueError("Inconsistent dimensions between x0 and P0")
        
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
        """Convert numpy arrays to JAX arrays."""
        # Convert observations and related data
        self.jax_data         = jnp.array(self.data)
        self.jax_data_errors  = jnp.array(self.data_errors)
        self.jax_psr_indices  = jnp.array(self.psr_indices)
        self.jax_t_diffs      = jnp.array(self.t_diffs)
        
        # Convert initial state and covariance
        self.jax_x0          = jnp.array(self.x0.reshape(-1, 1))
        self.jax_P0          = jnp.array(self.P0)
        
        # Convert H matrices
        self.jax_H_matrices  = jnp.array([h for h in self.model.H_matrix_list])

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
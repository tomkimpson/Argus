"""Module which implements JAX-based Kalman filter algorithm."""

import numpy as np

from argus.jmath import precompute_F_matrices, precompute_Q_matrices,precompute_R_matrices,compute_predicted_covariance,compute_predicted_state
from functools import partial
import jax
import jax.numpy as jnp
from jax import lax 


def is_positive_definite(matrix):
    try:
        # If Cholesky decomposition succeeds, matrix is positive definite
        jax.numpy.linalg.cholesky(matrix)
        return True
    except:
        return False


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
    #xp = compute_predicted_state(F_list, x, 72, 72)
    #Pp = compute_predicted_covariance(P,F_list,Q_list,72,72)

    dim_x = 4
    xp = compute_predicted_state(F_list, x, dim_x, dim_x)
    Pp = compute_predicted_covariance(P,F_list,Q_list,dim_x,dim_x)


    
    return xp, Pp


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

    jax.debug.print("P matrix is posdef? {}", is_positive_definite(Pp), ordered=True)
    #jax.debug.print("The P matrix input to the update function is: {}", Pp, ordered=True)


    y = z - H @ xp                                  
    S = H @ Pp @ H.T + R                           
    K = Pp @ H.T / S                               
    x = xp + K * y                                 
    P = (jnp.eye(len(xp)) - K @ H) @ Pp           
    
    #jax.debug.print("Innovation (y): {}", y, ordered=True)
    jax.debug.print("Innovation covariance (S): {}", S[0,0], ordered=True)
    jax.debug.print("Innovation covariance product (HPH): {}", H@Pp@H.T, ordered=True)
    jax.debug.print("R matrix (R): {}", R, ordered=True)


    cond = jax.numpy.linalg.cond(Pp)
    jax.debug.print("Condition number of P matrix: {}", cond, ordered=True)


    HPH = H @ Pp @ H.T
    jax.debug.print("HPH: {}", HPH, ordered=True)
    jax.debug.print("Min eigenvalue of HPH: {}", jax.numpy.linalg.eigvalsh(HPH).min(), ordered=True)


    jax.debug.print("Log10 max(abs(Pp)): {}", jax.numpy.log10(jax.numpy.abs(Pp)).max(), ordered=True)
    jax.debug.print("Log10 max(abs(H)): {}", jax.numpy.log10(jax.numpy.abs(H)).max(), ordered=True)



    #jax.debug.print("Innovation covariance components: {} {} {}", H, Pp, R)
    #jax.debug.print("Kalman gain min/max: {} to {}", jnp.min(K), jnp.max(K), ordered=True)
    #jax.debug.print("Updated state min/max: {} to {}", jnp.min(x), jnp.max(x), ordered=True)
    #jax.debug.print("Updated covariance min/max: {} to {}", jnp.min(P), jnp.max(P), ordered=True)
    

    #jax.debug.print("P matrix condition {}", jnp.linalg.cond(P), ordered=True)
    return x, P, y, S


def _compute_sigma_matrix(h2, γa, Γ):
    return (h2 / 6) * γa * Γ


@jax.named_call
@partial(jax.jit, static_argnames=('Npsr', 'M_sum'))
def _run_kalman_filter_scan(θ, data, data_errors, psr_indices, H_matrices, Npsr, M_sum,hellings_downs_matrix, dt_array, x0, P0):
    """Run the Kalman filter algorithm over all observations and return a log likelihood."""
    
    #Define the 
    σa2 = _compute_sigma_matrix(θ.h2, θ.γa, hellings_downs_matrix)

    jax.debug.print("The length of the data is: {}", len(data), ordered=True)
    jax.debug.print("The size of σa2 is: {}", σa2, ordered=True)
    jax.debug.print("The size of σp2 is: {}", θ.σp**2, ordered=True)
    jax.debug.print("The dt_array is: {}", dt_array, ordered=True)

    
    # Precompute all matrices for this parameter set
    jax.debug.print("### Precomputing matrices ###", ordered=True)
    F_matrices = precompute_F_matrices(θ.γa, θ.γp, dt_array, Npsr, M_sum)
    Q_matrices = precompute_Q_matrices(θ.γa,σa2, θ.γp,θ.σp**2, dt_array, Npsr, M_sum, θ.σeps)
    R_matrices = precompute_R_matrices(data_errors,θ.EFAC, θ.EQUAD, psr_indices)
    
    jax.debug.print("R matrices min/max: {} to {}", jnp.min(R_matrices), jnp.max(R_matrices), ordered=True)

    # First update
    jax.debug.print("### Calling the update function for the first time ###", ordered=True)

    H = H_matrices[0]
    x, P, y, S = _update(xp=x0, Pp=P0, H=H, R=R_matrices[0], z=data[0])
    ll0 = _log_likelihood(y, S)
    
    # Check initial likelihood for NaN
    jax.debug.print("Initial likelihood: {}", ll0)
    is_valid = ~jnp.isnan(ll0)
    
    def step(carry, inputs):
        x, P = carry
        dt_idx, z, R, H = inputs
                #jax.debug.print("Step {} likelihood: {}", dt_idx, ll, ordered=True)


        jax.debug.print("### STEP NUMBER: {} ###", dt_idx, ordered=True)

        # Get precomputed matrices for this timestep
        F_gw_at_timestep = F_matrices[0][dt_idx]
        F_spin_at_timestep = F_matrices[1][dt_idx]
        F = (F_gw_at_timestep, F_spin_at_timestep)

        Q_gw_at_timestep = Q_matrices[0][dt_idx]
        Q_spin_at_timestep = Q_matrices[1][dt_idx]
        Q_timing_at_timestep = Q_matrices[2][dt_idx]
        Q = (Q_gw_at_timestep, Q_spin_at_timestep, Q_timing_at_timestep)


        x_predict, P_predict = _predict(x, P, F, Q)
        x_new, P_new, y, S = _update(x_predict, P_predict, H, R, z)
        ll = _log_likelihood(y, S)
        
        #jax.debug.print("Step {} likelihood: {}", dt_idx, ll, ordered=True)
        #jax.debug.print("--------------------------------", ordered=True)

        
        return (x_new, P_new), ll

    # Pack inputs for scan
    # Take only first 5 timesteps of each input
    n_steps = 5 #len(data) - 1
    inputs = (jnp.arange(n_steps), 
             data[1:n_steps+1], 
             R_matrices[1:n_steps+1], 
             H_matrices[1:n_steps+1])

    # Run scan loop
    (xf, Pf), ll_arr = lax.scan(step, (x, P), inputs)
    
    total_ll = ll0 + jnp.sum(ll_arr)
    jax.debug.print("Final likelihood: {}", total_ll, ordered=True)
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

        print("The data is:", self.data)
        print("The data errors are:", self.data_errors)

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

        # Convert hellings downs matrix
        self.hellings_downs_matrix = jnp.array(self.model.hd_correlation_matrix)





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
            hellings_downs_matrix=self.hellings_downs_matrix,
            dt_array=self.jax_t_diffs,
            x0=self.jax_x0,
            P0=self.jax_P0
        ) 
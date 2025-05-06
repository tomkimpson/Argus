"""Module which implements JAX-based Kalman filter algorithm."""

import logging # Added import

import numpy as np

from argus.jmath import F_matrices_non_precomputed, Q_matrices_non_precomputed,precompute_R_matrices,compute_predicted_covariance,compute_predicted_state
from functools import partial
import jax
import jax.numpy as jnp
from jax import lax 
from jax.scipy.linalg import block_diag


# Get a logger for this module
logger = logging.getLogger(__name__)


def _log_likelihood(y: jax.Array, cov: jax.Array) -> jax.Array:
    """Calculate the log likelihood given innovation and innovation covariance.
    
    Args:
        y: Innovation term (measurement residual), shape (n,)
        cov: Innovation covariance matrix, shape (n,n)
        
    Returns
    -------
        float: Log likelihood value
    """
    sign, logdet = jnp.linalg.slogdet(2.0 * jnp.pi * cov)
    quadratic_term = y.T @ jnp.linalg.solve(cov, y)
    return -0.5 * (logdet + quadratic_term)

def _predict(x: jax.Array, P: jax.Array, F_list: tuple, Q_list: tuple, dim_x: int) -> tuple[jax.Array, jax.Array]:
    """Predict the next state and covariance.
    
    Args:
        x: Current state vector
        P: Current covariance matrix
        F_list: Tuple of state transition matrices
        Q_list: Tuple of process noise matrices
        dim_x: Dimension of the state vector
        
    Returns
    -------
        tuple: (predicted state, predicted covariance)
    """
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
    # Ensure z is a column vector
    z = z.reshape(-1, 1) # todo, remove this. I think we can adjust how we load the data to avoid this
    y = z - H @ xp              
    S = H @ Pp @ H.T + R
    Sinv = jnp.linalg.inv(S)                               
    K = Pp @ H.T @ Sinv                               
    x = xp + K @ y    
                            
 
    #Following FilterPy https://github.com/rlabbe/filterpy/blob/master/filterpy/kalman/EKF.py by using
    #Joseph form for numerically stable update of the covariance matrix
    # P = (I-KH)P(I-KH)' + KRK' which is more numerically stable
    # and works for non-optimal K vs the equation
    # P = (I-KH)P usually seen in the literature.   
    I_KH = jnp.eye(len(xp)) - K @ H
    P = I_KH @ Pp @ I_KH.T + K@R@K.T


    # Optional: enforce symmetry for numerical stability
    #P = 0.5 * (P + P.T)
    
    return x, P, y, S


def _compute_sigma_matrix(h2, γa, Γ):
    return (h2 / 12) * γa * Γ


def _initialize_kalman_filter(nx,Npsr,P_eps,σa2,γa,σp2,γp):
    """Initialize the state vector (x0) and covariance matrix (P0).

    This function sets up the initial conditions for the Kalman filter based on
    the assumed structure of the state vector and prior knowledge about the
    system noise properties (GW, spin noise, measurement noise).

    The state vector `x` is assumed to be structured block-wise:
    `x = [GW states (2*Npsr), Spin states (2*Npsr), Epsilon states (approx. 10*Npsr)]`

    Args:
        nx: Total dimension of the state vector.
        Npsr: Number of pulsars in the array.
        P_eps: Initial covariance matrix for the epsilon (measurement white noise)
               states block. Shape depends on epsilon state definition, e.g., (Npsr, Npsr).
               Represents initial uncertainty associated with terms like EFAC/EQUAD.
        h2: Squared characteristic strain amplitude (h_c^2) of the expected GW background.
            Used to calculate the stationary variance of the GW 'a' state component.
        γa: Damping constant (1 / correlation time) for the Ornstein-Uhlenbeck (OU)
            process modeling the GW 'a' state component.

    Returns
    -------
        tuple[jax.Array, jax.Array]: A tuple containing:
            - x0: Initial state vector, shape (nx, 1). Initialized to zeros, assuming
                  states represent perturbations around a known mean (or zero).
            - P0: Initial state covariance matrix, shape (nx, nx). Constructed by
                  combining covariance blocks for GW, Spin, and Epsilon states.
    """
    # Initialize the states
    x0 = jnp.zeros((nx, 1)) # Initialize as column vector. jnp.zeros(nx) # Initial state vector. δφ=0,δf=0, etc. As all the states are effecitvely perturbations, this is a reasonable guess.


    # Initialize the covariance matrices

    ## 1. The GW block "r/a"
    P_GW = jnp.zeros((Npsr * 2, Npsr * 2))


    #1.1 Set diagonal variances for 'r' states (indices 0, 2, 4, ...)
    #Set P[2n, 2n] = 1e-40 (very small initial variance)
    r_indices = jnp.arange(0, Npsr * 2, 2)
    P_GW = P_GW.at[r_indices, r_indices].set(1e-40)


    # 1.2 Set the P_aa block (indices 1, 3, 5, ...)
    # Sets P[2n+1, 2m+1] = P_aa_init[n, m]
    P_aa_init = σa2 / (2.0 * γa)
    P_GW = P_GW.at[1::2, 1::2].set(P_aa_init)

    ## 2. The spin block "phi / f "
    P_spin = jnp.zeros((Npsr * 2, Npsr * 2))

    #2.1 Set diagonal variances for 'phi' states (indices 0, 2, 4, ...)
    # Set P[2n, 2n] = 1e-40
    phi_indices = jnp.arange(0, Npsr * 2, 2)
    P_spin = P_spin.at[phi_indices, phi_indices].set(1e-40)

    # 2.2 Set diagonal variances for 'f' states (indices 1, 3, 5, ...)
    # Eq: Var(f) = sigma2_spin[n] / (2 * gamma_spin[n])
    # This is element-wise calculation resulting in a vector of length Npsr
    spin_variance_values = σp2 / (2.0 * γp)
    f_indices = jnp.arange(1, Npsr * 2, 2)
    P_spin = P_spin.at[f_indices, f_indices].set(spin_variance_values)

    P0 = block_diag(P_GW, P_spin, P_eps)

    return x0, P0

@jax.named_call
@partial(jax.jit, static_argnames=('Npsr', 'M_sum', 'dim_x','n_states'))
def _run_kalman_filter_scan(θ, data, data_errors, H_matrices, Npsr, M_sum,hellings_downs_matrix, dt_array, dim_x,n_states,P_eps):
    """Run the Kalman filter algorithm over all observations and return a log likelihood."""
    σa2 = _compute_sigma_matrix(θ.ha**2, θ.γa, hellings_downs_matrix)
    
    x0,P0 = _initialize_kalman_filter(n_states,Npsr,P_eps,σa2, θ.γa,θ.σp**2, θ.γp)

    # Precompute the R matrix for this parameter set and these data errors    
    R_matrices = precompute_R_matrices(data_errors,θ.EFAC, θ.EQUAD)


    # First update
    x, P, y, S = _update(xp=x0, Pp=P0, H=H_matrices[0,:,:], R=R_matrices[0,:,:], z=data[0])
    ll0 = _log_likelihood(y, S)

    
    def step(carry, inputs):
        x, P = carry
        dt_idx, z, R, H = inputs
        # Get dt for this step and precompute matrices just for this step
        dt = dt_array[dt_idx]
        # Compute F and Q matrices for this specific timestep only
        F_gw, F_spin = F_matrices_non_precomputed(θ.γa, θ.γp, dt, Npsr, M_sum)
        F = (F_gw, F_spin)
        
        Q_gw, Q_spin =Q_matrices_non_precomputed(θ.γa, σa2, θ.γp, θ.σp**2, dt)
        Q = (Q_gw, Q_spin)

     
        x_predict, P_predict = _predict(x, P, F, Q, dim_x)
     
        x_new, P_new, y, S = _update(x_predict, P_predict, H, R, z)
        ll = _log_likelihood(y, S)
        return (x_new, P_new), ll

    # Pack inputs for scan - iterate over first 10 timesteps
    inputs = (jnp.arange(len(data)-1), 
             data[1:], 
             R_matrices[1:], 
             H_matrices[1:])




    # Run scan loop
    (xf, Pf), ll_arr = lax.scan(step, (x, P), inputs)
    
    total_ll = ll0 + jnp.sum(ll_arr)
    return total_ll[0][0]

class JaxKalmanFilter:
    """A class to implement the linear Kalman filter on scalar inputs using JAX.

    Args:
        model: Class which defines all the Kalman machinery e.g. state transition models, covariance matrices etc.
        observations: 2D array which holds the noisy observations recorded at the detector
        x0: A 1D array which holds the initial guess of the initial states
        P0: The uncertainty in the guess of P0
    """

    def __init__(self, model, observations: np.ndarray, Peps: np.ndarray):
        """Initialize the class."""
        logger.info("Initializing JaxKalmanFilter...") # Log entry point

        self.model = model
        self.observations = observations
        self.P_eps = Peps

        # Extract the observations into separate arrays
        self.toa = self.observations[0]
        self.data = self.observations[1]
        self.data_errors = self.observations[2]

        self.t_diffs = np.diff(self.toa)

        logger.info(f"Total number of observations: {len(self.data)}")
        logger.info(f"Starting dt (days): {self.t_diffs[0]/86400}")
        logger.info(f"Ending dt (days): {self.t_diffs[-1]/86400}")
        logger.info(f"The errors at t=1 are: {self.data_errors[:,0]}")

        # Precompute the observation matrices and assign them to model.H_matrix_list
        self.Hmat = self.model.precompute_H_matrix()
    
        # Convert to JAX arrays for faster processing
        self._prepare_jax_arrays()





    def _prepare_jax_arrays(self): 
        """Convert numpy arrays to JAX arrays and verify they are 64-bit."""
        # Convert observations and related data
        self.jax_data = jnp.array(self.data)
        self.jax_data_errors = jnp.array(self.data_errors)
        #self.jax_psr_indices = jnp.array(self.psr_indices)
        self.jax_t_diffs = jnp.array(self.t_diffs)
                
        # Convert H matrices
        self.jax_H_matrices = jnp.array(self.Hmat)

        # Convert hellings downs matrix
        self.hellings_downs_matrix = jnp.array(self.model.hd_correlation_matrix)

        # Verify all floating-point arrays are 64-bit
        float_arrays = [
            ('jax_data', self.jax_data),
            ('jax_data_errors', self.jax_data_errors),
            ('jax_t_diffs', self.jax_t_diffs),
            ('jax_H_matrices', self.jax_H_matrices),
            ('hellings_downs_matrix', self.hellings_downs_matrix)
        ]
        
        for name, arr in float_arrays:
            if arr.dtype != jnp.float64:
                raise ValueError(f"{name} is {arr.dtype}, expected {jnp.float64}. The Kalman filter requires floats at standard precision for numerical stability.")


    def get_likelihood(self, θ):
        """Run the Kalman filter algorithm over all observations and return a log likelihood."""
        return _run_kalman_filter_scan(
            θ=θ,
            data=self.jax_data,
            data_errors=self.jax_data_errors,
            H_matrices=self.jax_H_matrices,
            Npsr=self.model.Npsr,
            M_sum=self.model.M_sum,
            hellings_downs_matrix=self.hellings_downs_matrix,
            dt_array=self.jax_t_diffs,
            dim_x=2*self.model.Npsr,
            n_states=self.model.nx,
            P_eps=self.P_eps
        ) 
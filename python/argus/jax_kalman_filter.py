"""Module which implements JAX-based Kalman filter algorithm."""

import numpy as np

from argus.jmath import get_Q_block,precompute_R_matrices,compute_predicted_covariance,compute_predicted_state,precompute_Q_matrices,precompute_F_matrices,get_F_block#,precompute_Q_matrices_non_vectorised,precompute_F_matrices_non_vectorised
from functools import partial
import jax
import jax.numpy as jnp
from jax import lax 

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
    jax.debug.print('log_likelihood: {x}, y: {y}, cov: {cov}', x=log_likelihood, y=y, cov=cov,ordered=True)
    return log_likelihood







def _predict(x: jax.Array, P: jax.Array, F_list: tuple, Q_list: tuple, dim_x: int,Npsr:int,M_sizes:int,psr_idx:int) -> tuple[jax.Array, jax.Array]:
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

    jax.debug.print("This is the predict step. Some useful numbers are surfaced below.",ordered=True)

    xp = compute_predicted_state(F_list, x, dim_x, dim_x)
    Pp = compute_predicted_covariance(P,F_list,Q_list,dim_x,dim_x)


    jax.debug.print("Pp max diag: {max}", max=jnp.max(jnp.diag(Pp)))

    evals = jnp.linalg.eigvalsh(Pp)
    jax.debug.print("Pp eigs: {e}", e=evals[-5:])


    idx_max = jnp.argmax(jnp.diag(Pp))
    jax.debug.print("Pp max index: {i}, value: {v}", i=idx_max, v=jnp.diag(Pp)[idx_max])

    idx_max = jnp.argmax(jnp.diag(Pp))
    jax.debug.print("Max variance at index {i}: {v}", i=idx_max, v=jnp.diag(Pp)[idx_max])



    Pp = 0.5 * (Pp + Pp.T)  # Symmetrize
   # Pp += 1e-12 * jnp.eye(Pp.shape[0])  # Regularize
    
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

    #jax.debug.print("The H matrix for this timestep is: {H}", H=H,ordered=True)
    #jax.debug.print("H row: {h}", h=H)

    #jax.debug.print("The HPH magnitude is: {H}", H=H @ Pp @ H.T,ordered=True)
    #jax.debug.print("The R matrix for this timestep is: {R}", R=R,ordered=True)
    #jax.debug.print("The Pp matrix for this timestep is: {Pp}", Pp=Pp,ordered=True)
    jax.debug.print("This is the update step.",ordered=True)

    y = z - H @ xp                                  
    S = H @ Pp @ H.T + R     
    K = Pp @ H.T / S                               
    x = xp + K * y    
    
                                 





    #Following FilterPy https://github.com/rlabbe/filterpy/blob/master/filterpy/kalman/EKF.py by using
    #Joseph form for numerically stable update of the covariance matrix
    # P = (I-KH)P(I-KH)' + KRK' which is more numerically stable
    # and works for non-optimal K vs the equation
    # P = (I-KH)P usually seen in the literature.   
    I_KH = jnp.eye(len(xp)) - K @ H
    P = I_KH @ Pp @ I_KH.T + R*(K@ K.T)
    
    return x, P, y, S


def _compute_sigma_matrix(h2, γa, Γ):
    return (h2 / 6) * γa * Γ



def create_block_sparse_transition_matrix(p_idx, block, Npsr):
    """
    Creates a block-sparse transition matrix of the form:
    F = blockdiag(I, ..., block, ..., I)
    
    Where F has dimension 2N×2N, and the 2×2 block occurs at position p_idx.
    
    Args:
        p_idx: The pulsar index where the block should be placed (0-indexed)
        block: The 2×2 block to insert
        Npsr: Total number of pulsars (N)
        
    Returns:
        A 2N×2N block diagonal matrix with the special block at position p_idx
    """
    # Create indices for the locations we want to modify
    row_idx = jnp.array([2*p_idx, 2*p_idx+1])
    col_idx = jnp.array([2*p_idx, 2*p_idx+1])
    
    # Flatten the block for scatter update
    updates = block.reshape(-1)
    
    # Create the indices for scatter update
    idx = jnp.stack([
        jnp.repeat(row_idx, 2),
        jnp.tile(col_idx, 2)
    ], axis=1)
    
    # Start with identity matrix
    matrix = jnp.eye(2 * Npsr)
    
    # Use scatter to update just the relevant part
    return matrix.at[idx[:, 0], idx[:, 1]].set(updates)


def create_block_sparse_Q_spin(p_idx, block, Npsr):
    """
    Creates a block-sparse Q_spin matrix with zeros everywhere except a single
    2×2 block at position p_idx.
    
    Args:
        p_idx: The pulsar index (0-indexed)
        block: The 2×2 block to insert
        Npsr: Total number of pulsars
        
    Returns:
        A 2N×2N matrix with the block at position p_idx
    """
    # Create indices for the locations we want to modify
    row_idx = jnp.array([2*p_idx, 2*p_idx+1])
    col_idx = jnp.array([2*p_idx, 2*p_idx+1])
    
    # Flatten the block for scatter update
    updates = block.reshape(-1)
    
    # Create the indices for scatter update
    idx = jnp.stack([
        jnp.repeat(row_idx, 2),
        jnp.tile(col_idx, 2)
    ], axis=1)
    
    # Start with zeros matrix
    matrix = jnp.zeros((2 * Npsr, 2 * Npsr))
    
    # Use scatter to update just the relevant part
    return matrix.at[idx[:, 0], idx[:, 1]].set(updates)



def precompute_Q_eps_matrices(M_sizes, σ_eps, total_dim, Npsr):
    """
    Pre-computes Q_eps matrices for all possible pulsar indices.
    
    Args:
        M_sizes: Array containing the sizes of M for each pulsar
        σ_eps: The scaling factor
        total_dim: Total dimension of the resulting matrix
        Npsr: Number of pulsars
        
    Returns:
        A stack of matrices, one for each possible pulsar index
    """
    M_sizes_arr = jnp.array(M_sizes)
    
    # Compute offsets for all pulsars at once
    cumsum = jnp.cumsum(jnp.pad(M_sizes_arr, (1, 0), 'constant')[:-1])
    
    # Function to create a single Q_eps matrix for a given pulsar index
    def create_single_Q_eps(p_idx):
        offset = cumsum[p_idx]
        block_size = M_sizes_arr[p_idx]
        
        # Create diagonal indices
        diag_indices = jnp.arange(total_dim)
        
        # Create mask for diagonal elements
        diag_mask = (diag_indices >= offset) & (diag_indices < offset + block_size)
        
        # Create matrix with diagonal set according to mask
        return jnp.diag(σ_eps * diag_mask)
    
    # Use vmap to create matrices for all pulsar indices
    return jax.vmap(create_single_Q_eps)(jnp.arange(Npsr))









@jax.named_call
@partial(jax.jit, static_argnames=('Npsr', 'M_sum', 'dim_x'))
def _run_kalman_filter_scan(θ, data, data_errors, psr_indices, H_matrices, Npsr, M_sum,M_sizes,hellings_downs_matrix, timestamps, x0, P0, dim_x):
    """Run the Kalman filter algorithm over all observations and return a log likelihood.
    
    TK: Regarding the commented out F and Q matrices, I am still undecided whether to precompute them or not.
    Computing them on the fly is more memory efficient, but precomputing them might be faster.
    We are hitting some memory issues when we try to run with NUTS and construct the AD Jacobian, so for now we precompute them.
    """
    # Limit to first 10 timesteps
    idx =  700 #int(0.5*len(data))
    data = data[:idx]
    data_errors = data_errors[:idx]
    psr_indices = psr_indices[:idx]
    H_matrices = H_matrices[:idx]
    #dt_array = dt_array[:idx-1]  # dt_array is one shorter than data
    
    σa2 = _compute_sigma_matrix(θ.ha**2, θ.γa, hellings_downs_matrix)

    all_Q_eps_matrices = precompute_Q_eps_matrices(M_sizes, θ.σeps, M_sum, Npsr)

    
    # Precompute all matrices for this parameter set
    #F_matrices = precompute_F_matrices(θ.γa, θ.γp, dt_array, Npsr, M_sum)
    #Q_matrices = precompute_Q_matrices(θ.γa,σa2, θ.γp,θ.σp**2, dt_array, Npsr, M_sum, θ.σeps)
    R_matrices = precompute_R_matrices(data_errors,θ.EFAC, θ.EQUAD, psr_indices)

    # First update
    #H = H_matrices[0]
    #x, P, y, S = _update(xp=x0, Pp=P0, H=H, R=R_matrices[0], z=data[0])
    #ll0 = _log_likelihood(y, S)

    # Initialize last_obs_time
    first_obs_time = 0.0
    last_obs_time = jnp.full((Npsr,), -1e10)

    def step(carry, inputs):
        x, P,last_obs_time = carry
        dt_idx, z, R, H = inputs


        p_idx = psr_indices[dt_idx]
        t_now = timestamps[dt_idx]  # current time delta
        t_last = last_obs_time[p_idx]
        dt = t_now - t_last


        is_first_obs = last_obs_time[p_idx] < 0
        jax.debug.print('dt_idx: {dt_idx}, pulsar: {p_idx}, dt: {dt}, is_first_obs: {is_first_obs}', dt_idx=dt_idx, p_idx=p_idx, dt=dt / (24 * 3600), is_first_obs=is_first_obs, ordered=True)


        # Build full-size sparse transition matrices for GW and spin
        # Build GW and spin transition blocks
        gw_block = get_F_block(θ.γa, dt)
        spin_block = get_F_block(θ.γp[p_idx], dt)

        # Create block-sparse matrices more efficiently
        F_gw = create_block_sparse_transition_matrix(p_idx, gw_block, Npsr)
        F_spin = create_block_sparse_transition_matrix(p_idx, spin_block, Npsr)
        
        F = (F_gw, F_spin)



        #and the Q matrix
        Q_gw_block = get_Q_block(θ.γa, dt)
        Q_gw = jnp.kron(σa2, Q_gw_block) #the full Q-matrix



        #Q_spin_block = get_Q_spin(θ.γp[p_idx], dt, θ.σp[p_idx]**2)
        Q_spin_block = θ.σp[p_idx]**2*get_Q_block(θ.γp[p_idx], dt)
        Q_spin = create_block_sparse_Q_spin(p_idx, Q_spin_block, Npsr)
    
        # For Q_eps, calculate total dimension
         # For Q_eps, calculate total dimension
        Q_eps = all_Q_eps_matrices[p_idx]

        Q = (Q_gw, Q_spin, Q_eps)



        #F = get_F_matrices_single(θ.γa, θ.γp[p_idx], dt, Npsr, M_sum, p_idx)
        #Q = get_Q_matrices_single(θ.γa, σa2, θ.γp[p_idx], θ.σp**2, dt, Npsr, M_sum, θ.σeps, p_idx)

        def do_predict(_):
            return _predict(x, P, F, Q, dim_x, Npsr, M_sizes, p_idx)

        # Apply predict only if not first observation
        x, P = lax.cond(is_first_obs, lambda _: (x, P), do_predict, operand=None)

        # Full update step (x and P)
        x, P, y, S = _update(x, P, H, R, z)

        # Log likelihood contribution
        ll = _log_likelihood(y, S)

        # Update last_obs_time
        last_obs_time = last_obs_time.at[p_idx].set(t_now)

        jax.debug.print("-------------------------------------------")

        return (x, P, last_obs_time), ll

    # Pack inputs for scan
    # inputs = (jnp.arange(len(data) - 1), 
    #          data[1:], 
    #          R_matrices[1:], 
    #          H_matrices[1:])


    inputs = (jnp.arange(len(data)), 
             data, 
             R_matrices, 
             H_matrices)




    # Run scan loop
    (xf, Pf,_), ll_arr = lax.scan(step, (x0, P0,last_obs_time), inputs)
    
    total_ll = jnp.sum(ll_arr)
    return total_ll#[0][0]

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
        #self.timestamps = self.observations[:, 0]
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
        self.jax_timestamps   = jnp.array(self.toa)
        # Convert initial state and covariance
        self.jax_x0          = jnp.array(self.x0.reshape(-1, 1))
        self.jax_P0          = jnp.array(self.P0)
        
        # Convert H matrices
        self.jax_H_matrices  = jnp.array([h for h in self.model.H_matrix_list])

        # Convert hellings downs matrix
        self.hellings_downs_matrix = jnp.array(self.model.hd_correlation_matrix)


    def get_likelihood(self, θ):
        """Run the Kalman filter algorithm over all observations and return a log likelihood."""
        # Create a modified version of x0 where elements after 4Npsr are set to thetaIC
        return _run_kalman_filter_scan(
            θ=θ,
            data=self.jax_data,
            data_errors=self.jax_data_errors,
            psr_indices=self.jax_psr_indices,
            H_matrices=self.jax_H_matrices,
            Npsr=self.model.Npsr,
            M_sum=self.model.M_sum,
            M_sizes=self.model.M_sizes,
            hellings_downs_matrix=self.hellings_downs_matrix,
            timestamps=self.jax_timestamps,
            x0=self.jax_x0,
            P0=self.jax_P0,
            dim_x=2*self.model.Npsr
        ) 

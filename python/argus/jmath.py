"""Module which implements JAX-based math operations for Kalman filtering.

This module contains three main groups of functions:
1. Basic block matrix operations (get_F_block, get_Q_block)
2. Component-specific operations (get_F_spin, get_Q_spin)
3. Full system operations (get_F, get_Q, get_Pp_blocks)
"""

import jax
import jax.numpy as jnp
from jax.scipy.linalg import block_diag
from jax import vmap
from functools import partial
from typing import Tuple
from jax import lax


def get_F_block(gamma: float, dt: float) -> jax.Array:
    """Compute 2x2 state transition block matrix for a single component.
    
    Args:
        gamma: Decay rate parameter
        dt: Time step
        
    Returns
    -------
        jax.Array: 2x2 state transition matrix
    """
    exp_term = jnp.exp(-gamma * dt)
    return jnp.array([[1.0, (1-exp_term)/gamma],
                     [0.0, exp_term]])

# def get_Q_block(γ: float, dt: float) -> jax.Array:
#     """Compute Q block matrix using JAX.
    
#     Note: For very small γ or dt values, exponential terms may need 
#     special handling to maintain numerical stability.
#     """
#     exp_term = jnp.exp(-γ * dt)
#     exp_2term = jnp.exp(-2 * γ * dt)

#     q11 = (dt - 2 * (1 - exp_term) / γ + (1 - exp_2term) / (2 * γ)) / γ**3
#     q12 = ((1 - exp_term) - (1 - exp_2term) / 2) / (γ**2)
#     q22 = (1 - exp_2term) / (2 * γ)

#     return jnp.array([[q11, q12], [q12, q22]])



def get_Q_block(γ: float, dt: float) -> jnp.ndarray:
    """Compute Q block matrix for the Ornstein–Uhlenbeck process.

    Uses numerically stable formulas and switches to Taylor approximation for small γ*dt.
    """
    γdt = γ * dt

    def stable_case(args):
        γ, dt, γdt = args
        e1 = -jnp.expm1(-γdt)
        e2 = -jnp.expm1(-2 * γdt)

        q11 = (dt - 2 * e1 / γ + e2 / (2 * γ)) / γ**3
        q12 = (e1 - e2 / 2) / γ**2
        q22 = e2 / (2 * γ)

        return jnp.array([[q11, q12], [q12, q22]])

    def taylor_case(args):
        γ, dt, γdt = args
        # These are the moments of the integrated Wiener process
        q11 = dt**3 / 3
        q12 = dt**2 / 2
        q22 = dt

        return jnp.array([[q11, q12], [q12, q22]])






    return lax.cond(
        γdt < 1e-3,
        taylor_case,
        stable_case,
        operand=(γ, dt, γdt)
    )










def get_F_spin(gamma: jax.Array, dt: float) -> jax.Array:
    """Compute block diagonal state transition matrix for spin noise.
    
    Args:
        gamma: Array of decay rates for each component
        dt: Time step
        
    Returns
    -------
        jax.Array: Block diagonal matrix composed of 2x2 blocks
    """
    res = vmap(lambda x: get_F_block(x, dt))(gamma)
    return block_diag(*res)

@partial(jax.jit, static_argnums=(3,4))
def get_F(gamma, gamma_spin, dt, Npsr, M_sum):
    """Get transition matrices using JAX."""
    F_gw_block = get_F_block(gamma, dt)
    F_gw = jnp.kron(jnp.eye(Npsr), F_gw_block)
    F_spin = get_F_spin(gamma_spin, dt)
    return F_gw, F_spin


def get_Q_matrices_single_blocks(gamma_gw, sigma_matrix, gamma_spin, sigma_spin2, dt, Npsr, M_sum, sigma_eps, p_idx, M_sizes):
    """Constructs a full-size Q matrix with nonzero blocks for the observed pulsar only."""
    nx = 4 * Npsr + M_sum
    Q = jnp.zeros((nx, nx))

    # GW block (size 2x2)
    Q_gw_block = get_Q_block(gamma_gw, dt)
    Q_gw = sigma_matrix[p_idx, p_idx] * Q_gw_block
    i0 = 2 * p_idx
    i1 = i0 + 2
    Q = Q.at[i0:i1, i0:i1].set(Q_gw)

    # Spin block (2x2)
    Q_spin = sigma_spin2 * get_Q_block(gamma_spin, dt)
    s0 = 2 * Npsr + 2 * p_idx
    s1 = s0 + 2
    Q = Q.at[s0:s1, s0:s1].set(Q_spin)

    # Epsilon block (variable size)
    eps_offset = jnp.sum(jnp.array(M_sizes[:p_idx]))
    e0 = 4 * Npsr + eps_offset
    e1 = e0 + M_sizes[p_idx]
    Q_eps = jnp.eye(M_sizes[p_idx]) * sigma_eps
    Q = Q.at[e0:e1, e0:e1].set(Q_eps)

    return Q_gw, Q_spin, Q_eps



def get_Q_spin(gamma, dt,sigma_p):
    """Compute Q spin matrix using JAX."""
    res = vmap(lambda g, s: get_Q_block(g, dt) * s)(gamma, sigma_p)
    return block_diag(*res)

@partial(jax.jit, static_argnums=(5,6))
def get_Q(gamma,σa2, gamma_spin,σp2, dt, Npsr, M_sum, eps):
    """Get process noise matrices using JAX."""
    Q_gw_block = get_Q_block(gamma, dt)
    Q_gw = jnp.kron(σa2, Q_gw_block)
    Q_spin = get_Q_spin(gamma_spin, dt, σp2)
    Q_timing = jnp.eye(M_sum) * eps**2 #assume these are known exactly. We will tune the initial condition
    return Q_gw, Q_spin, Q_timing

@partial(jax.jit, static_argnums=(2,3))
def compute_predicted_state(F_list, x, gw_size, spin_size):
    """Compute the predicted state vector by applying transition matrices to state blocks.
    
    Args:
        F_list: Tuple of (F_gw, F_spin) transition matrices for GW and spin components
        x: Current state vector containing GW, spin and timing components
        gw_size: Size of gravitational wave state block
        spin_size: Size of spin state block
        
    Returns
    -------
        jax.Array: Predicted state vector with same structure as input, computed by:
            - Applying F_gw transition to GW states
            - Applying F_spin transition to spin states  
            - Keeping timing states unchanged
            
    Note:
        The state vector x is assumed to have structure [x_gw, x_spin, x_timing]
        where each component has size determined by gw_size and spin_size parameters.
    """
    F_gw, F_spin = F_list
    x_gw = x[:gw_size]
    x_spin = x[gw_size:gw_size+spin_size]
    x_timing = x[gw_size+spin_size:]
    
    return jnp.vstack([F_gw@x_gw, F_spin@x_spin, x_timing])

@partial(jax.jit, static_argnums=(3,4))
def compute_predicted_covariance(P: jax.Array,
                               F_list: Tuple[jax.Array, jax.Array],
                               Q_list: Tuple[jax.Array, ...],
                               gw_size: int,
                               spin_size: int) -> jax.Array:
    """Compute predicted covariance matrix in one operation.
    
    Args:
        P: Full covariance matrix
        F_list: Tuple of (F_gw, F_spin) transition matrices
        Q_list: Tuple of (Q_gw, Q_spin, Q_timing) process noise matrices
        gw_size: Size of GW block
        spin_size: Size of spin block
        
    Returns
    -------
        jax.Array: Combined predicted covariance matrix
        
    Note:
        Computing the predicted covariance by slicing the matrix into blocks and doing
        individual matrix products is significantly faster than doing the full matrix
        multiplication FPF^T + Q. This is because the block structure allows us to avoid
        many unnecessary multiplications with zero elements.
    """
    F1, F2 = F_list
    Q1, Q2, Q3 = Q_list


    #jax.debug.print("Qgw block values: q11={q11}, q22={q22}", q11=Q1[0,0], q22=Q1[1,1])
    #jax.debug.print("Qspin block values: q11={q11}, q22={q22}", q11=Q2[0,0], q22=Q2[1,1])
    
    # Extract blocks directly from P
    P1 = P[:gw_size, :gw_size]
    P2 = P[gw_size:gw_size+spin_size, gw_size:gw_size+spin_size]
    P3 = P[gw_size+spin_size:, gw_size+spin_size:]
    P4 = P[:gw_size, gw_size:gw_size+spin_size]
    P5 = P[gw_size:gw_size+spin_size, gw_size+spin_size:]
    P6 = P[:gw_size, gw_size+spin_size:]
    
    # Compute individual blocks
    PF1 = F1 @ P1 @ F1.T + Q1
    PF2 = F2 @ P2 @ F2.T + Q2
    PF4 = F1 @ P4 @ F2.T
    PF5 = F2 @ P5
    PF6 = F1 @ P6

    # Assemble full matrix
    return jnp.block([[PF1,   PF4,   PF6],
                     [PF4.T,  PF2,   PF5],
                     [PF6.T,  PF5.T, P3 + Q3]])

@partial(jax.jit, static_argnums=(3, 4))
def precompute_F_matrices(gamma_a: float, 
                         gamma_p: jax.Array, 
                         dt_array: jax.Array, 
                         Npsr: int, 
                         M_sum: int) -> Tuple[jax.Array, jax.Array]:
    """Precompute all F matrices for a given parameter set and sequence of dt values.
    
    This computes F matrices for all time steps at once with JAX vectorization.
    
    Args:
        gamma_a: GWB parameter
        gamma_p: Pulsar-specific parameters, shape (n_components,)
        dt_array: Time differences between observations, shape (n_timesteps,)
        Npsr: Number of pulsars
        M_sum: Sum of model components
        
    Returns
    -------
        tuple: (F_gw_matrices, F_spin_matrices) containing matrices for all timesteps
               F_gw_matrices.shape = (n_timesteps, n_gw, n_gw)
               F_spin_matrices.shape = (n_timesteps, n_spin, n_spin)
    """
    def get_F_for_dt(dt):
        F_gw, F_spin = get_F(gamma_a, gamma_p, dt, Npsr, M_sum)
        return F_gw, F_spin
    
    return jax.vmap(get_F_for_dt)(dt_array)


@partial(jax.jit, static_argnums=(5, 6))
def precompute_Q_matrices(gamma_a, σa2, gamma_p,σp2, dt_array, Npsr, M_sum, eps):
    """Precompute all Q matrices for a given parameter set and sequence of dt values.
    
    Args:
        gamma_a: float, GWB parameter
        gamma_p: array, pulsar-specific parameters
        dt_array: array of time differences between observations
        Npsr: int, number of pulsars
        M_sum: int, sum of model components
        eps: float, timing parameter
        
    Returns
    -------
        tuple: (Q_gw_matrices, Q_spin_matrices, Q_timing_matrices) where each element 
               is a JAX array containing matrices for all timesteps
    """
    def get_Q_for_dt(dt):
        Q_gw, Q_spin, Q_timing = get_Q(gamma_a,σa2, gamma_p,σp2, dt, Npsr, M_sum, eps)
        return Q_gw, Q_spin, Q_timing
    
    return jax.vmap(get_Q_for_dt)(dt_array)





@partial(jax.jit, static_argnums=(3, 4))
def precompute_F_matrices_non_vectorised(gamma_a: float, 
                         gamma_p: jax.Array, 
                         dt_array: jax.Array, 
                         Npsr: int, 
                         M_sum: int) -> Tuple[jax.Array, jax.Array]:
    """Precompute F matrices for a single timestep only.
    
    Instead of vectorizing over all timesteps, this function only handles one timestep.
    
    Args:
        gamma_a: GWB parameter
        gamma_p: Pulsar-specific parameters, shape (n_components,)
        dt_array: Time difference for ONE observation, scalar
        Npsr: Number of pulsars
        M_sum: Sum of model components
        
    Returns
    -------
        tuple: (F_gw_matrix, F_spin_matrix) for the single timestep
    """
    # Get F matrices for a single dt (not vectorized)
    F_gw, F_spin = get_F(gamma_a, gamma_p, dt_array, Npsr, M_sum)
    return F_gw, F_spin



@partial(jax.jit, static_argnums=(5, 6))
def precompute_Q_matrices_non_vectorised(gamma_a, σa2, gamma_p, σp2, dt_array, Npsr, M_sum, eps):
    """Precompute Q matrices for a single timestep only.
    
    Args:
        gamma_a: float, GWB parameter
        σa2: array, GW covariance matrix 
        gamma_p: array, pulsar-specific parameters
        σp2: array, pulsar-specific noise parameters
        dt_array: scalar, time difference for ONE observation
        Npsr: int, number of pulsars
        M_sum: int, sum of model components
        eps: float, timing parameter
        
    Returns
    -------
        tuple: (Q_gw_matrix, Q_spin_matrix, Q_timing_matrix) for the single timestep
    """
    # Get Q matrices for a single dt (not vectorized)
    Q_gw, Q_spin, Q_timing = get_Q(gamma_a, σa2, gamma_p, σp2, dt_array, Npsr, M_sum, eps)
    return Q_gw, Q_spin, Q_timing





@jax.jit
def precompute_R_matrices(σ: jax.Array, EFAC: jax.Array, EQUAD: jax.Array, psr_indices: int) -> jax.Array:
    """Build the measurement-noise covariance matrix R for the pulsars observed at a given epoch.

    For pulsar n, the measurement noise variance is (σt[n])².
    Currently, this method returns a scalar
    or a per-pulsar value.
    """
    return jnp.square(σ* EFAC[psr_indices]) + jnp.square(EQUAD[psr_indices])






############## Scratch space 
# def get_Q_block(gamma: float, dt: float, eps: float = 1e-3) -> jnp.ndarray:
#     """
#     Compute a numerically stable 2x2 process noise covariance block Q
#     for an Ornstein–Uhlenbeck process with decay rate gamma and time step dt.

#     Uses series expansion for small gamma*dt to avoid numerical instability.

#     Parameters
#     ----------
#     gamma : float
#         Decay rate parameter.
#     dt : float
#         Time step.
#     eps : float
#         Threshold for switching between Taylor expansion and exponentials.

#     Returns
#     -------
#     jnp.ndarray
#         2x2 process noise covariance matrix.
#     """
#     γdt = gamma * dt
#     use_series = γdt < eps

#     # Series expansions
#     exp1_series = 1 - γdt + 0.5 * γdt**2 - (1/6) * γdt**3
#     exp2_series = 1 - 2*γdt + 2 * γdt**2 - (4/3) * γdt**3

#     exp1 = jnp.where(use_series, exp1_series, jnp.exp(-γdt))
#     exp2 = jnp.where(use_series, exp2_series, jnp.exp(-2 * γdt))

#     # Guard against zero division
#     gamma_safe = jnp.maximum(gamma, 1e-12)

#     q11 = (dt - 2 * (1 - exp1) / gamma_safe + (1 - exp2) / (2 * gamma_safe)) / gamma_safe**3
#     q12 = ((1 - exp1) - (1 - exp2) / 2) / gamma_safe**2
#     q22 = (1 - exp2) / (2 * gamma_safe)

#     Q = jnp.array([[q11, q12], [q12, q22]])

#     # Regularize and symmetrize to ensure positive-definiteness
#     Q = 0.5 * (Q + Q.T)
#     Q += 1e-12 * jnp.eye(2)

#     return Q








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


def get_F_block(γ: float, dt: float) -> jax.Array:
    """Compute 2x2 state transition block matrix for a single component.
    
    Uses expm1 for improved numerical stability when γ*dt is small.
    Assumes γ != 0 based on prior constraints.

    Args:
        γ: Decay rate parameter (non-zero)
        dt: Time step

    Returns
    -------
        jax.Array: 2x2 state transition matrix
    """
    neg_gamma_dt = -γ * dt
    exp_term = jnp.exp(neg_gamma_dt) 

    # Calculate (1 - exp(-gamma*dt)) / gamma = - (exp(-gamma*dt) - 1) / gamma
    F12 = -jnp.expm1(neg_gamma_dt) / γ 

    return jnp.array([[1.0, F12],
                     [0.0, exp_term]])

def get_Q_block(γ: float, dt: float) -> jax.Array:
    """Compute Q block matrix using JAX.

    Uses expm1 for improved numerical stability when gamma*dt is small.
    Assumes gamma != 0 based on prior constraints.

    Note: For very small γ or dt values, exponential terms may need 
    special handling to maintain numerical stability.
    """
    neg_gamma_dt = -γ * dt
    neg_2gamma_dt = -2 * γ * dt

    # Using expm1: (1 - exp(x)) = -expm1(x)
    one_minus_exp_term = -jnp.expm1(neg_gamma_dt)
    one_minus_exp_2term = -jnp.expm1(neg_2gamma_dt)

    # Calculate terms assuming gamma != 0
    q11 = (dt - 2 * one_minus_exp_term / γ + one_minus_exp_2term / (2 * γ)) / γ**3
    q12 = (one_minus_exp_term - one_minus_exp_2term / 2) / (γ**2)
    q22 = one_minus_exp_2term / (2 * γ)

    return jnp.array([[q11, q12], [q12, q22]])

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

def get_Q_spin(gamma, dt,sigma_p):
    """Compute Q spin matrix using JAX."""
    res = vmap(lambda g, s: get_Q_block(g, dt) * s)(gamma, sigma_p)
    return block_diag(*res)

@partial(jax.jit, static_argnums=(3,4))
def get_F(gamma, gamma_spin, dt, Npsr, M_sum):
    """Get transition matrices using JAX."""
    F_gw_block = get_F_block(gamma, dt)
    F_gw = jnp.kron(jnp.eye(Npsr), F_gw_block)
    F_spin = get_F_spin(gamma_spin, dt)
    return F_gw, F_spin

@jax.jit
def get_Q(gamma,σa2, gamma_spin,σp2, dt):
    """Get process noise matrices using JAX."""
    Q_gw_block = get_Q_block(gamma, dt)
    Q_gw = jnp.kron(σa2, Q_gw_block)
    Q_spin = get_Q_spin(gamma_spin, dt, σp2)
    return Q_gw, Q_spin

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
    Q1, Q2 = Q_list
    
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
                     [PF6.T,  PF5.T, P3]])

@jax.jit
def precompute_R_matrices(σ: jax.Array, EFAC: jax.Array, EQUAD: jax.Array) -> jax.Array:
    """Build the measurement-noise covariance matrix R for the pulsars observed at a given epoch.

    For pulsar n, the measurement noise variance is (σt[n])².
    Currently, this method returns a scalar
    or a per-pulsar value.
    """
   # Calculate all diagonal elements for all observations using broadcasting
    diagonals = jnp.square(EFAC * σ ) + jnp.square(EQUAD) # Shape: (Nobs, ny)
    R = jax.vmap(jnp.diag)(diagonals)
    #jax.debug.print('R.shape: {shape}',shape=R.shape,ordered=True)
    return R

# @partial(jax.jit, static_argnums=(3, 4))
# def F_matrices_non_precomputed(gamma_a: float, 
#                          gamma_p: jax.Array, 
#                          dt_array: jax.Array, 
#                          Npsr: int, 
#                          M_sum: int) -> Tuple[jax.Array, jax.Array]:
#     """Precompute F matrices for a single timestep only.
    
#     Instead of vectorizing over all timesteps, this function only handles one timestep.
    
#     Args:
#         gamma_a: GWB parameter
#         gamma_p: Pulsar-specific parameters, shape (n_components,)
#         dt_array: Time difference for ONE observation, scalar
#         Npsr: Number of pulsars
#         M_sum: Sum of model components
        
#     Returns
#     -------
#         tuple: (F_gw_matrix, F_spin_matrix) for the single timestep
#     """
#     # Get F matrices for a single dt (not vectorized)
#     F_gw, F_spin = get_F(gamma_a, gamma_p, dt_array, Npsr, M_sum)
#     return F_gw, F_spin

# @jax.jit
# def Q_matrices_non_precomputed(gamma_a, σa2, gamma_p, σp2, dt_array):
#     """Precompute Q matrices for a single timestep only.
    
#     Args:
#         gamma_a: float, GWB parameter
#         σa2: array, GW covariance matrix 
#         gamma_p: array, pulsar-specific parameters
#         σp2: array, pulsar-specific noise parameters
#         dt_array: scalar, time difference for ONE observation
#         Npsr: int, number of pulsars
#         M_sum: int, sum of model components
#         eps: float, timing parameter
        
#     Returns
#     -------
#         tuple: (Q_gw_matrix, Q_spin_matrix, Q_timing_matrix) for the single timestep
#     """
#     # Get Q matrices for a single dt (not vectorized)
#     Q_gw, Q_spin = get_Q(gamma_a, σa2, gamma_p, σp2, dt_array)
#     return Q_gw, Q_spin









### SCRATCH SPACE




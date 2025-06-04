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
import numpy as np


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

def get_Q_spin(gamma, dt, sigma_p):
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

def compute_H_matrix_for_step(time_step_index: int,
                            Npsr: int,
                            nx: int,
                            M_start_indices: np.ndarray,
                            pulsar_design_matrices: list,
                            use_gw: bool,
                            f0: np.ndarray) -> np.ndarray:
    """
    Compute the observation matrix H for the current time step using NumPy.

    This matrix maps the full state vector to the vector of observations
    from all pulsars at the given time step.

    Parameters
    ----------
    time_step_index : int
        The index corresponding to the current observation time step.
    Npsr : int
        Number of pulsars
    nx : int
        Size of the full state vector
    M_start_indices : np.ndarray
        Array containing the starting indices for each pulsar's timing model parameters
    pulsar_design_matrices : list
        List of design matrices for each pulsar
    use_gw : bool
        Whether gravitational wave terms should be included
    f0 : np.ndarray
        Array of pulsar frequencies

    Returns
    -------
    np.ndarray
        Observation matrix H of shape (Npsr, nx) for the current step.
    """
    # Initialize the H matrix with zeros using NumPy
    H = np.zeros((Npsr, nx))

    # Loop over each pulsar to build the corresponding row of H
    for psr_idx in range(Npsr):
        # Indices in the state vector 'x' relevant to this pulsar (psr_idx)
        redshift_idx = 2 * psr_idx
        spin_idx = Npsr * 2 + 2 * psr_idx
        tm_start_idx = M_start_indices[psr_idx]
        tm_end_idx = M_start_indices[psr_idx + 1]

        # Get the relevant row from this pulsar's precomputed design matrix
        design_row = pulsar_design_matrices[psr_idx][time_step_index, :]

        # Update Redshift term coefficient (-1.0) only if use_gw is True
        if use_gw:
            H[psr_idx, redshift_idx] = -1.0

        # Update Spin noise term coefficient (1 / f0_n)
        H[psr_idx, spin_idx] = 1.0 / f0[psr_idx]

        # Update Timing model term coefficients (design matrix row)
        H[psr_idx, tm_start_idx:tm_end_idx] = design_row

    return H

def precompute_H_matrix(Npsr: int,
                       nx: int,
                       M_start_indices: np.ndarray,
                       pulsar_design_matrices: list,
                       use_gw: bool,
                       f0: np.ndarray) -> np.ndarray:
    """
    Compute the observation matrix H for all time steps using NumPy.

    This iterates through all time steps, calling compute_H_matrix_for_step
    for each one, using the appropriate row from the time-varying design matrices.

    Parameters
    ----------
    Npsr : int
        Number of pulsars
    nx : int
        Size of the full state vector
    M_start_indices : np.ndarray
        Array containing the starting indices for each pulsar's timing model parameters
    pulsar_design_matrices : list
        List of design matrices for each pulsar
    use_gw : bool
        Whether gravitational wave terms should be included
    f0 : np.ndarray
        Array of pulsar frequencies

    Returns
    -------
    np.ndarray
        A NumPy array of shape (num_time_steps, Npsr, nx) containing all H matrices.
    """
    if not pulsar_design_matrices or Npsr == 0:
        print("Warning: No pulsars or design matrices found. Returning empty list.")
        return []

    try:
        num_time_steps = pulsar_design_matrices[0].shape[0]
    except (IndexError, AttributeError):
        raise ValueError("Cannot determine number of time steps. "
                        "Ensure pulsar_design_matrices is a list of NumPy arrays "
                        "with at least one element.")

    # Optional consistency check
    for i in range(1, Npsr):
        if pulsar_design_matrices[i].shape[0] != num_time_steps:
            raise ValueError(f"Inconsistent number of time steps found (Pulsar 0: {num_time_steps}, Pulsar {i}: {pulsar_design_matrices[i].shape[0]})")


    all_H = []
    for t_idx in range(num_time_steps):
        H_step = compute_H_matrix_for_step(t_idx, Npsr, nx, M_start_indices,
                                         pulsar_design_matrices, use_gw, f0)
        all_H.append(H_step)

    return np.stack(all_H, axis=0) 
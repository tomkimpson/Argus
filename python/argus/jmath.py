import jax
import jax.numpy as jnp
from jax.scipy.linalg import block_diag
from jax import vmap
from functools import partial
from line_profiler import profile


#### F-matrix computations ####

def get_F_block(gamma, dt):
    exp_term = jnp.exp(-gamma * dt)
    return jnp.array([[1.0, (1-exp_term)/gamma],
                      [0.0, exp_term]])

def get_F_spin(gamma, dt):
    res = vmap(lambda x: get_F_block(x, dt))(gamma)
    return block_diag(*res)

#@partial(jax.jit, static_argnums=(3)) # static_argnums=(3, 4) tells JAX that the 4th and 5th arguments of get_F (Npsr and M_sum) are static
def get_F(gamma, gamma_spin, dt, nx):

    Npsr = len(gamma_spin)
    M_sum = nx - (4*Npsr) 

    F_gw_block = get_F_block(gamma, dt)
    F_gw       = jnp.kron(jnp.eye(Npsr), F_gw_block)
    F_spin     = get_F_spin(gamma_spin, dt)
    F_timing   = jnp.eye(M_sum)
    return block_diag(F_gw, F_spin, F_timing)



# Pre-compute F matrices for each dt in your dataset
@partial(jax.jit, static_argnums=(3,))
def precompute_F_matrices(gamma_a, gamma_p, dt_array, nx):
    """Precompute all F matrices for a given parameter set and sequence of dt values.
    
    This computes F matrices for all time steps at once with a given parameter set.
    """
    return jax.vmap(lambda dt: get_F(gamma_a, gamma_p, dt, nx))(dt_array)









#### Q-matrix computations ####
def get_Q_block(γ, dt):
    exp_term = jnp.exp(-γ * dt)
    exp_2term = jnp.exp(-2 * γ * dt)

    q11 = (dt - 2 * (1 - exp_term) / γ + (1 - exp_2term) / (2 * γ)) / γ**3
    q12 = ((1 - exp_term) - (1 - exp_2term) / 2) / (γ**2)
    q22 = (1 - exp_2term) / (2 * γ)

    return jnp.array([[q11, q12], [q12, q22]])

def get_Q_spin(gamma, dt):
    res = vmap(lambda x: get_Q_block(x, dt))(gamma)
    return block_diag(*res)

@partial(jax.jit, static_argnums=(3,))
def get_Q(gamma, gamma_spin, dt, nx, eps):
    
    Npsr = len(gamma_spin)
    M_sum = nx - (4*Npsr) 

    Q_gw_block = get_Q_block(gamma, dt)
    Q_gw = jnp.kron(jnp.eye(Npsr), Q_gw_block)
    Q_spin = get_Q_spin(gamma_spin, dt)
    Q_timing = jnp.eye(M_sum) * eps**2
    
    return block_diag(Q_gw, Q_spin, Q_timing)


@partial(jax.jit, static_argnums=(3,))
def precompute_Q_matrices(gamma_a, gamma_p, dt_array, nx, sigma_eps):
    return jax.vmap(lambda dt: get_Q(gamma_a, gamma_p, dt, nx, sigma_eps))(dt_array)

# @partial(jax.jit, static_argnums=(2,3))
# def get_xp(F_list, x, gw_size, spin_size):
#     F_gw, F_spin = F_list
#     x_gw = x[:gw_size]
#     x_spin = x[gw_size:gw_size+spin_size]
#     x_timing = x[gw_size+spin_size:]
    
#     return jnp.vstack([F_gw@x_gw, F_spin@x_spin, x_timing])

# @partial(jax.jit, static_argnums=(1,2))
# def get_P_blocks(P, gw_size, spin_size):
#     P1 = P[:gw_size, :gw_size]
#     P2 = P[gw_size:gw_size+spin_size, gw_size:gw_size+spin_size]
#     P3 = P[gw_size+spin_size:, gw_size+spin_size:]
#     P4 = P[:gw_size, gw_size:gw_size+spin_size]
#     P5 = P[gw_size:gw_size+spin_size, gw_size+spin_size:]
#     P6 = P[:gw_size, gw_size+spin_size:]

#     return P1, P2, P3, P4, P5, P6

# @jax.jit
# def get_Pp_blocks(list_A, list_B, list_C):
#     F1, F2 = list_A
#     P1, P2, P3, P4, P5, P6 = list_B
#     Q1, Q2, Q3 = list_C
#     # breakpoint()
#     PF1 = F1@P1@F1.T + Q1
#     PF2 = F2@P2@F2.T + Q2
#     PF4 = F1@P4@F1.T
#     PF5 = F2@P5
#     PF6 = F1@P6
    
#     block_row1 = jnp.hstack([PF1, PF4, PF6])
#     block_row2 = jnp.hstack([PF4.T, PF2, PF5])
#     block_row3 = jnp.hstack([PF6.T, PF5.T, P3 + Q3])

#     return jnp.vstack([block_row1, block_row2, block_row3])

# @partial(jax.jit, static_argnums=(3,))
# def get_H(psr_index, M_cumsum, Npsr, M, f0_i):
#     """Construct measurement matrix H for the current update step.
    
#     Parameters
#     ----------
#     psr_index : int
#         Index of the pulsar being updated
#     M_cumsum : array_like
#         Cumulative sum of model parameters for each pulsar
#     Npsr : int
#         Number of pulsars
#     M : array_like
#         Current row of design matrix for this pulsar
#     f0_i : float
#         Frequency of current pulsar
    
#     Returns
#     -------
#     H : array_like
#         Measurement matrix for current update
#     """
#     # Get total state vector size (2*Npsr + sum(M_sizes))
#     total_size = 2*Npsr + M_cumsum[-1]
    
#     # Initialize H vector with zeros
#     H = jnp.zeros(total_size)
    
#     # Set GW component (-r term)
#     H = H.at[2*psr_index].set(-1.0)
    
#     # Set spin component (δφ/f0 term)
#     H = H.at[2*psr_index + 1].set(1.0/f0_i)
    
#     # Set epsilon components (M @ δε term)
#     eps_start = 2*Npsr + M_cumsum[psr_index]
#     eps_end = 2*Npsr + M_cumsum[psr_index + 1]
#     H = H.at[eps_start:eps_end].set(M)
    
#     return H


import numpy as np

def precompute_all_H(num_measurements, dim_x, num_pulsars, len_epsilon, f0, M, obs_sequence):
    """
    Precompute the measurement matrix H for pulsar timing observations.
    
    Parameters:
    -----------
    num_measurements : int
        Number of total observations across all pulsars.
    dim_x : int
        Total dimension of the state vector.
    num_pulsars : int
        Number of pulsars in the model.
    len_epsilon : list
        List containing the dimension of epsilon for each pulsar.
    f0 : list
        List containing the F0 frequency for each pulsar.
    M : list of lists
        Matrix coefficients for each pulsar and observation.
    obs_sequence : list
        Sequence of pulsar IDs for each observation.
        
    Returns:
    --------
    H_all : numpy.ndarray
        Complete measurement matrix with shape (num_measurements, dim_x).
    """
    # Compute offsets and dimensions for each pulsar
    pulsar_offsets = {}
    pulsar_dims = {}

    current_offset = 0
    for p in range(num_pulsars):
        # Dimension of epsilon for this pulsar
        L_p = len_epsilon[p]
        
        # Store the offset for this pulsar's parameters
        pulsar_offsets[p] = current_offset
        
        # The dimension for this pulsar is 4 + L_p
        pulsar_dims[p] = 4 + L_p
        
        # Update offset for next pulsar
        current_offset += pulsar_dims[p]
    
    # Verify that current_offset matches dim_x
    if current_offset != dim_x:
        raise ValueError(f"Computed dimension {current_offset} does not match input dim_x {dim_x}")

    # Track how many times each pulsar has been observed and collect rows
    pulsar_counts = {i: 0 for i in range(num_pulsars)}
    result_rows = []
    
    for pulsar_id in obs_sequence:
        count = pulsar_counts[pulsar_id]
        if count >= len(M[pulsar_id]):
            raise ValueError(f"Not enough M matrices for pulsar {pulsar_id}")
        row = M[pulsar_id][count]
        result_rows.append(row)
        pulsar_counts[pulsar_id] += 1
    
    # Verify we have correct number of rows
    if len(result_rows) != num_measurements:
        raise ValueError(f"Expected {num_measurements} rows, got {len(result_rows)}")

    # Parameter indices
    INDEX_R = 0
    INDEX_A = 1  # Not used in this function but kept for clarity
    INDEX_PHI = 2
    INDEX_F = 3  # Not used in this function but kept for clarity
    INDEX_EPS = 4

    # Prepare the measurement matrix
    H_all = np.zeros((num_measurements, dim_x))

    for i in range(num_measurements):
        p = obs_sequence[i]
        
        # The offset block for this pulsar
        offset = pulsar_offsets[p]
        
        # Set values for r and phi
        H_all[i, offset + INDEX_R] = -1
        H_all[i, offset + INDEX_PHI] = f0[p]
        
        # Set values for epsilon
        M_vector = result_rows[i]
        H_all[i, offset + INDEX_EPS : offset + INDEX_EPS + len(M_vector)] = M_vector

    return H_all
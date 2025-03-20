"""Module which implements Kalman filter algorithm using functional programming paradigm."""

import numpy as np
from tqdm import tqdm
from line_profiler import profile
import jax.numpy as jnp
from jax import jit
from argus.jmath import get_F, get_Q

def get_ith_pair(x, i):
    """Return the ith entries of x."""
    idx = 2 * i
    return x[idx], x[idx + 1]

def get_ith_vector(x, M_cumsum, i):
    """Return the ith vector slice based on cumulative sum boundaries."""
    return x[M_cumsum[i]:M_cumsum[i + 1]]

def log_likelihood(y, cov):
    """Calculate log likelihood given innovation and innovation covariance."""
    return -0.5 * (np.log(2.0 * np.pi * cov) + (y * y) / cov)

def predict(dt, θ, x_list, P_list, Npsr, M_sum):
    """Predict next state and covariance."""
    x_gw, x_spin, x_eps = x_list
    P_gw, P_spin, P_eps, P_gw_spin, P_gw_eps, P_spin_eps = P_list

    # Define the F matrices for each block
    F_gw, F_spin, F_eps = get_F(θ["γa"], θ["γp"], dt, Npsr, M_sum)

    # Predict the next state
    x_gw_predict   = F_gw   @ x_gw
    x_spin_predict = F_spin @ x_spin
    x_eps_predict  = F_eps  @ x_eps

    # Define the Q matrices for each block
    Q_gw, Q_spin, Q_eps = get_Q(θ["γa"], θ["γp"], dt, Npsr, M_sum, θ["σeps"])

    # Predict the next covariance
    ## auto covariance terms
    P_gw_predict   = F_gw   @ P_gw   @ F_gw.T   + Q_gw
    P_spin_predict = F_spin @ P_spin @ F_spin.T + Q_spin
    P_eps_predict  = F_eps  @ P_eps  @ F_eps.T  + Q_eps

    ## cross covariance terms
    P_gw_spin_predict  = F_gw   @ P_gw_spin @ F_spin.T
    P_gw_eps_predict   = F_gw   @ P_gw_eps  @ F_eps.T
    P_spin_eps_predict = F_spin @ P_spin_eps @ F_eps.T

    x_list_predict = [x_gw_predict, x_spin_predict, x_eps_predict]
    P_list_predict = [P_gw_predict, P_spin_predict, P_eps_predict,
                     P_gw_spin_predict, P_gw_eps_predict, P_spin_eps_predict]

    return x_list_predict, P_list_predict

def update(psr_index, θ, design_matrix_counter, pulsar_design_matrices, M_cumsum, f0,
          x_list, P_list, R, y):
    """Perform one Kalman update for a single scalar measurement."""
    x_gw, x_spin, x_eps = x_list
    P_gw, P_spin, P_eps, P_gw_spin, P_gw_eps, P_spin_eps = P_list

    P_spin_gw  = P_gw_spin.T
    P_eps_gw   = P_gw_eps.T
    P_eps_spin = P_spin_eps.T

    row_idx    = design_matrix_counter[psr_index]
    M          = pulsar_design_matrices[psr_index][row_idx,:]

    # Innovation calculation
    r   = get_ith_pair(x_gw, psr_index)[0]
    δφ  = get_ith_pair(x_spin, psr_index)[0]
    δε  = get_ith_vector(x_eps, M_cumsum, psr_index)
    nu  = y - (-r + δφ/f0[psr_index] + M @ δε)

    # Calculate update vectors
    u_gw   = -P_gw[:,psr_index]      + P_gw_spin[:,psr_index]/f0[psr_index]  + P_gw_eps[:, M_cumsum[psr_index]:M_cumsum[psr_index+1]]   @ M
    u_spin = -P_spin_gw[:,psr_index] + P_spin[:,psr_index]/f0[psr_index]     + P_spin_eps[:, M_cumsum[psr_index]:M_cumsum[psr_index+1]] @ M
    u_eps  = -P_eps_gw[:,psr_index]  + P_eps_spin[:,psr_index]/f0[psr_index] + P_eps[:, M_cumsum[psr_index]:M_cumsum[psr_index+1]]      @ M

    # Innovation variance
    u_gw_value   = get_ith_pair(u_gw, psr_index)[0]
    u_spin_value = get_ith_pair(u_spin, psr_index)[0]
    u_eps_value  = get_ith_vector(u_eps, M_cumsum, psr_index)
    S            = (-1 * u_gw_value) + (1.0/f0[psr_index] * u_spin_value) + (M @ u_eps_value) + R

    # Kalman gain scale
    alpha = 1.0 / S

    # Updated state
    x_gw_up   = x_gw   + alpha * u_gw   * nu
    x_spin_up = x_spin + alpha * u_spin * nu
    x_eps_up  = x_eps  + alpha * u_eps  * nu

    # Covariance updates
    P_gw_up       = P_gw       - alpha * jnp.outer(u_gw, u_gw)
    P_gw_spin_up  = P_gw_spin  - alpha * jnp.outer(u_gw, u_spin)
    P_spin_up     = P_spin     - alpha * jnp.outer(u_spin, u_spin)
    P_gw_eps_up   = P_gw_eps   - alpha * jnp.outer(u_gw, u_eps)
    P_spin_eps_up = P_spin_eps - alpha * jnp.outer(u_spin, u_eps)
    P_eps_up      = P_eps      - alpha * jnp.outer(u_eps, u_eps)

    x_list_up = [x_gw_up, x_spin_up, x_eps_up]
    P_list_up = [P_gw_up, P_spin_up, P_eps_up,
                 P_gw_spin_up, P_gw_eps_up, P_spin_eps_up]

    return x_list_up, P_list_up

@jit
def get_likelihood(θ, data, data_errors, psr_indices, t_diffs,
                  pulsar_design_matrices, design_matrix_counter, M_cumsum, f0,
                  x0_list, P0_list, Npsr, M_sum):
    """Run Kalman filter algorithm over all observations and return log likelihood."""
    # Initialize likelihood
    ll = 0.0

    # First update step
    i = 0
    x_list, P_list = update(
        psr_index              = psr_indices[i],
        θ                      = θ,
        design_matrix_counter  = design_matrix_counter,
        pulsar_design_matrices = pulsar_design_matrices,
        M_cumsum              = M_cumsum,
        f0                    = f0,
        x_list                = x0_list,
        P_list                = P0_list,
        R                     = data_errors[i],
        y                     = data[i]
    )

    # Main filter loop
    for i in tqdm(range(1, len(data)), desc="Processing timesteps"):
        dt = t_diffs[i - 1]

        # Predict step
        x_list_predict, P_list_predict = predict(
            dt      = dt,
            θ       = θ,
            x_list  = x_list,
            P_list  = P_list,
            Npsr    = Npsr,
            M_sum   = M_sum
        )

        # Update step
        x_list, P_list = update(
            psr_index              = psr_indices[i],
            θ                      = θ,
            design_matrix_counter  = design_matrix_counter,
            pulsar_design_matrices = pulsar_design_matrices,
            M_cumsum              = M_cumsum,
            f0                    = f0,
            x_list                = x_list_predict,
            P_list                = P_list_predict,
            R                     = data_errors[i],
            y                     = data[i]
        )

    return ll

import jax
import jax.numpy as jnp
from jax.scipy.linalg import block_diag
from jax import vmap
from functools import partial
from line_profiler import profile

@profile
def get_F_block(gamma, dt):
    exp_term = jnp.exp(-gamma * dt)
    return jnp.array([[1.0, (1-exp_term)/gamma],
                      [0.0, exp_term]])

@profile
def get_F_spin(gamma, dt):
    res = vmap(lambda x: get_F_block(x, dt))(gamma)
    return block_diag(*res)

@profile
@partial(jax.jit, static_argnums=(3,4))
def get_F(gamma, gamma_spin, dt, Npsr, M_sum):
    F_gw_block = get_F_block(gamma, dt)
    F_gw = jnp.kron(jnp.eye(Npsr), F_gw_block)
    F_spin = get_F_spin(gamma_spin, dt)
    return F_gw, F_spin

@profile
def get_Q_block(γ, dt):
    exp_term = jnp.exp(-γ * dt)
    exp_2term = jnp.exp(-2 * γ * dt)

    q11 = (dt - 2 * (1 - exp_term) / γ + (1 - exp_2term) / (2 * γ)) / γ**3
    q12 = ((1 - exp_term) - (1 - exp_2term) / 2) / (γ**2)
    q22 = (1 - exp_2term) / (2 * γ)

    return jnp.array([[q11, q12], [q12, q22]])

@profile
def get_Q_spin(gamma, dt):
    res = vmap(lambda x: get_Q_block(x, dt))(gamma)
    return block_diag(*res)

@profile
@partial(jax.jit, static_argnums=(3,4))
def get_Q(gamma, gamma_spin, dt, Npsr, M_sum, eps):
    Q_gw_block = get_Q_block(gamma, dt)
    Q_gw = jnp.kron(jnp.eye(Npsr), Q_gw_block)
    Q_spin = get_Q_spin(gamma_spin, dt)
    Q_timing = jnp.eye(M_sum) * eps**2
    return Q_gw, Q_spin, Q_timing

@profile
@partial(jax.jit, static_argnums=(2,3))
def get_xp(F_list, x, gw_size, spin_size):
    F_gw, F_spin = F_list
    x_gw = x[:gw_size]
    x_spin = x[gw_size:gw_size+spin_size]
    x_timing = x[gw_size+spin_size:]
    
    return jnp.vstack([F_gw@x_gw, F_spin@x_spin, x_timing])

@profile
@partial(jax.jit, static_argnums=(1,2))
def get_P_blocks(P, gw_size, spin_size):
    P1 = P[:gw_size, :gw_size]
    P2 = P[gw_size:gw_size+spin_size, gw_size:gw_size+spin_size]
    P3 = P[gw_size+spin_size:, gw_size+spin_size:]
    P4 = P[:gw_size, gw_size:gw_size+spin_size]
    P5 = P[gw_size:gw_size+spin_size, gw_size+spin_size:]
    P6 = P[:gw_size, gw_size+spin_size:]

    return P1, P2, P3, P4, P5, P6

@profile
@jax.jit
def get_Pp_blocks(list_A, list_B, list_C):
    F1, F2 = list_A
    P1, P2, P3, P4, P5, P6 = list_B
    Q1, Q2, Q3 = list_C
    
    PF1 = F1@P1@F1.T + Q1
    PF2 = F2@P2@F2.T + Q2
    PF4 = F1@P4@F1.T
    PF5 = F2@P5
    PF6 = F1@P6
    
    block_row1 = jnp.hstack([PF1, PF4, PF6])
    block_row2 = jnp.hstack([PF4.T, PF2, PF5])
    block_row3 = jnp.hstack([PF6.T, PF5.T, P3 + Q3])

    return jnp.vstack([block_row1, block_row2, block_row3])


# Pre-compute F matrices for each dt
@profile
@partial(jax.jit, static_argnums=(3, 4))
def precompute_F_matrices(gamma_a, gamma_p, dt_array, Npsr, M_sum):
    """Precompute all F matrices for a given parameter set and sequence of dt values.
    
    This computes F matrices for all time steps at once with a given parameter set.
    Returns a tuple of arrays where each array contains matrices for all timesteps.
    """
    # Define a function that returns the F matrices for a single dt
    def get_F_for_dt(dt):
        F_gw, F_spin = get_F(gamma_a, gamma_p, dt, Npsr, M_sum)
        return F_gw, F_spin
    
    # Vectorize this function over all dt values
    return jax.vmap(get_F_for_dt)(dt_array)



# Similarly for Q matrices
@profile
@partial(jax.jit, static_argnums=(3, 4))
def precompute_Q_matrices(gamma_a, gamma_p, dt_array, Npsr, M_sum, eps):
    """Precompute all Q matrices for a given parameter set and sequence of dt values."""
    def get_Q_for_dt(dt):
        Q_gw, Q_spin, Q_timing = get_Q(gamma_a, gamma_p, dt, Npsr, M_sum, eps)
        return Q_gw, Q_spin, Q_timing
    
    return jax.vmap(get_Q_for_dt)(dt_array)








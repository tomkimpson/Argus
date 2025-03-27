import jax
import jax.numpy as jnp
from jax.scipy.linalg import block_diag
from jax import vmap
from functools import partial
from line_profiler import profile


def get_F_block(gamma, dt):
    exp_term = jnp.exp(-gamma * dt)
    return jnp.array([[1.0, (1-exp_term)/gamma],
                      [0.0, exp_term]])

def get_F_spin(gamma, dt):
    res = vmap(lambda x: get_F_block(x, dt))(gamma)
    return block_diag(*res)

@partial(jax.jit, static_argnums=(3,4))
def get_F(gamma, gamma_spin, dt, Npsr, M_sum):
    F_gw_block = get_F_block(gamma, dt)
    F_gw = jnp.kron(jnp.eye(Npsr), F_gw_block)
    F_spin = get_F_spin(gamma_spin, dt)
    F_timing = jnp.eye(M_sum)
    return F_gw, F_spin, F_timing

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

@partial(jax.jit, static_argnums=(3,4))
def get_Q(gamma, gamma_spin, dt, Npsr, M_sum, eps):
    Q_gw_block = get_Q_block(gamma, dt)
    Q_gw = jnp.kron(jnp.eye(Npsr), Q_gw_block)
    Q_spin = get_Q_spin(gamma_spin, dt)
    Q_timing = jnp.eye(M_sum) * eps**2
    # return block_diag(Q_gw, Q_spin, Q_timing)
    return Q_gw, Q_spin, Q_timing

@jax.jit
def get_xp(F_list, x_list):
    F_gw, F_spin, F_eps = F_list
    x_gw, x_spin, x_eps = x_list
    
    return F_gw@x_gw, F_spin@x_spin, x_eps


@jax.jit
def get_Pp_blocks(F_list, P_list, Q_list):
    F_gw, F_spin, F_eps = F_list
    P_gw, P_gw_spin, P_gw_eps, P_spin, P_spin_eps, P_eps = P_list
    Q_gw, Q_spin, Q_eps = Q_list
   
    Pp_gw = F_gw@P_gw@F_gw.T + Q_gw
    Pp_spin = F_spin@P_spin@F_spin.T + Q_spin
    Pp_eps = P_eps+ Q_eps
    Pp_gw_spin = F_gw@P_gw_spin@F_spin.T 
    Pp_gw_eps = F_gw@P_gw_eps
    Pp_spin_eps = F_spin@P_spin_eps

    return Pp_gw, Pp_spin, Pp_eps, Pp_gw_spin, Pp_gw_eps, Pp_spin_eps

@jax.jit
def get_kalman(P_list, psr_index, f0, H_eps):
    P_gw, P_gw_spin, P_gw_eps, P_spin, P_spin_eps, P_eps = P_list
    u_gw   = -P_gw[:,psr_index]      + P_gw_spin[:,psr_index]/f0  + P_gw_eps@H_eps.reshape(-1)
    u_spin = -P_gw_spin[psr_index,:] + P_spin[:,psr_index]/f0     + P_spin_eps@H_eps.reshape(-1)
    u_eps  = -P_gw_eps[psr_index,:]  + P_spin_eps[psr_index,:]/f0 + P_eps@H_eps.reshape(-1)
    return u_gw, u_spin, u_eps

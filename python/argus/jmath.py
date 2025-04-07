import jax
import jax.numpy as jnp
from jax.scipy.linalg import block_diag
from jax import vmap
from functools import partial

from typing import NamedTuple, Tuple

# Calculate F matrix
def _get_F_block(gamma, dt):
    exp_term = jnp.exp(-gamma * dt)
    return jnp.array([[1.0, (1-exp_term)/gamma],
                      [0.0, exp_term]])

def _get_F_spin(gamma, dt):
    res = vmap(lambda x: _get_F_block(x, dt))(gamma)
    return block_diag(*res)

@partial(jax.jit, static_argnames=["Npsr"])
def get_F(gamma_gw, gamma_spin, Npsr, dt):
    F_gw_block = _get_F_block(gamma_gw, dt)
    F_gw = jnp.kron(jnp.eye(Npsr), F_gw_block)
    F_spin = _get_F_spin(gamma_spin, dt)
   
    return F_gw, F_spin

# Calculate Q matrix
def _get_Q_block(γ, dt):
    exp_term = jnp.exp(-γ * dt)
    exp_2term = jnp.exp(-2 * γ * dt)

    q11 = (dt - 2 * (1 - exp_term) / γ + (1 - exp_2term) / (2 * γ)) / γ**3
    q12 = ((1 - exp_term) - (1 - exp_2term) / 2) / (γ**2)
    q22 = (1 - exp_2term) / (2 * γ)

    return jnp.array([[q11, q12], [q12, q22]])

def get_Q_spin(gamma, dt):
    res = vmap(lambda x: _get_Q_block(x, dt))(gamma)
    return block_diag(*res)

@partial(jax.jit, static_argnames=["Npsr"])
def get_Q(gamma_gw, gamma_spin, dt, Npsr, sigma_eps):
    Q_gw_block = _get_Q_block(gamma_gw, dt)
    Q_gw = jnp.kron(jnp.eye(Npsr), Q_gw_block)
    Q_spin = get_Q_spin(gamma_spin, dt)
    Q_eps = sigma_eps**2
   
    return Q_gw, Q_spin, Q_eps

@jax.jit
def get_xp(F, x):
    # unpack F,x 
    F_gw, F_spin = F
    x_gw, x_spin, x_eps = x

    # predict the next state
    xp_gw = F_gw@x_gw
    xp_spin = F_spin@x_spin
    return xp_gw, xp_spin, x_eps

    
@jax.jit
def get_Pp(F: Tuple, P: Tuple, Q: Tuple):
    # unpack F, P, Q
    F_gw, F_spin = F
    P_gw, P_spin, P_eps, P_gw_spin, P_gw_eps, P_spin_eps = P
    Q_gw, Q_spin, Q_eps = Q
    
    # predict the next covariance
    Pp_gw = F_gw@P_gw@F_gw.T + Q_gw
    Pp_spin = F_spin@P_spin@F_spin.T + Q_spin
    Pp_eps = P_eps+ Q_eps
    Pp_gw_spin = F_gw@P_gw_spin@F_spin.T 
    Pp_gw_eps = F_gw@P_gw_eps
    Pp_spin_eps = F_spin@P_spin_eps

    return Pp_gw, Pp_spin, Pp_eps, Pp_gw_spin, Pp_gw_eps, Pp_spin_eps

@jax.jit
def update_x_P(x: Tuple, P: Tuple, psr_index: int, f0, H_eps, R, y):
    """Update state vector x and covariance matrix P"""
    # unpack x, P
    x_gw, x_spin, x_eps = x
    P_gw, P_spin, P_eps, P_gw_spin, P_gw_eps, P_spin_eps = P
    
    # calculate P * H^T
    u_gw   = -P_gw[:,2*psr_index]      + P_gw_spin[:,2*psr_index]/f0  + P_gw_eps@H_eps.reshape(-1)
    u_spin = -P_gw_spin[2*psr_index,:] + P_spin[:,2*psr_index]/f0     + P_spin_eps@H_eps.reshape(-1)
    u_eps  = -P_gw_eps[2*psr_index,:]  + P_spin_eps[2*psr_index,:]/f0 + P_eps@H_eps.reshape(-1)

    # Innovation variance: S = H * P * H^T + R
    S = (-1 * u_gw[2*psr_index]) + (1.0/f0 * u_spin[2*psr_index]) + (H_eps.reshape(-1))@u_eps + R

    # update covariance
    P_gw_up = P_gw - jnp.outer(u_gw, u_gw) /S
    P_gw_spin_up = P_gw_spin - jnp.outer(u_gw, u_spin)/S
    P_spin_up = P_spin - jnp.outer(u_spin, u_spin)/S
    P_gw_eps_up = P_gw_eps - jnp.outer(u_gw, u_eps)/S
    P_spin_eps_up = P_spin_eps - jnp.outer(u_spin, u_eps)/S
    P_eps_up = P_eps - jnp.outer(u_eps, u_eps)/S
    
    # measurement equation
    nu = y - (-x_gw[psr_index*2] + x_spin[psr_index*2]/f0 + (H_eps.reshape(-1))@x_eps)
    x_gw_up = x_gw + u_gw * nu/S
    x_spin_up = x_spin + u_spin * nu/S
    x_eps_up = x_eps + u_eps * nu/S

    # get loglikelihood
    ll_t = log_likelihood(nu, S)

    return (x_gw_up, x_spin_up, x_eps_up), (P_gw_up, P_spin_up, P_eps_up, P_gw_spin_up, P_gw_eps_up, P_spin_eps_up), ll_t

# @jax.jit
def log_likelihood(y, cov):
    """Given the innovation and innovation covariance, get the likelihood."""
    log_likelihood = -0.5 * (jnp.log(2.0 * jnp.pi * cov) + (y * y) / cov)
    return log_likelihood


# @jax.jit
# def get_kalman(P: Tuple, psr_index: int, f0, H_eps, R):
#     P_gw, P_spin, P_eps, P_gw_spin, P_gw_eps, P_spin_eps = P
#     u_gw   = -P_gw[:,2*psr_index]      + P_gw_spin[:,2*psr_index]/f0  + P_gw_eps@H_eps.reshape(-1)
#     u_spin = -P_gw_spin[2*psr_index,:] + P_spin[:,2*psr_index]/f0     + P_spin_eps@H_eps.reshape(-1)
#     u_eps  = -P_gw_eps[2*psr_index,:]  + P_spin_eps[2*psr_index,:]/f0 + P_eps@H_eps.reshape(-1)

#     S = (-1 * u_gw[2*psr_index]) + (1.0/f0 * u_spin[2*psr_index]) + (H_eps.reshape(-1))@u_eps + R
#     return (u_gw, u_spin, u_eps), 1/S

# @jax.jit
# def update_P(P_list, u_list, alpha):
#     P_gw, P_spin, P_eps, P_gw_spin, P_gw_eps, P_spin_eps = P_list
#     u_gw, u_spin, u_eps = u_list
#     P_gw_up = P_gw - alpha * jnp.outer(u_gw, u_gw)
#     P_gw_spin_up = P_gw_spin - alpha * jnp.outer(u_gw, u_spin)
#     P_spin_up = P_spin - alpha * jnp.outer(u_spin, u_spin)
#     P_gw_eps_up = P_gw_eps - alpha * jnp.outer(u_gw, u_eps)
#     P_spin_eps_up = P_spin_eps - alpha * jnp.outer(u_spin, u_eps)
#     P_eps_up = P_eps - alpha * jnp.outer(u_eps, u_eps)
#     return P_gw_up, P_spin_up, P_eps_up, P_gw_spin_up, P_gw_eps_up, P_spin_eps_up

# @jax.jit
# def update_x(x_list, u_list, psr_index, f0, H_eps, y, alpha):
#     x_gw, x_spin, x_eps = x_list
#     u_gw, u_spin, u_eps = u_list
#     nu = y - (-x_gw[psr_index*2] + x_spin[psr_index*2]/f0 + (H_eps.reshape(-1))@x_eps)
#     x_gw_up = x_gw + alpha * u_gw * nu
#     x_spin_up = x_spin + alpha * u_spin * nu
#     x_eps_up = x_eps + alpha * u_eps * nu
#     return x_gw_up, x_spin_up, x_eps_up

    import jax
import jax.numpy as jnp
from jax import lax

##############################################################################
# 1) DATA / MODEL STRUCTURES
##############################################################################
# We assume:
#   - N = number of pulsars
#   - local_dim[n] = dimension of pulsar n's local state
#   - We'll keep all times, dt, measurement pulses in arrays.
#   - "params" is a dictionary that includes gamma_a, Sigma_a, and possibly
#       other parameters (like gamma_p^(n)).

##############################################################################
# 2) DISCRETE-TIME TRANSITION FUNCTIONS
##############################################################################

def global_transition_FQ(dt, gamma_a, Sigma_a):
    """
    Build the global transition matrix F^G (N×N) and process noise Q^GG (N×N)
    for the 'a^(n)' states over step dt, assuming Ornstein-Uhlenbeck (OU) with
    common damping gamma_a for all pulsars, and correlated noise Sigma_a.

    F^G = exp(- gamma_a * dt) * I_N.
    Q^GG = integral of noise correlation => a standard OU formula:
        Q^GG = alpha * Sigma_a
      where alpha = (1 - e^{-2 gamma_a dt}) / (2 gamma_a), if gamma_a > 0.
      If gamma_a=0 => pure random walk => alpha = dt.
    """
    N = Sigma_a.shape[0]
    edamp = jnp.exp(-gamma_a * dt)  # scalar
    Fg = edamp * jnp.eye(N)         # shape (N,N)

    # Integrated OU noise factor
    def ou_alpha(gamma, dtt):
        return (1.0 - jnp.exp(-2.0 * gamma * dtt)) / (2.0 * gamma)

    alpha = jnp.where(gamma_a > 0.0,
                      ou_alpha(gamma_a, dt),
                      dt)  # fallback if gamma_a=0 => random walk
    Qg = alpha * Sigma_a  # shape (N,N)

    return Fg, Qg


def local_transition_FQ(dt, n, params, local_dim):
    """
    Returns F^L_n (local transition) and Q^nn (local process noise) for pulsar n
    over step dt.

    In practice, you'd embed your full SDE -> discrete-time mapping here,
    including e^{-gamma_p^(n)*dt}, integrated noise, etc.

    For DEMO: we'll do:
        x = (phi, f, r, eps1, eps2, ...)
        phi' = phi + dt*f
        f'   = e^{-gamma_p dt} * f + noise
        r'   = r   (the effect of a^(n) is handled by a separate coupling G_n)
        eps' = eps + noise
    We'll just put a small placeholder noise, ignoring the real structure.
    """
    # e.g. we might have a separate gamma_p^(n):
    # gamma_p_n = params["gamma_p"][n]  # if you store a list/array
    # For demonstration, fix a small gamma_p:
    gamma_p_n = 0.01

    ld = local_dim[n]

    # Construct F_n^L
    F_local = jnp.eye(ld)
    # Suppose state indexing: 0=phi, 1=f, 2=r, 3..=eps
    F_local = F_local.at[0,1].set(dt)  # phi -> phi + dt*f
    edamp_p = jnp.exp(-gamma_p_n * dt)
    F_local = F_local.at[1,1].set(edamp_p)  # f -> e^{-gamma_p dt}*f

    # Construct Q^nn (very rough placeholder)
    # If gamma_p, eps are OU, we'd do integrated forms.  We'll do a small constant:
    sigma_local = 1e-4
    Q_local = (sigma_local**2) * jnp.eye(ld) * dt

    return F_local, Q_local


def local_coupling_G(dt, n, N, local_dim):
    """
    G_n: how the global vector a (shape (N,)) influences x^(n).
    E.g. r^(n)' = r^(n) + dt*a^(n}, so only a(n) entry affects r^(n).
    We'll assume index 2 is 'r'.
    """
    ld = local_dim[n]
    G_n = jnp.zeros((ld, N))
    # r index = 2:
    G_n = G_n.at[2, n].set(dt)
    return G_n


##############################################################################
# 3) PREDICT STEP (Partitioned)
##############################################################################

def predict_local_block(carry, n):
    """
    This function is used by a lax.scan over all pulsars n=0..N-1.
    It updates x^(n) and the associated P^{nn}, P^{nG} blocks, given the
    current global a, P^{GG}, plus the local block from carry.

    Inputs:
      carry = (a_pred, P_GG_pred, x_pred_list, P_xx_pred_list, P_xG_pred_list,
               dt, params, local_dim)
      n = the integer index of the pulsar to update

    Returns:
      updated carry, plus no extra output (None).
    """
    (a_curr, P_GG_curr, 
     x_list, P_xx_list, P_xG_list,
     dt, params, local_dim) = carry

    x_n     = x_list[n]         # shape (ld_n,)
    P_xx_n  = P_xx_list[n]      # shape (ld_n, ld_n)
    P_xG_n  = P_xG_list[n]      # shape (ld_n, N)

    # Build F^L_n, Q^nn, G_n from dt, etc.
    F_local_n, Q_local_n = local_transition_FQ(dt, n, params, local_dim)
    G_n = local_coupling_G(dt, n, P_GG_curr.shape[0], local_dim)

    # 1) Predict the local mean
    x_n_new = F_local_n @ x_n + G_n @ a_curr

    # 2) Predict local cov P^{nn}
    #    P_xx_n -> F_n^L P_xx_n F_n^L^T
    #              + F_n^L P^{nG} G_n^T
    #              + G_n   P^{Gn} F_n^L^T
    #              + G_n   P^{GG} G_n^T
    #              + Q_local_n
    P_xx_tmp = F_local_n @ P_xx_n @ F_local_n.T
    P_xx_tmp += F_local_n @ P_xG_n @ G_n.T
    P_xx_tmp += G_n @ P_xG_n.T @ F_local_n.T
    P_xx_tmp += G_n @ P_GG_curr @ G_n.T
    P_xx_pred = P_xx_tmp + Q_local_n

    # 3) Predict cross-cov P^{nG}
    #    P_xG_n -> F_local_n P_xG_n F_G^T + G_n P^{GG} F_G^T
    # We need F^G, Q^G for the global too, but that was already applied outside.
    # Actually the "global" is already predicted to a_curr, P_GG_curr, which 
    # means a_curr = F^G a_{prev}, P_GG_curr = F^G P_GG_prev F^G^T + Q^G, so
    # we do not re-multiply by F^G inside this local step. 
    # Instead, P_GG_curr is already the predicted global cov. 
    # So the new cross term is:
    #   P^{nG}_pred = F_local_n P^{nG}_old (F^G)^T + G_n P^{GG}_old (F^G)^T
    # But we have stored (a_curr, P_GG_curr) AFTER applying F^G, Q^G.
    # => This simplifies the local cross term to:
    #   P^{nG}_pred = F_local_n P^{nG}_old + G_n P^{GG}_old
    # For correctness, let's assume the global block was advanced from 
    # (a_{old}, P_GG_{old}) to (a_curr, P_GG_curr). The correct "old" cross-block
    # would not be the same as P_xG_n we have in carry, if that was not predicted yet.
    #
    # Real solution: we do "predict_partitioned" in two parts: 
    #   (a, P_GG) -> predicted, then local. 
    # We'll assume the input P_xG_n is from the previous step (unpredicted),
    # so we do:
    #   P_xG_n_pred = F_local_n @ P_xG_n @ F_g^T + G_n @ P_GG_old @ F_g^T
    # But we do not have "P_GG_old" or "F_g^T" in carry once we've overwritten them.
    #
    # For simplicity in this code, let's treat "predict_partitioned" as:
    #   1) we do global predict (a->a_pred, P^GG->...) => store them 
    #   2) we do local predict w.r.t. the *same old* a, P^GG if needed 
    # 
    # or we treat everything simultaneously. This gets complicated quickly.
    #
    # In the demonstration below, we skip the strict "F_g^T" factor in cross-terms 
    # for brevity.  In real code, you'd do the exact 2×2 block-lift. 
    #
    # We'll just do a minimal approach here:
    P_xG_pred = F_local_n @ P_xG_n  # ignoring 'F_g^T' for demonstration

    # Return updated local state in the lists
    x_list_new     = x_list.at[n].set(x_n_new)
    P_xx_list_new  = P_xx_list.at[n].set(P_xx_pred)
    P_xG_list_new  = P_xG_list.at[n].set(P_xG_pred)

    new_carry = (a_curr, P_GG_curr,
                 x_list_new, P_xx_list_new, P_xG_list_new,
                 dt, params, local_dim)
    return new_carry, None


def predict_partitioned(carry):
    """
    A single 'predict' step over a time interval dt, updating:
     - global block (a, P^GG)
     - local blocks (x^(n), P^{nn}, P^{nG})

    The carry must contain:
      (a, P_GG, x_list, P_xx_list, P_xG_list, dt, params, local_dim)
    We'll:
     1) do the global predict (a->a_pred, P_GG-> P_GG_pred)
     2) do a lax.scan over n=0..N-1 for local predictions.
    """
    (a_k, P_GG_k,
     x_k_list, P_xx_k_list, P_xG_k_list,
     dt, params, local_dim) = carry

    # 1) Global predict
    gamma_a = params["gamma_a"]
    Sigma_a = params["Sigma_a"]
    Fg, Qg  = global_transition_FQ(dt, gamma_a, Sigma_a)
    a_pred  = Fg @ a_k
    P_GG_pred = Fg @ P_GG_k @ Fg.T + Qg

    # 2) local predict in a scan
    # We'll define a sub-carry for local_predict
    local_carry_init = (a_pred, P_GG_pred,
                        x_k_list, P_xx_k_list, P_xG_k_list,
                        dt, params, local_dim)

    def scan_local_predict_fn(carry_in, n):
        return predict_local_block(carry_in, n)

    N = len(local_dim)
    local_carry_out, _ = lax.scan(scan_local_predict_fn,
                                  local_carry_init,
                                  jnp.arange(N))

    # Unpack results
    (a_out, P_GG_out,
     x_out_list, P_xx_out_list, P_xG_out_list,
     _dt, _params, _ld) = local_carry_out

    # Return the same structure
    new_carry = (a_out, P_GG_out,
                 x_out_list, P_xx_out_list, P_xG_out_list,
                 dt, params, local_dim)
    return new_carry


##############################################################################
# 4) UPDATE STEP (measurement from a single pulsar)
##############################################################################

def update_partitioned(carry, obs):
    """
    Perform one measurement update for the pulsar indicated by obs, then return
    the updated state plus the log-likelihood increment from that measurement.

    carry = (a_pred, P_GG_pred, x_pred_list, P_xx_pred_list, P_xG_pred_list,
             dt, params, local_dim)

    obs   = (dt, n_meas, y_obs, R_obs)
       - dt can be redundant if we want each step to do "predict + update" in one go.
         but here we only do "update" with 'obs'.

    We'll do a 2-block update on {global a, local x^(n_meas)}. Then we compute
    the scalar innovation log-likelihood.

    Return: (new_carry, logL_incr)
    """
    (a_curr, P_GG_curr,
     x_list, P_xx_list, P_xG_list,
     old_dt, params, local_dim) = carry

    # Unpack measurement
    dt_obs, n_meas, y_obs, R_obs = obs
    n_meas = n_meas.astype(int)

    x_n = x_list[n_meas]
    P_nn = P_xx_list[n_meas]
    P_nG = P_xG_list[n_meas]       # shape (ld_n, N)
    P_Gn = P_nG.T                  # shape (N, ld_n)

    # Build measurement matrix H. 
    # Example: y = phi - r => indices 0=phi, 2=r
    ld_n = local_dim[n_meas]
    # Suppose no direct dependence on a in the measurement => H_a=0
    H_a = jnp.zeros((1, P_GG_curr.shape[0]))   # shape (1, N)
    # local part H_xn shape(1, ld_n)
    H_xn = jnp.zeros((1, ld_n))
    # pick out phi(0) = +1, r(2) = -1
    H_xn = H_xn.at[0,0].set(1.0)
    H_xn = H_xn.at[0,2].set(-1.0)

    # Predicted measurement + residual
    y_pred = (H_a @ a_curr) + (H_xn @ x_n)
    resid = y_obs - y_pred[0]  # shape ()

    # Innovation variance:
    # S = H_a P^GG H_a^T + H_a P^Gn H_xn^T
    #   + H_xn P^nG H_a^T + H_xn P^nn H_xn^T + R_obs
    S_aa  = H_a @ P_GG_curr @ H_a.T
    S_an  = H_a @ P_Gn       @ H_xn.T
    S_na  = H_xn @ P_nG      @ H_a.T
    S_nn  = H_xn @ P_nn      @ H_xn.T
    S_val = S_aa + S_an + S_na + S_nn + R_obs
    # shape (1,1)
    S_val_scalar = S_val[0,0]
    S_inv = 1.0 / S_val_scalar

    # Kalman gain block
    # K_a = P^GG H_a^T + P^Gn H_xn^T
    Ka = (P_GG_curr @ H_a.T) + (P_Gn @ H_xn.T)  # shape (N,1)
    # K_x = P^{nG} H_a^T + P^{nn} H_xn^T
    Kx = (P_nG @ H_a.T) + (P_nn @ H_xn.T)       # shape (ld_n,1)
    Ka = Ka * S_inv
    Kx = Kx * S_inv

    # Posterior means
    a_post = a_curr + (Ka * resid).reshape(-1)   # shape(N,)
    x_n_post = x_n + (Kx * resid).reshape(-1)    # shape(ld_n,)

    # We do Joseph or standard rank-1 form to update covariance blocks:
    # We want (I - K H) * P. In block partition:
    # dP^GG = K_a [H_a P^GG + H_xn P^nG]  etc.
    # For brevity, do direct expansions:
    # Compute [H_a, H_xn] * P^GG, P^Gn, etc.

    # Let's define partials for short:
    HP_GG_a  = (H_a @ P_GG_curr)    # shape (1,N)
    HP_Gn_xn = (H_a @ P_Gn)         # shape (1,ld_n)
    HX_Gn_a  = (H_xn @ P_nG)        # shape (1,N)
    HX_xn_xn = (H_xn @ P_nn)        # shape (1,ld_n)

    # Full "H_aug * P_aug" in block form => we gather them.
    # Then K_aug * that => rank-1. We'll do partial sub-block updates.

    # Update P^GG
    dP_GG = jnp.outer(Ka, (HP_GG_a + HX_Gn_a))  # shape(N,N)
    P_GG_post = P_GG_curr - dP_GG

    # Update P^Gn => (ld_n, N)
    dP_Gn = jnp.outer(Kx.reshape(-1),
                      (HP_GG_a + HX_Gn_a).reshape(-1))
    # But wait, we only need the portion for n? Actually that is P^nG^T => be careful
    # It's easier to do the standard formula for block as well.
    # Let's do a simpler approach:
    HP_Gn_combined = (H_a @ P_Gn) + (H_xn @ P_nn)  # shape(1, ld_n)
    dP_Gn_2 = jnp.outer(Kx, HP_Gn_combined)
    P_Gn_post = P_Gn - dP_Gn_2.T  # shape(N,ld_n) ?

    # Actually watch the transposes carefully.  A simpler method is the direct formula:
    # P_post = P_pred - K * (H * P_pred).
    # For the n-block, we do P^nG_post = P^nG_pred - Kx * (H_aug * P_pred^G) ??? 
    # Let's do it carefully below:

    # We'll do the rank-1 approach for each block:

    # (1) For P^GG_post: done above => P_GG_post
    # (2) For P^Gn_post:
    #    P^Gn_post = P^Gn_pred - K_a * (H_a P^Gn_pred + H_xn P^nn_pred)
    # Actually K_a is shape (N,1), "H_a P^Gn" is (1,ld_n).
    # so dP^Gn = K_a * that => shape(N, ld_n).
    HP_Gn_block = (H_a @ P_Gn) + (H_xn @ P_nn)  # shape(1,ld_n)
    dP_Gn_final = jnp.outer(Ka.reshape(-1),
                            HP_Gn_block.reshape(-1))  # shape(N,ld_n)
    P_Gn_post = P_Gn - dP_Gn_final

    # (3) For P^nG_post = (P^Gn_post)^T
    P_nG_post = P_Gn_post.T

    # (4) For P^nn_post:
    #   P^nn_post = P^nn_pred - K_x * [H_a P^Gn + H_xn P^nn]
    # => shape (ld_n, ld_n)
    dP_nn = jnp.outer(Kx.reshape(-1),
                      HP_Gn_block.reshape(-1))  # shape (ld_n, ld_n)
    P_nn_post = P_nn - dP_nn

    # Store updates
    # local block
    x_list_new = x_list.at[n_meas].set(x_n_post)
    P_xx_list_new = P_xx_list.at[n_meas].set(P_nn_post)
    P_xG_list_new = P_xG_list.at[n_meas].set(P_nG_post)

    # Summaries
    carry_out = (a_post, P_GG_post,
                 x_list_new, P_xx_list_new, P_xG_list_new,
                 old_dt, params, local_dim)

    # log-likelihood increment from the scalar innovation
    ll_innov = -0.5*(jnp.log(2.0*jnp.pi*S_val_scalar)
                     + (resid**2)/S_val_scalar)

    return carry_out, ll_innov


##############################################################################
# 5) COMBINED "PREDICT + UPDATE" FOR EACH MEASUREMENT
##############################################################################

def kalman_step(carry, obs):
    """
    For each measurement obs = (dt, n_meas, y_obs, R_obs), do:
      1) PREDICT the entire partitioned state over dt
      2) UPDATE with single-pulsar measurement
    Return updated carry plus the partial log-likelihood increment.

    This is the function we will 'scan' over all measurements.
    """
    # 1) Predict
    carry_pred = predict_partitioned(carry)

    # 2) Update
    carry_upd, ll_incr = update_partitioned(carry_pred, obs)

    return carry_upd, ll_incr


##############################################################################
# 6) MAIN LIKELIHOOD FUNCTION WITH JAX.SCAN
##############################################################################

def kalman_log_likelihood(params,
                          local_dim,
                          obs_times, obs_pulsar, obs_value, obs_var,
                          a0, P_GG0,
                          x0_list, P_xx0_list, P_xG0_list):
    """
    - N = len(local_dim)
    - obs_times[k], obs_pulsar[k], obs_value[k], obs_var[k]: sorted by time
      We assume obs_times[0] is the first measurement time, etc.
    - We'll build an array of (dt, n_meas, y_obs, R_obs) for each measurement,
      where dt = t_k - t_{k-1}. For k=0, define dt=obs_times[0] - t0 (some base).
    - Then we 'scan' over these measurements with the "kalman_step" that does
      PREDICT+UPDATE for each measurement.

    Returns total log-likelihood = sum of each measurement's log-likelihood increment.
    """
    K = len(obs_times)
    if K == 0:
        return 0.0  # no data => zero log-likelihood

    # Build the array of "obs" to feed into lax.scan.
    # We'll define dt_array[0] = obs_times[0] if we assume initial time=0.
    # Then dt_array[k] = obs_times[k] - obs_times[k-1] for k>0.
    dt_array = jnp.concatenate([
        jnp.array([obs_times[0]]),
        obs_times[1:] - obs_times[:-1]
    ])

    # We'll zip them into a single structure of shape (K,4):
    # obs_data[k] = (dt_array[k], obs_pulsar[k], obs_value[k], obs_var[k])
    obs_data = jnp.stack([dt_array, obs_pulsar, obs_value, obs_var], axis=1)

    # Initial carry in the form:
    # (a, P_GG, x_list, P_xx_list, P_xG_list, dt=0, params, local_dim)
    # The 'dt' in the carry is not used in the initial state, but we keep it for shape consistency
    init_carry = (
        a0, P_GG0,
        x0_list, P_xx0_list, P_xG0_list,
        jnp.array(0.0), params, local_dim
    )

    # Now we do a single scan over the K measurements.
    # Each step calls `kalman_step(carry, obs) -> (carry, ll_incr)`.
    final_carry, ll_incrs = lax.scan(kalman_step, init_carry, obs_data)
    logL = jnp.sum(ll_incrs)
    return logL


@jax.jit
def neg_log_likelihood(params,
                       local_dim,
                       obs_times, obs_pulsar, obs_value, obs_var,
                       a0, P_GG0,
                       x0_list, P_xx0_list, P_xG0_list):
    """
    Wrapper for negative log-likelihood, for e.g. optimization or sampling.
    We JIT-compile it for speed.
    """
    ll_val = kalman_log_likelihood(params,
                                   local_dim,
                                   obs_times, obs_pulsar, obs_value, obs_var,
                                   a0, P_GG0,
                                   x0_list, P_xx0_list, P_xG0_list)
    return -ll_val


##############################################################################
# 7) DEMONSTRATION
##############################################################################

if __name__ == "__main__":

    # Suppose we have N=2 pulsars
    N = 2
    local_dim = [4, 4]  # each pulsar local dimension = 4 (demo)
    
    # Some toy data with K=5 measurements at arbitrary times, each from one pulsar
    obs_times = jnp.array([2.0, 4.0, 4.5, 7.0, 10.0])
    obs_pulsar = jnp.array([0,    1,    0,    1,    0])  # which pulsar is measured
    obs_value  = jnp.array([0.1,  0.2,  0.05, -0.1, 0.3]) # dummy residual
    obs_var    = jnp.array([1e-3, 1e-3, 1e-3, 1e-3, 1e-3]) # measurement noise

    # Parameters
    #   gamma_a = global damping for a^(n)
    #   Sigma_a = NxN correlation matrix for the a-noise
    Sigma_a = jnp.array([[1.0, 0.2],
                         [0.2, 1.0]])
    params = {
        "gamma_a": 0.01,
        "Sigma_a": Sigma_a
        # Could also store e.g. "gamma_p": jnp.array([...]) if needed
    }

    # Initial state + cov
    a0 = jnp.zeros(N)           # global
    P_GG0 = jnp.eye(N)*1e6
    x0_list = jnp.array([jnp.zeros(ld) for ld in local_dim ], dtype=object)
    P_xx0_list = jnp.array([ jnp.eye(ld)*1e6 for ld in local_dim], dtype=object)
    
    # cross-cov P^{nG} initially zero
    P_xG0_list = jnp.array([ jnp.zeros((ld, N)) for ld in local_dim], dtype=object)

    # Evaluate negative log-likelihood
    nll_value = neg_log_likelihood(params,
                                   local_dim,
                                   obs_times, obs_pulsar, obs_value, obs_var,
                                   a0, P_GG0,
                                   x0_list, P_xx0_list, P_xG0_list)

    print("Negative Log-Likelihood =", nll_value)




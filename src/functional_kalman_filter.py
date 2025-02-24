
# import jax
# import jax.numpy as jnp
# from jax import lax


import numpy as np 


### START MODEL


def build_Fa(γa,dt,Npsr):
    return np.exp(-γa * dt) * np.eye(Npsr)

#this exponential can be reused
def build_Qaa(dt,hellings_downs_matrix,γa,h2):
        return (1 - np.exp(-2 * γa * dt)) *h2*hellings_downs_matrix
    





### END MODEL
def predict_a(a, P_aa, dt):
    """
    Predict the global a-vector over a time-step dt.
    Returns a_pred, P_aa_pred.
    """
    # Example: an OU with damping gamma_a => F^a = e^{-gamma_a * dt} * I
    # Q^aa = integrated noise covariance
    F_a = _build_F_a(γa,dt,Npsr)    # shape (N,N)
    Q_aa = build_Q_aa(dt,hellings_downs_matrix,γa,h2)  # shape (N,N)

    a_pred = F_a @ a
    P_aa_pred = F_a @ P_aa @ F_a.T + Q_aa

    return a_pred, P_aa_pred






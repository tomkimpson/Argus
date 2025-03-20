"""Module which implements standard Kalman filter algorithm using functional programming paradigm."""
import jax.numpy as jnp
from jax import jit
from jax import lax 
from argus.jmath import precompute_F_matrices, precompute_Q_matrices
from jax import debug 

def log_likelihood(y, S):
    """Calculate log likelihood given innovation and innovation covariance."""
    return -0.5 * (jnp.log(2.0 * jnp.pi) + jnp.log(S) + (y * y) / S)


def predict(x, P, F, Q):
    """Predict using precomputed matrices."""
    x_predict = jnp.dot(F, x)
    P_predict = jnp.dot(jnp.dot(F, P), F.T) + Q
    return x_predict, P_predict

def update(x, P, H, R, z):
    """Perform Kalman update step.
    
    Parameters
    ----------
    x : array_like
        Prior state estimate
    P : array_like
        Prior state covariance
    H : array_like
        Measurement matrix
    R : float
        Measurement noise variance (scalar)
    z : float
        Scalar measurement
    
    Returns
    -------
    x_up : array_like
        Updated state estimate
    P_up : array_like
        Updated state covariance
    y : float
        Innovation
    S : float
        Innovation variance
    """
    # Innovation calculation
    y = z - H @ x
    
    # Innovation variance (scalar)
    S = H @ P @ H.T + R
    
    # Kalman gain (using scalar division since S is scalar)
    K = (P @ H.T) / S
    
    # State update
    x_up = x + K * y
    
    # Covariance update using Joseph form for numerical stability
    I = jnp.eye(P.shape[0])
    P_up = (I - K @ H) @ P @ (I - K @ H).T + K * R * K.T
    
    return x_up, P_up, y, S

@jit
def get_likelihood(θ, data, data_errors, dt_array, x0, P0, H_arrays):
    """Run Kalman filter algorithm over all observations and return log likelihood.
    
    Parameters
    ----------
    θ : dict
        Parameter dictionary
    data : array_like
        Measurement data
    data_errors : array_like
        Measurement error variances
    dt_array : array_like
        Array of time differences between measurements
    x0 : array_like
        Initial state estimate
    P0 : array_like
        Initial state covariance
    H_arrays : array_like
        Array of H matrices for each observation

    Returns
    -------
    ll : float
        Log likelihood of the data given the parameters
    """

    # Precompute all F matrices for this parameter set
    F_matrices = precompute_F_matrices(θ.γa, θ.γp, dt_array, x0.shape[0])
    Q_matrices = precompute_Q_matrices(θ.γa, θ.γp, dt_array, x0.shape[0], θ.σeps)


    # First update
    H = H_arrays[0]
    x, P, y, S = update(x=x0, P=P0, H=H, R=data_errors[0], z=data[0])
    ll0 = log_likelihood(y, S)

    def step(carry, inputs):
        x, P = carry
        dt_idx, z, R, H = inputs

        #Predict
        F = F_matrices[dt_idx]
        Q = Q_matrices[dt_idx]
        x_predict, P_predict = predict(x, P, F, Q)

        #Update
        x, P, y, S = update(x_predict, P_predict, H, R, z)

        #Likelihood
        ll = log_likelihood(y, S)
        return (x, P), ll

    # Pack inputs for scan
    inputs = (jnp.arange(len(dt_array)), data[1:], data_errors[1:], H_arrays[1:])

    # Run scan loop
    (xf, Pf), ll_arr = lax.scan(step, (x, P), inputs)

    total_ll = ll0 + jnp.sum(ll_arr)
    return total_ll
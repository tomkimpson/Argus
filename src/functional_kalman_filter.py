# import jax
# import jax.numpy as jnp
# from jax import lax


import numpy as np 
from typing import Tuple, TypeVar
from scipy.linalg import block_diag


### START MODEL


def build_Fa(γa: float, dt: float, Npsr: int) -> np.ndarray:
    """Build state transition matrix for the 'a' component."""
    return np.exp(-γa * dt) * np.eye(Npsr)

#this exponential can be reused
def build_Qaa(dt: float, hellings_downs_matrix: np.ndarray, γa: float, h2: float) -> np.ndarray:
    """Build process noise covariance matrix for the 'a' component."""
    return (1 - np.exp(-2 * γa * dt)) * h2 * hellings_downs_matrix
    





### END MODEL

# Type aliases for better readability
State = TypeVar('State', bound=np.ndarray)
Covariance = TypeVar('Covariance', bound=np.ndarray)
Measurement = TypeVar('Measurement', bound=np.ndarray)

def predict_step(
    state_tuple: Tuple[np.ndarray, np.ndarray],
    F: np.ndarray,
    Q: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Pure function for Kalman filter prediction step.
    
    Parameters
    ----------
    state_tuple : Tuple[np.ndarray, np.ndarray]
        Tuple of (state_vector, covariance_matrix)
    F : np.ndarray
        State transition matrix
    Q : np.ndarray
        Process noise covariance matrix
    """
    x, P = state_tuple
    x_pred = F @ x
    P_pred = F @ P @ F.T + Q
    
    return (x_pred, P_pred)

def update_step(
    state: Tuple[np.ndarray, np.ndarray],
    measurement: np.ndarray,
    H: np.ndarray,
    R: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Pure function for Kalman filter update step.
    
    Parameters
    ----------
    state : Tuple[np.ndarray, np.ndarray]
        Tuple of (state_vector, covariance_matrix)
    measurement : np.ndarray
        Measurement vector
    H : np.ndarray
        Measurement matrix
    R : np.ndarray
        Measurement noise covariance
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Updated state and covariance
    """
    x, P = state
    # Innovation and its covariance
    y = measurement - H @ x
    S = H @ P @ H.T + R
    
    # Kalman gain
    K = P @ H.T @ np.linalg.inv(S)
    
    # Update state and covariance
    x_update = x + K @ y
    P_update = P - K @ H @ P
    
    return (x_update, P_update)

def kalman_step(
    state_tuple: Tuple[np.ndarray, np.ndarray],
    measurement: np.ndarray,
    F: np.ndarray,
    H: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray
) -> Tuple[Tuple[np.ndarray, np.ndarray], float]:
    """Complete Kalman filter step combining prediction and update.
    Also computes log-likelihood.
    
    Parameters
    ----------
    state_tuple : Tuple[np.ndarray, np.ndarray]
        Tuple of (state_vector, covariance_matrix)
    measurement : np.ndarray
        Measurement vector
    F : np.ndarray
        State transition matrix
    H : np.ndarray
        Measurement matrix
    Q : np.ndarray
        Process noise covariance
    R : np.ndarray
        Measurement noise covariance
    
    Returns
    -------
    Tuple[Tuple[np.ndarray, np.ndarray], float]
        Updated state and log-likelihood contribution
    """
    # Predict
    predicted_state = predict_step(state_tuple, F, Q)
    x_pred, P_pred = predicted_state
    
    # Innovation and its covariance
    y = measurement - H @ x_pred
    S = H @ P_pred @ H.T + R
    
    # Log-likelihood computation
    n = len(y)
    log_likelihood = -0.5 * (
        n * np.log(2 * np.pi) +
        np.log(np.linalg.det(S)) +
        y.T @ np.linalg.inv(S) @ y
    )
    
    # Update
    K = P_pred @ H.T @ np.linalg.inv(S)
    x_update = x_pred + K @ y
    P_update = P_pred - K @ H @ P_pred
    
    return (x_update, P_update), log_likelihood

def run_kalman_filter(
    initial_state: Tuple[np.ndarray, np.ndarray],
    measurements: np.ndarray,
    get_F: callable,
    get_H: callable,
    get_Q: callable,
    get_R: callable,
    dt: float
) -> Tuple[list[Tuple[np.ndarray, np.ndarray]], float]:
    """Run Kalman filter over a sequence of measurements.
    
    Parameters
    ----------
    initial_state : Tuple[np.ndarray, np.ndarray]
        Initial state and covariance
    measurements : np.ndarray
        Sequence of measurements
    get_F, get_H, get_Q, get_R : callable
        Functions that return the respective matrices
    dt : float
        Time step
        
    Returns
    -------
    Tuple[list[Tuple[np.ndarray, np.ndarray]], float]
        List of states and total log-likelihood
    """
    states = [initial_state]
    total_log_likelihood = 0.0
    
    for measurement in measurements:
        # Get matrices for this step
        F = get_F(dt)
        H = get_H()
        Q = get_Q(dt)
        R = get_R()
        
        # Perform Kalman step
        new_state, log_likelihood = kalman_step(
            states[-1],
            measurement,
            F, H, Q, R
        )
        
        states.append(new_state)
        total_log_likelihood += log_likelihood
    
    return states, total_log_likelihood

def neg_log_likelihood(params: dict, Npsr: int, hellings_downs_matrix: np.ndarray) -> float:
    """Compute negative log-likelihood for parameter optimization."""
    # Unpack parameters
    γa = params["gamma_a"]
    h2 = params["h_a"]**2  # Using h_a squared for h2
    
    # TODO: Implement full likelihood computation
    return 0.0  # Placeholder

def build_state_transition_blocks(γp: float, γa: float, dt: float, M: int) -> np.ndarray:
    """Build state transition blocks for a single pulsar.
    
    Parameters
    ----------
    γp : float
        Spin noise damping rate
    γa : float
        GW damping rate
    dt : float
        Time step
    M : int
        Number of timing model parameters
    
    Returns
    -------
    np.ndarray
        Block diagonal matrix containing the state transition blocks
    """
    # Block for (δφ, δf)
    F_spin = np.array([
        [1.0, (1 - np.exp(-γp * dt))/γp],
        [0.0, np.exp(-γp * dt)]
    ])
    
    # Block for (r, a)
    F_gw = np.array([
        [1.0, (1 - np.exp(-γa * dt))/γa],
        [0.0, np.exp(-γa * dt)]
    ])
    
    # Block for timing model parameters
    F_eps = np.eye(M)
    
    return block_diag(F_spin, F_gw, F_eps)

def build_process_noise_blocks(
    γp: float, 
    γa: float, 
    σp: float,
    σa: float,
    σeps: float,
    dt: float, 
    M: int
) -> np.ndarray:
    """Build process noise covariance blocks for a single pulsar.
    
    Parameters
    ----------
    γp : float
        Spin noise damping rate
    γa : float
        GW damping rate
    σp : float
        Spin noise amplitude
    σa : float
        GW noise amplitude
    σeps : float
        Timing model parameter noise amplitude
    dt : float
        Time step
    M : int
        Number of timing model parameters
    
    Returns
    -------
    np.ndarray
        Block diagonal matrix containing the process noise blocks
    """
    # Block for (δφ, δf)
    exp_γp_dt = np.exp(-γp * dt)
    exp_2γp_dt = np.exp(-2 * γp * dt)
    Q_spin = np.array([
        [(dt/γp**2 - 2*(1-exp_γp_dt)/γp**3 + (1-exp_2γp_dt)/(2*γp**3)) * σp**2,
         ((1-exp_γp_dt)/γp**2 - (1-exp_2γp_dt)/(2*γp**2)) * σp**2],
        [((1-exp_γp_dt)/γp**2 - (1-exp_2γp_dt)/(2*γp**2)) * σp**2,
         (1-exp_2γp_dt)/(2*γp) * σp**2]
    ])
    
    # Block for (r, a)
    exp_γa_dt = np.exp(-γa * dt)
    exp_2γa_dt = np.exp(-2 * γa * dt)
    Q_gw = np.array([
        [(dt/γa**2 - 2*(1-exp_γa_dt)/γa**3 + (1-exp_2γa_dt)/(2*γa**3)) * σa**2,
         ((1-exp_γa_dt)/γa**2 - (1-exp_2γa_dt)/(2*γa**2)) * σa**2],
        [((1-exp_γa_dt)/γa**2 - (1-exp_2γa_dt)/(2*γa**2)) * σa**2,
         (1-exp_2γa_dt)/(2*γa) * σa**2]
    ])
    
    # Block for timing model parameters
    Q_eps = σeps**2 * dt * np.eye(M)
    
    return block_diag(Q_spin, Q_gw, Q_eps)

def build_F_matrix(
    γp_list: np.ndarray,
    γa: float,
    M_list: np.ndarray,
    dt: float
) -> np.ndarray:
    """Build the complete state transition matrix for all pulsars.
    
    Parameters
    ----------
    γp_list : np.ndarray
        Array of spin noise damping rates for each pulsar
    γa : float
        GW damping rate
    M_list : np.ndarray
        Array of number of timing model parameters for each pulsar
    dt : float
        Time step
    
    Returns
    -------
    np.ndarray
        Complete state transition matrix
    """
    blocks = [
        build_state_transition_blocks(γp, γa, dt, M)
        for γp, M in zip(γp_list, M_list)
    ]
    return block_diag(*blocks)

def build_Q_matrix(
    γp_list: np.ndarray,
    γa: float,
    σp_list: np.ndarray,
    h2: float,
    σeps: float,
    hellings_downs_matrix: np.ndarray,
    M_list: np.ndarray,
    dt: float
) -> np.ndarray:
    """Build the complete process noise covariance matrix for all pulsars.
    
    Parameters
    ----------
    γp_list : np.ndarray
        Array of spin noise damping rates for each pulsar
    γa : float
        GW damping rate
    σp_list : np.ndarray
        Array of spin noise amplitudes for each pulsar
    h2 : float
        Mean square GW strain
    σeps : float
        Timing model parameter noise amplitude
    hellings_downs_matrix : np.ndarray
        Matrix of Hellings-Downs correlations
    M_list : np.ndarray
        Array of number of timing model parameters for each pulsar
    dt : float
        Time step
    
    Returns
    -------
    np.ndarray
        Complete process noise covariance matrix
    """
    Npsr = len(γp_list)
    
    # First build the block diagonal matrix without GW correlations
    σa_auto = np.sqrt((h2/6) * γa)  # Auto-correlation amplitude
    blocks = [
        build_process_noise_blocks(γp, γa, σp, σa_auto, σeps, dt, M)
        for γp, σp, M in zip(γp_list, σp_list, M_list)
    ]
    Q = block_diag(*blocks)
    
    # Now add the GW correlations between different pulsars
    for i in range(Npsr):
        for j in range(i+1, Npsr):
            # Calculate the cross-correlation amplitude
            σa_cross = np.sqrt((h2/6) * γa * hellings_downs_matrix[i,j])
            
            # Build the cross-correlation block
            exp_γa_dt = np.exp(-γa * dt)
            exp_2γa_dt = np.exp(-2 * γa * dt)
            Q_cross = np.array([
                [(dt/γa**2 - 2*(1-exp_γa_dt)/γa**3 + (1-exp_2γa_dt)/(2*γa**3)) * σa_cross**2,
                 ((1-exp_γa_dt)/γa**2 - (1-exp_2γa_dt)/(2*γa**2)) * σa_cross**2],
                [((1-exp_γa_dt)/γa**2 - (1-exp_2γa_dt)/(2*γa**2)) * σa_cross**2,
                 (1-exp_2γa_dt)/(2*γa) * σa_cross**2]
            ])
            
            # Calculate indices for the GW blocks in the full matrix
            i_start = sum(4 + M_list[k] for k in range(i)) + 2  # Skip spin noise block
            j_start = sum(4 + M_list[k] for k in range(j)) + 2
            
            # Insert the cross-correlation blocks
            Q[i_start:i_start+2, j_start:j_start+2] = Q_cross
            Q[j_start:j_start+2, i_start:i_start+2] = Q_cross.T
    
    return Q
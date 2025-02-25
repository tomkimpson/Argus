import numpy as np
from scipy.linalg import block_diag
from typing import Tuple, List, Dict, Any, Optional
from dataclasses import dataclass
import time
from scipy import sparse

@dataclass(frozen=True)
class KalmanState:
    """Immutable container for Kalman filter state"""
    x: np.ndarray  # State vector
    P: np.ndarray  # Covariance matrix

@dataclass
class PartitionedState:
    """Container for partitioned state variables"""
    a: np.ndarray              # Global state (N,)
    P_aa: np.ndarray          # Global covariance (N, N)
    x_list: List[np.ndarray]  # Local states [(d_n,) for each pulsar]
    P_xx_list: List[np.ndarray]  # Local covariances [(d_n, d_n)]
    P_xa_list: List[np.ndarray]  # Cross covariances [(d_n, N)]

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
    """Build the complete process noise covariance matrix for all pulsars."""
    Npsr = len(γp_list)
    
    # First build the block diagonal matrix without GW correlations
    σa_auto_squared = (h2/6) * γa  # Auto-correlation amplitude squared
    blocks = [
        build_process_noise_blocks(γp, γa, σp, np.sqrt(σa_auto_squared), σeps, dt, M)
        for γp, σp, M in zip(γp_list, σp_list, M_list)
    ]
    Q = block_diag(*blocks)
    
    # Now add the GW correlations between different pulsars
    for i in range(Npsr):
        for j in range(i+1, Npsr):
            # Calculate the cross-correlation amplitude squared directly
            σa_cross_squared = (h2/6) * γa * hellings_downs_matrix[i,j]
            
            # Build the cross-correlation block
            exp_γa_dt = np.exp(-γa * dt)
            exp_2γa_dt = np.exp(-2 * γa * dt)
            Q_cross = np.array([
                [(dt/γa**2 - 2*(1-exp_γa_dt)/γa**3 + (1-exp_2γa_dt)/(2*γa**3)) * σa_cross_squared,
                 ((1-exp_γa_dt)/γa**2 - (1-exp_2γa_dt)/(2*γa**2)) * σa_cross_squared],
                [((1-exp_γa_dt)/γa**2 - (1-exp_2γa_dt)/(2*γa**2)) * σa_cross_squared,
                 (1-exp_2γa_dt)/(2*γa) * σa_cross_squared]
            ])
            
            # Calculate indices for the GW blocks in the full matrix
            i_start = sum(4 + M_list[k] for k in range(i)) + 2
            j_start = sum(4 + M_list[k] for k in range(j)) + 2
            
            # Insert the cross-correlation blocks
            Q[i_start:i_start+2, j_start:j_start+2] = Q_cross
            Q[j_start:j_start+2, i_start:i_start+2] = Q_cross.T
    
    return Q

def predict_step(
    state: KalmanState,
    F: np.ndarray,
    Q: np.ndarray
) -> KalmanState:
    """Pure function for Kalman filter prediction step."""
    # x is a column vector (n x 1)
    x = state.x.reshape(-1, 1)
    # F is (n x n), P is (n x n)
    x_pred = F @ x  # Result: (n x 1)
    P_pred = F @ state.P @ F.T + Q  # Result: (n x n)
    
    return KalmanState(x_pred.flatten(), P_pred)

def build_H_matrix(f0: float, M: int, state_size: int, total_state_size: int, state_start: int) -> np.ndarray:
    """Build measurement matrix for a single pulsar."""
    # H is (1 x total_state_size) for scalar measurements
    H = np.zeros((1, total_state_size))
    H[0, state_start] = 1/f0      # δφ term
    H[0, state_start + 2] = -1    # r term
    return H

def update_step(
    state: KalmanState,
    measurement: float,  # Single scalar measurement
    H: np.ndarray,      # (1 x n) measurement matrix
    R: float           # Scalar measurement noise
) -> Tuple[KalmanState, float]:
    """Pure function for Kalman filter update step."""
    # Ensure state vector is column vector (n x 1)
    x = state.x.reshape(-1, 1)
    
    # Innovation (scalar measurement)
    y = measurement - (H @ x).item()  # Scalar
    
    # Innovation covariance (scalar)
    S = (H @ state.P @ H.T + R).item()  # Scalar
    
    # Log-likelihood computation (scalar)
    log_likelihood = -0.5 * (
        np.log(2 * np.pi) +
        np.log(S) +
        (y * y) / S
    )
    
    # Kalman gain (n x 1)
    K = (state.P @ H.T) / S
    
    # Update state and covariance
    x_update = x + K * y
    P_update = state.P - K @ H @ state.P
    
    return KalmanState(x_update.flatten(), P_update), log_likelihood

def evaluate_likelihood(
    params: Dict[str, Any],
    measurements: List[np.ndarray],
    times: List[np.ndarray],
    f0_list: np.ndarray,
    M_list: np.ndarray,
    hellings_downs_matrix: np.ndarray,
    measurement_errors: np.ndarray,
    initial_state: KalmanState = None
) -> float:
    """Evaluate the log-likelihood for given parameters using Kalman filter.
    
    Parameters
    ----------
    params : Dict[str, Any]
        Dictionary containing model parameters
    measurements : List[np.ndarray]
        List of measurements for each pulsar
    times : List[np.ndarray]
        List of observation times for each pulsar (potentially different for each)
    f0_list : np.ndarray
        Array of pulsar frequencies
    M_list : np.ndarray
        Array of number of timing parameters per pulsar
    hellings_downs_matrix : np.ndarray
        Matrix of Hellings-Downs correlations
    measurement_errors : np.ndarray
        Array of measurement errors for each pulsar
    initial_state : KalmanState, optional
        Initial state for the filter. If None, starts with zeros.
    
    Returns
    -------
    float
        Total log-likelihood
    """
    # Unpack parameters
    γa = params["gamma_a"]
    h2 = params["h_a"]**2
    γp_list = params["gamma_p"]
    σp_list = params["sigma_p"]
    σeps = params["sigma_eps"]
    
    Npsr = len(M_list)
    total_state_size = sum(4 + M for M in M_list)
    
    # Initialize state if not provided
    if initial_state is None:
        x0 = np.zeros(total_state_size).reshape(-1, 1)  # Make column vector
        P0 = np.eye(total_state_size) * 1e-6
        initial_state = KalmanState(x0.flatten(), P0)
    
    total_log_likelihood = 0.0
    current_state = initial_state
    
    # Create a merged timeline of all observation times
    all_times = np.unique(np.concatenate(times))
    all_times.sort()
    
    # Create lookup dictionaries for measurements at each time
    measurement_dict = [{} for _ in range(Npsr)]
    for i in range(Npsr):
        for t, m in zip(times[i], measurements[i]):
            measurement_dict[i][t] = m
    
    # Loop over time steps
    for i in range(1, len(all_times)):
        dt = all_times[i] - all_times[i-1]
        
        # Build system matrices for this dt
        F = build_F_matrix(γp_list, γa, M_list, dt)
        Q = build_Q_matrix(γp_list, γa, σp_list, h2, σeps, hellings_downs_matrix, M_list, dt)
        
        # Predict step
        predicted_state = predict_step(current_state, F, Q)
        current_state = predicted_state
        
        # Update step for each pulsar that has a measurement at this time
        t = all_times[i]
        for j in range(Npsr):
            if t in measurement_dict[j]:  # Check if pulsar j has measurement at time t
                # Calculate start index for this pulsar's state
                state_start = sum(4 + M_list[k] for k in range(j))
                state_size = 4 + M_list[j]
                
                # Build H matrix for this pulsar
                H = build_H_matrix(f0_list[j], M_list[j], state_size, total_state_size, state_start)
                
                # Get measurement and its error
                y = measurement_dict[j][t]
                R = measurement_errors[j]**2
                
                # Update state and accumulate likelihood
                current_state, log_like = update_step(predicted_state, y, H, R)
                total_log_likelihood += log_like
    
    return total_log_likelihood

def print_data_stats(
    times: List[np.ndarray],
    M_list: np.ndarray,
    verbose: bool = True
) -> Dict[str, float]:
    """Print statistics about the dataset."""
    stats = {}
    
    # Basic counts
    stats['n_pulsars'] = len(times)
    stats['n_observations'] = [len(t) for t in times]
    stats['total_observations'] = sum(stats['n_observations'])
    
    # Timing stats
    stats['timespan'] = [t[-1] - t[0] for t in times]
    stats['mean_cadence'] = [np.mean(np.diff(t)) for t in times]
    stats['min_cadence'] = [np.min(np.diff(t)) for t in times]
    stats['max_cadence'] = [np.max(np.diff(t)) for t in times]
    
    # State space stats
    stats['state_size_per_pulsar'] = [4 + m for m in M_list]  # 4 = 2(spin) + 2(GW)
    stats['total_state_size'] = sum(stats['state_size_per_pulsar'])
    
    if verbose:
        print("\nDataset Statistics:")
        print(f"Number of pulsars: {stats['n_pulsars']}")
        print(f"Observations per pulsar: {stats['n_observations']}")
        print(f"Total observations: {stats['total_observations']}")
        print(f"\nTimespans (years):")
        for i, span in enumerate(stats['timespan']):
            print(f"  Pulsar {i}: {span:.2f}")
        print(f"\nMean cadence (days):")
        for i, cad in enumerate(stats['mean_cadence']):
            print(f"  Pulsar {i}: {cad*365.25:.2f}")
        print(f"\nState dimensions:")
        print(f"  Per pulsar: {stats['state_size_per_pulsar']}")
        print(f"  Total: {stats['total_state_size']}")
    
    return stats

def time_likelihood_evaluation(
    params: Dict[str, Any],
    measurements: List[np.ndarray],
    times: List[np.ndarray],
    f0_list: np.ndarray,
    M_list: np.ndarray,
    hellings_downs_matrix: np.ndarray,
    measurement_errors: np.ndarray,
    initial_state: Optional[KalmanState] = None,
    verbose: bool = True
) -> Tuple[float, Dict[str, float]]:
    """Time the different components of likelihood evaluation."""
    print("\nStarting likelihood evaluation...")
    timing = {}
    
    print("Computing data statistics...")
    t0 = time.time()
    stats = print_data_stats(times, M_list, verbose=verbose)
    timing['stats_computation'] = time.time() - t0
    
    print("Merging timelines...")
    t0 = time.time()
    all_times = np.unique(np.concatenate(times))
    all_times.sort()
    timing['timeline_merge'] = time.time() - t0
    print(f"Total number of unique timestamps: {len(all_times)}")
    
    print("Building measurement dictionary...")
    t0 = time.time()
    measurement_dict = [{} for _ in range(len(M_list))]
    for i in range(len(M_list)):
        for t, m in zip(times[i], measurements[i]):
            measurement_dict[i][t] = m
    timing['dict_creation'] = time.time() - t0
    
    print("Initializing state...")
    t0 = time.time()
    current_state = initial_state or KalmanState(
        np.zeros(sum(4 + M for M in M_list)),
        np.eye(sum(4 + M for M in M_list)) * 1e-6
    )
    timing['state_initialization'] = time.time() - t0
    
    # Initialize timing counters
    timing['F_matrix_total'] = 0.0
    timing['Q_matrix_total'] = 0.0
    timing['prediction_total'] = 0.0
    timing['H_matrix_total'] = 0.0
    timing['update_total'] = 0.0
    
    print("\nStarting main Kalman filter loop...")
    t_start = time.time()
    total_log_likelihood = 0.0
    n_predictions = 0
    n_updates = 0
    
    # Progress tracking
    n_steps = len(all_times) - 1
    progress_interval = max(1, n_steps // 20)  # Show progress ~20 times
    
    for i in range(1, len(all_times)):
        if i % progress_interval == 0:
            print(f"Progress: {i}/{n_steps} steps ({100*i/n_steps:.1f}%)")
            print(f"Current processing rate: {i/((time.time() - t_start) or 1e-6):.1f} steps/second")
        
        dt = all_times[i] - all_times[i-1]
        
        # Matrix construction and prediction
        F = build_F_matrix(params["gamma_p"], params["gamma_a"], M_list, dt)
        Q = build_Q_matrix(
            params["gamma_p"], params["gamma_a"], 
            params["sigma_p"], params["h_a"]**2,
            params["sigma_eps"], hellings_downs_matrix,
            M_list, dt
        )
        predicted_state = predict_step(current_state, F, Q)
        current_state = predicted_state
        n_predictions += 1
        
        # Updates
        t = all_times[i]
        updates_this_step = 0
        for j in range(len(M_list)):
            if t in measurement_dict[j]:
                state_start = sum(4 + M_list[k] for k in range(j))
                state_size = 4 + M_list[j]
                H = build_H_matrix(f0_list[j], M_list[j], state_size, sum(4 + M for M in M_list), state_start)
                y = measurement_dict[j][t]
                R = measurement_errors[j]**2
                current_state, log_like = update_step(predicted_state, y, H, R)
                total_log_likelihood += log_like
                n_updates += 1
                updates_this_step += 1
        
        if updates_this_step > 0 and i % progress_interval == 0:
            print(f"  Processed {updates_this_step} measurements at t = {t:.2f} years")
    
    timing['total'] = time.time() - t_start
    print("\nKalman filter loop completed!")
    
    if verbose:
        print("\nDetailed Performance Metrics:")
        print(f"Total time: {timing['total']*1000:.1f} ms")
        print("\nSetup costs:")
        print(f"  Timeline merge: {timing['timeline_merge']*1000:.1f} ms")
        print(f"  Dictionary creation: {timing['dict_creation']*1000:.1f} ms")
        print(f"  State initialization: {timing['state_initialization']*1000:.1f} ms")
        print("\nPer-step operations:")
        print(f"  F matrix construction: {timing['F_matrix_total']*1000:.1f} ms total, {timing['F_matrix_total']*1000/n_predictions:.3f} ms per step")
        print(f"  Q matrix construction: {timing['Q_matrix_total']*1000:.1f} ms total, {timing['Q_matrix_total']*1000/n_predictions:.3f} ms per step")
        print(f"  Prediction steps: {timing['prediction_total']*1000:.1f} ms total, {timing['prediction_total']*1000/n_predictions:.3f} ms per step")
        print(f"  H matrix construction: {timing['H_matrix_total']*1000:.1f} ms total, {timing['H_matrix_total']*1000/n_updates:.3f} ms per step")
        print(f"  Update steps: {timing['update_total']*1000:.1f} ms total, {timing['update_total']*1000/n_updates:.3f} ms per step")
        print(f"\nOperation counts:")
        print(f"  Number of predictions: {n_predictions}")
        print(f"  Number of updates: {n_updates}")
        print(f"  Updates per prediction: {n_updates/n_predictions:.1f}")
        print(f"\nProcessing rate: {stats['total_observations']/timing['total']:.1f} observations/second")
    
    return total_log_likelihood, timing

def hellings_downs(theta: float) -> float:
    """Compute Hellings-Downs correlation for angle theta (in radians)."""
    if theta == 0:
        return 1.0
    x = (1 - np.cos(theta)) / 2
    return 3 * x * np.log(x) - x/4 + 1/2

def precompute_indices_restructured(M_list: np.ndarray) -> Dict[str, np.ndarray]:
    """Precompute indices for restructured state space.
    New structure:
    [a1, a2, ..., aN, δφ1, δf1, r1, δε1, δφ2, δf2, r2, δε2, ...]
    """
    Npsr = len(M_list)
    
    # First N states are 'a' terms
    gw_amp_indices = np.arange(Npsr)
    
    # Rest are organized by pulsar
    state_sizes = 3 + M_list  # 3 = (δφ, δf, r)
    uncorr_start = Npsr  # Start after all 'a' terms
    start_idx = uncorr_start + np.cumsum([0] + list(state_sizes[:-1]))
    
    indices = {
        'gw_amplitudes': gw_amp_indices,
        'starts': start_idx,
        'spin_blocks': [(i, i+2) for i in start_idx],
        'r_indices': [i+2 for i in start_idx],
        'param_blocks': [(i+3, i+3+M) for i, M in zip(start_idx, M_list)]
    }
    return indices

def build_F_restructured(
    γp_list: np.ndarray,
    γa: float,
    M_list: np.ndarray,
    dt: float,
    indices: Dict[str, np.ndarray]
) -> np.ndarray:
    """Build F matrix for restructured state space."""
    Npsr = len(M_list)
    total_size = Npsr + sum(3 + M for M in M_list)
    F = np.eye(total_size)
    
    # GW amplitude evolution
    F[:Npsr, :Npsr] = np.eye(Npsr) * np.exp(-γa * dt)
    
    # Evolution for each pulsar's uncorrelated states
    for i in range(Npsr):
        # Spin noise block
        spin_start, spin_end = indices['spin_blocks'][i]
        F[spin_start:spin_end, spin_start:spin_end] = np.array([
            [1.0, (1 - np.exp(-γp_list[i] * dt))/γp_list[i]],
            [0.0, np.exp(-γp_list[i] * dt)]
        ])
        
        # Connect 'a' term to 'r' term
        r_idx = indices['r_indices'][i]
        a_idx = indices['gw_amplitudes'][i]
        F[r_idx, a_idx] = (1 - np.exp(-γa * dt))/γa
    
    return F

def build_Q_restructured(
    γa: float,
    h2: float,
    hellings_downs_matrix: np.ndarray,
    γp_list: np.ndarray,
    σp_list: np.ndarray,
    σeps: float,
    M_list: np.ndarray,
    dt: float,
    indices: Dict[str, np.ndarray]
) -> np.ndarray:
    """Build Q matrix for restructured state space."""
    Npsr = len(M_list)
    total_size = Npsr + sum(3 + M for M in M_list)
    Q = np.zeros((total_size, total_size))
    
    # GW amplitude correlations
    σa_squared = (h2/6) * γa
    exp_2γa_dt = np.exp(-2 * γa * dt)
    Q[:Npsr, :Npsr] = σa_squared * (1 - exp_2γa_dt) * hellings_downs_matrix
    
    # Uncorrelated blocks for each pulsar
    for i in range(Npsr):
        # Spin noise
        spin_start, spin_end = indices['spin_blocks'][i]
        exp_γp_dt = np.exp(-γp_list[i] * dt)
        exp_2γp_dt = np.exp(-2 * γp_list[i] * dt)
        
        Q[spin_start:spin_end, spin_start:spin_end] = np.array([
            [(dt/γp_list[i]**2 - 2*(1-exp_γp_dt)/γp_list[i]**3 + 
              (1-exp_2γp_dt)/(2*γp_list[i]**3)) * σp_list[i]**2,
             ((1-exp_γp_dt)/γp_list[i]**2 - 
              (1-exp_2γp_dt)/(2*γp_list[i]**2)) * σp_list[i]**2],
            [((1-exp_γp_dt)/γp_list[i]**2 - 
              (1-exp_2γp_dt)/(2*γp_list[i]**2)) * σp_list[i]**2,
             (1-exp_2γp_dt)/(2*γp_list[i]) * σp_list[i]**2]
        ])
        
        # Timing parameters
        param_start, param_end = indices['param_blocks'][i]
        Q[param_start:param_end, param_start:param_end] = σeps**2 * dt * np.eye(M_list[i])
    
    return Q

def build_local_transition(γp: float, dt: float, M: int) -> np.ndarray:
    """Build local state transition matrix F_n."""
    # Size is 3+M: [δφ, δf, r, δε(M)]
    F_local = np.eye(3 + M)
    # δφ evolution
    F_local[0, 1] = (1 - np.exp(-γp * dt))/γp
    # δf evolution
    F_local[1, 1] = np.exp(-γp * dt)
    return F_local

def build_global_coupling(γa: float, dt: float, n_local: int) -> np.ndarray:
    """Build coupling matrix G_n from global to local states."""
    # Only r depends on a
    G = np.zeros((n_local, 1))
    G[2] = (1 - np.exp(-γa * dt))/γa  # r evolution
    return G

def predict_partitioned_step(
    state: PartitionedState,
    γa: float,
    γp_list: List[float],
    h2: float,
    σp_list: List[float],
    σeps: float,
    hellings_downs_matrix: np.ndarray,
    M_list: List[int],
    dt: float
) -> PartitionedState:
    """Prediction step for partitioned Kalman filter.
    
    The state evolution follows:
    da/dt = -γa*a + χa(t)  [correlated GW amplitudes]
    dδφ/dt = δf           [phase]
    dδf/dt = -γp*δf + χp(t) [frequency]
    dr/dt = a             [GW response]
    dδε/dt = χε(t)        [timing parameters]
    """
    Npsr = len(state.x_list)
    
    # 1. Global state prediction (GW amplitudes)
    # F_a = exp(-γa*dt)
    F_a = np.exp(-γa * dt) * np.eye(Npsr)
    a_pred = F_a @ state.a
    
    # Q_aa from correlated GW noise
    σa_squared = (h2/6) * γa
    exp_2γa_dt = np.exp(-2 * γa * dt)
    Q_aa = σa_squared * (1 - exp_2γa_dt) * hellings_downs_matrix
    P_aa_pred = F_a @ state.P_aa @ F_a.T + Q_aa
    
    # 2. Local state predictions
    x_pred_list = []
    P_xx_pred_list = []
    P_xa_pred_list = []
    
    for n in range(Npsr):
        # Get local state size
        M = M_list[n]
        
        # Build local transition matrix
        F_local = np.eye(3 + M)  # [δφ, δf, r, δε]
        F_local[0, 1] = (1 - np.exp(-γp_list[n] * dt))/γp_list[n]  # δφ evolution
        F_local[1, 1] = np.exp(-γp_list[n] * dt)  # δf evolution
        
        # Build coupling matrix (how r depends on a)
        G = np.zeros((3 + M, 1))
        G[2] = (1 - np.exp(-γa * dt))/γa  # r evolution
        
        # Local state prediction
        x_pred = (F_local @ state.x_list[n] + 
                 G @ state.a[n:n+1])
        
        # Build local process noise
        Q_local = np.zeros((3 + M, 3 + M))
        
        # Spin noise block
        exp_γp_dt = np.exp(-γp_list[n] * dt)
        exp_2γp_dt = np.exp(-2 * γp_list[n] * dt)
        Q_spin = np.array([
            [(dt/γp_list[n]**2 - 2*(1-exp_γp_dt)/γp_list[n]**3 + 
              (1-exp_2γp_dt)/(2*γp_list[n]**3)) * σp_list[n]**2,
             ((1-exp_γp_dt)/γp_list[n]**2 - 
              (1-exp_2γp_dt)/(2*γp_list[n]**2)) * σp_list[n]**2],
            [((1-exp_γp_dt)/γp_list[n]**2 - 
              (1-exp_2γp_dt)/(2*γp_list[n]**2)) * σp_list[n]**2,
             (1-exp_2γp_dt)/(2*γp_list[n]) * σp_list[n]**2]
        ])
        Q_local[:2, :2] = Q_spin
        
        # Timing parameter noise
        Q_local[3:, 3:] = σeps**2 * dt * np.eye(M)
        
        # Local covariance prediction
        P_xx_pred = (F_local @ state.P_xx_list[n] @ F_local.T + 
                    G @ state.P_aa[n:n+1, n:n+1] @ G.T +
                    Q_local)
        
        # Cross-covariance prediction
        P_xa_pred = (F_local @ state.P_xa_list[n] @ F_a.T + 
                    G @ state.P_aa[n:n+1, :])
        
        x_pred_list.append(x_pred)
        P_xx_pred_list.append(P_xx_pred)
        P_xa_pred_list.append(P_xa_pred)
    
    return PartitionedState(
        a_pred, P_aa_pred,
        x_pred_list, P_xx_pred_list, P_xa_pred_list
    )

def update_partitioned(
    state: PartitionedState,
    n_meas: int,
    y_obs: float,
    R_obs: float,
    f0: float,
    M_vec: np.ndarray
) -> Tuple[PartitionedState, float]:
    """Update step for partitioned Kalman filter."""
    # Build measurement matrix for pulsar n_meas
    d_n = len(state.x_list[n_meas])
    H_n = np.zeros(d_n)
    H_n[0] = 1/f0  # δφ term
    H_n[2] = -1    # r term
    H_n[3:] = M_vec  # timing parameters
    
    # Innovation
    y_pred = (H_n @ state.x_list[n_meas])
    v = y_obs - y_pred
    
    # Innovation covariance
    S = (H_n @ state.P_xx_list[n_meas] @ H_n + 
         R_obs)
    
    # Kalman gains
    K_n = state.P_xx_list[n_meas] @ H_n / S
    K_a = state.P_xa_list[n_meas].T @ H_n / S
    
    # State updates
    x_new = state.x_list[n_meas] + K_n * v
    a_new = state.a + K_a * v
    
    # Covariance updates
    I_n = np.eye(d_n)
    I_N = np.eye(len(state.a))
    
    P_xx_new = (I_n - np.outer(K_n, H_n)) @ state.P_xx_list[n_meas]
    P_aa_new = state.P_aa - np.outer(K_a, H_n @ state.P_xa_list[n_meas])
    P_xa_new = state.P_xa_list[n_meas] - np.outer(K_n, H_n @ state.P_xa_list[n_meas])
    
    # Only pulsar n_meas is updated
    x_list_new = state.x_list.copy()
    P_xx_list_new = state.P_xx_list.copy()
    P_xa_list_new = state.P_xa_list.copy()
    
    x_list_new[n_meas] = x_new
    P_xx_list_new[n_meas] = P_xx_new
    P_xa_list_new[n_meas] = P_xa_new
    
    # Log-likelihood contribution
    log_like = -0.5 * (np.log(2*np.pi) + np.log(S) + v**2/S)
    
    return PartitionedState(
        a_new, P_aa_new,
        x_list_new, P_xx_list_new, P_xa_list_new
    ), log_like

def evaluate_likelihood_partitioned(
    params: Dict[str, Any],
    measurements: List[np.ndarray],
    times: List[np.ndarray],
    f0_list: np.ndarray,
    M_list: List[int],
    hellings_downs_matrix: np.ndarray,
    measurement_errors: np.ndarray,
    initial_state: Optional[PartitionedState] = None,
    verbose: bool = True
) -> Tuple[float, Dict[str, float]]:
    """Evaluate likelihood using partitioned Kalman filter."""
    print("\nStarting partitioned likelihood evaluation...")
    timing = {}
    
    # Get data statistics
    t0 = time.time()
    stats = print_data_stats(times, M_list, verbose=verbose)
    timing['stats'] = time.time() - t0
    
    # Merge timelines
    print("Merging timelines...")
    t0 = time.time()
    all_times = np.unique(np.concatenate(times))
    all_times.sort()
    timing['timeline'] = time.time() - t0
    print(f"Total timestamps: {len(all_times)}")
    
    # Create measurement lookup
    print("Building measurement dictionary...")
    t0 = time.time()
    measurement_dict = [{} for _ in range(len(M_list))]
    for i in range(len(M_list)):
        for t, m in zip(times[i], measurements[i]):
            measurement_dict[i][t] = m
    timing['dict_creation'] = time.time() - t0
    
    # Initialize state if not provided
    print("Initializing state...")
    t0 = time.time()
    Npsr = len(M_list)
    if initial_state is None:
        # Initialize global state
        a0 = np.zeros(Npsr)
        P_aa0 = np.eye(Npsr) * 1e-6
        
        # Initialize local states
        x_list0 = [np.zeros(3 + M) for M in M_list]  # [δφ, δf, r, δε(M)]
        P_xx_list0 = [np.eye(3 + M) * 1e-6 for M in M_list]
        P_xa_list0 = [np.zeros((3 + M, Npsr)) for M in M_list]
        
        initial_state = PartitionedState(a0, P_aa0, x_list0, P_xx_list0, P_xa_list0)
    timing['initialization'] = time.time() - t0
    
    # Main filter loop
    print("\nStarting main Kalman filter loop...")
    t_start = time.time()
    current_state = initial_state
    total_log_likelihood = 0.0
    n_predictions = 0
    n_updates = 0
    
    # Progress tracking
    n_steps = len(all_times) - 1
    progress_interval = max(1, n_steps // 20)
    
    timing['predict'] = 0.0
    timing['update'] = 0.0
    
    for i in range(1, len(all_times)):
        if i % progress_interval == 0:
            print(f"Progress: {i}/{n_steps} steps ({100*i/n_steps:.1f}%)")
            rate = i/((time.time() - t_start) or 1e-6)
            print(f"Processing rate: {rate:.1f} steps/second")
        
        dt = all_times[i] - all_times[i-1]
        
        # Prediction step
        t0 = time.time()
        current_state = predict_partitioned_step(
            current_state,
            params["gamma_a"],
            params["gamma_p"],
            params["h_a"]**2,
            params["sigma_p"],
            params["sigma_eps"],
            hellings_downs_matrix,
            M_list,
            dt
        )
        timing['predict'] += time.time() - t0
        n_predictions += 1
        
        # Update steps
        t = all_times[i]
        updates_this_step = 0
        t0 = time.time()
        
        for j in range(Npsr):
            if t in measurement_dict[j]:
                # Get measurement and noise
                y = measurement_dict[j][t]
                R = measurement_errors[j]**2
                
                # Update state
                current_state, log_like = update_partitioned(
                    current_state,
                    j,  # pulsar index
                    y,  # measurement
                    R,  # measurement noise
                    f0_list[j],  # pulsar frequency
                    np.ones(M_list[j])  # measurement matrix for timing params
                )
                
                total_log_likelihood += log_like
                n_updates += 1
                updates_this_step += 1
        
        timing['update'] += time.time() - t0
        
        if updates_this_step > 0 and i % progress_interval == 0:
            print(f"  Processed {updates_this_step} measurements at t = {t:.2f} years")
    
    timing['total'] = time.time() - t_start
    print("\nKalman filter loop completed!")
    
    if verbose:
        print("\nPerformance Metrics:")
        print(f"Total time: {timing['total']*1000:.1f} ms")
        print(f"  Prediction time: {timing['predict']*1000:.1f} ms ({timing['predict']/timing['total']*100:.1f}%)")
        print(f"  Update time: {timing['update']*1000:.1f} ms ({timing['update']/timing['total']*100:.1f}%)")
        print(f"\nOperation counts:")
        print(f"  Predictions: {n_predictions}")
        print(f"  Updates: {n_updates}")
        print(f"  Updates per prediction: {n_updates/n_predictions:.1f}")
        print(f"\nProcessing rate: {stats['total_observations']/timing['total']:.1f} obs/second")
    
    return total_log_likelihood, timing

# Example usage:
if __name__ == "__main__":
    # More realistic parameters
    Npsr = 50  # Number of pulsars
    years = 10  # Total timespan
    nominal_cadence = 7/365.25  # ~weekly observations in years
    
    # Random parameters for each pulsar
    γp_list = np.random.uniform(0.1, 0.3, Npsr)  # Spin noise damping rates
    γa = 0.05  # GW damping rate
    σp_list = np.random.uniform(1e-9, 3e-9, Npsr)  # Spin noise amplitudes
    h2 = 1e-30  # Mean square GW strain
    σeps = 1e-12  # Timing model parameter noise
    M_list = np.random.randint(2, 5, Npsr)  # Random number of timing parameters
    f0_list = np.random.uniform(100.0, 300.0, Npsr)  # Frequencies in Hz
    measurement_errors = np.random.uniform(1e-7, 2e-7, Npsr)  # Measurement errors
    
    # Generate Hellings-Downs matrix from random sky positions
    # First generate random positions on the sky
    phi = np.random.uniform(0, 2*np.pi, Npsr)   # Right ascension
    cos_theta = np.random.uniform(-1, 1, Npsr)  # Cos of declination
    theta = np.arccos(cos_theta)                # Declination

    # Compute angular separations and correlations
    hellings_downs_matrix = np.zeros((Npsr, Npsr))
    for i in range(Npsr):
        for j in range(i, Npsr):
            if i == j:
                hellings_downs_matrix[i,j] = 1.0
            else:
                # Compute angular separation between pulsars i and j
                cos_ang = (np.sin(theta[i]) * np.sin(theta[j]) * 
                          np.cos(phi[i] - phi[j]) + 
                          np.cos(theta[i]) * np.cos(theta[j]))
                # Ensure cos_ang is in [-1, 1] to avoid numerical errors
                cos_ang = np.clip(cos_ang, -1.0, 1.0)
                ang = np.arccos(cos_ang)
                
                # Compute correlation
                hd = hellings_downs(ang)
                hellings_downs_matrix[i,j] = hd
                hellings_downs_matrix[j,i] = hd
    
    # Generate irregular observation times for each pulsar
    times = []
    measurements = []
    for i in range(Npsr):
        # Base number of observations (weekly for 10 years) with some randomness
        n_obs = int(years / nominal_cadence * (1 + 0.2 * np.random.randn()))
        
        # Generate slightly irregular observation times
        base_times = np.linspace(0, years, n_obs)
        # Add jitter to make observations irregular
        jitter = np.random.uniform(-nominal_cadence/2, nominal_cadence/2, n_obs)
        t = base_times + jitter
        # Sort and remove any negative times or duplicates
        t = np.unique(t[t > 0])
        
        # Generate measurements (simple noise for now)
        m = np.random.normal(0, measurement_errors[i], len(t))
        
        times.append(t)
        measurements.append(m)
    
    # Parameter dictionary
    params = {
        "gamma_a": γa,
        "h_a": np.sqrt(h2),
        "gamma_p": γp_list,
        "sigma_p": σp_list,
        "sigma_eps": σeps
    }
    
    # Time the partitioned likelihood evaluation
    log_like, timing = evaluate_likelihood_partitioned(
        params,
        measurements,
        times,
        f0_list,
        M_list,
        hellings_downs_matrix,
        measurement_errors,
        verbose=True
    )
    
    print("Log-likelihood:", log_like) 
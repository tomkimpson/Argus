#!/usr/bin/env python

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, NamedTuple, Any
import time
from common import Measurement, PulsarState, KalmanState, compute_hellings_downs_matrix
from utils import generate_test_data, print_data_summary, compute_state_dimensions
from tqdm import tqdm

###############################################################################
#               PART 1: Data Structures and Utilities
###############################################################################

class Measurement(NamedTuple):
    """Single measurement from a pulsar."""
    time: float
    value: float
    error: float
    pulsar_idx: int

class PulsarState(NamedTuple):
    """State for a single pulsar."""
    phi: float      # Phase offset
    freq: float     # Frequency offset
    r: float        # GW response
    eps: np.ndarray # Timing parameters

class KalmanState(NamedTuple):
    """Complete state of the system using efficient array operations.
    
    Attributes
    ----------
    a : ndarray, shape (N,)
        GW amplitudes for N pulsars
    
    P_aa : ndarray, shape (N, N)
        GW amplitude covariance matrix
    
    pulsar_states : ndarray, shape (N, max_state_size)
        Combined state array for all pulsars where max_state_size = 3 + max(M_list)
        Each row contains: [phi, freq, r, eps_1, ..., eps_M]
        Note: If a pulsar has fewer than max(M_list) parameters, extra entries are unused
    
    P_xx : ndarray, shape (N, N, max_state_size, max_state_size)
        Pulsar-pulsar covariances
        P_xx[i,j] gives the covariance between pulsar i's state and pulsar j's state
    
    P_ax : ndarray, shape (N, N, max_state_size)
        GW-pulsar cross covariances
        P_ax[i,j] gives covariance between GW amplitude i and pulsar j's state
    
    state_sizes : ndarray, shape (N,)
        Number of state variables for each pulsar (3 + M_i)
    
    state_starts : ndarray, shape (N,)
        Starting index for each pulsar's state in the full state vector
    """
    a: np.ndarray              
    P_aa: np.ndarray          
    pulsar_states: np.ndarray  
    P_xx: np.ndarray          
    P_ax: np.ndarray          
    state_sizes: np.ndarray    
    state_starts: np.ndarray   

###############################################################################
#               PART 1: Utility Functions
###############################################################################

###############################################################################
#               PART 2: Building Model Blocks
###############################################################################

def build_F_a_block(gamma_a: float, dt: float, N: int) -> np.ndarray:
    """Build state transition matrix for the a-block.
    
    Parameters
    ----------
    gamma_a : float
        GW damping rate
    dt : float
        Time step
    N : int
        Number of pulsars
        
    Returns
    -------
    np.ndarray
        NxN state transition matrix for a-block
    """
    return np.exp(-gamma_a*dt)*np.eye(N)

def build_Q_a_block(gamma_a: float, h2: float, dt: float, 
                    hellings_downs_matrix: np.ndarray) -> np.ndarray:
    """Build process noise matrix for the a-block.
    
    Parameters
    ----------
    gamma_a : float
        GW damping rate
    h2 : float
        Mean square GW strain
    dt : float
        Time step
    hellings_downs_matrix : np.ndarray
        Matrix of Hellings-Downs correlations
        
    Returns
    -------
    np.ndarray
        NxN process noise matrix for a-block
    """
    fac_a = (1.0 - np.exp(-2.0*gamma_a*dt))/(2.0*gamma_a)
    return fac_a * (h2/6) * hellings_downs_matrix

def build_F_p_block(gamma_p: float, dt: float, M: int) -> np.ndarray:
    """Build state transition matrix for a single pulsar block.
    
    Parameters
    ----------
    gamma_p : float
        Spin noise damping rate
    dt : float
        Time step
    M : int
        Number of timing model parameters
        
    Returns
    -------
    np.ndarray
        State transition matrix for pulsar block (3+M x 3+M)
    """
    # Size is 3+M: [δφ, δf, r, δε(M)]
    F_p = np.eye(3 + M)  # Changed from 4 to 3+M
    F_p[0,1] = dt  # delta_phi evolution
    F_p[1,1] = np.exp(-gamma_p*dt)  # delta_f evolution
    return F_p

def build_G_block(dt: float, N: int, n: int, M: int) -> np.ndarray:
    """Build coupling matrix for a single pulsar.
    
    Parameters
    ----------
    dt : float
        Time step
    N : int
        Total number of pulsars
    n : int
        Index of current pulsar
    M : int
        Number of timing model parameters
        
    Returns
    -------
    np.ndarray
        Coupling matrix G^(n) (3+M x N)
    """
    # Size is (3+M x N) for pulsar n
    G = np.zeros((3 + M, N))  # Changed dimensions to match F_p
    G[2,n] = dt  # r_{k+1} = r_k + dt*a_k^(n)
    return G

def build_Q_p_block(gamma_p: float, sigma_p: float, 
                    sigma_eps: float, dt: float, M: int) -> np.ndarray:  # Added M parameter
    """Build process noise matrix for a single pulsar block.
    
    Parameters
    ----------
    gamma_p : float
        Spin noise damping rate
    sigma_p : float
        Spin noise amplitude
    sigma_eps : float
        Timing model parameter noise amplitude
    dt : float
        Time step
    M : int
        Number of timing model parameters
        
    Returns
    -------
    np.ndarray
        Process noise matrix for pulsar block (3+M x 3+M)
    """
    Q_p = np.zeros((3 + M, 3 + M))  # Changed from 4x4 to (3+M)x(3+M)
    
    # Spin noise block
    fac_f = (1.0 - np.exp(-2.0*gamma_p*dt))/(2.0*gamma_p)
    Q_p[1,1] = sigma_p**2 * fac_f
    
    # Timing model parameter noise
    Q_p[3:,3:] = sigma_eps**2 * dt * np.eye(M)  # Changed indexing for M parameters
    
    return Q_p

###############################################################################
#   PART 3: Partitioned Predict - Two-Stage Update (a first, then x_p^(n))
###############################################################################

def predict_pulsar_cov(
    n: int,
    m: int,
    P_aa: np.ndarray,        # GW amplitude covariance
    P_xx_old: Dict[Tuple[int,int], np.ndarray],  # Old pulsar-pulsar covariances
    P_ax_list: List[np.ndarray],  # Old a-to-x cross covariances
    P_xa_list: List[np.ndarray],  # Old x-to-a cross covariances
    F_pn: np.ndarray,        # State transition for pulsar n
    F_pm: np.ndarray,        # State transition for pulsar m
    Gn: np.ndarray,          # GW coupling for pulsar n
    Gm: np.ndarray,          # GW coupling for pulsar m
    Qpn: np.ndarray = None   # Process noise (only if n==m)
) -> np.ndarray:
    """Predict covariance block P_{n,m}(k+1) for pulsars n,m."""
    
    # Get old blocks
    P_nm_old = P_xx_old[(n,m)]
    P_na = P_xa_list[n]  # P_{n,a}(k)
    P_am = P_ax_list[m]  # P_{a,m}(k)
    
    # Build new block
    P_nm_new = (F_pn @ P_nm_old @ F_pm.T     # Base evolution
                + F_pn @ P_na @ Gm.T          # Cross with n->a->m
                + Gn @ P_am @ F_pm.T          # Cross with n<-a<-m
                + Gn @ P_aa @ Gm.T)          # Cross through a
    
    # Add process noise only for diagonal blocks
    if n == m and Qpn is not None:
        P_nm_new += Qpn
        
    return P_nm_new

def predict_ax_cov(
    n: int,
    F_a: np.ndarray,         # GW amplitude transition
    P_aa: np.ndarray,        # GW amplitude covariance
    P_ax_old: np.ndarray,    # Old a-to-x cross covariance
    F_pn: np.ndarray,        # State transition for pulsar n
    Gn: np.ndarray,          # GW coupling for pulsar n
) -> np.ndarray:
    """Predict cross covariance P_{a,x^n}(k+1)."""
    
    # P_{a_{k+1}, x_{k+1}^n} = F_a @ P_{a,x^n} @ F_pn.T + F_a @ P_aa @ Gn.T
    P_ax_new = F_a @ P_ax_old @ F_pn.T + F_a @ P_aa @ Gn.T
    return P_ax_new

def predict_step(
    state: KalmanState,
    dt: float,
    params: Dict[str, Any],
    hellings_downs_matrix: np.ndarray
) -> KalmanState:
    """Predict step maintaining separate a and x states using efficient array operations."""
    N = len(state.a)
    max_state_size = state.pulsar_states.shape[1]
    
    # 1. Predict GW amplitudes (a)
    F_a = np.exp(-params["gamma_a"]*dt) * np.eye(N)
    a_pred = F_a @ state.a
    
    Q_a = ((1.0 - np.exp(-2.0*params["gamma_a"]*dt))/(2.0*params["gamma_a"]) * 
           (params["h_a"]**2/6) * hellings_downs_matrix)
    P_aa_pred = F_a @ state.P_aa @ F_a.T + Q_a
    
    # 2. Build transition and noise matrices for all pulsars
    pulsar_states_pred = state.pulsar_states.copy()
    P_xx_pred = state.P_xx.copy()
    P_ax_pred = state.P_ax.copy()
    
    # Process each pulsar
    for n in range(N):
        state_size = state.state_sizes[n]
        gamma_p = params["gamma_p"][n]
        
        # Build F_p block (pad to max_state_size)
        F_p = np.zeros((max_state_size, max_state_size))
        F_p[:state_size, :state_size] = np.eye(state_size)
        F_p[0,1] = dt  # phi evolution
        F_p[1,1] = np.exp(-gamma_p*dt)  # freq evolution
        
        # Build Q_p block (pad to max_state_size)
        Q_p = np.zeros((max_state_size, max_state_size))
        Q_p[1,1] = params["sigma_p"][n]**2 * (1.0 - np.exp(-2.0*gamma_p*dt))/(2.0*gamma_p)
        Q_p[3:state_size,3:state_size] = params["sigma_eps"]**2 * dt * np.eye(state_size - 3)
        
        # Build G block (coupling matrix)
        G = np.zeros((max_state_size, N))
        G[2,n] = dt  # r evolution coupled to a
        
        # Predict pulsar state
        pulsar_states_pred[n,:state_size] = (
            F_p[:state_size,:state_size] @ state.pulsar_states[n,:state_size] + 
            G[:state_size,:] @ state.a
        )
        
        # Update diagonal covariance block
        P_xx_pred[n,n] = (
            F_p @ state.P_xx[n,n] @ F_p.T + 
            F_p @ state.P_ax[n].T @ G.T +
            G @ state.P_ax[n] @ F_p.T +
            G @ state.P_aa @ G.T +
            Q_p
        )
        
        # Update cross covariances
        P_ax_pred[n] = (
            F_a @ state.P_ax[n] @ F_p.T +
            F_a @ state.P_aa @ G.T
        )
        
        # Update off-diagonal blocks
        for m in range(n+1, N):
            state_size_m = state.state_sizes[m]
            
            # Build F_m block (pad to max_state_size)
            F_m = np.zeros((max_state_size, max_state_size))
            F_m[:state_size_m, :state_size_m] = np.eye(state_size_m)
            F_m[0,1] = dt
            F_m[1,1] = np.exp(-params["gamma_p"][m]*dt)
            
            # Build G_m block
            G_m = np.zeros((max_state_size, N))
            G_m[2,m] = dt
            
            P_xx_pred[n,m] = (
                F_p @ state.P_xx[n,m] @ F_m.T +
                F_p @ state.P_ax[m].T @ G_m.T +
                G @ state.P_ax[m] @ F_m.T +
                G @ state.P_aa @ G_m.T
            )
            P_xx_pred[m,n] = P_xx_pred[n,m].T
    
    return KalmanState(
        a=a_pred,
        P_aa=P_aa_pred,
        pulsar_states=pulsar_states_pred,
        P_xx=P_xx_pred,
        P_ax=P_ax_pred,
        state_sizes=state.state_sizes,
        state_starts=state.state_starts
    )


###############################################################################
#               PART 4: A Simple Kalman Update (for measurement)
###############################################################################

def kalman_update_scalar(
    state: KalmanState,
    value: float,
    error: float,
    psr_idx: int,
    f0_list: List[float]
) -> KalmanState:
    """Scalar measurement update handling a and x components separately."""
    # Build measurement matrix for this pulsar
    N = len(state.a)  # Number of pulsars
    state_size = state.state_sizes[psr_idx]  # Size of this pulsar's state
    
    # Build measurement vectors (more efficient than building full H matrix)
    H_a = np.zeros(N)
    H_a[psr_idx] = state.pulsar_states[psr_idx, 2]  # r component
    
    H_x = np.zeros(state.pulsar_states.shape[1])  # max_state_size
    H_x[0] = 1/f0_list[psr_idx]      # phi term
    H_x[2] = -1                       # r term
    H_x[3:state_size] = 1.0          # timing parameters
    
    # Compute predicted measurement (scalar)
    y_pred_a = H_a @ state.a
    y_pred_x = H_x[:state_size] @ state.pulsar_states[psr_idx, :state_size]
    y_pred = y_pred_a + y_pred_x
    
    # Innovation (scalar)
    inn = value - y_pred
    
    # Innovation covariance (scalar)
    S = error**2 + 1e-10
    S += float(H_a @ state.P_aa @ H_a)  # a contribution
    S += float(H_x[:state_size] @ state.P_xx[psr_idx,psr_idx,:state_size,:state_size] @ H_x[:state_size])  # x contribution
    S += 2 * float(H_a @ state.P_ax[:,psr_idx,:state_size] @ H_x[:state_size])  # cross terms
    
    # Kalman gains
    K_a = (state.P_aa @ H_a + state.P_ax[:,psr_idx,:state_size] @ H_x[:state_size]) / S  # Shape (N,)
    K_x = (state.P_xx[psr_idx,psr_idx,:state_size,:state_size] @ H_x[:state_size] + 
           state.P_ax[:,psr_idx,:state_size].T @ H_a) / S  # Shape (state_size,)
    
    # Update means
    a_new = state.a + K_a * inn
    pulsar_states_new = state.pulsar_states.copy()
    pulsar_states_new[psr_idx,:state_size] += K_x * inn
    
    # Update covariances
    # P_aa update
    P_aa_new = state.P_aa - np.outer(K_a, H_a @ state.P_aa)
    
    # P_xx update (only the block for measured pulsar)
    P_xx_new = state.P_xx.copy()
    P_xx_new[psr_idx,psr_idx,:state_size,:state_size] -= (
        np.outer(K_x, H_x[:state_size] @ state.P_xx[psr_idx,psr_idx,:state_size,:state_size])
    )
    
    # P_ax update
    P_ax_new = state.P_ax.copy()
    P_ax_new[:,psr_idx,:state_size] -= np.outer(K_a, H_x[:state_size] @ state.P_xx[psr_idx,psr_idx,:state_size,:state_size])
    
    return KalmanState(
        a=a_new,
        P_aa=P_aa_new,
        pulsar_states=pulsar_states_new,
        P_xx=P_xx_new,
        P_ax=P_ax_new,
        state_sizes=state.state_sizes,
        state_starts=state.state_starts
    )


###############################################################################
#               PART 5: DEMO / EXAMPLE
###############################################################################

def run_kalman_filter(
    dt_values: np.ndarray,       # Pre-computed time differences
    values: np.ndarray,          # Measurement values
    errors: np.ndarray,          # Measurement errors
    psr_indices: np.ndarray,     # Pulsar indices for each measurement
    params: Dict[str, Any],
    initial_state: KalmanState,
    f0_list: List[float],        # Pass frequencies directly
    hellings_downs_matrix: np.ndarray,
    verbose: bool = True
) -> Tuple[List[KalmanState], float]:
    """Run complete Kalman filter over all data."""
    # Initialize storage
    states = [initial_state]
    total_log_likelihood = 0.0
    current_state = initial_state
    
    # Main Kalman filter loop with tqdm progress bar
    pbar = tqdm(range(len(dt_values)), disable=not verbose)
    
    for i in pbar:
        # Predict step
        current_state = predict_step(current_state, dt_values[i], params, hellings_downs_matrix)
        
        # Update step (measurement matrix built inside)
        current_state = kalman_update_scalar(
            current_state, 
            values[i+1], 
            errors[i+1], 
            psr_indices[i+1],
            f0_list
        )
        
        # Update likelihood
        total_log_likelihood += 1  # Placeholder for now
        states.append(current_state)
    
    if verbose:
        total_time = pbar.format_dict["elapsed"]
        iterations_per_sec = len(dt_values) / total_time
        print(f"\nTotal runtime: {total_time:.2f} seconds")
        print(f"Iterations per second: {iterations_per_sec:.1f}")
    
    return states, float(total_log_likelihood)


if __name__ == "__main__":
    # Generate test data
    Npsr = 50
    years = 10
    cadence = 90/365.25
    
    measurements, params, M_list, f0_list, hellings_downs_matrix = generate_test_data(
        Npsr=Npsr,
        years=years,
        nominal_cadence=cadence
    )
    
    # Compute state dimensions
    max_state_size = 3 + max(M_list)  # Maximum size of any pulsar's state
    state_sizes = np.array([3 + M for M in M_list])  # Size of each pulsar's state
    state_starts = np.array([Npsr + sum(state_sizes[:i]) for i in range(Npsr)])
    
    # Initialize state arrays
    pulsar_states = np.zeros((Npsr, max_state_size))  # All states start at 0
    
    # Initialize covariance arrays
    P_aa = 0.01 * np.eye(Npsr)  # GW amplitude covariances
    
    # Initialize pulsar-pulsar covariances (4D array)
    P_xx = np.zeros((Npsr, Npsr, max_state_size, max_state_size))
    for i in range(Npsr):
        # Diagonal blocks have small initial uncertainty
        P_xx[i,i,:state_sizes[i],:state_sizes[i]] = 0.01 * np.eye(state_sizes[i])
    
    # Initialize GW-pulsar cross covariances (3D array)
    P_ax = np.zeros((Npsr, Npsr, max_state_size))
    
    # Create initial state
    initial_state = KalmanState(
        a=np.zeros(Npsr),
        P_aa=P_aa,
        pulsar_states=pulsar_states,
        P_xx=P_xx,
        P_ax=P_ax,
        state_sizes=state_sizes,
        state_starts=state_starts
    )
    
    # Print data summary
    print_data_summary(
        Npsr=Npsr,
        years=years,
        cadence=cadence,
        measurements=measurements,
        params=params,
        M_list=M_list,
        f0_list=f0_list,
        errors=[m.error for m in measurements[:Npsr]],
        total_state_size=sum(state_sizes)
    )
    
    # Extract and pre-compute arrays from measurements
    times = np.array([m.time for m in measurements])
    dt_values = np.diff(times)  # Pre-compute time differences
    values = np.array([m.value for m in measurements])
    errors = np.array([m.error for m in measurements])
    psr_indices = np.array([m.pulsar_idx for m in measurements])
    
    # Run filter with f0_list instead of H_matrices
    states, log_like = run_kalman_filter(
        dt_values=dt_values,
        values=values,
        errors=errors,
        psr_indices=psr_indices,
        params=params,
        initial_state=initial_state,
        f0_list=f0_list,  # Pass frequencies instead of H_matrices
        hellings_downs_matrix=hellings_downs_matrix
    )
    
    # Print summary statistics
    final_state = states[-1]
    print("\nFinal state summary:")
    print("GW amplitudes:", np.array2string(final_state.a, precision=3))
    
    # Compute total state norm across all components
    total_norm = np.sqrt(
        np.sum(final_state.a**2) + 
        sum(ps.phi**2 + ps.freq**2 + ps.r**2 + np.sum(ps.eps**2) 
            for ps in final_state.pulsar_states)
    )
    print(f"State vector norm: {total_norm:.3e}")
    
    # Compute total covariance trace
    total_trace = (
        np.trace(final_state.P_aa) + 
        sum(np.trace(final_state.P_xx[(n,n)]) 
            for n in range(len(final_state.pulsar_states)))
    )
    print(f"Covariance trace: {total_trace:.3e}")

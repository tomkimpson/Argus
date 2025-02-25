#!/usr/bin/env python

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, NamedTuple, Any
import time
from common import Measurement, PulsarState, KalmanState, compute_hellings_downs_matrix
from utils import generate_test_data, print_data_summary

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
    """Complete state of the system."""
    a: np.ndarray              # GW amplitudes (N,)
    P_aa: np.ndarray          # GW amplitude covariance (N,N)
    pulsar_states: List[PulsarState]  # List of N pulsar states
    P_xx: Dict[Tuple[int,int], np.ndarray]  # Pulsar-pulsar covariances
    P_ax: List[np.ndarray]    # GW-pulsar cross covariances

###############################################################################
#               PART 1: Utility Functions
###############################################################################

def compute_hellings_downs_matrix(phi: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Compute Hellings-Downs correlation matrix from pulsar sky positions."""
    Npsr = len(phi)
    matrix = np.zeros((Npsr, Npsr))
    
    for i in range(Npsr):
        for j in range(i, Npsr):
            if i == j:
                matrix[i,j] = 1.0
            else:
                cos_ang = (np.sin(theta[i]) * np.sin(theta[j]) * 
                          np.cos(phi[i] - phi[j]) + 
                          np.cos(theta[i]) * np.cos(theta[j]))
                cos_ang = np.clip(cos_ang, -1.0, 1.0)
                ang = np.arccos(cos_ang)
                hd = 3 * (1-np.cos(ang))/2 * np.log((1-np.cos(ang))/2) - (1-np.cos(ang))/4 + 1/2
                matrix[i,j] = hd
                matrix[j,i] = hd
    return matrix

def compute_state_dimensions(N: int, dims_p: List[int]) -> Tuple[List[int], List[int], int]:
    """Compute various state space dimensions.
    
    Returns
    -------
    pulsar_state_dims : List[int]
        Size of each pulsar's state vector (3 + M)
    state_starts : List[int]
        Starting index for each pulsar's state
    total_size : int
        Total size of state vector
    """
    pulsar_state_dims = [3 + M for M in dims_p]
    state_starts = [N + sum(pulsar_state_dims[:i]) for i in range(N)]
    total_size = N + sum(pulsar_state_dims)
    return pulsar_state_dims, state_starts, total_size

def build_measurement_matrices(
    N: int,
    dims_p: List[int],
    f0_list: List[float],
    state_starts: List[int],
    total_size: int
) -> List[np.ndarray]:
    """Build measurement matrices for all pulsars."""
    H_matrices = []
    
    start_idx = N  # Start after GW amplitudes
    for j in range(N):
        H = np.zeros((1, total_size))
        # GW amplitude contribution
        H[0, j] = 0  # Usually zero, set in predict step
        
        # Pulsar state contribution
        state_size = 3 + dims_p[j]
        H[0, state_starts[j]] = 1/f0_list[j]      # phi term
        H[0, state_starts[j] + 2] = -1            # r term
        H[0, state_starts[j] + 3:state_starts[j] + state_size] = 1.0  # timing parameters
        
        start_idx += state_size
        H_matrices.append(H)
    
    return H_matrices

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
    """Predict step maintaining separate a and x states."""
    N = len(state.pulsar_states)
    
    # 1. Predict GW amplitudes
    F_a = np.exp(-params["gamma_a"]*dt) * np.eye(N)
    a_pred = F_a @ state.a
    
    Q_a = ((1.0 - np.exp(-2.0*params["gamma_a"]*dt))/(2.0*params["gamma_a"]) * 
           (params["h_a"]**2/6) * hellings_downs_matrix)
    P_aa_pred = F_a @ state.P_aa @ F_a.T + Q_a
    
    # 2. Build transition matrices for all pulsars
    F_p_list = []
    G_list = []
    Q_p_list = []
    for n in range(N):
        gamma_p = params["gamma_p"][n]
        M = len(state.pulsar_states[n].eps)
        
        F_p_list.append(build_F_p_block(gamma_p, dt, M))
        G_list.append(build_G_block(dt, N, n, M))
        Q_p_list.append(build_Q_p_block(
            gamma_p, params["sigma_p"][n], params["sigma_eps"], dt, M))
    
    # 3. Predict pulsar states
    pulsar_states_pred = []
    for n in range(N):
        ps = state.pulsar_states[n]
        F_pn = F_p_list[n]
        Gn = G_list[n]
        
        # Predict mean
        phi_pred = ps.phi + dt * ps.freq
        freq_pred = np.exp(-params["gamma_p"][n]*dt) * ps.freq
        r_pred = ps.r + dt * state.a[n]
        eps_pred = ps.eps  # Random walk
        
        pulsar_states_pred.append(PulsarState(
            phi_pred, freq_pred, r_pred, eps_pred))
    
    # 4. Predict covariances
    P_xx_pred = {}
    P_ax_pred = []
    P_xa_pred = []
    
    # 4a. Predict pulsar-pulsar covariances
    for n in range(N):
        for m in range(N):
            P_xx_pred[(n,m)] = predict_pulsar_cov(
                n, m,
                state.P_aa,
                state.P_xx,
                state.P_ax,
                [P.T for P in state.P_ax],  # P_xa = P_ax.T
                F_p_list[n],
                F_p_list[m],
                G_list[n],
                G_list[m],
                Q_p_list[n] if n==m else None
            )
    
    # 4b. Predict a-x cross covariances
    for n in range(N):
        P_ax_new = predict_ax_cov(
            n, F_a, state.P_aa, state.P_ax[n],
            F_p_list[n], G_list[n]
        )
        P_ax_pred.append(P_ax_new)
        P_xa_pred.append(P_ax_new.T)
    
    return KalmanState(
        a=a_pred,
        P_aa=P_aa_pred,
        pulsar_states=pulsar_states_pred,
        P_xx=P_xx_pred,
        P_ax=P_ax_pred
    )


###############################################################################
#               PART 4: A Simple Kalman Update (for measurement)
###############################################################################

def merge_state_components(state: KalmanState) -> Tuple[np.ndarray, np.ndarray]:
    """Merge partitioned state into single state vector and covariance matrix."""
    # Merge state vector
    X = np.concatenate([
        state.a,
        *[np.concatenate([
            [ps.phi, ps.freq, ps.r],
            ps.eps
        ]) for ps in state.pulsar_states]
    ])
    
    # Get dimensions
    N = len(state.pulsar_states)
    dims = [3 + len(ps.eps) for ps in state.pulsar_states]
    total_size = state.a.size + sum(dims)
    
    # Build full covariance matrix
    P = np.zeros((total_size, total_size))
    
    # Fill GW amplitude block
    P[:state.a.size, :state.a.size] = state.P_aa
    
    # Fill pulsar-pulsar blocks
    start_idx = state.a.size
    for i in range(N):
        for j in range(N):
            P[start_idx:start_idx + dims[i], 
              state.a.size + sum(dims[:j]):state.a.size + sum(dims[:j+1])] = state.P_xx[(i,j)]
        start_idx += dims[i]
    
    # Fill cross-covariance blocks
    start_idx = state.a.size
    for i in range(N):
        P[:state.a.size, start_idx:start_idx + dims[i]] = state.P_ax[i]
        P[start_idx:start_idx + dims[i], :state.a.size] = state.P_ax[i].T
        start_idx += dims[i]
    
    return X, P

def split_state_components(X: np.ndarray, P: np.ndarray, N: int, dims_p: List[int]) -> KalmanState:
    """Split state vector and covariance matrix back into partitioned components."""
    # Split state vector
    a = X[:N]
    start_idx = N
    pulsar_states = []
    for M in dims_p:
        pulsar_states.append(PulsarState(
            phi=X[start_idx],
            freq=X[start_idx + 1],
            r=X[start_idx + 2],
            eps=X[start_idx + 3:start_idx + 3 + M]
        ))
        start_idx += 3 + M
    
    # Split covariance matrix
    P_aa = P[:N, :N]
    P_xx = {}
    P_ax = []
    
    start_idx = N
    for i, M_i in enumerate(dims_p):
        dim_i = 3 + M_i
        # Extract cross-covariance
        P_ax.append(P[:N, start_idx:start_idx + dim_i])
        
        # Extract pulsar-pulsar blocks
        col_idx = N
        for j, M_j in enumerate(dims_p):
            dim_j = 3 + M_j
            P_xx[(i,j)] = P[start_idx:start_idx + dim_i, 
                           col_idx:col_idx + dim_j]
            col_idx += dim_j
        start_idx += dim_i
    
    return KalmanState(a, P_aa, pulsar_states, P_xx, P_ax)

def kalman_update_scalar(
    state: KalmanState,
    H: np.ndarray,
    R: float,
    y: float
) -> KalmanState:
    """Scalar measurement update returning new state."""
    # Merge state components
    X, P = merge_state_components(state)
    
    # Standard Kalman update
    y_pred = (H @ X).item()
    inn = y - y_pred
    S = (H @ P @ H.T).item() + R + 1e-10
    K = (P @ H.T)/S
    X_upd = X + K.flatten()*inn
    P_upd = P - np.outer(K, H @ P)
    P_upd = (P_upd + P_upd.T)/2
    
    # Split back into components
    return split_state_components(X_upd, P_upd, len(state.pulsar_states), 
                                [len(ps.eps) for ps in state.pulsar_states])


###############################################################################
#               PART 5: DEMO / EXAMPLE
###############################################################################

def run_kalman_filter(
    measurements: List[Measurement],
    params: Dict[str, Any],
    initial_state: KalmanState,
    N: int,
    dims_p: List[int],
    f0_list: List[float],
    hellings_downs_matrix: np.ndarray,
    verbose: bool = True
) -> Tuple[List[KalmanState], float]:
    """Run complete Kalman filter over all data."""
    start_time = time.time()
    
    # Pre-compute dimensions and matrices
    pulsar_state_dims, state_starts, total_size = compute_state_dimensions(N, dims_p)
    H_matrices = build_measurement_matrices(N, dims_p, f0_list, state_starts, total_size)
    
    # Initialize storage
    states = [initial_state]
    total_log_likelihood = 0.0
    
    # Main Kalman filter loop
    current_state = initial_state
    prev_time = measurements[0].time
    
    for i, m in enumerate(measurements[1:], 1):
        if verbose and i % max(1, len(measurements)//10) == 0:
            print(f"Step {i}/{len(measurements)} ({100*i/len(measurements):.1f}%)")
        
        # Predict step
        dt = m.time - prev_time
        current_state = predict_step(
            current_state, dt, params, hellings_downs_matrix
        )
        
        # Update step
        H = H_matrices[m.pulsar_idx]
        current_state = kalman_update_scalar(
            current_state, H, m.error**2, m.value
        )
        
        # Update likelihood
        X, P = merge_state_components(current_state)  # Merge for likelihood calculation
        y_pred = (H @ X).item()
        inn = m.value - y_pred
        S = (H @ P @ H.T).item() + m.error**2 + 1e-10
        total_log_likelihood += -0.5 * (np.log(2*np.pi) + np.log(S) + inn**2/S)
        
        states.append(current_state)
        prev_time = m.time
    
    if verbose:
        elapsed_time = time.time() - start_time
        print(f"\nTotal runtime: {elapsed_time:.2f} seconds")
        print(f"Measurements per second: {len(measurements)/elapsed_time:.1f}")
    
    return states, float(total_log_likelihood)


if __name__ == "__main__":
    # Generate test data
    Npsr = 5
    years = 10
    cadence = 14/365.25  # Bi-weekly observations
    
    measurements, params, M_list, f0_list, hellings_downs_matrix = generate_test_data(
        Npsr=Npsr,
        years=years,
        nominal_cadence=cadence
    )
    
    # Initialize state
    total_state_size = Npsr + sum(3 + M for M in M_list)
    
    # Create initial pulsar states
    initial_pulsar_states = []
    P_xx = {}
    P_ax = []
    
    for n in range(Npsr):
        # Initialize pulsar state
        initial_pulsar_states.append(PulsarState(
            phi=0.0,
            freq=0.0,
            r=0.0,
            eps=np.zeros(M_list[n])
        ))
        
        # Initialize pulsar-pulsar covariances
        for m in range(Npsr):
            P_xx[(n,m)] = 0.01 * np.eye(3 + M_list[n]) if n == m else np.zeros((3 + M_list[n], 3 + M_list[m]))
        
        # Initialize a-x cross covariances
        P_ax.append(np.zeros((Npsr, 3 + M_list[n])))
    
    initial_state = KalmanState(
        a=np.zeros(Npsr),
        P_aa=0.01 * np.eye(Npsr),
        pulsar_states=initial_pulsar_states,
        P_xx=P_xx,
        P_ax=P_ax
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
        total_state_size=total_state_size
    )
    
    # Run complete Kalman filter
    states, log_like = run_kalman_filter(
        measurements, params, initial_state, 
        Npsr, M_list, f0_list, hellings_downs_matrix
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

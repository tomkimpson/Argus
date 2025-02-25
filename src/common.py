import numpy as np
from typing import NamedTuple, List, Dict, Tuple

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
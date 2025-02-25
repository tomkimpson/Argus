import numpy as np
from typing import List, Dict, Tuple, Any
from common import Measurement, compute_hellings_downs_matrix

def generate_test_data(
    Npsr: int = 10,
    years: float = 2,
    nominal_cadence: float = 14/365.25
) -> Tuple[List[Measurement], Dict[str, Any], List[int], List[float], np.ndarray]:
    """Generate synthetic test data for PTA analysis."""
    # Generate parameters
    γp_list = np.random.uniform(0.1, 0.3, Npsr)
    γa = 0.05
    σp_list = np.random.uniform(1e-9, 3e-9, Npsr)
    h2 = 1e-30
    σeps = 1e-12
    M_list = np.random.randint(2, 5, Npsr)
    f0_list = np.random.uniform(100.0, 300.0, Npsr)
    errors = np.random.uniform(1e-7, 2e-7, Npsr)
    
    # Generate sky positions and correlation matrix
    phi = np.random.uniform(0, 2*np.pi, Npsr)
    theta = np.arccos(np.random.uniform(-1, 1, Npsr))
    hellings_downs_matrix = compute_hellings_downs_matrix(phi, theta)
    
    # Generate measurements
    measurements = []
    for i in range(Npsr):
        n_obs = int(years / nominal_cadence * (1 + 0.2 * np.random.randn()))
        base_times = np.linspace(0, years, n_obs)
        jitter = np.random.uniform(-nominal_cadence/2, nominal_cadence/2, n_obs)
        times = base_times + jitter
        times = times[times > 0]
        values = np.random.normal(0, errors[i], len(times))
        
        for t, v in zip(times, values):
            measurements.append(Measurement(t, v, errors[i], i))
    
    measurements.sort(key=lambda m: m.time)
    
    params = {
        "gamma_a": γa,
        "h_a": np.sqrt(h2),
        "gamma_p": γp_list,
        "sigma_p": σp_list,
        "sigma_eps": σeps,
    }
    
    return measurements, params, M_list, f0_list, hellings_downs_matrix

def print_data_summary(
    Npsr: int,
    years: float,
    cadence: float,
    measurements: List[Measurement],
    params: Dict[str, Any],
    M_list: List[int],
    f0_list: List[float],
    errors: List[float],
    total_state_size: int
) -> None:
    """Print summary of PTA data and model parameters."""
    print("\nGenerating synthetic PTA data:")
    print(f"Number of pulsars: {Npsr}")
    print(f"Timespan: {years} years")
    print(f"Nominal cadence: {cadence*365.25:.1f} days")
    
    print("\nData characteristics:")
    print("Pulsar frequencies (Hz):", 
          np.array2string(np.array(f0_list), precision=1))
    print("Timing model parameters per pulsar:", M_list)
    
    # Count observations per pulsar
    obs_per_pulsar = [0] * Npsr
    for m in measurements:
        obs_per_pulsar[m.pulsar_idx] += 1
    print("\nObservations per pulsar:")
    for i, n_obs in enumerate(obs_per_pulsar):
        print(f"  Pulsar {i}: {n_obs} observations")
    print(f"Total observations: {len(measurements)}")
    
    print("\nNoise parameters:")
    print(f"GW damping rate (γa): {params['gamma_a']:.3f}")
    print(f"GW strain amplitude: {params['h_a']:.2e}")
    print("Spin noise damping rates (γp):", 
          np.array2string(params["gamma_p"], precision=3))
    print("Spin noise amplitudes (σp):", 
          np.array2string(params["sigma_p"], precision=3, suppress_small=True))
    print(f"Timing parameter noise (σε): {params['sigma_eps']:.2e}")
    print("Measurement errors:", 
          np.array2string(np.array(errors), precision=3, suppress_small=True))
    
    print(f"\nState space dimensions:")
    print(f"GW amplitudes: {Npsr}")
    print(f"Total state dimension: {total_state_size}")

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
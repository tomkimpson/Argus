import os
import glob
from argus import data_loader, models, jax_kalman_filter, gravitational_waves
import numpy as np
import pandas as pd
import time
from flax import struct
import jax.numpy as jnp
import jax

# Check available devices
devices = jax.devices()
print("\nAvailable devices:", devices)

# Check if GPU is available
gpu_devices = [d for d in devices if d.platform == 'gpu']
if gpu_devices:
    print(f"\nGPU is available! Found {len(gpu_devices)} GPU device(s):")
    for i, d in enumerate(gpu_devices):
        print(f"  GPU {i}: {d}")
else:
    print("\nNo GPU devices found - running on CPU")

# Print default device
default_device = jax.devices()[0]
print(f"Default device: {default_device}\n")

@struct.dataclass
class Parameters:
    γa: float  # s⁻¹
    γp: jnp.ndarray  # Pulsar-specific gamma values
    σp: jnp.ndarray  # Pulsar-specific sigma values
    h2: float  # GWB amplitude
    σeps: jnp.ndarray  # Measurement noise
    f0: jnp.ndarray  # Frequencies (Hz)
    EFAC: jnp.ndarray  # Error factors
    EQUAD: jnp.ndarray  # Extra quadrature noise

def test_filter_run():
    """Test the JAX KalmanFilter class by loading data, initializing the model, setting parameters, and running the filter."""
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the invariant directory path
    data_path = "../data/IPTA_MockDataChallenge/IPTA_Challenge1_open/Challenge_Data/Dataset2/"
    directory = os.path.join(
        script_dir,
        data_path
    )

    # Get all .par and .tim files in the directory
    par_files = sorted(glob.glob(directory + "*.par"))
    tim_files = sorted(glob.glob(directory + "*.tim"))
    assert len(par_files) == len(tim_files), "Mismatch between .par and .tim file counts."

    # Get the data
    print(f"Getting the data. Loading {len(par_files)} pulsars from {data_path}")
    pulsar_residuals, pulsar_metadata, pulsar_design_matrices = (
        data_loader.LoadWidebandPulsarData.read_multiple_par_tim(par_files, tim_files)
    )

    # Get the separation angles and compute HD correlation
    ra = pulsar_metadata["RA"].to_numpy(dtype=float)
    dec = pulsar_metadata["DEC"].to_numpy(dtype=float)
    angular_separation_matrix = data_loader.LoadWidebandPulsarData.pairwise_angular_separation(ra, dec)
    hd_correlation_matrix = gravitational_waves.hellings_downs(angular_separation_matrix)

    # Post-process the residuals
    processed_pulsar_residuals = data_loader.LoadWidebandPulsarData.post_process_residuals(pulsar_residuals)

    print("Total length of the data is ", len(processed_pulsar_residuals))
    print("Total number of pulsars is ", len(pulsar_metadata))

    print("Initializing the model")
    model = models.StochasticGWBackgroundModel(pulsar_metadata, hd_correlation_matrix, pulsar_design_matrices)

    # Initialize the JAX Kalman Filter
    x0 = jnp.zeros(model.nx)
    P0 = jnp.eye(model.nx) * 1e-12

    KF = jax_kalman_filter.JaxScalarKalmanFilter(
        model=model, 
        observations=processed_pulsar_residuals, 
        x0=x0, 
        P0=P0
    )

    # Set global parameters
    params = Parameters(
        γa=1e-1,
        γp=jnp.ones(len(pulsar_metadata)) * 1e-1,
        σp=jnp.ones(len(pulsar_metadata)) * 1e-20,
        h2=1e-12,
        σeps=jnp.ones(model.M_sum) * 1e-20,
        f0=jnp.ones(len(pulsar_metadata)) * 100,
        EFAC=jnp.ones(len(pulsar_metadata)),
        EQUAD=jnp.ones(len(pulsar_metadata))
    )

    print("Running the filter")
    start_time = time.time()
    ll = KF.get_likelihood(params)
    end_time = time.time()
    
    print(f"Log-likelihood: {ll}")
    print(f"Time taken: {end_time - start_time:.4f} seconds")


    print("Running the filter again")
    start_time = time.time()
    ll = KF.get_likelihood(params)
    end_time = time.time()
    
    print(f"Log-likelihood: {ll}")
    print(f"Time taken: {end_time - start_time:.4f} seconds")


if __name__ == "__main__":
    test_filter_run() 
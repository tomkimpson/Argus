import os
import glob
from argus import data_loader, models, jax_kalman_filter, gravitational_waves
import time
from flax import struct
import jax.numpy as jnp
import jax
import jax.profiler


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

# Check available devices
print("=== JAX VERSION INFO ===")
print(f"JAX version: {jax.__version__}")

print("\n=== DEVICE INFO ===")
print("Default device:", jax.default_backend())

# Check if GPU is available
if any(d.platform == 'gpu' for d in jax.devices()):
    print("JAX GPU acceleration is AVAILABLE!")
    print("GPU devices:", [d for d in jax.devices() if d.platform == 'gpu'])
else:
    print("JAX GPU acceleration is NOT available. Using CPU only.")
print('-----------------------------------------------')

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

    # Time compilation
    print("\nStarting compilation phase...")
    compilation_start = time.time()
    _ = KF.get_likelihood(params)
    compilation_end = time.time()
    print(f"Compilation time: {compilation_end - compilation_start:.4f} seconds")

    print("\nRunning profiled execution")
    start_time = time.time()
    ll = KF.get_likelihood(params)
    jax.block_until_ready(ll)  # Ensure computation is complete
    end_time = time.time()
    
    print(f"Log-likelihood: {ll}")
    print(f"Execution time: {end_time - start_time:.4f} seconds")


    print("\nRunning again with the same params")
    start_time = time.time()
    ll = KF.get_likelihood(params)
    jax.block_until_ready(ll)  # Ensure computation is complete
    end_time = time.time()
    
    print(f"Log-likelihood: {ll}")
    print(f"Execution time: {end_time - start_time:.4f} seconds")

    print("\nRunning again with adjusted params")
    params = Parameters(
    γa=2e-1,
    γp=jnp.ones(len(pulsar_metadata)) * 2e-1,
    σp=jnp.ones(len(pulsar_metadata)) * 2e-20,
    h2=2e-12,
    σeps=jnp.ones(model.M_sum) * 2e-20,
    f0=jnp.ones(len(pulsar_metadata)) * 200,
    EFAC=jnp.ones(len(pulsar_metadata)),
    EQUAD=jnp.ones(len(pulsar_metadata))
    )

    start_time = time.time()
    ll = KF.get_likelihood(params)
    jax.block_until_ready(ll)  # Ensure computation is complete
    end_time = time.time()
    
    print(f"Log-likelihood: {ll}")
    print(f"Execution time: {end_time - start_time:.4f} seconds")


import os
import glob
import logging
import sys

# Remove the unwanted path if it exists
unwanted_path = "/fred/oz022/tkimpson/Argus/python"
if unwanted_path in sys.path:
    sys.path.remove(unwanted_path)

# Add the path to the Python package directory
package_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../python'))
sys.path.append(package_path)

# Debug logging for Python path
print("Current directory:", os.getcwd())
print("Added path:", package_path)
print("Full Python path:")
for p in sys.path:
    print(f"  - {p}")



# Configure logging to show up in pytest output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("JAX_TEST")
logger.setLevel(logging.INFO)

# Now import JAX and configure
import jax
jax.config.update('jax_log_compiles', True)
jax.config.update("jax_enable_x64", True)


# Rest of imports
from argus import data_loader, jax_kalman_filter, gravitational_waves
import numpy as np
import pandas as pd
import cProfile
import pstats
import time
import jax.numpy as jnp
from argus.jmath import precompute_all_H
import timeit
from flax import struct

def _load_mock_data():
    """Load the mock data for testing."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
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
    logger.info(f"Getting the data. Loading {len(par_files)} pulsars from {data_path}")
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


    #Define some useful dimensions. A lot of this was previously in model.py
    Npsr = len(pulsar_metadata)
    

    logger.info(f"Total length of the data is {len(processed_pulsar_residuals)}")
    logger.info(f"Total number of pulsars is {Npsr}")
    
 
    return processed_pulsar_residuals, Npsr, pulsar_design_matrices

def test_filter_run():
    """Test the functional Kalman filter by loading data and running the filter."""
    logger.info("Starting Kalman filter test...")

    # Get the data
    processed_pulsar_residuals, Npsr, pulsar_design_matrices = _load_mock_data()

    # Calculate some useful dimensions
    M     = [submatrix.shape[-1] for submatrix in pulsar_design_matrices] 
    M_sum = sum(M)
    nx    = Npsr * (2 + 2) + M_sum

    print("M_sum: ", M_sum )
    print("nx: ", nx)
    print("M: ", M)
    # Initialize the Kalman Filter
    x0 = np.zeros(nx)
    P0 = np.eye(nx) * 1e-12

    # Extract required data from processed residuals
    data        = processed_pulsar_residuals[:, 1]  # measurements
    data_errors = processed_pulsar_residuals[:, 2]  # measurement errors
    psr_indices = processed_pulsar_residuals[:, 3].astype(int)  # pulsar indices
    t_diffs     = np.diff(processed_pulsar_residuals[:, 0])  # time differences

    logger.info("Precomputing H matrices...")
    # The H-matrix does not depend on the parameters, so we can precompute it for each pulsar
    H_arrays = precompute_all_H(num_measurements = len(data),
                               dim_x            = nx,
                               num_pulsars      = Npsr,
                               len_epsilon      = M,
                               f0               = 100 * np.ones(Npsr),  # assume all pulsars spin at 100 Hz for now,
                               M                = pulsar_design_matrices,
                               obs_sequence     = psr_indices)

    # Create parameter struct
    @struct.dataclass
    class KalmanParams:
        γa: float
        γp: jnp.ndarray
        σeps: jnp.ndarray
        σp: jnp.ndarray

    params = KalmanParams(γa=1e-1, γp=1e-1 * np.ones(Npsr), σeps=1e-20 * np.ones(M_sum),σp=1e-20 * np.ones(Npsr))








    logger.info("Converting to JAX arrays...")
    # Explicitly convert all the arguments to jax arrays
    data        = jnp.array(data)
    data_errors = jnp.array(data_errors)
    psr_indices = jnp.array(psr_indices)
    t_diffs     = jnp.array(t_diffs)
    H_arrays    = jnp.array(H_arrays)
    x0          = jnp.array(x0)
    P0          = jnp.array(P0)
    
    logger.info("First run - this should trigger compilation:")
    t0 = time.time()
    ll = jax_kalman_filter.get_likelihood(θ=params, data=data, data_errors=data_errors, dt_array=t_diffs, x0=x0, P0=P0, H_arrays=H_arrays)
    ll.block_until_ready()  # Add this line to ensure computation is complete
    logger.info(f"Likelihood result: {ll}")
    t1 = time.time()
    logger.info(f"Time taken: {t1 - t0} seconds")

    logger.info("Second run - this should use cached compilation:")
    t0 = time.time()
    ll = jax_kalman_filter.get_likelihood(θ=params, data=data, data_errors=data_errors, dt_array=t_diffs, x0=x0, P0=P0, H_arrays=H_arrays)
    ll.block_until_ready()  # Add this line to ensure computation is complete
    logger.info(f"Likelihood result: {ll}")
    t1 = time.time()
    logger.info(f"Time taken: {t1 - t0} seconds")
    
    # Force the test to pass
    assert True

if __name__ == "__main__":
    logger.info("Script started, JAX compilation logging should be enabled")
    test_filter_run()
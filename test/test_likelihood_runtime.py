





#Jax stuff
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy.linalg import block_diag

#argus stuff

# Assume the file is named jax_kalman_filter.py
from argus import jax_kalman_filter as jk
from argus import data_loader,gravitational_waves,model




#external imports
import pytest
import time
import numpy as np
from flax import struct # Import flax struct
import os
import glob




# --- Define the Flax Dataclass for Parameters ---
@struct.dataclass
class Parameters:
    """Define a struct to store the parameters of the Kalman filter model"""

    #GW parameters
    γa: float  # s⁻¹
    ha: float  # GWB amplitude

    #Pulsar parameters for the OU process
    γp: jnp.ndarray  # Pulsar-specific gamma values
    σp: jnp.ndarray  # Pulsar-specific sigma values

    #Measurement noise parameters
    EFAC: jnp.ndarray  # Error factors
    EQUAD: jnp.ndarray # Extra quadrature noise


@pytest.fixture(scope="module")
def IPTA_MDC2_data():
    #Get the data. We will use the mock data for this test

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the invariant directory path
    data_path = os.path.join(script_dir, "../data/IPTA_MockDataChallenge2/dataset_2b/")


    # Get all .par and .tim files in the directory
    par_files = sorted(glob.glob(data_path + "*.par"))
    tim_files = sorted(glob.glob(data_path + "*.tim"))
    assert len(par_files) == len(tim_files)

    #Exclude PR J1640+2224 as it has an exponent which is to small for the OU process to be valid
    par_files = [f for f in par_files if "J1640" not in f]
    tim_files = [f for f in tim_files if "J1640" not in f]



    # Get the data
    pulsar_residuals, pulsar_metadata, pulsar_design_matrices,P_eps_matrices = (
        data_loader.LoadWidebandPulsarData.read_multiple_par_tim(par_files, tim_files)
    )

    # Get the separation angles and compute HD correlation
    ra = pulsar_metadata["RA"].to_numpy(dtype=float)
    dec = pulsar_metadata["DEC"].to_numpy(dtype=float)
    angular_separation_matrix = gravitational_waves.pairwise_angular_separation(ra, dec)
    hd_correlation_matrix = gravitational_waves.hellings_downs(angular_separation_matrix)

    # Post-process the residuals    
    processed_pulsar_residuals = data_loader.LoadWidebandPulsarData.process_pulsar_residuals_by_epoch(pulsar_residuals)

    return processed_pulsar_residuals, pulsar_metadata, pulsar_design_matrices,P_eps_matrices,hd_correlation_matrix




# --- Timing Tolerances ---
TOLERANCE_FIRST_RUN_S = 10.0  # Allow more time for the first run (JIT compilation)
TOLERANCE_SECOND_RUN_S = 2 # Expect much faster execution after compilation

@pytest.mark.gpu
def test_likelihood_timing_and_jit_speedup(IPTA_MDC2_data):
    """
    Tests the likelihood calculation time using pytest.

    Expects JIT compilation on the first run and faster execution on the second.
    Relies on the 'IPTA_MDC2_data' fixture.
    """
    print("\n--- Starting test_likelihood_timing_and_jit_speedup ---")


    # --- 0. Check for GPU availability ---
    print("Checking for available GPU devices...")
    try:
        gpu_devices = jax.devices('gpu')
        if not gpu_devices:
            pytest.skip("No GPU device found. Skipping GPU-dependent test.")
        else:
            print(f"Found GPU devices: {gpu_devices}")
            # Optional: You could force JAX to use the GPU if multiple device types exist
            # jax.config.update("jax_platform_name", "gpu")
    except Exception as e:
        # Handle potential errors during device lookup, though unlikely for 'gpu'
        pytest.fail(f"Error checking for JAX GPU devices: {e}")









    # --- 1. Unpack data from the fixture ---

    #Load the data   
    processed_pulsar_residuals, pulsar_metadata, pulsar_design_matrices, P_eps_matrices, hd_correlation_matrix = IPTA_MDC2_data
    


    # Initialize the model
    model = models.StochasticGWBackgroundModel(pulsar_metadata, hd_correlation_matrix, pulsar_design_matrices)

    P0 = jnp.array(block_diag(*P_eps_matrices)) # Convert back to JAX array

    # Initialize Kalman Filter
    kf = jk.JaxKalmanFilter(
        model=model,
        observations=processed_pulsar_residuals,
        Peps=P0 
    )

    # Define Parameters
    params = Parameters(
        # GW parameters
        γa=1e-9,
        ha=1e-15,
        # Spin parameters
        γp=jnp.ones(model.Npsr) * 1e-8,
        σp=jnp.ones(model.Npsr) * 1e-15,
        # Measurement noise parameters
        EFAC=jnp.ones(model.Npsr),
        EQUAD=jnp.ones(model.Npsr) * 1e-6
    )
    print("Initialization complete.")



    # --- 3. First call (includes JIT compilation) ---
    print("Performing first likelihood call (compilation expected)...")
    start_time_1 = time.perf_counter()
    try:
        log_likelihood_1 = kf.get_likelihood(params)
        # IMPORTANT: Wait for JAX computation to finish before stopping timer
        log_likelihood_1.block_until_ready()
    except Exception as e:
        pytest.fail(f"kf.get_likelihood(params) failed on first call: {e}")
    end_time_1 = time.perf_counter()
    duration_1 = end_time_1 - start_time_1
    print(f"First call duration: {duration_1:.4f} seconds")
    print(f"Log Likelihood (1st call): {float(log_likelihood_1)}")

    # --- 4. Second call (should use compiled version) ---
    print("Performing second likelihood call (should be faster)...")
    start_time_2 = time.perf_counter()
    try:
        log_likelihood_2 = kf.get_likelihood(params)
        # IMPORTANT: Wait for JAX computation to finish
        log_likelihood_2.block_until_ready()
    except Exception as e:
        pytest.fail(f"kf.get_likelihood(params) failed on second call: {e}")
    end_time_2 = time.perf_counter()
    duration_2 = end_time_2 - start_time_2
    print(f"Second call duration: {duration_2:.4f} seconds")
    print(f"Log Likelihood (2nd call): {float(log_likelihood_2)}")

    # --- 5. Assertions ---
    print("Performing assertions...")
    assert isinstance(log_likelihood_1, jax.Array), f"Likelihood 1 type is {type(log_likelihood_1)}, expected jax.Array"
    assert isinstance(log_likelihood_2, jax.Array), f"Likelihood 2 type is {type(log_likelihood_2)}, expected jax.Array"

    assert duration_1 < TOLERANCE_FIRST_RUN_S, \
        f"First call ({duration_1:.4f}s) exceeded tolerance ({TOLERANCE_FIRST_RUN_S}s)"

    assert duration_2 < TOLERANCE_SECOND_RUN_S, \
        f"Second call ({duration_2:.4f}s) exceeded tolerance ({TOLERANCE_SECOND_RUN_S}s)"

    assert duration_2 < duration_1, \
        f"Second call ({duration_2:.4f}s) was not faster than the first call ({duration_1:.4f}s)"

    # Check if likelihood values are consistent (they should be identical)
    # Use np.isclose for robust floating-point comparison
    assert np.isclose(float(log_likelihood_1), float(log_likelihood_2), rtol=1e-5, atol=1e-8), \
        f"Likelihood values differ significantly: {float(log_likelihood_1)} vs {float(log_likelihood_2)}"





import json 
import pandas as pd
def _get_efac_equad_injections():

    # Load the noise parameters from the json file
    with open("../data/IPTA_MockDataChallenge2/group1_psr_noise.json", "r") as f:
        noise_params = json.load(f)

    # Extract EFAC and EQUAD values for each pulsar
    efac_values = []
    equad_values = []

    for psr in noise_params:

        if  "J1640" not in psr:
            efac_values.append(noise_params[psr]["efac"])
            equad_values.append(10**noise_params[psr]["equad"]) # Convert from log10 to linear

    # Convert to JAX arrays
    efac_array = jnp.array(efac_values)
    equad_array = jnp.array(equad_values)


    return efac_array, equad_array

def _get_psr_noise_injections():

    df = pd.read_pickle('../notebooks/approximate_spin_injections.pkl')
    condition = df['psr'] != 'J1640+2224'



    # 2. Use the condition to select rows and create a new DataFrame
    df_filtered = df[condition]


    sigma_p_injected = df_filtered['optimal_sigma'].values
    gamma_p_injected = df_filtered['optimal_gamma'].values

    return jnp.array(sigma_p_injected), jnp.array(gamma_p_injected)






def test_likelihood_value(IPTA_MDC2_data):
    """
    Tests the likelihood evaluation is correct. The "correct" value is inserted by hand to ensure consistency between edits

    """

    # --- 0. Check for GPU availability ---
    print("Checking for available GPU devices...")
    try:
        gpu_devices = jax.devices('gpu')
        if not gpu_devices:
            pytest.skip("No GPU device found. Skipping GPU-dependent test.")
        else:
            print(f"Found GPU devices: {gpu_devices}")
            # Optional: You could force JAX to use the GPU if multiple device types exist
            # jax.config.update("jax_platform_name", "gpu")
    except Exception as e:
        # Handle potential errors during device lookup, though unlikely for 'gpu'
        pytest.fail(f"Error checking for JAX GPU devices: {e}")



    #Get the data
    #processed_pulsar_residuals, pulsar_metadata, pulsar_design_matrices, P_eps_matrices, hd_correlation_matrix = IPTA_MDC2_data

    script_dir = os.path.dirname(os.path.abspath(__file__))
    directory = os.path.join(script_dir, "../data/IPTA_MockDataChallenge2/dataset_2b/")
    pulsar_data = data_loader.LoadWidebandPulsarData.get_processed_residuals(directory,excluded_psrs=['J1640+2224'])


    processed_pulsar_residuals = pulsar_data['processed_residuals']
    pulsar_metadata = pulsar_data['metadata']
    pulsar_design_matrices = pulsar_data['design_matrices']
    P_eps_matrices = pulsar_data['parameter_covariances']
    hd_correlation_matrix = pulsar_data['hd_correlation']




    #Get efac and equad
    efac_array, equad_array = _get_efac_equad_injections()
    assert len(efac_array) == len(equad_array) == len(pulsar_metadata)

    #Get psr noise 
    sigma_p_injected, gamma_p_injected = _get_psr_noise_injections()
    assert len(sigma_p_injected) == len(gamma_p_injected) == len(pulsar_metadata)


    #Calculate P0 based on the maximum value of the design matrix, and a delta tolerance
    #model = models.StochasticGWBackgroundModel(pulsar_metadata, hd_correlation_matrix, pulsar_design_matrices)

    alpha = 1 #scale slightly 
    P0 = alpha*block_diag(*P_eps_matrices)



    KF = jk.JaxKalmanFilter(
        df_psr=pulsar_metadata,
        observations=processed_pulsar_residuals, 
        Peps=P0,
        hd_correlation_matrix=hd_correlation_matrix,
        pulsar_design_matrices=pulsar_design_matrices,
        use_gw=True
    )


    γa = 1e-9 
    ha = 1e-15

    #Set the parameters
    params = Parameters(
        #GW parameters
        γa=γa,
        ha=ha,

        #Spin parameters
        γp=gamma_p_injected,
        σp=sigma_p_injected,

        #Measurement noise parameters
        EFAC=efac_array,
        EQUAD=equad_array
    )

    t1 = time.perf_counter()
    log_likelihood = KF.get_likelihood(params)
    assert log_likelihood == 55963.86071845221 #this is only true on OzStar GPU. On NT this value is 55963.87289660473. There seems to be a small difference between GPU implemenations.
    
    log_likelihood.block_until_ready()
    t2 = time.perf_counter()
    print(f"Time taken: {t2 - t1:.4f} seconds")





    t1 = time.perf_counter()
    log_likelihood = KF.get_likelihood(params)
    assert log_likelihood == 55963.86071845221 #this is only true on OzStar GPU. On NT this value is 55963.87289660473. There seems to be a small difference between GPU implemenations.
    
    log_likelihood.block_until_ready()
    t2 = time.perf_counter()
    print(f"Time taken: {t2 - t1:.4f} seconds")






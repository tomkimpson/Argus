import os
import glob
from argus import data_loader, models, jax_kalman_filter, gravitational_waves
import time
from flax import struct
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from jax.scipy.linalg import block_diag

jax.config.update("jax_enable_x64", True)

@struct.dataclass
class Parameters:
    
    #GW parameters
    γa: float  # s⁻¹
    ha: float  # GWB amplitude

    #Pulsar parameters for the OU process
    γp: jnp.ndarray  # Pulsar-specific gamma values
    σp: jnp.ndarray  # Pulsar-specific sigma values 

    #Timing model noise parameters
    σeps: jnp.ndarray 

    #Measurement noise parameters
    EFAC: jnp.ndarray  # Error factors
    EQUAD: jnp.ndarray # Extra quadrature noise


def _get_processed_residuals(data_path):
    """Test the JAX KalmanFilter class by loading data, initializing the model, setting parameters, and running the filter."""
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the invariant directory path
    # data_path = "../data/IPTA_MockDataChallenge/IPTA_Challenge1_open/Challenge_Data/Dataset2/"

    directory = os.path.join(
        script_dir,
        data_path
    )

    # Get all .par and .tim files in the directory
    par_files = sorted(glob.glob(directory + "*.par"))
    tim_files = sorted(glob.glob(directory + "*.tim"))

    assert len(par_files) == len(tim_files), "Mismatch between .par and .tim file counts."


    #Load just one pulsar and check everything looks reasonable
    psr = data_loader.LoadWidebandPulsarData.read_par_tim(par_files[0], tim_files[0])
    plt.plot(psr.toas, psr.residuals)
    plt.savefig("residuals.png")




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
    processed_pulsar_residuals = processed_pulsar_residuals[:len(processed_pulsar_residuals)//2]

    print("Total length of the data is ", len(processed_pulsar_residuals))
    print("Total number of pulsars is ", len(pulsar_metadata))

    return processed_pulsar_residuals, pulsar_metadata, pulsar_design_matrices,hd_correlation_matrix



def initialize_kalman_filter(nx,Npsr,M_sum):

    # Initialize the JAX Kalman Filter
    x0 = jnp.zeros(nx) # Initial state vector. δφ=0,δf=0, etc. As all the states are effecitvely perturbations, this is a reasonable guess.


    #Initialize the covariance matrices
    P_GW = jnp.eye(Npsr * 2)
    P_GW = P_GW.at[1::2, 1::2].multiply(1e-12) # All the odd diagonal elements, (1,1), (3,3) etc. are set to 1e-12
    P_GW = P_GW.at[0::2, 0::2].multiply(1e-18) # All the even diagonal elements, (0,0), (2,2) etc. are set to 1e-18


    P_spin = jnp.eye(Npsr * 2)
    P_spin = P_spin.at[1::2, 1::2].multiply(1e-8) # All the odd diagonal elements, (1,1), (3,3) etc. are set to 1e-12
    P_spin = P_spin.at[0::2, 0::2].multiply(1e-18) # All the even diagonal elements, (0,0), (2,2) etc. are set to 1e-18


    P_eps = jnp.eye(M_sum) * 1e-1

    P0 = block_diag(P_GW, P_spin, P_eps)

    return x0, P0



def benchmark_jax_runtime():

    #Get the data
    data_path = "../data/IPTA_MockDataChallenge2/dataset_1b/" # https://github.com/ipta/mdc2/tree/master
    processed_pulsar_residuals, pulsar_metadata, pulsar_design_matrices,hd_correlation_matrix = _get_processed_residuals(data_path)
    print("Got the data")
    #Initialize the model
    model = models.StochasticGWBackgroundModel(pulsar_metadata, hd_correlation_matrix, pulsar_design_matrices)
    print("Initialized the model")

    x0,P0 = initialize_kalman_filter(model.nx,model.Npsr,model.M_sum) #this could go inside the model class....
    print("Initialized the Kalman filter")

    KF = jax_kalman_filter.JaxScalarKalmanFilter(
        model=model, 
        observations=processed_pulsar_residuals, 
        x0=x0, 
        P0=P0
    )

 

    # Guess of the model parameters
    # See notebooks/PSD_for_OU_process.ipynb for discussion on the parameter values
    params = Parameters(
        #GW parameters
        γa=1e-9,
        ha=1e-12,

        #Spin parameters
        γp=jnp.ones(model.Npsr) * 1e-8, #1/year timescale. Assumed the same for all pulsars
        σp=jnp.ones(model.Npsr) * 1e-14, #For now, assume the same noise for all pulsars

        #Timing model noise parameters
        σeps=jnp.ones(model.M_sum) * 1e-12, #TBD a good value for the timing model noise. There are some rough estimates in data_loader.py, but not sure how accurate they are.

        #Measurement noise parameters
        EFAC=jnp.ones(model.Npsr),
        EQUAD=jnp.ones(model.Npsr) * (-6.7)
    )


    # Time compilation
    compilation_start = time.time()
    _ = jax.block_until_ready(KF.get_likelihood(params))  # Ensure computation is complete
    compilation_end = time.time()
    print(f"Compilation time: {compilation_end - compilation_start:.4f} seconds")


    # Get memory after compilation
    jax.profiler.save_device_memory_profile("outputs/memory_after_compilation.prof")


    start_time = time.time()
    ll = KF.get_likelihood(params)
    jax.block_until_ready(ll)  # Ensure computation is complete
    end_time = time.time()
    
    print(f"Log-likelihood: {ll}")
    print(f"Execution time: {end_time - start_time:.4f} seconds")

    # Get memory after execution
    jax.profiler.save_device_memory_profile("outputs/memory_after_execution.prof")






if __name__ == "__main__":

    # Check available devices
    print("=== JAX VERSION INFO ===")
    print(f"JAX version: {jax.__version__}")

    print("\n=== DEVICE INFO ===")
    print("Default device:", jax.default_backend())

    print("\n=== JAX CONFIG SETTINGS ===")
    for name, value in sorted(jax.config.values.items()):
        print(f"{name}: {value}")

    # Check if GPU is available
    if any(d.platform == 'gpu' for d in jax.devices()):
        print("\nJAX GPU acceleration is AVAILABLE!")
        print("GPU devices:", [d for d in jax.devices() if d.platform == 'gpu'])
    else:
        print("\nJAX GPU acceleration is NOT available. Using CPU only.")
    print('-----------------------------------------------')



    #go
    benchmark_jax_runtime() 
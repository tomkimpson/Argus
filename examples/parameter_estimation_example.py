import time
import jax
import jax.numpy as jnp
from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC,NUTS
import jax.profiler 
from numpyro.infer import SA
from argus import models, jax_kalman_filter,data_loader,gravitational_waves
from datetime import datetime
from flax import struct
import os
import glob
import matplotlib.pyplot as plt
from jax.scipy.linalg import block_diag
from numpyro.infer import Predictive # Import Predictive
from numpyro import handlers
import arviz as az

# Configure JAX
jax.config.update("jax_enable_x64", True)
numpyro.set_platform(jax.default_backend())
numpyro.set_host_device_count(len(jax.devices()))

@struct.dataclass
class Parameters:

    """Define a struct to store the parameters of the Kalman filter model"""
    
    #GW parameters
    γa: float  # s⁻¹
    ha: float  # GWB amplitude

    #Pulsar parameters for the OU process
    γp: jnp.ndarray  # Pulsar-specific gamma values
    σp: jnp.ndarray  # Pulsar-specific sigma values 

    #Timing model parameters
    x0_timing: jnp.ndarray # Sampled initial state for timing params


    #Measurement noise parameters
    EFAC: jnp.ndarray  # Error factors
    EQUAD: jnp.ndarray # Extra quadrature noise


def _get_processed_residuals(data_path):
    """Get the processed residuals from the data."""
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
    plt.savefig("outputs/exmple_residuals_plot.png")




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

    return processed_pulsar_residuals, pulsar_metadata, pulsar_design_matrices,hd_correlation_matrix

def _initialize_kalman_filter_covariance(nx,Npsr,M_sum):

    """
    Specify the initial state vector x0 and the covariance matrix P0 for the Kalman filter.
    """

    # Initialize the JAX Kalman Filter
    x0 = jnp.zeros(nx) # Initial state vector. δφ=0,δf=0, etc. As all the states are effecitvely perturbations, this is a reasonable guess.

    #Initialize the covariance matrices
    P_GW = jnp.eye(Npsr * 2)
    P_GW = P_GW.at[1::2, 1::2].multiply(1e-12) # All the odd diagonal elements, (1,1), (3,3) etc. are set to 1e-12
    P_GW = P_GW.at[0::2, 0::2].multiply(1e-18) # All the even diagonal elements, (0,0), (2,2) etc. are set to 1e-18


    P_spin = jnp.eye(Npsr * 2)
    P_spin = P_spin.at[1::2, 1::2].multiply(1e-8) # All the odd diagonal elements, (1,1), (3,3) etc. are set to 1e-12
    P_spin = P_spin.at[0::2, 0::2].multiply(1e-18) # All the even diagonal elements, (0,0), (2,2) etc. are set to 1e-18


    P_eps = jnp.eye(M_sum) * 0.0 

    P0 = block_diag(P_GW, P_spin, P_eps)

    return P0



def _priors(Npsr,M_sum):

    """
    Define the priors for the parameters.
    """

    # Parameters of the GW background
    γa = numpyro.sample("γa", dist.LogUniform(1e-11, 1e-6))
    ha = numpyro.sample("ha", dist.LogUniform(1e-16, 1e-11))

    #Parameters of the pulsar process
    γp = numpyro.sample("γp", dist.LogUniform(1e-11, 1e-6),sample_shape=(Npsr,))
    σp = numpyro.sample("σp", dist.LogUniform(1e-16, 1e-11),sample_shape=(Npsr,))

    #Timing model noise parameters
    #These are states which are tracked.
    x0_timing_std = 1e-7
    x0_timing = numpyro.sample("x0_timing", dist.Normal(0., x0_timing_std), sample_shape=(M_sum,))

    
    #Measurement noise parameters
    EFAC = numpyro.sample("EFAC", dist.Uniform(0.5, 2),sample_shape=(Npsr,))
    EQUAD = numpyro.sample("EQUAD", dist.LogUniform(1e-8, 1e-5),sample_shape=(Npsr,))


    # Construct the Parameters object
    params = Parameters(
        γa=γa,
        ha=ha,
        γp=γp,
        σp=σp,
        x0_timing=x0_timing,
        EFAC=EFAC,
    EQUAD=EQUAD
    )
    
    return params









def jax_parameter_estimation():

    #Get the data
    data_path = "../data/IPTA_MockDataChallenge2/dataset_1b/" # https://github.com/ipta/mdc2/tree/master
    processed_pulsar_residuals, pulsar_metadata, pulsar_design_matrices,hd_correlation_matrix = _get_processed_residuals(data_path)

    #Initialize the model
    model = models.StochasticGWBackgroundModel(pulsar_metadata, hd_correlation_matrix, pulsar_design_matrices)


    P0 = _initialize_kalman_filter_covariance(model.nx,model.Npsr,model.M_sum) #this could go inside the model class....


    KF = jax_kalman_filter.JaxScalarKalmanFilter(
        model=model, 
        observations=processed_pulsar_residuals, 
        P0=P0
    )




    #pre-compile
    Npsr = model.Npsr
    M_sum = model.M_sum
    seed_value = 42
    rng_key = random.PRNGKey(seed_value)

    print(f"Sampling one draw using seed {seed_value} with handlers.seed:")
    with handlers.seed(rng_seed=rng_key):
        params = _priors(Npsr=Npsr, M_sum=M_sum)

    print("Sampled Parameters object:")
    print(params)
    ll = KF.get_likelihood(params)
    print("Log likelihood of precompilation was: ", ll)

    #Now do parameter estimation
    print("Starting NumPyro inference")
    
    # Check NumPyro device usage
    print("\n=== NUMPYRO DEVICE INFO ===")
    print(f"NumPyro version: {numpyro.__version__}")
    print("--------------------------------")



    # NumPyro model
    def numpyro_model(kf):

        params = _priors(Npsr=Npsr, M_sum=M_sum)
        log_likelihood = kf.get_likelihood(params)
        numpyro.factor("likelihood", log_likelihood)
    
    # Run MCMC

    
    nuts_kernel = NUTS(numpyro_model)
    mcmc = MCMC(nuts_kernel, num_samples=1000, num_warmup=500, num_chains=4, progress_bar=True)
    rng_key   = random.PRNGKey(0)
    print("Starting MCMC")
    mcmc.run(rng_key, kf=KF)
    mcmc.print_summary()  # Posterior estimates


    print("Completed. Saving results to disk...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"outputs/parameter_estimation_results_example_{timestamp}.nc"
    inf_data = az.from_numpyro(mcmc)
    print(f"Saving results to {output_filename}")
    inf_data.to_netcdf(output_filename)





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
    print(jax.devices())



    #go
    jax_parameter_estimation() 

   
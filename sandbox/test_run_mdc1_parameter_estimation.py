import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy.linalg import block_diag
from jax import random


import os 
import glob
import sys 
import json
import pandas as pd
import numpy as np 
from flax import struct
import numpyro.distributions as dist
from datetime import datetime

sys.path.append('../python/argus')
from argus import data_loader
from argus import models
from argus import jax_kalman_filter
from argus import gravitational_waves
from argus import utils

#NumPyro
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC,SA,NUTS

#Arviz
import arviz as az

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




def _get_processed_residuals(directory):
    """Get the processed residuals from the data."""

    # Get all .par and .tim files in the directory
    par_files = sorted(glob.glob(directory + "*.par"))
    tim_files = sorted(glob.glob(directory + "*.tim"))

    assert len(par_files) == len(tim_files), "Mismatch between .par and .tim file counts."

    #Exclude PR J1640+2224 as it has an exponent which is to small for the OU process to be valid
    par_files = [f for f in par_files if "J1640" not in f]
    tim_files = [f for f in tim_files if "J1640" not in f]



    # Get the data
    print(f"Getting the data. Loading {len(par_files)} pulsars from {directory}")
    pulsar_residuals, pulsar_metadata, pulsar_design_matrices,P_eps_matrices = (
        data_loader.LoadWidebandPulsarData.read_multiple_par_tim(par_files, tim_files)
    )

    # Get the separation angles and compute HD correlation
    ra = pulsar_metadata["RA"].to_numpy(dtype=float)
    dec = pulsar_metadata["DEC"].to_numpy(dtype=float)
    angular_separation_matrix = data_loader.LoadWidebandPulsarData.pairwise_angular_separation(ra, dec)
    hd_correlation_matrix = gravitational_waves.hellings_downs(angular_separation_matrix)

    # Post-process the residuals




    
    processed_pulsar_residuals = data_loader.LoadWidebandPulsarData.post_process_residuals(pulsar_residuals)

    print("Total length of the data is ", len(processed_pulsar_residuals[1]))
    print("Total number of pulsars is ", len(pulsar_metadata))

    return processed_pulsar_residuals, pulsar_metadata, pulsar_design_matrices,P_eps_matrices,hd_correlation_matrix



def _initialize_kalman_filter(nx,Npsr,P_eps):

    """
    Specify the initial state vector x0 and the covariance matrix P0 for the Kalman filter.
    """

    # Initialize the JAX Kalman Filter
    x0 = jnp.zeros(nx) # Initial state vector. δφ=0,δf=0, etc. As all the states are effecitvely perturbations, this is a reasonable guess.


    #Initialize the covariance matrices

    ## GW block "r/a"
    P_GW = jnp.eye(Npsr * 2)
    P_GW = P_GW.at[0::2, 0::2].multiply(1e-40) #r(0), integrated: set tiny variance. All the even diagonal elements, (0,0), (2,2) etc. are set to 1e-40
    
    
    
    # h2 = (1e-12)**2
    # γa = 1e-9
    # sigma2 =  (h2 / 12) * γa 
    #P_GW = P_GW.at[1::2, 1::2].multiply(sigma2 / (2 * γa)) 
    P_GW = P_GW.at[1::2, 1::2].multiply(1e-25) #Set 'a' components (odd indices) to stationary OU variance


    utils.check_cholesky(P_GW,"The initial PGW-matrix")
    utils.check_min_eigenvalue(P_GW, "The initial PGW-matrix")
    utils.check_symmetry(P_GW, "The initial PGW-matrix")
    utils.check_condition_number(P_GW, "The initial PGW-matrix")


    P_spin = jnp.eye(Npsr * 2)
    P_spin = P_spin.at[0::2, 0::2].multiply(1e-40) # All the even diagonal elements, (0,0), (2,2) etc. are set to X
    P_spin = P_spin.at[1::2, 1::2].multiply(1e-20) # All the odd diagonal elements, (1,1), (3,3) etc. are set to Y


    utils.check_cholesky(P_spin,"The initial Pspin-matrix")
    utils.check_min_eigenvalue(P_spin, "The initial Pspin-matrix")
    utils.check_symmetry(P_spin, "The initial Pspin-matrix")
    utils.check_condition_number(P_spin, "The initial Pspin-matrix")


    P1 = P_eps
    utils.check_cholesky(P1,"The initial Peps-matrix")
    utils.check_min_eigenvalue(P1, "The initial Peps-matrix")
    utils.check_symmetry(P1, "The initial Peps-matrix")
    utils.check_condition_number(P1, "The initial Peps-matrix")








    P0 = block_diag(P_GW, P_spin, P_eps)

    return x0, P0






def parameter_estimation():

    #Get the data
    data_path = "../data/IPTA_MockDataChallenge/IPTA_Challenge1_open/Challenge_Data/Dataset1/" 
    processed_pulsar_residuals, pulsar_metadata, pulsar_design_matrices,P_eps_matrices,hd_correlation_matrix = _get_processed_residuals(data_path)




    #Calculate P0 based on the maximum value of the design matrix, and a delta tolerance
    model = models.StochasticGWBackgroundModel(pulsar_metadata, hd_correlation_matrix, pulsar_design_matrices)



    alpha = 10
    P0 = alpha*block_diag(*P_eps_matrices)

    #Initialize the model
    x_init,P_init = _initialize_kalman_filter(model.nx,model.Npsr,P0) #this could go inside the model class....


    KF = jax_kalman_filter.JaxKalmanFilter(
        model=model, 
        observations=processed_pulsar_residuals, 
        x0=x_init, 
        P0=P_init
    )


    γa = 1e-6 
    ha = 10**(-12.78) #1e-12

    #Set the parameters
    params = Parameters(
        #GW parameters
        γa=γa,
        ha=ha,

        #Spin parameters
        γp=jnp.ones(model.Npsr)*1e-15,
        σp=jnp.ones(model.Npsr)*0.0,

        #Measurement noise parameters
        EFAC=jnp.ones(model.Npsr),
        EQUAD=jnp.zeros(model.Npsr)
    )

    print("First call to get_likelihood")
    ll = KF.get_likelihood(params)
    ll.block_until_ready()
    print("Likelihood: ",ll)


    #Now do parameter estimation
    print("Starting NumPyro")

    # Check NumPyro device usage
    print("\n=== NUMPYRO DEVICE INFO ===")
    print(f"NumPyro version: {numpyro.__version__}")
    print("--------------------------------")



    # NumPyro model
    def numpyro_model(kf):


        # Parameters of the GW background
        γa = numpyro.deterministic("γa", 1e-9)
        ha = numpyro.sample("ha", dist.LogUniform(1e-18, 1e-4))

        #Parameters of the pulsar process
        γp = numpyro.deterministic("γp", jnp.ones(model.Npsr)*1e-15,)
        σp = numpyro.deterministic("σp", jnp.zeros(model.Npsr))

        
        #Measurement noise parameters
        EFAC = numpyro.deterministic("EFAC", jnp.ones(model.Npsr))
        EQUAD = numpyro.deterministic("EQUAD", jnp.zeros(model.Npsr))


        # Construct the Parameters object
        params = Parameters(
            γa=γa,
            ha=ha,
            γp=γp,
            σp=σp,
            EFAC=EFAC,
            EQUAD=EQUAD
        )
        
        # Call the likelihood
        #jax.profiler.save_device_memory_profile("memory_during_likelihood_call.prof")
        log_likelihood = kf.get_likelihood(params)
        numpyro.factor("likelihood", log_likelihood)



    # Parameter estimation with numpyro
    print("Starting inference ")
    rng_key = random.PRNGKey(0)
    kernel = NUTS(numpyro_model)
    sampler = MCMC(kernel, num_samples=1000, num_warmup=500,progress_bar=True,num_chains=4)
    sampler.run(rng_key, kf=KF)
    sampler.print_summary()  # Posterior estimates


    print("Completed. Saving results to disk...")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    inf_data = az.from_numpyro(sampler)
    fname = f"outputs/mdc1_parameter_estimation_results_{timestamp}.nc" 
    inf_data.to_netcdf(fname)
    print(f"Saved results to {fname}")







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
    parameter_estimation() 

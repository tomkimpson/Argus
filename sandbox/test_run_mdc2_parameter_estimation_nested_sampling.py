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
from datetime import datetime


sys.path.append('../python/argus')
from argus import data_loader
from argus import models
from argus import jax_kalman_filter
from argus import gravitational_waves
#from argus import utils



#NumPyro
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC,SA,NUTS


from numpyro.contrib.nested_sampling import NestedSampler
from jaxns import plot_cornerplot,plot_diagnostics,save_results


#Arviz
import arviz as az


import numpyro.distributions as dist


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
    angular_separation_matrix = gravitational_waves.pairwise_angular_separation(ra, dec)
    hd_correlation_matrix = gravitational_waves.hellings_downs(angular_separation_matrix)

    # Post-process the residuals    
    processed_pulsar_residuals = data_loader.LoadWidebandPulsarData.process_pulsar_residuals_by_epoch(pulsar_residuals)

    print("Total length of the data is ", len(processed_pulsar_residuals[1]))
    print("Total number of pulsars is ", len(pulsar_metadata))

    return processed_pulsar_residuals, pulsar_metadata, pulsar_design_matrices,P_eps_matrices,hd_correlation_matrix

def get_efac_equad_injections():

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

def get_psr_noise_injections():

    df = pd.read_pickle('../notebooks/approximate_spin_injections.pkl')
    condition = df['psr'] != 'J1640+2224'



    # 2. Use the condition to select rows and create a new DataFrame
    df_filtered = df[condition]


    sigma_p_injected = df_filtered['optimal_sigma'].values
    gamma_p_injected = df_filtered['optimal_gamma'].values

    return jnp.array(sigma_p_injected), jnp.array(gamma_p_injected)




def parameter_estimation():
    #Get the data
    data_path = "../data/IPTA_MockDataChallenge2/dataset_2b/" 
    processed_pulsar_residuals, pulsar_metadata, pulsar_design_matrices,P_eps_matrices,hd_correlation_matrix = _get_processed_residuals(data_path)

    #Get efac and equad
    efac_array, equad_array = get_efac_equad_injections()
    assert len(efac_array) == len(equad_array) == len(pulsar_metadata)

    #Get psr noise 
    sigma_p_injected, gamma_p_injected = get_psr_noise_injections()
    assert len(sigma_p_injected) == len(gamma_p_injected) == len(pulsar_metadata)


    #Calculate P0 based on the maximum value of the design matrix, and a delta tolerance
    model = models.StochasticGWBackgroundModel(pulsar_metadata, hd_correlation_matrix, pulsar_design_matrices)

    alpha = 1 #scale slightly 
    P0 = alpha*block_diag(*P_eps_matrices)


    #placeholders, not actually used
    x_init = np.zeros(model.nx)
    P_init = P0

    KF = jax_kalman_filter.JaxKalmanFilter(
        model=model, 
        observations=processed_pulsar_residuals, 
        x0=x_init, 
        P0=P_init,
        Peps=P0
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

    print("First call to get_likelihood. Just for precompilation")
    ll = KF.get_likelihood(params)
    ll.block_until_ready()
    print("Likelihood: ",ll)




    #Now do parameter estimation
    print("Starting Nested sampling with jaxns")


    # NumPyro model
    def numpyro_model(kf):


        # Parameters of the GW background
        γa = numpyro.deterministic("γa", 1e-9)
    

        log10_ha = numpyro.sample("log10_ha", dist.Uniform(-17.0, -14.0))
        # Convert back to ha for the physics calculation
        ha = numpyro.deterministic("ha", 10**log10_ha)

        #Parameters of the pulsar process
        log10_γp = numpyro.sample("log10_γp", dist.Uniform(-11.0, -6.0),sample_shape=(model.Npsr,))
        γp = numpyro.deterministic("γp", 10**log10_γp)

        log10_σp = numpyro.sample("log10_σp", dist.Uniform(-18.0, -12.0),sample_shape=(model.Npsr,))
        σp = numpyro.deterministic("σp", 10**log10_σp)

        
        #Measurement noise parameters
        EFAC = numpyro.deterministic("EFAC", efac_array)
        EQUAD = numpyro.deterministic("EQUAD", equad_array)


        # Construct the Parameters object
        params = Parameters(
            γa=γa,
            ha=ha,
            γp=γp,
            σp=σp,
            EFAC=EFAC,
            EQUAD=EQUAD
        )
        log_likelihood = kf.get_likelihood(params)
        numpyro.factor("likelihood", log_likelihood)


    
    constructor_args = {
        #'num_live_points': 10,
        'verbose': True #does this work?         
    }

    #This seems to have a different notation than the docs? See TerminationCondition = numpyro.contrib.nested_sampling.TerminationCondition; help(TerminationCondition)
    termination_args = {
        'dlogZ': 0.1  
    }

    # Pass the dictionaries to the correct keyword arguments
    ns = NestedSampler(
        numpyro_model,
        constructor_kwargs=constructor_args,
        termination_kwargs=termination_args
    )

    ns.run(random.PRNGKey(2), kf=KF)


    print("Getting samples")
    ns.get_samples(random.PRNGKey(3), num_samples=1000)
    
    print("Summary")
    ns.print_summary()  # Posterior estimates


    #Plots
    print("Generating plots")
    plot_diagnostics(ns._results,save_name='NS_diagnostics')
    plot_cornerplot(ns._results,variables =['log10_ha','γa'],save_name='NS_cornerplot')




    print("Completed. Saving results to disk...")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


    save_results(ns._results,f"mdc2_parameter_estimation_nested_sampling_results_{timestamp}")

    # inf_data = az.from_numpyro(ns)
    # fname = f"outputs/mdc2_parameter_estimation_nested_sampling_results_{timestamp}.nc" 
    # inf_data.to_netcdf(fname)
    # print(f"Saved results to {fname}")




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


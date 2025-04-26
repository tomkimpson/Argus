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

    alpha = 1.0 #scale slightly 
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




    print("doing some gradient checks")



        # Define a helper function that takes only 'ha' and returns the log likelihood scalar
    def log_likelihood_for_log10_ha(log10_ha_value):
        # Ensure input is a JAX array
        log10_ha_value = jnp.asarray(log10_ha_value, dtype=jnp.float64)
        # Convert back to ha
        ha_value = 10**log10_ha_value

        # Construct the Parameters object with the test ha_value and fixed deterministics
        params = Parameters(
            γa=jnp.array(1e-9), # Ensure these are JAX arrays too if needed inside kf
            ha=ha_value,
            γp=jnp.array(gamma_p_injected),
            σp=jnp.array(sigma_p_injected),
            EFAC=jnp.array(efac_array),
            EQUAD=jnp.array(equad_array)
        )
        # Ensure the output is a scalar
        loglik = KF.get_likelihood(params) 
        # If loglik is an array with one element, extract the scalar
        return jnp.squeeze(loglik)


    # Get the gradient function with respect to log10_ha
    grad_wrt_log10_ha_fn = jax.grad(log_likelihood_for_log10_ha)

    # --- Testing ---
    # Original ha values used previously
    ha_test_points = jnp.array([1.1e-16, 1.0e-15, 2.0e-15, 5.0e-15, 9.0e-15], dtype=jnp.float64)
    # Corresponding log10(ha) values
    log10_ha_test_points = jnp.log10(ha_test_points)

    print("Checking gradients w.r.t. log10(ha):")
    for ha_val, log10_ha_val in zip(ha_test_points, log10_ha_test_points):
        try:
            # Calculate likelihood using the new function
            loglik = log_likelihood_for_log10_ha(log10_ha_val) 
            # Calculate gradient w.r.t. log10_ha
            grad_val_log10 = grad_wrt_log10_ha_fn(log10_ha_val)


            print(f"ha = {ha_val:.3e} (log10_ha = {log10_ha_val:.3f}):")
            print(f"  loglik = {loglik:.4e}")
            print(f"  Grad w.r.t. log10(ha) = {grad_val_log10:.4e} (Finite: {jnp.isfinite(grad_val_log10).all()})")

        except Exception as e:
            print(f"ha = {ha_val:.3e} (log10_ha = {log10_ha_val:.3f}): Error during calculation: {e}")


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
        #ha = numpyro.sample("ha", dist.LogUniform(1e-16, 1e-14))

        log10_ha = numpyro.sample("log10_ha", dist.Uniform(-17.0, -14.0))
        # Convert back to ha for the physics calculation
        ha = numpyro.deterministic("ha", 10**log10_ha)

        #Parameters of the pulsar process
        γp = numpyro.deterministic("γp", gamma_p_injected)
        σp = numpyro.deterministic("σp", sigma_p_injected)

        
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
        #jax.debug.print("log_likelihood: {log_likelihood}",log_likelihood=log_likelihood)
        numpyro.factor("likelihood", log_likelihood)


    # Parameter estimation with numpyro
    print("Starting inference ")
    rng_key = random.PRNGKey(0)
    kernel = NUTS(numpyro_model)
    sampler = MCMC(kernel, num_samples=2000, num_warmup=2000,progress_bar=True,num_chains=2)
    sampler.run(rng_key, kf=KF)
    sampler.print_summary()  # Posterior estimates


    print("Completed. Saving results to disk...")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    inf_data = az.from_numpyro(sampler)
    fname = f"outputs/mdc2_parameter_estimation_NUTS_results_{timestamp}.nc" 
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


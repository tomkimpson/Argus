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





#jaxns 

import tensorflow_probability.substrates.jax as tfp
tfpd = tfp.distributions

from jaxns import Prior, Model, NestedSampler # Import necessary components

from utils import _get_processed_residuals, get_efac_equad_injections, get_psr_noise_injections




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


def parameter_estimation():


    print("Inside the parameter estimation function")

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


    KF = jax_kalman_filter.JaxKalmanFilter(
        model=model, 
        observations=processed_pulsar_residuals, 
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


    def prior_model():
        """Defines the prior distributions for the parameters."""
        # Scalar parameter
        log10_ha = yield Prior(tfpd.Uniform(-17.0, -14.0), name='log10_ha')

        # Vector parameters - Use jnp.full or pass arrays to low/high for TFP Uniform
        log10_γp = yield Prior(
            tfpd.Uniform(low=jnp.full(model.Npsr, -11.0), high=jnp.full(model.Npsr, -6.0)),
            name='log10_γp'
        )

        log10_σp = yield Prior(
            tfpd.Uniform(low=jnp.full(model.Npsr, -18.0), high=jnp.full(model.Npsr, -12.0)),
            name='log10_σp'
        )
 
        return log10_ha, log10_γp, log10_σp



    # JAXNS model
    def jaxns_log_likelihood(log10_ha, log10_γp, log10_σp):

        # Fixed values
        γa = 1e-9
        EFAC = efac_array
        EQUAD = equad_array


       # Calculate derived parameters
        ha = 10.**log10_ha
        γp = 10.**log10_γp # Will have shape (Npsr,)
        σp = 10.**log10_σp # Will have shape (Npsr,)

        # Construct the Parameters object
        params = Parameters(
            γa=γa,
            ha=ha,
            γp=γp,
            σp=σp,
            EFAC=EFAC,
            EQUAD=EQUAD
        )
        return KF.get_likelihood(params)     



    print("Initializing Model")
    jax_model = Model(prior_model=prior_model, log_likelihood=jaxns_log_likelihood)
    
    # https://jaxns.readthedocs.io/en/latest/api/jaxns/index.html#jaxns.NestedSampler
    print("Initializing NestedSampler")
    ns = NestedSampler(
        model=jax_model,
        num_live_points=10,
        verbose=True)

    print("Running NestedSampler")
    termination_reason, state = jax.jit(ns)(random.PRNGKey(432345987))

    print("Converting to results")
    results = ns.to_results(termination_reason=termination_reason, state=state)

    print("Summary")
    ns.print_summary()  # Posterior estimates




    #Plots
    print("Generating plots")
    ns.plot_diagnostics(results,save_name='outputs/images/example_NS_diagnostics')
    ns.plot_cornerplot(results,variables =['log10_ha','γa'],save_name='outputs/images/example_NS_cornerplot')

    print("Completed. Saving results to disk...")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    ns.save_results(results,f"outputs/datafiles/example_nested_sampling_results_{timestamp}")
   


if __name__ == "__main__":


    print("Double check this is the correct script")
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
    print("go: parameter NS estimatin")
    parameter_estimation() 


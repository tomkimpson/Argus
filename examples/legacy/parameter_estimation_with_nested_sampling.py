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
from argus import jax_kalman_filter
from argus import bayesian_inference

import time 

# For creating zero-argument prior functions
import functools

#jaxns 

import tensorflow_probability.substrates.jax as tfp
tfpd = tfp.distributions

from jaxns import Prior, Model, NestedSampler # Import necessary components

from utils import get_efac_equad_injections, get_psr_noise_injections


def parameter_estimation():

    #Get the data
    directory = "../data/IPTA_MockDataChallenge2/dataset_2b/" 
    pulsar_data = data_loader.LoadWidebandPulsarData.get_processed_residuals(directory,excluded_psrs=['J1640+2224'])

    #Extract the arrays. todo. this is a bit messy and we should update the Kalman filter to take in a dictionary of data
    processed_pulsar_residuals = pulsar_data['processed_residuals']
    pulsar_metadata = pulsar_data['metadata']
    pulsar_design_matrices = pulsar_data['design_matrices']
    P_eps_matrices = pulsar_data['parameter_covariances']
    hd_correlation_matrix = pulsar_data['hd_correlation']


    #Get efac and equad
    efac_array, equad_array = get_efac_equad_injections()
    assert len(efac_array) == len(equad_array) == len(pulsar_metadata)

    #Get psr noise 
    sigma_p_injected, gamma_p_injected = get_psr_noise_injections()
    assert len(sigma_p_injected) == len(gamma_p_injected) == len(pulsar_metadata)

    alpha = 1 #scale slightly 
    P0 = alpha*block_diag(*P_eps_matrices)


    KF = jax_kalman_filter.JaxKalmanFilter(
        df_psr=pulsar_metadata,
        observations=processed_pulsar_residuals, 
        Peps=P0,
        hd_correlation_matrix=hd_correlation_matrix,
        pulsar_design_matrices=pulsar_design_matrices,
        use_gw=True
    )



    prior_model = functools.partial(
        bayesian_inference.configurable_prior_model,
        Npsr=len(pulsar_metadata),
        log10_ha_spec=-15.0,
        gamma_a_spec=1e-9,                       # Default fixed gamma_a
        log10_gamma_p_spec=gamma_p_injected, # Pass the distribution for log10_gamma_p
        log10_sigma_p_spec=sigma_p_injected, # Pass the distribution for log10_sigma_p
        efac_spec=efac_array,                 # Pass the array for efac
        equad_spec=equad_array      # Pass the array for equad (converted from log10)
    )


    loglik_fn = lambda log10_ha, γa, log10_γp, log10_σp, efac, equad: \
    bayesian_inference.jaxns_log_likelihood(KF, log10_ha, γa, log10_γp, log10_σp, efac, equad)

    #Initialise the jaxns model
    jax_model = Model(prior_model=prior_model, log_likelihood=loglik_fn)


    #Call the likelihood function a couple of times to precompile and check everything looks ok

    u = jax_model.sample_U(key=random.PRNGKey(432345987))  # Unit cube sample
    θ = jax_model.transform(u)                             # Transform to physical parameter space
    params = bayesian_inference.Parameters(
        γa=θ['γa'],
        ha=10**θ['log10_ha'],
        γp=10**θ['log10_γp'],
        σp=10**θ['log10_σp'],
        EFAC=θ['efac'],
        EQUAD=θ['equad']
    )


    print("The sampled parameters are:")
    bayesian_inference.print_parameters(params)









    # γa = 1e-9 
    # ha = 1e-15

    # #Set the parameters
    # params = Parameters(
    #     #GW parameters
    #     γa=γa,
    #     ha=ha,

    #     #Spin parameters
    #     γp=gamma_p_injected,
    #     σp=sigma_p_injected,

    #     #Measurement noise parameters
    #     EFAC=efac_array,
    #     EQUAD=equad_array
    # )

    print("First call to get_likelihood. Just for precompilation")
    t1 = time.time()
    ll = KF.get_likelihood(params)
    ll.block_until_ready()
    t2 = time.time()
    print("Likelihood: ",ll)
    print("Time taken for precompilation: ",t2-t1)


    print("Second call to get_likelihood. Checking precompilation")
    t1 = time.time()
    ll = KF.get_likelihood(params)
    ll.block_until_ready()
    t2 = time.time()
    print("Likelihood: ",ll)
    print("Time taken for second call: ",t2-t1)






    #Now do parameter estimation
    print("Starting Nested sampling with jaxns")


    def prior_model():
        """Defines the prior distributions for the parameters."""
        # Scalar parameter
        log10_ha = yield Prior(tfpd.Uniform(-17.0, -14.0), name='log10_ha')

        # # Vector parameters - Use jnp.full or pass arrays to low/high for TFP Uniform
        # log10_γp = yield Prior(
        #     tfpd.Uniform(low=jnp.full(model.Npsr, -11.0), high=jnp.full(model.Npsr, -6.0)),
        #     name='log10_γp'
        # )

        # log10_σp = yield Prior(
        #     tfpd.Uniform(low=jnp.full(model.Npsr, -18.0), high=jnp.full(model.Npsr, -12.0)),
        #     name='log10_σp'
        #)
 
        return log10_ha #, log10_γp, log10_σp



    # JAXNS model
    def jaxns_log_likelihood(log10_ha):

        # Fixed values
        γa = 1e-9
        EFAC = efac_array
        EQUAD = equad_array


       # Calculate derived parameters
        ha = 10.0**log10_ha
        #γp = 10.**log10_γp # Will have shape (Npsr,)
        #σp = 10.**log10_σp # Will have shape (Npsr,)

        # Construct the Parameters object
        params = Parameters(
            γa=γa,
            ha=ha,
            γp=gamma_p_injected,
            σp=sigma_p_injected,
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


    #Plots
    print("Generating plots")
    ns.plot_diagnostics(results,save_name='outputs/images/example_NS_diagnostics')
    ns.plot_cornerplot(results,variables =['log10_ha','γa'],save_name='outputs/images/example_NS_cornerplot')

    print("Completed. Saving results to disk...")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    ns.save_results(results,f"outputs/example_nested_sampling_results_{timestamp}")
   


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


import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax.scipy.linalg import block_diag
from jax import random

import sys 

sys.path.append('../python/argus')
from argus import models
from argus import jax_kalman_filter
from argus import bayesian_inference

import tensorflow_probability.substrates.jax as tfp
tfpd = tfp.distributions

from jaxns import Model

from utils import _get_processed_residuals

import os 
import glob
import json
import pandas as pd
import numpy as np 
from datetime import datetime

import time 

def parameter_estimation():


    #Get the data
    data_path = "../data/IPTA_MockDataChallenge2/dataset_2b/" 
    processed_pulsar_residuals, pulsar_metadata, pulsar_design_matrices,P_eps_matrices,hd_correlation_matrix = _get_processed_residuals(data_path)

    GW_model = models.StochasticGWBackgroundModel(pulsar_metadata, hd_correlation_matrix, pulsar_design_matrices)

    alpha = 1 #scale slightly 
    P0 = alpha*block_diag(*P_eps_matrices)

    KF = jax_kalman_filter.JaxKalmanFilter(
        model=GW_model, 
        observations=processed_pulsar_residuals, 
        Peps=P0
    )

    loglik_fn = lambda log10_ha, γa, log10_γp, log10_σp, efac, equad: \
    bayesian_inference.jaxns_log_likelihood(KF, log10_ha, γa, log10_γp, log10_σp, efac, equad)
    
    jax_model = Model(prior_model=bayesian_inference.gw_prior_model, log_likelihood=loglik_fn)

    u = jax_model.sample_U(key=random.PRNGKey(432345987))  # Unit cube sample
    θ = jax_model.transform(u)                       # Transform to physical parameter space

    print("The sampled parameters are:")
    print(θ)
  

    # print("First call to get_likelihood. Just for precompilation")
    # t1 = time.time()
    # ll = KF.get_likelihood(params)
    # ll.block_until_ready()
    # t2 = time.time()
    # print("Likelihood: ",ll)
    # print("Time taken for precompilation: ",t2-t1)


    # print("Second call to get_likelihood. Checking precompilation")
    # t1 = time.time()
    # ll = KF.get_likelihood(params)
    # ll.block_until_ready()
    # t2 = time.time()
    # print("Likelihood: ",ll)
    # print("Time taken for second call: ",t2-t1)


    # #Now do parameter estimation
    # print("Starting Nested sampling with jaxns")





 



   


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


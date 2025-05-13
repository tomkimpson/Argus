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
from argus import model
from argus import jax_kalman_filter
from argus import gravitational_waves
from argus import bayesian_inference

import time 


#jaxns 

import tensorflow_probability.substrates.jax as tfp
tfpd = tfp.distributions

from jaxns import Prior, Model, NestedSampler,TerminationCondition # Import necessary components





def parameter_estimation(P_eps_scaling):


    #Get the data 
    data_path = "../data/IPTA_MockDataChallenge2/dataset_2b/" 
    data = data_loader.get_processed_residuals(data_path)
    
    #Initialise the Kalman filter
    KF = jax_kalman_filter.JaxKalmanFilter(data_dict=data,P0_scaling=1.0,use_gw=True)



    sys.exit()

    #loglik_fn = lambda log10_ha, log10_γp, log10_σp,efac_array,equad_array: bayesian_inference.jaxns_log_likelihood(KF,log10_ha, log10_γp, log10_σp, efac_array,equad_array)
    
    loglik_fn = lambda log10_ha, γa, log10_γp, log10_σp, efac, equad: \
    bayesian_inference.jaxns_log_likelihood(KF, log10_ha, γa, log10_γp, log10_σp, efac, equad)
    
    






# ### TO BE REMOVED--------------
# import json
# Npsr = 32
# def get_efac_equad_injections():

#     # Load the noise parameters from the json file
#     with open("../data/IPTA_MockDataChallenge2/group1_psr_noise.json", "r") as f:
#         noise_params = json.load(f)

#     # Extract EFAC and EQUAD values for each pulsar
#     efac_values = []
#     equad_values = []

#     for psr in noise_params:

#         if  "J1640" not in psr:
#             efac_values.append(noise_params[psr]["efac"])
#             equad_values.append(10**noise_params[psr]["equad"]) # Convert from log10 to linear

#     # Convert to JAX arrays
#     efac_array = jnp.array(efac_values)
#     equad_array = jnp.array(equad_values)


#     return efac_array, equad_array

# efac_array, equad_array = get_efac_equad_injections()
# ### TO BE REMOVED--------------















    
    jax_model = Model(prior_model=bayesian_inference.gw_prior_model, log_likelihood=loglik_fn)

   # params = jax_model.sample_U(key=random.PRNGKey(432345987))

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
    P_eps_scaling = 1.0
    parameter_estimation(P_eps_scaling) 


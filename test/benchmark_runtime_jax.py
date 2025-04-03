import os
import glob
from argus import data_loader, models, jax_kalman_filter, gravitational_waves
import numpy as np
import pandas as pd
import time
from flax import struct
import jax.numpy as jnp
import jax
from jax.profiler import trace
import contextlib
from jax.experimental.compilation_cache import compilation_cache as cc
import jax.profiler
import socket
from contextlib import closing

import sys 
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






def benchmark_jax_runtime():
    """Test the JAX KalmanFilter class by loading data, initializing the model, setting parameters, and running the filter."""
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the invariant directory path
    data_path = "../data/IPTA_MockDataChallenge/IPTA_Challenge1_open/Challenge_Data/Dataset2/"
    directory = os.path.join(
        script_dir,
        data_path
    )

    # Get all .par and .tim files in the directory
    par_files = sorted(glob.glob(directory + "*.par"))
    tim_files = sorted(glob.glob(directory + "*.tim"))
    assert len(par_files) == len(tim_files), "Mismatch between .par and .tim file counts."

    #select only 2 pulsars
    # par_files = par_files[:2]
    # tim_files = tim_files[:2]


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

    print("Initializing the model")
    model = models.StochasticGWBackgroundModel(pulsar_metadata, hd_correlation_matrix, pulsar_design_matrices)

    # Initialize the JAX Kalman Filter
    x0 = jnp.ones(model.nx)
    P0 = jnp.eye(model.nx) * 1e1

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


    #utils.estimate_memory_usage(model, params)


    # Time compilation
    print("\nStarting compilation phase...")
    compilation_start = time.time()
    _ = KF.get_likelihood(params)
    compilation_end = time.time()
    print(f"Compilation time: {compilation_end - compilation_start:.4f} seconds")

    print("--------------------------------")
    print("--------------------------------")
    print("--------------------------------")
    print("--------------------------------")
    print("--------------------------------")
    print("--------------------------------")
    print("--------------------------------")
    print("--------------------------------")
    print("\n Running compiled execution")
    start_time = time.time()
    ll = KF.get_likelihood(params)
    jax.block_until_ready(ll)  # Ensure computation is complete
    end_time = time.time()
    
    print(f"Log-likelihood: {ll}")
    print(f"Execution time: {end_time - start_time:.4f} seconds")



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
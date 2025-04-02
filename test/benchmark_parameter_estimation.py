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
from jax import random

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

@struct.dataclass
class Parameters:
    
    #GW parameters
    γa: float  # s⁻¹
    h2: float  # GWB amplitude

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
    x0 = jnp.zeros(model.nx)
    P0 = jnp.eye(model.nx) * 1e-1

    KF = jax_kalman_filter.JaxScalarKalmanFilter(
        model=model, 
        observations=processed_pulsar_residuals, 
        x0=x0, 
        P0=P0
    )

    # Guess of the model parameters
    params = Parameters(
        #GW parameters
        γa=1e-10,
        h2=1e-12,

        #Spin parameters
        γp=jnp.ones(model.Npsr) * 1e-13,
        σp=jnp.ones(model.Npsr) * 1e-20,

        #Timing model noise parameters
        σeps=jnp.ones(model.M_sum) * 1e-20,

        #Measurement noise parameters
        EFAC=jnp.ones(model.Npsr),
        EQUAD=jnp.ones(model.Npsr) * (-6.7)
    )

    # Time compilation
    print("\nStarting compilation phase...")
    compilation_start = time.time()
    _ = KF.get_likelihood(params)
    compilation_end = time.time()
    print(f"Compilation time: {compilation_end - compilation_start:.4f} seconds")

    print("\nRunning profiled execution")
    start_time = time.time()
    ll = KF.get_likelihood(params)
    jax.block_until_ready(ll)  # Ensure computation is complete
    end_time = time.time()
    print(f"Log-likelihood: {ll}")
    print(f"Execution time: {end_time - start_time:.4f} seconds")



    # print("Starting NumPyro")

    # # NumPyro model
    # def model(kf):

    #     # Parameters of the GW background. These are just scalars
    #     γa = numpyro.sample("γa", dist.LogUniform(1e-11, 1e-6))
    #     h2 = numpyro.sample("h2", dist.LogUniform(1e-16, 1e-11)) 

    #     # # Sample array parameters with appropriate shapes
    #     # γp = numpyro.sample("γp", dist.Normal(-2, 1), sample_shape=(n_pulsars,))
    #     # σp = numpyro.sample("σp", dist.LogNormal(-20, 1), sample_shape=(n_pulsars,))
    #     # σeps = numpyro.sample("σeps", dist.LogNormal(-20, 1), sample_shape=(m_sum,))
    #     # f0 = numpyro.sample("f0", dist.Normal(100, 10), sample_shape=(n_pulsars,))
    #     # EFAC = numpyro.sample("EFAC", dist.Normal(1, 0.1), sample_shape=(n_pulsars,))
    #     # EQUAD = numpyro.sample("EQUAD", dist.Normal(1, 0.1), sample_shape=(n_pulsars,))

    #     # # Construct the Parameters object
    #     params = Parameters(
    #         γa=γa,
    #         h2=h2,
    #     )

    #     # Call the likelihood
    #     log_likelihood = kf.get_likelihood(params)
    #     numpyro.factor("likelihood", log_likelihood)

    # # Run MCMC
    # rng_key = random.PRNGKey(0)
    # nuts_kernel = NUTS(model)
    # mcmc = MCMC(nuts_kernel, num_samples=1000, num_warmup=500)
    # mcmc.run(rng_key, kf=KF)
    # mcmc.print_summary()  # Posterior estimates




if __name__ == "__main__":

    # Check available devices
    print("=== JAX VERSION INFO ===")
    print(f"JAX version: {jax.__version__}")

    print("\n=== DEVICE INFO ===")
    print("Default device:", jax.default_backend())

    # Check if GPU is available
    if any(d.platform == 'gpu' for d in jax.devices()):
        print("JAX GPU acceleration is AVAILABLE!")
        print("GPU devices:", [d for d in jax.devices() if d.platform == 'gpu'])
    else:
        print("JAX GPU acceleration is NOT available. Using CPU only.")
    print('-----------------------------------------------')



    #go
    benchmark_jax_runtime() 
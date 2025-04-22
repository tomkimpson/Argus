import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy.linalg import block_diag
from jax import random
from functools import partial # Useful for wrapping functions

import os
import glob
import sys
import json
import pandas as pd
import numpy as np
from flax import struct
from datetime import datetime

# NumPyro (Only needed for distributions if not replaced by jaxns.utils or scipy.stats)
# import numpyro
# import numpyro.distributions as dist
# from numpyro.infer import MCMC,SA,NUTS
# --> We will replace NumPyro sampling parts

# jaxns
import jaxns


# Arviz
import arviz as az


# Argus (Assuming this path is correct relative to your execution)
sys.path.append('../python/argus')
from argus import data_loader
from argus import models
from argus import jax_kalman_filter
from argus import gravitational_waves


@struct.dataclass
class Parameters:
    """Define a struct to store the parameters of the Kalman filter model"""
    # GW parameters
    γa: float  # s⁻¹ - Fixed in this model setup
    ha: float  # GWB amplitude

    # Pulsar parameters for the OU process
    γp: jnp.ndarray  # Pulsar-specific gamma values
    σp: jnp.ndarray  # Pulsar-specific sigma values

    # Measurement noise parameters
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
    print("Total length of the data is ", len(processed_pulsar_residuals[1]))
    print("Total number of pulsars is ", len(pulsar_metadata))
    return processed_pulsar_residuals, pulsar_metadata, pulsar_design_matrices,hd_correlation_matrix

def _initialize_kalman_filter(nx,Npsr,P_eps):
    """
    Specify the initial state vector x0 and the covariance matrix P0 for the Kalman filter.
    """
    # Initialize the JAX Kalman Filter
    x0 = jnp.zeros(nx) # Initial state vector. δφ=0,δf=0, etc. As all the states are effecitvely perturbations, this is a reasonable guess.

    #Initialize the covariance matrices
    P_GW = jnp.eye(Npsr * 2)
    P_GW = P_GW.at[1::2, 1::2].multiply(1e-20) # All the odd diagonal elements, (1,1), (3,3) etc. are set to 1e-12 # Note: Original comment said 1e-12, code uses 1e-20
    P_GW = P_GW.at[0::2, 0::2].multiply(1e-25) # All the even diagonal elements, (0,0), (2,2) etc. are set to 1e-18 # Note: Original comment said 1e-18, code uses 1e-25


    P_spin = jnp.eye(Npsr * 2)
    P_spin = P_spin.at[1::2, 1::2].multiply(1e-12) # All the odd diagonal elements, (1,1), (3,3) etc. are set to 1e-12
    P_spin = P_spin.at[0::2, 0::2].multiply(1e-20) # All the even diagonal elements, (0,0), (2,2) etc. are set to 1e-18 # Note: Original comment said 1e-18, code uses 1e-20


    P0 = block_diag(P_GW, P_spin, np.diag(P_eps)) # Need to ensure P_eps is numpy array here if using np.diag

    return x0, P0

# --- JAXNS specific functions ---
def prior_transform(u, Npsr):
    """
    Transforms samples from the unit cube `u` to the physical parameter space
    based on the prior distributions, using only JAX numpy.

    Args:
        u (dict): Dictionary of unit cube samples (values are JAX arrays).
                  Keys are parameter names ('ha', 'γp', 'σp', 'EFAC', 'EQUAD').
        Npsr (int): Number of pulsars.

    Returns:
        dict: Dictionary of parameters in the physical space.
    """
    theta = {}

    # ha: LogUniform(1e-14, 1e-10)
    low_ha, high_ha = 1e-14, 1e-10
    theta['ha'] = low_ha * (high_ha / low_ha)**u['ha'] # Direct LogUniform transformation

    # γp: LogUniform(1e-11, 1e-6), shape (Npsr,)
    low_gp, high_gp = 1e-11, 1e-6
    theta['γp'] = low_gp * (high_gp / low_gp)**u['γp'] # Applied element-wise

    # σp: LogUniform(1e-18, 1e-12), shape (Npsr,)
    low_sp, high_sp = 1e-18, 1e-12
    theta['σp'] = low_sp * (high_sp / low_sp)**u['σp'] # Applied element-wise

    # EFAC: Uniform(0.5, 2), shape (Npsr,)
    low_efac, high_efac = 0.5, 2.0
    theta['EFAC'] = low_efac + u['EFAC'] * (high_efac - low_efac) # Direct Uniform transformation

    # EQUAD: LogUniform(1e-7, 1e-6), shape (Npsr,)
    low_eq, high_eq = 1e-7, 1e-6
    theta['EQUAD'] = low_eq * (high_eq / low_eq)**u['EQUAD'] # Applied element-wise

    return theta


def log_likelihood_jaxns(theta, kf, fixed_params):
    """
    Calculates the log likelihood for jaxns.

    Args:
        theta (dict): Dictionary of sampled parameters in physical space.
        kf (JaxKalmanFilter): The initialized Kalman filter object.
        fixed_params (dict): Dictionary containing fixed parameters (like γa).

    Returns:
        float: The log likelihood value.
    """
    # Construct the Parameters object
    params = Parameters(
        γa=fixed_params['γa'],
        ha=theta['ha'],
        γp=theta['γp'],
        σp=theta['σp'],
        EFAC=theta['EFAC'],
        EQUAD=theta['EQUAD']
    )

    # Call the likelihood function from the Kalman Filter object
    log_likelihood = kf.get_likelihood(params)
    return log_likelihood

# --- Main parameter estimation function ---

def parameter_estimation():

    #Get the data
    data_path = "../data/IPTA_MockDataChallenge2/dataset_1b/" # https://github.com/ipta/mdc2/tree/master
    processed_pulsar_residuals, pulsar_metadata, pulsar_design_matrices,hd_correlation_matrix = _get_processed_residuals(data_path)


    #Initialize the model
    model = models.StochasticGWBackgroundModel(pulsar_metadata, hd_correlation_matrix, pulsar_design_matrices)


    #Initialize the covariance matrix for \delta \epsilon
    delta = 1e-3 #milliseconds
    # Ensure P0 components are standard numpy arrays for block_diag if needed later
    P0_eps_list = [delta**2  / np.max(pulsar_design_matrices[i],axis=0)**2 for i in range(len(pulsar_design_matrices))]
    for i in range(len(P0_eps_list)):
        assert len(P0_eps_list[i]) == model.M[i]

    P0_eps = np.concatenate(P0_eps_list) # This is now a numpy array
    assert len(P0_eps) == model.M_sum

    #Initialize the model state and covariance
    x_init,P_init = _initialize_kalman_filter(model.nx,model.Npsr,P0_eps) # Pass numpy array P0_eps

    # Initialize Kalman Filter Object
    KF = jax_kalman_filter.JaxKalmanFilter(
        model=model,
        observations=processed_pulsar_residuals,
        x0=x_init,
        P0=P_init # P_init is now a JAX array after block_diag
    )

    # --- Define fixed parameters and parameter names for sampling ---
    fixed_gamma_a = 1e-9
    fixed_params = {'γa': fixed_gamma_a}

    # Define the parameters to be sampled and their shapes for the prior transform
    # The shapes correspond to the unit cube input `u`
    param_names_and_shapes = {
        'ha': jaxns.PriorParam(shape=(), low=0., high=1.), # Scalar
        'γp': jaxns.PriorParam(shape=(model.Npsr,), low=0., high=1.),
        'σp': jaxns.PriorParam(shape=(model.Npsr,), low=0., high=1.),
        'EFAC': jaxns.PriorParam(shape=(model.Npsr,), low=0., high=1.),
        'EQUAD': jaxns.PriorParam(shape=(model.Npsr,), low=0., high=1.)
    }
    param_names = list(param_names_and_shapes.keys())


    # --- Pre-compile the likelihood (optional but good practice) ---
    print("Compiling likelihood function...")
    # Create dummy unit samples
    dummy_u = {name: jnp.full(shape.shape, 0.5) for name, shape in param_names_and_shapes.items()}
    # Transform to physical space
    dummy_theta = prior_transform(dummy_u, model.Npsr)
     # Wrap the log_likelihood function with fixed arguments
    compiled_loglik_func = partial(log_likelihood_jaxns, kf=KF, fixed_params=fixed_params)
    # JIT compile
    jitted_loglik = jax.jit(compiled_loglik_func)
    ll = jitted_loglik(dummy_theta)
    jax.block_until_ready(ll)  # Ensure computation is complete
    print(f"Likelihood on compilation run is {ll}")


    # --- Setup and run jaxns ---
    print("\nStarting jaxns Nested Sampling")

    # Wrap the prior transform function to include Npsr
    wrapped_prior_transform = partial(prior_transform, Npsr=model.Npsr)

    # Instantiate the Nested Sampler
    # Choose the number of live points - this is crucial for accuracy vs computation time
    num_live_points = 1000 # Example value, adjust based on dimensionality and desired accuracy
    print(f"Using {num_live_points} live points.")

    ns = jaxns.NestedSampler(
        log_likelihood_func=jitted_loglik, # Use the JIT-compiled version
        prior_transform_func=wrapped_prior_transform,
        param_names_and_shapes=param_names_and_shapes,
        num_live_points=num_live_points,
        sampler_name='slice' # Or 'multi_ellipsoid', slice is often robust
    )

    # Define termination condition
    # Stop when the estimated remaining evidence contribution is small
    term_cond = jaxns.TerminationCondition(live_evidence_frac=1e-3) # Example value

    # Run the sampler
    rng_key = random.PRNGKey(0)
    results, state = ns(rng_key, term_cond=term_cond) # Use ns() instead of ns.run() for newer jaxns versions

    print("jaxns run complete.")
    results.print_summary(sampler_name='jaxns')

    # --- Process and save results ---
    print("Completed. Saving results to disk...")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    fname_base = f"outputs/jaxns_parameter_estimation_results_{timestamp}"

    # 1. Save raw jaxns results object (recommended)
    # You might need a library like 'pickle' or 'cloudpickle' if saving complex JAX objects
    try:
        import cloudpickle
        with open(f"{fname_base}_results.pkl", "wb") as f:
            cloudpickle.dump(results, f)
        print(f"Saved raw jaxns results to {fname_base}_results.pkl")
    except ImportError:
        print("Install 'cloudpickle' to save the raw jaxns results object.")
    except Exception as e:
        print(f"Could not save raw jaxns results: {e}")


    # 2. Convert to ArviZ InferenceData (Optional, for plotting/compatibility)
    print("Converting to ArviZ InferenceData...")
    try:
        # jaxns results contain weighted samples. Resample to get ~equally weighted posterior samples.
        num_posterior_samples = 5000 # Choose desired number
        posterior_samples_dict = results.get_posterior_samples(rng_key, num_posterior_samples)

        # Prepare data for ArviZ
        arviz_data = {
            'posterior': posterior_samples_dict,
            # Add prior samples if needed (can generate using prior_transform)
            # Add observed data if needed (KF.observations)
            # Add sampler stats (like log_evidence)
            'log_likelihood': {'log_likelihood': results.log_L_samples}, # LogL per sample *before* resampling
            'sampler_stats': {'log_evidence': results.log_Z,
                              'log_evidence_err': results.log_Z_err}
        }

        # Need to structure observed_data correctly if adding
        # observed_data_dict = {'y': KF.observations} # This might need flattening or reshaping

        inf_data = az.from_dict(
             posterior=arviz_data['posterior'],
             log_likelihood=arviz_data['log_likelihood'], # Check shape compatibility with posterior
             # prior=...
             # observed_data=observed_data_dict,
             coords={'pulsar': np.arange(model.Npsr)}, # Assuming order matches
             dims={'γp': ['pulsar'], 'σp': ['pulsar'], 'EFAC': ['pulsar'], 'EQUAD': ['pulsar']},
             # Add sampler_stats separately as dataset attributes might be better
        )
        # Add log evidence to dataset attributes
        inf_data.attrs['log_evidence'] = results.log_Z
        inf_data.attrs['log_evidence_err'] = results.log_Z_err


        fname_nc = f"{fname_base}.nc"
        inf_data.to_netcdf(fname_nc)
        print(f"Saved ArviZ results to {fname_nc}")

        # Print ArviZ summary
        print("\nArviZ Summary (from resampled posterior):")
        print(az.summary(inf_data, round_to=3))

    except Exception as e:
        print(f"Could not convert or save ArviZ InferenceData: {e}")
        print("Raw jaxns results might still be available if saved.")


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

    # Ensure output directory exists
    os.makedirs("outputs", exist_ok=True)

    #go
    parameter_estimation()
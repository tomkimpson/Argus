"""A working example to use this package."""

import os
import glob
from argus import data_loader, models, kalman_filter, gravitational_waves
import numpy as np
import jax.numpy as jnp
import timeit

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = (
    "../data/IPTA_MockDataChallenge/IPTA_Challenge1_open/Challenge_Data/Dataset2/"
)
directory = os.path.join(script_dir, data_path)

# # Get all .par and .tim files in the directory
par_files = sorted(glob.glob(directory + "*.par"))
tim_files = sorted(glob.glob(directory + "*.tim"))
assert len(par_files) == len(tim_files), "Mismatch between .par and .tim file counts."

# # Get the data
print(f"Getting the data. Loading {len(par_files)} pulsars from {data_path}")
pulsar_residuals, pulsar_metadata, pulsar_design_matrices = (
    data_loader.LoadWidebandPulsarData.read_multiple_par_tim(par_files, tim_files)
)

# # Get the separation angles and compute HD correlation
ra = pulsar_metadata["RA"].to_numpy(dtype=float)
dec = pulsar_metadata["DEC"].to_numpy(dtype=float)
angular_separation_matrix = (
    data_loader.LoadWidebandPulsarData.pairwise_angular_separation(ra, dec)
)
hd_correlation_matrix = gravitational_waves.hellings_downs(angular_separation_matrix)

# # Post-process the residuals.
processed_pulsar_residuals = data_loader.LoadWidebandPulsarData.post_process_residuals(
    pulsar_residuals
)

print("Total length of the data is ", len(processed_pulsar_residuals))
print("Total number of pulsars is ", len(pulsar_metadata))

print("Initializing the model")
model = models.StochasticGWBackgroundModel(
    pulsar_metadata, hd_correlation_matrix, pulsar_design_matrices
)
State = model.stateclass
Covariance = model.covclass

# Initialize the Kalman Filter
x_gw0 = jnp.zeros((2 * model.Npsr))
x_spin0 = jnp.zeros((2 * model.Npsr))
x_eps0 = jnp.zeros((model.M_sum))
x0 = State.prior(gw=x_gw0, spin=x_spin0, eps=x_eps0)

# Initialise the covariance matrices
# P_gw0   = np.zeros((2*model.Npsr, 2*model.Npsr))
# P_spin0 = np.zeros((2*model.Npsr, 2*model.Npsr))
# P_eps0  = np.zeros((model.M_sum, model.M_sum))

P_gw0 = jnp.eye(2 * model.Npsr) * 1e-12
P_spin0 = jnp.eye(2 * model.Npsr) * 1e-12
P_eps0 = jnp.eye(model.M_sum) * 1e-12

P_gw_spin0 = jnp.zeros((2 * model.Npsr, 2 * model.Npsr))
P_gw_eps0 = jnp.zeros((2 * model.Npsr, model.M_sum))
P_spin_eps0 = jnp.zeros((2 * model.Npsr, model.M_sum))

P0 = Covariance.prior(
    gw=P_gw0,
    spin=P_spin0,
    eps=P_eps0,
    gw_spin=P_gw_spin0,
    gw_eps=P_gw_eps0,
    spin_eps=P_spin_eps0,
)

KF = kalman_filter.ScalarKalmanFilter(
    model=model, observations=processed_pulsar_residuals, x0=x0, P0=P0
)

# # Set global parameters. In an inference run we will search for the best parameters.
params = {
    "γa": 1e-1,  # s⁻¹
    "γp": 1e-1 * np.ones(len(pulsar_metadata)),
    "σp": 1e-20 * np.ones(len(pulsar_metadata)),
    "h2": 1e-12,
    "σeps": 1e-20 * np.ones(model.M_sum),
    "f0": 100 * np.ones(len(pulsar_metadata)),  # everything is 100 Hz for now
    "EFAC": np.ones(len(pulsar_metadata)),
    "EQUAD": np.ones(len(pulsar_metadata)),
}

ll = KF.get_likelihood(params)

t = timeit.timeit(lambda: KF.get_likelihood(params), number=10)
avg = t / 10
print(f"average running time: {avg}")

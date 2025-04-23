import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy.linalg import block_diag


import os 
import glob
import sys 
import json
import pandas as pd
import numpy as np 
from flax import struct
import numpyro.distributions as dist

sys.path.append('../python/argus')
from argus import data_loader
from argus import models
from argus import jax_kalman_filter
from argus import gravitational_waves



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
    P_GW = P_GW.at[1::2, 1::2].multiply(1e-20) # All the odd diagonal elements, (1,1), (3,3) etc. are set to 1e-12
    P_GW = P_GW.at[0::2, 0::2].multiply(1e-25) # All the even diagonal elements, (0,0), (2,2) etc. are set to 1e-18


    P_spin = jnp.eye(Npsr * 2)
    P_spin = P_spin.at[1::2, 1::2].multiply(1e-12) # All the odd diagonal elements, (1,1), (3,3) etc. are set to 1e-12
    P_spin = P_spin.at[0::2, 0::2].multiply(1e-20) # All the even diagonal elements, (0,0), (2,2) etc. are set to 1e-18


    P0 = block_diag(P_GW, P_spin, np.diag(P_eps))

    return x0, P0



#Get the data
data_path = "../data/IPTA_MockDataChallenge/IPTA_Challenge1_open/Challenge_Data/Dataset1/" 
processed_pulsar_residuals, pulsar_metadata, pulsar_design_matrices,hd_correlation_matrix = _get_processed_residuals(data_path)




#Calculate P0 based on the maximum value of the design matrix, and a delta tolerance
model = models.StochasticGWBackgroundModel(pulsar_metadata, hd_correlation_matrix, pulsar_design_matrices)

delta = 1e-3 #milliseconds
P0 = [delta**2  / np.max(pulsar_design_matrices[i],axis=0)**2 for i in range(len(pulsar_design_matrices))]

for i in range(len(P0)):
    assert len(P0[i]) == model.M[i]

P0 = np.concatenate(P0)
assert len(P0) == model.M_sum


#Initialize the model
x_init,P_init = _initialize_kalman_filter(model.nx,model.Npsr,P0) #this could go inside the model class....


KF = jax_kalman_filter.JaxKalmanFilter(
    model=model, 
    observations=processed_pulsar_residuals, 
    x0=x_init, 
    P0=P_init
)


γa = 1e-6 
ha = 1e-12




#Set the parameters
params = Parameters(
    #GW parameters
    γa=γa,
    ha=ha,

    #Spin parameters
    γp=jnp.ones(model.Npsr)*1e-9,
    σp=jnp.ones(model.Npsr)*1e-15,

    #Measurement noise parameters
    EFAC=jnp.ones(model.Npsr),
    EQUAD=jnp.zeros(model.Npsr)
)

print("First call to get_likelihood")
ll = KF.get_likelihood(params)
ll.block_until_ready()
print("Likelihood: ",ll)
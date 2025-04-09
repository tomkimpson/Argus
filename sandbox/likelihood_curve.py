import time
import jax
import jax.numpy as jnp
from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC
import jax.profiler 
from numpyro.infer import SA

import arviz as az

import matplotlib.pyplot as plt

# Configure JAX
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", 'cpu')

# Configure NumPyro to use the same device as JAX
numpyro.set_platform(jax.default_backend())
numpyro.set_host_device_count(len(jax.devices()))

from benchmark_runtime_jax import Parameters, _get_processed_residuals, initialize_kalman_filter
from argus import models, jax_kalman_filter




def likelihood_curve():
    #Get the data
    data_path = "../data/IPTA_MockDataChallenge2/dataset_1b/" # https://github.com/ipta/mdc2/tree/master
    processed_pulsar_residuals, pulsar_metadata, pulsar_design_matrices,hd_correlation_matrix = _get_processed_residuals(data_path)

    #Initialize the model
    model = models.StochasticGWBackgroundModel(pulsar_metadata, hd_correlation_matrix, pulsar_design_matrices)


    x0,P0 = initialize_kalman_filter(model.nx,model.Npsr,model.M_sum) #this could go inside the model class....

    print(P0)

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
        ha=6e-14,

        #Spin parameters
        γp=jnp.ones(model.Npsr) * 1e-13, #1/year timescale. Assumed the same for all pulsars
        σp=jnp.ones(model.Npsr) * 1e-22, #For now, assume the same noise for all pulsars

        #Timing model noise parameters
        σeps=jnp.ones(model.M_sum) * 1e-12, #TBD a good value for the timing model noise. There are some rough estimates in data_loader.py, but not sure how accurate they are.
        #Measurement noise parameters
        EFAC=jnp.ones(model.Npsr)*1e10,
        EQUAD=jnp.ones(model.Npsr) * (-6.7)
    )


    #Load some "truth" parameters from file
    import pandas as pd
    truth_params = pd.read_pickle("../notebooks/spindown_results.pkl")









    #pre-compile
    ll1 = KF.get_likelihood(params)
    print(ll1)
    import numpy as np
    num = 3
    plot_y = np.zeros(num)
    #plot_x = np.logspace(-11, -6, num)
    plot_x = np.array([1e-11, 1e-9,1e-6])
    for i in range(num):

        gamma_a = plot_x[i]


        params = Parameters(
            #GW parameters
            γa=gamma_a,
            ha=1e-14,

            #Spin parameters
            γp=jnp.ones(model.Npsr) * 1e-13, #1/year timescale. Assumed the same for all pulsars
            σp=jnp.ones(model.Npsr) * 1e-22, #For now, assume the same noise for all pulsars

            #Timing model noise parameters
            σeps=jnp.ones(model.M_sum) * 1e-10, #TBD a good value for the timing model noise. There are some rough estimates in data_loader.py, but not sure how accurate they are.

            #Measurement noise parameters
            EFAC=jnp.ones(model.Npsr),
            EQUAD=jnp.ones(model.Npsr) * (-6.7)
        )
        ll = KF.get_likelihood(params)
        plot_y[i] = ll

        print(f"gamma_a: {plot_x[i]}, ll: {ll}")


    plt.plot(plot_x, plot_y)
    plt.xscale("log")
    plt.xlim(1e-11, 1e-6)
    plt.ylim(np.min(plot_y), np.max(plot_y))
    print("saving")
    plt.savefig("outputs/likelihood_curve.png")
    plt.show()







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
    likelihood_curve() 

import time
import jax
import jax.numpy as jnp
from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC
import jax.profiler 
#from numpyro.infer import HMC  # Less memory-intensive than NUTS
from numpyro.infer import SA
import arviz as az


# Configure JAX
jax.config.update("jax_enable_x64", True)
#jax.config.update("jax_enable_x64", False)

# Configure NumPyro to use the same device as JAX
numpyro.set_platform(jax.default_backend())
numpyro.set_host_device_count(len(jax.devices()))

from benchmark_runtime_jax import Parameters, _get_processed_residuals, initialize_kalman_filter
from argus import models, jax_kalman_filter



def benchmark_jax_parameter_estimation():

    #Get the data
    data_path = "../data/IPTA_MockDataChallenge2/dataset_1b/" # https://github.com/ipta/mdc2/tree/master
    processed_pulsar_residuals, pulsar_metadata, pulsar_design_matrices,hd_correlation_matrix = _get_processed_residuals(data_path)

    #Initialize the model
    model = models.StochasticGWBackgroundModel(pulsar_metadata, hd_correlation_matrix, pulsar_design_matrices)


    x0,P0 = initialize_kalman_filter(model.nx,model.Npsr,model.M_sum,hd_correlation_matrix) #this could go inside the model class....


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
        EQUAD=jnp.ones(model.Npsr) * 10**(-6.7)
    )


    #pre-compile

    _ = KF.get_likelihood(params)
    #jax.profiler.save_device_memory_profile("memory_precompile.prof")

    ll = KF.get_likelihood(params)
    jax.block_until_ready(ll)  # Ensure computation is complete
    #jax.profiler.save_device_memory_profile("memory_postcompile.prof")


   


    new_params = Parameters(
        #GW parameters
        γa=2e-9,
        ha=2e-12,

        #Spin parameters
        γp=jnp.ones(model.Npsr) * 2e-8, #1/year timescale. Assumed the same for all pulsars
        σp=jnp.ones(model.Npsr) * 1e-14, #For now, assume the same noise for all pulsars

        #Timing model noise parameters
        σeps=jnp.ones(model.M_sum) * 1e-12, #TBD a good value for the timing model noise. There are some rough estimates in data_loader.py, but not sure how accurate they are.



        #Measurement noise parameters
        EFAC=jnp.ones(model.Npsr),
        EQUAD=jnp.ones(model.Npsr) * 10**(-6.7)
    )






    ll = KF.get_likelihood(params)
    jax.block_until_ready(ll)  # Ensure computation is complete
    #jax.profiler.save_device_memory_profile("memory_postcompile_second.prof")

    #Now do parameter estimation
    print("Starting NumPyro")
    
    # Check NumPyro device usage
    print("\n=== NUMPYRO DEVICE INFO ===")
    print(f"NumPyro version: {numpyro.__version__}")
    print("--------------------------------")



    # NumPyro model
    def numpyro_model(kf):
        #jax.profiler.save_device_memory_profile("memory_during_model_call.prof")

        # Parameters of the GW background
        γa = numpyro.sample("γa", dist.LogUniform(1e-11, 1e-6))
        ha = numpyro.sample("ha", dist.LogUniform(1e-16, 1e-11))

        #Parameters of the pulsar process
        γp = numpyro.sample("γp", dist.LogUniform(1e-11, 1e-6),sample_shape=(model.Npsr,))
        σp = numpyro.sample("σp", dist.LogUniform(1e-18, 1e-13),sample_shape=(model.Npsr,))

        #Timing model noise parameters
        #σeps = numpyro.sample("σeps", dist.LogUniform(1e-16, 1e-11),sample_shape=(model.M_sum,))
        #σeps


        known_σeps = jnp.ones(model.M_sum) * 1e-10
        σeps = numpyro.deterministic("σeps", known_σeps)
        
        #Measurement noise parameters
        EFAC = numpyro.sample("EFAC", dist.Uniform(0.5, 2),sample_shape=(model.Npsr,))

        EQUAD = numpyro.sample("EQUAD", dist.LogUniform(1e-8, 1e-5),sample_shape=(model.Npsr,))

 


        # Construct the Parameters object
        params = Parameters(
            γa=γa,
            ha=ha,
            γp=γp,
            σp=σp,
            σeps=σeps,
            EFAC=EFAC,
            EQUAD=EQUAD
        )
        
        # Call the likelihood
        #jax.profiler.save_device_memory_profile("memory_during_likelihood_call.prof")

        log_likelihood = kf.get_likelihood(params)


        numpyro.factor("likelihood", log_likelihood)
    
    # Run MCMC
    print("RNG")
    rng_key = random.PRNGKey(0)
    
    
    
    
    # nuts_kernel = NUTS(numpyro_model)
    # jax.profiler.save_device_memory_profile("memory_nut_kernel.prof")
    # mcmc = MCMC(nuts_kernel, num_samples=1000, num_warmup=500,num_chains=1,progress_bar=True)
    
    

    # Use Simulated Annealing kernel
    print("SA kernel")
    sa_kernel = SA(numpyro_model)
    print("MCMC")
    mcmc = MCMC(sa_kernel, num_samples=1000, num_warmup=500, num_chains=1, progress_bar=True)

    mcmc.run(rng_key, kf=KF)
    
    
    

    mcmc.run(rng_key, kf=KF)
    mcmc.print_summary()  # Posterior estimates







    print("Completed. Saving results to disk...")
    inf_data = az.from_numpyro(mcmc)
    inf_data.to_netcdf("outputs/SA_data_4_chains.nc")










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
    benchmark_jax_parameter_estimation() 

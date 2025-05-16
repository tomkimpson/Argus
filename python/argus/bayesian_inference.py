"""Bayesian inference module for pulsar timing array analysis.

This module provides functionality for performing Bayesian parameter estimation
on pulsar timing array data using nested sampling. It includes:

- Prior definitions for gravitational wave background and pulsar noise parameters
- Likelihood calculations using a Kalman filter implementation
- Nested sampling routines using the JAXNS package

The module is designed to work with the JAX framework for automatic differentiation
and GPU acceleration. It handles parameters like:

- Gravitational wave background amplitude (ha) and spectral index (γa)
- Pulsar-specific red noise parameters (γp, σp)
- White noise parameters (EFAC, EQUAD)

The implementation uses the Hellings-Downs correlation pattern for the 
gravitational wave background and models pulsar red noise as an 
Ornstein-Uhlenbeck process.
"""

#Jax
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

#Jaxns
from jaxns import Prior, Model, NestedSampler,TerminationCondition # Import necessary components

#Tensorflow - for setting up the prior distributions
import tensorflow_probability.substrates.jax as tfp
tfpd = tfp.distributions


#Flax
from flax import struct



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



def configurable_prior_model(
    Npsr: int, # Number of pulsars, needed for array shapes if defaults are used
    # --- Specifications for each parameter ---
    # Each *_spec can be a TFP distribution object or a fixed jnp.ndarray/float.
    # If None, a default distribution will be used for some, or an error raised for others.
    log10_ha_spec = tfpd.Uniform(-17.0, -14.0),
    gamma_a_spec = 1e-9, # Typically fixed
    log10_gamma_p_spec = None,
    log10_sigma_p_spec = None,
    efac_spec = None,
    equad_spec = None
):
    """
    Defines prior distributions for the parameters.
    Each parameter's specification (e.g., `log10_ha_spec`) can be:
    - A TFP distribution object (e.g., tfpd.Uniform(...)) for sampling.
    - A scalar or jnp.ndarray for a fixed value.
    - None, in which case a default distribution is used (for pulsar params)
      or an error is raised (for efac/equad which must be specified).
    """

    # GW parameters
    # If log10_ha_spec is a value (e.g. -15.0), Prior will use it as fixed.
    # If log10_ha_spec is tfpd.Uniform(...), Prior will sample from it.
    log10_ha = yield Prior(log10_ha_spec, name='log10_ha')
    γa = yield Prior(gamma_a_spec, name='γa') # Often fixed

    # PSR vector parameters: γp and σp.
    # Use default uniform distributions if no specific spec is provided.
    _log10_gamma_p_spec_to_use = log10_gamma_p_spec
    if _log10_gamma_p_spec_to_use is None:
        _log10_gamma_p_spec_to_use = tfpd.Uniform(low=jnp.full(Npsr, -11.0), high=jnp.full(Npsr, -6.0)) # Default
    
    _log10_sigma_p_spec_to_use = log10_sigma_p_spec
    if _log10_sigma_p_spec_to_use is None:
        _log10_sigma_p_spec_to_use = tfpd.Uniform(low=jnp.full(Npsr, -18.0), high=jnp.full(Npsr, -12.0)) # Default

    log10_γp = yield Prior(_log10_gamma_p_spec_to_use, name='log10_γp')
    log10_σp = yield Prior(_log10_sigma_p_spec_to_use, name='log10_σp')
    
    # Measurement noise parameters: EFAC and EQUAD.
    # These usually come from pulsar data files or are specific to the analysis,
    # so we require them to be explicitly passed. They can be fixed arrays or distributions.
    if efac_spec is None:
        raise ValueError("efac_spec must be provided to configurable_prior_model. "
                         "It can be a fixed array or a TFP distribution.")
    if equad_spec is None:
        raise ValueError("equad_spec must be provided to configurable_prior_model. "
                         "It can be a fixed array or a TFP distribution.")

    efac = yield Prior(efac_spec, name='efac')
    equad = yield Prior(equad_spec, name='equad')

    return log10_ha, γa, log10_γp, log10_σp, efac, equad    


















### TO BE REMOVED--------------
import json
Npsr = 32
def get_efac_equad_injections():

    # Load the noise parameters from the json file
    with open("../data/IPTA_MockDataChallenge2/group1_psr_noise.json", "r") as f:
        noise_params = json.load(f)

    # Extract EFAC and EQUAD values for each pulsar
    efac_values = []
    equad_values = []

    for psr in noise_params:

        if  "J1640" not in psr:
            efac_values.append(noise_params[psr]["efac"])
            equad_values.append(10**noise_params[psr]["equad"]) # Convert from log10 to linear

    # Convert to JAX arrays
    efac_array = jnp.array(efac_values)
    equad_array = jnp.array(equad_values)


    return efac_array, equad_array

efac_array, equad_array = get_efac_equad_injections()
### TO BE REMOVED--------------



def gw_prior_model():

    """Defines the prior distributions for the parameters."""
    
    # GW parameters: ha and γa. We fix gamma to be 1e-9 and use a log transform for ha
    log10_ha = yield Prior(tfpd.Uniform(-17.0, -14.0), name='log10_ha')
    γa = yield Prior(1e-9, name='γa')



    #PSR vector parameters: γp and σp. We use a uniform prior for the log of the parameters
    log10_γp = yield Prior(
                            tfpd.Uniform(low=jnp.full(Npsr, -11.0), high=jnp.full(Npsr, -6.0)), #logU(-11,-6)
                            name='log10_γp'
                        )

    log10_σp = yield Prior(
                            tfpd.Uniform(low=jnp.full(Npsr, -18.0), high=jnp.full(Npsr, -12.0)), #logU(-18,-12)
                            name='log10_σp'
                        )
    

    #Measurement noise parameters: EFAC and EQUAD. We use a uniform prior for the log of the parameters
    efac = yield Prior(efac_array, name='efac')
    equad = yield Prior(equad_array, name='equad')

    return log10_ha,γa, log10_γp, log10_σp,efac,equad

def null_prior_model(Npsr,efac_array,equad_array):

    """Defines the prior distributions for the parameters for the null (no GW) model."""
    
    # GW parameters: ha and γa. We fix gamma to be 1e-9 and use a log transform for ha
    log10_ha = yield Prior(-15, name='log10_ha')
    γa = yield Prior(1e-9, name='γa')



    #PSR vector parameters: γp and σp. We use a uniform prior for the log of the parameters
    log10_γp = yield Prior(
                            tfpd.Uniform(low=jnp.full(Npsr, -11.0), high=jnp.full(Npsr, -6.0)), #logU(-11,-6)
                            name='log10_γp'
                        )

    log10_σp = yield Prior(
                            tfpd.Uniform(low=jnp.full(Npsr, -18.0), high=jnp.full(Npsr, -12.0)), #logU(-18,-12)
                            name='log10_σp'
                        )
    

    #Measurement noise parameters: EFAC and EQUAD. We use a uniform prior for the log of the parameters
    efac = yield Prior(efac_array, name='efac')
    equad = yield Prior(equad_array, name='equad')

    return log10_ha,γa, log10_γp, log10_σp,efac,equad


# JAXNS model
def jaxns_log_likelihood(KF, log10_ha, γa, log10_γp, log10_σp, efac, equad):
    ha = 10.0 ** log10_ha
    γp = 10.0 ** log10_γp
    σp = 10.0 ** log10_σp

    params = Parameters(
        γa=γa,
        ha=ha,
        γp=γp,
        σp=σp,
        EFAC=efac,
        EQUAD=equad
    )

    return KF.get_likelihood(params)



def print_parameters(params: Parameters):
    """Print all entries of a Parameters struct."""
    for field in params.__dataclass_fields__:
        value = getattr(params, field)
        print(f"{field}: {value}")
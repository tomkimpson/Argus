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


def simple_prior_model(Npsr,efac_array,equad_array,sigma_p_log, gamma_p_log):

    """Defines the prior distributions for the parameters."""
    
    # GW parameters: ha and γa. We fix gamma to be 1e-9 and use a log transform for ha
    log10_ha = yield Prior(tfpd.Uniform(-17.0, -14.0), name='log10_ha')
    γa = yield Prior(1e-9, name='γa')

    #PSR vector parameters: γp and σp. We use a uniform prior for the log of the parameters
    log10_γp = yield Prior(gamma_p_log,name='gamma_p')
    log10_σp = yield Prior(sigma_p_log,name='sigma_p')
    

    #Measurement noise parameters: EFAC and EQUAD. We use a uniform prior for the log of the parameters
    efac = yield Prior(efac_array, name='efac')
    equad = yield Prior(equad_array, name='equad')

    return log10_ha,γa, log10_γp, log10_σp,efac,equad




def gw_prior_model(Npsr,efac_array,equad_array):

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
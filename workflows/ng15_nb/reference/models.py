"""Discovery models for the NG15 narrowband reference (spec D3), with a traced ε-ORF.

Every model is NANOGrav's 15-yr GWB likelihood (Agazie et al. 2023): per-backend
EFAC/EQUAD and ECORR fixed at the feather noise dictionaries, the full timing model
(DMX included) marginalised under Discovery's 1e40 improper prior, a per-pulsar red
noise process on ``rn_components`` Fourier modes, and a common process on
``gw_components`` modes. What changes between models is only the spectral shape:

- ``kernel="pl"``: NANOGrav's power laws (C3).
- ``kernel="ou"``: Argus's integrated-OU spectra (``ou_psd.py``) for both the
  per-pulsar and the common process (C4, C5).

The common process carries the correlation-path ORF Γ(ε) = (1 − ε) I + ε Γ_HD, with ε
an ordinary likelihood parameter (``orf_epsilon``). That is what makes the path-sampling
integrand ∂lnL/∂ε available by autodiff, exactly as in Argus. Discovery's own
``makeglobalgp_fourier`` bakes the ORF into numpy constants, so it is rebuilt here with
Γ(ε) in jax; the inverse keeps Discovery's Kronecker shortcut,
Φ⁻¹ = Γ⁻¹ ⊗ diag(1/φ), log|Φ| = n_f log|Γ| + n_psr Σ log φ.

Parameter names: red noise ``<psr>_red_noise_<arg>``; common process ``gw_<arg>``.
For ``kernel="ou"`` the red-noise argument is ``log10_sigma_r`` (residual units); the
sampler maps Argus's ``log10_sigma_p`` onto it with each pulsar's f0.
"""

import inspect

import discovery as ds
import jax
import jax.numpy as jnp
import numpy as np
from discovery import matrix

import ou_psd

GW_NAME = "gw"
RN_NAME = "red_noise"
EPS = "orf_epsilon"


def hd_matrix(psrs):
    return np.array([[ds.hd_orf(p1.pos, p2.pos) for p1 in psrs] for p2 in psrs])


def orf_path(hd, eps):
    """Γ(ε) = (1 − ε) I + ε Γ_HD. Γ_HD has a unit diagonal, so the diagonal stays 1."""
    return (1.0 - eps) * jnp.eye(hd.shape[0]) + eps * hd


def make_epsilon_gp(psrs, prior, components, T, hd=None, name=GW_NAME):
    """Common Fourier GP with ORF Γ(ε); ε is the likelihood parameter ``orf_epsilon``.

    ``prior(f, df, *args)`` returns the per-coefficient variance; its argument names
    become the parameters ``{name}_{arg}``.
    """
    hd = jnp.asarray(hd_matrix(psrs) if hd is None else hd)
    npsr = len(psrs)
    argnames = [a for a in _argnames(prior) if a not in ("f", "df")]
    argmap = [f"{name}_{a}" for a in argnames]

    fs, dfs, fmats = zip(*[ds.fourierbasis(psr, components, T=T) for psr in psrs])
    f, df = matrix.jnparray(fs[0]), matrix.jnparray(dfs[0])
    nf = f.shape[0]

    def phi_of(params):
        return prior(f, df, *[params[a] for a in argmap])

    def priorfunc(params):
        return jnp.kron(orf_path(hd, params[EPS]), jnp.diag(phi_of(params)))

    priorfunc.params = sorted(argmap + [EPS])
    priorfunc.type = jax.Array

    def invprior(params):
        orf = orf_path(hd, params[EPS])
        phi = phi_of(params)
        return jnp.kron(
            jnp.linalg.inv(orf), jnp.diag(1.0 / phi)
        ), nf * jnp.linalg.slogdet(orf)[1] + npsr * jnp.sum(jnp.log(phi))

    invprior.params = priorfunc.params
    invprior.type = jax.Array

    gp = matrix.GlobalVariableGP(matrix.NoiseMatrix12D_var(priorfunc), fmats)
    gp.Phi_inv, gp.factors = invprior, None
    gp.index = {
        f"{psr.name}_{name}_coefficients({nf})": slice(nf * i, nf * (i + 1))
        for i, psr in enumerate(psrs)
    }
    gp.pos = [psr.pos for psr in psrs]
    gp.name = [psr.name for psr in psrs]
    return gp


def pulsar_models(psrs):
    """NANOGrav's per-pulsar pieces: fixed white noise + ECORR, marginalised timing model."""
    return [
        ds.PulsarLikelihood(
            [
                psr.residuals,
                ds.makenoise_measurement(psr, psr.noisedict),
                ds.makegp_ecorr(psr, psr.noisedict),
                ds.makegp_timing(psr, svd=True),
            ]
        )
        for psr in psrs
    ]


def common_prior(kernel, gw_gamma=None):
    """The common-process prior; ``gw_gamma`` fixes the power-law index (e.g. 13/3)."""
    if kernel == "pl":
        if gw_gamma is None:
            return ds.powerlaw

        def powerlaw_fixed_gamma(f, df, log10_A):
            return ds.powerlaw(f, df, log10_A, gw_gamma)

        return powerlaw_fixed_gamma
    if kernel == "ou":
        if gw_gamma is not None:
            raise ValueError("gw_gamma only applies to the power-law kernel")
        return ou_psd.ou_gw_prior
    raise ValueError(f"unknown kernel {kernel!r}")


def red_noise_prior(kernel):
    return {"pl": ds.powerlaw, "ou": ou_psd.ou_rn_prior}[kernel]


def make_model(
    psrs, kernel="pl", rn_components=30, gw_components=14, gw_gamma=None, gw=True
):
    """The array likelihood. ``gw=False`` drops the common process (intrinsic noise only)."""
    T = ds.getspan(psrs)
    rngp = ds.makecommongp_fourier(
        psrs, red_noise_prior(kernel), rn_components, T=T, name=RN_NAME
    )
    globalgp = (
        make_epsilon_gp(psrs, common_prior(kernel, gw_gamma), gw_components, T)
        if gw
        else None
    )
    return ds.ArrayLikelihood(pulsar_models(psrs), commongp=rngp, globalgp=globalgp)


def _argnames(fn):
    return inspect.getfullargspec(fn).args

#!/usr/bin/env python
"""Single-pulsar Discovery-OU noise run on NG15 narrowband (spec task 3.2).

The kernel-matched reference for task 3.1: the per-pulsar pieces of the Phase 1 OU
model (`reference/models.py`), with fixed white noise and ECORR from the feather noise
dictionary, the SVD timing model under Discovery's improper prior, and spin red noise on
30 Fourier modes with Argus's integrated-OU PSD (`ou_psd.ou_rn_prior`). There is no
common process.

The Fourier basis uses the 67-pulsar array span T, as in the Phase 1 array model, not
the pulsar's own span, so the per-pulsar process is the one the array runs see and C4
compares at the same f = 1/T on both sides.

NUTS samples the same sites, under the same boxes and sampler block, as `run_argus.py`
(`common`), and writes outputs/single_psr/discovery/<psr>.nc and
results/single_psr/discovery/<psr>.json.

    conda activate discovery
    python workflows/ng15_nb/single_psr/run_discovery.py --pulsar B1855+09
    python workflows/ng15_nb/single_psr/run_discovery.py --index 0   # SLURM array index

`--gamma-p-max` raises the γp ceiling for the spot check behind the γp amendments
(spec §12, 2026-10-20 and 2026-10-22); those runs are written under the pipeline name
`discovery_gp<max>`. `--sigma-p-max` raises the σp ceiling as well (`_sp<max>`): the
likelihood is near-flat along the f⁻² ridge, where σp rises with γp, so raising γp's
ceiling alone moves the pile-up onto σp's.
`--rn-components` changes the Fourier mode count from 30 (`_m<n>`), to test a C4 outlier
for truncation.
`--free-white` adds one sampled per-pulsar white term on top of the fixed white noise: the
per-TOA variance becomes N_fixed + 10^(2 log10_wq), with log10_wq [s] uniform on WQ_BOX
(`_wq`). It is the flat-PSD (EQUAD-like) shape of spec §12 2026-11-13 step 1, a different
shape from the OU's f⁻² limit: if it carries the >30/T excess, γp should leave its ceiling.
At log10_wq = WQ_BOX[0] the term is ≤ 2e-4 of the smallest TOA variance (J1713+0747, 8 ns).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import discovery as ds  # noqa: E402
import numpy as np  # noqa: E402
import numpyro  # noqa: E402
import numpyro.distributions as dist  # noqa: E402
from numpyro.infer import MCMC, NUTS  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "reference"))
import common  # noqa: E402
import models  # noqa: E402
import ou_psd  # noqa: E402

DISCOVERY_DIR = os.environ.get(
    "NG15_DISCOVERY_DIR", "/fred/oz022/tkimpson/external/discovery"
)
RN_COMPONENTS = 30
WQ_SITE = "log10_wq"
WQ_BOX = (-10.0, -5.0)  # log10 s; NANOGrav's EQUAD box is [-8.5, -5]


def load(name):
    entry = next(
        e
        for e in json.loads(common.MANIFEST.read_text())["feathers"]
        if e["pulsar"] == name
    )
    return ds.Pulsar.read_feather(str(Path(DISCOVERY_DIR) / "data" / entry["file"]))


def free_white_model(psr):
    """`models.pulsar_models([psr])[0]` with one sampled white term added to the fixed one.

    The fixed per-TOA variance (NANOGrav's per-backend EFAC/EQUAD from the noise dictionary)
    is taken from the constant noise matrix and the term 10^(2 log10_wq) added to every TOA.
    ECORR and the timing model are unchanged.
    """
    from discovery import matrix

    fixed = ds.makenoise_measurement(psr, psr.noisedict)
    n_fixed = matrix.jnparray(np.asarray(fixed.N))
    param = f"{psr.name}_{WQ_SITE}"

    def getnoise(params):
        return n_fixed + 10.0 ** (2.0 * params[param])

    getnoise.params = [param]
    return ds.PulsarLikelihood(
        [
            psr.residuals,
            matrix.NoiseMatrix1D_var(getnoise),
            ds.makegp_ecorr(psr, psr.noisedict),
            ds.makegp_timing(psr, svd=True),
        ]
    )


def make_loglike(psr, span_s, free_white=False):
    rn = ds.makecommongp_fourier(
        [psr], ou_psd.ou_rn_prior, RN_COMPONENTS, T=span_s, name=models.RN_NAME
    )
    pulsar = [free_white_model(psr)] if free_white else models.pulsar_models([psr])
    logl = ds.ArrayLikelihood(pulsar, commongp=rn).logL
    key = lambda arg: f"{psr.name}_{models.RN_NAME}_{arg}"  # noqa: E731
    wq = f"{psr.name}_{WQ_SITE}"
    expected = {key("log10_gamma_p"), key("log10_sigma_r")} | ({wq} if free_white else set())
    if set(logl.params) != expected:
        raise RuntimeError(f"parameters {logl.params}, expected {sorted(expected)}")
    log10_f0 = np.log10(common.f0_of(psr.name))

    def loglike(log10_gamma_p, log10_sigma_p, log10_wq=None):
        p = {
            key("log10_gamma_p"): log10_gamma_p,
            key("log10_sigma_r"): log10_sigma_p - log10_f0,
        }
        if free_white:
            p[wq] = log10_wq
        return logl(p)

    return loglike


def run(
    name,
    num_warmup,
    num_samples,
    gamma_p_max=None,
    target_accept=None,
    sigma_p_max=None,
    free_white=False,
):
    psr = load(name)
    span_s = common.array_span_s(DISCOVERY_DIR)
    loglike = make_loglike(psr, span_s, free_white)
    box = dict(common.BOX)
    pipeline = "discovery"
    if gamma_p_max is not None:
        box["log10_gamma_p"] = (box["log10_gamma_p"][0], gamma_p_max)
        pipeline = f"discovery_gp{gamma_p_max:g}"
    if sigma_p_max is not None:
        box["log10_sigma_p"] = (box["log10_sigma_p"][0], sigma_p_max)
        pipeline += f"_sp{sigma_p_max:g}"
    if RN_COMPONENTS != 30:
        pipeline += f"_m{RN_COMPONENTS}"
    if target_accept is not None:
        pipeline += f"_ta{target_accept:g}"
    if free_white:
        box[WQ_SITE] = WQ_BOX
        pipeline += "_wq"

    def model():
        x = {s: numpyro.sample(s, dist.Uniform(*box[s])) for s in box}
        numpyro.factor(
            "loglik",
            loglike(x["log10_gamma_p"], x["log10_sigma_p"], x.get(WQ_SITE)),
        )

    nuts = dict(
        common.NUTS,
        target_accept_prob=target_accept or common.NUTS["target_accept_prob"],
    )
    mcmc = MCMC(
        NUTS(
            model,
            target_accept_prob=nuts["target_accept_prob"],
            max_tree_depth=nuts["max_tree_depth"],
        ),
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=nuts["num_chains"],
        chain_method="vectorized",
        progress_bar=False,
    )
    t0 = time.perf_counter()
    mcmc.run(
        jax.random.PRNGKey(common.seed_of(name)),
        extra_fields=("diverging", "num_steps"),
    )
    wall = time.perf_counter() - t0
    samples = {
        k: np.asarray(v) for k, v in mcmc.get_samples(group_by_chain=True).items()
    }
    extra = mcmc.get_extra_fields(group_by_chain=True)

    import arviz as az

    out_dir = common.OUTPUTS / pipeline
    out_dir.mkdir(parents=True, exist_ok=True)
    az.from_numpyro(mcmc).to_netcdf(out_dir / f"{name}.nc")

    rec = common.summarise(
        name,
        pipeline,
        samples,
        np.asarray(extra["diverging"]),
        span_s,
        extra={
            "rn_components": RN_COMPONENTS,
            "free_white": bool(free_white),
            "n_toa": int(len(psr.toas)),
            "num_warmup": num_warmup,
            "num_samples": num_samples,
            "mean_num_steps": float(np.mean(extra["num_steps"])),
            "sample_wall_s": wall,
            "device": str(jax.devices()[0]),
        },
        box=box,
        nuts=dict(nuts, num_samples=num_samples, num_warmup=num_warmup),
    )
    path = common.write(rec)
    print(json.dumps(rec, indent=2), flush=True)
    print(f"wrote {path}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    who = ap.add_mutually_exclusive_group(required=True)
    who.add_argument("--pulsar")
    who.add_argument("--index", type=int, help="manifest index (SLURM array)")
    ap.add_argument("--num-warmup", type=int, default=common.NUTS["num_warmup"])
    ap.add_argument("--num-samples", type=int, default=common.NUTS["num_samples"])
    ap.add_argument(
        "--target-accept",
        type=float,
        help="rerun at this NUTS target (written under <pipeline>_ta<value>)",
    )
    ap.add_argument(
        "--gamma-p-max", type=float, help="raise the γp ceiling (spot check)"
    )
    ap.add_argument(
        "--sigma-p-max", type=float, help="raise the σp ceiling (spot check)"
    )
    ap.add_argument(
        "--rn-components", type=int, help="Fourier modes (default 30; truncation test)"
    )
    ap.add_argument(
        "--free-white",
        action="store_true",
        help="add a sampled per-pulsar white term log10_wq (spec §12 2026-11-13 step 1)",
    )
    args = ap.parse_args()
    if args.rn_components:
        global RN_COMPONENTS
        RN_COMPONENTS = args.rn_components
    name = args.pulsar or common.pulsars()[args.index]
    print(f"{name}: device {jax.devices()[0]}, commit {common.git_sha()}", flush=True)
    run(
        name,
        args.num_warmup,
        args.num_samples,
        args.gamma_p_max,
        args.target_accept,
        args.sigma_p_max,
        args.free_white,
    )


if __name__ == "__main__":
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    main()

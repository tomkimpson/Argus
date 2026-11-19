#!/usr/bin/env python
"""67-pulsar NG15 narrowband NUTS run through the production harness (spec Phase 5).

`workflow.run_inference` is the MDC2 harness: ridge and uniform priors, checkpointing,
parallel chains, output layout. Its only data-dependent piece is the Kalman filter it
builds from `data_path`, and the stock filter cannot read the D2 compression. So, as
`run_scrambled.py` does for the ORF, two runtime overrides are installed with no library
edit:
- `workflow.setup_data_and_kalman_filter` returns `SparseArrayFilter` over the 67 pulsars
  in manifest order, with epoch cells for the pulsars `results/c1/epoch_rule.json` lists;
- `utils.get_noise_parameters` returns EFAC = 1 and EQUAD = 0, so the harness fixes the
  white-noise sites instead of sampling them (white noise and ECORR live inside the
  compression, A2).

The config must say so: `noise_params_path = fixed_in_compression`, `timing_prior =
diffuse`, an empty `spin_injections_path`. Anything else is refused. `data_path` names the
ingest-cache directory; `NG15_INGEST_DIR` overrides it, because the caches are gitignored
and a pinned worktree reads the main checkout's.

    conda activate Argus
    export PYTHONPATH=$PWD/python
    python workflows/ng15_nb/array/run_array.py workflows/ng15_nb/configs/frozen/ng15_nb_v1.ini --check
    python workflows/ng15_nb/array/run_array.py workflows/ng15_nb/configs/frozen/ng15_nb_v1.ini

`--check` builds the filter, times one jitted value-plus-gradient call at the centre of the
prior box (ε from the config), prints it and exits without sampling.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "single_psr"))
import common  # noqa: E402
from sparse_filter import SparseArrayFilter  # noqa: E402

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
from argus import utils, workflow  # noqa: E402
from argus.bayesian_inference import Parameters  # noqa: E402

WN_SENTINEL = "fixed_in_compression"


def load_config(path):
    config = utils.resolve_config_paths(utils.load_config(path), path)
    problems = []
    if os.path.basename(config.get("PriorModel", "noise_params_path")) != WN_SENTINEL:
        problems.append(f"noise_params_path must be {WN_SENTINEL}")
    if config.get("PriorModel", "timing_prior", fallback="").strip() != "diffuse":
        problems.append(
            "timing_prior must be diffuse (the sparse path is diffuse only)"
        )
    if config.get("PriorModel", "spin_injections_path", fallback="").strip():
        problems.append("spin_injections_path must be empty (red noise is sampled)")
    if config.get("Data", "signal_model", fallback="gwb").strip().lower() != "gwb":
        problems.append("signal_model must be gwb")
    if config.get("Data", "excluded_psrs", fallback="__NONE__").strip() != "__NONE__":
        problems.append("excluded_psrs is not supported: the array is the 67 pulsars")
    if problems:
        raise SystemExit(f"Refusing {path}:\n  - " + "\n  - ".join(problems))
    return config


def build_filter(config):
    ingest_dir = os.environ.get("NG15_INGEST_DIR", config.get("Data", "data_path"))
    epoch = json.loads(common.EPOCH_RULE.read_text())["needs_epoch_cells"]
    t = time.perf_counter()
    kf = SparseArrayFilter(ingest_dir, common.pulsars(), epoch)
    print(
        f"[SPARSE] {kf.Npsr} psr, {kf.n_obs} observations, M_sum {kf.M_sum}, "
        f"epoch cells {kf.epoch_cells}, built in {time.perf_counter() - t:.0f} s "
        f"from {ingest_dir}",
        flush=True,
    )
    return kf


def install(kf):
    def setup_data_and_kalman_filter(config, logger, use_gw, signal_model="gwb"):
        logger.info(f"[SPARSE] using SparseArrayFilter: {json.dumps(kf.summary())}")
        return {"metadata": kf.names}, kf

    def get_noise_parameters(config):
        return jnp.ones(kf.Npsr), jnp.zeros(kf.Npsr), None, None

    workflow.setup_data_and_kalman_filter = setup_data_and_kalman_filter
    utils.get_noise_parameters = get_noise_parameters


def box_centre(config, n):
    """[log10 h_a, log10 γa, log10 σp × n, log10 γp × n] at the centre of each box."""

    def mid(key):
        return 0.5 * (
            config.getfloat("PriorModel", f"{key}_min")
            + config.getfloat("PriorModel", f"{key}_max")
        )

    return jnp.concatenate(
        [
            jnp.array([-14.5, mid("log10_gamma_a")]),
            jnp.full(n, mid("log10_sigma_p")),
            jnp.full(n, mid("log10_gamma_p")),
        ]
    )


def sampled_sites(config, n):
    """The harness model's latent sites and sizes, traced with a zero likelihood."""
    from numpyro import handlers
    from argus import bayesian_inference, prior_models

    class Zero:
        def get_likelihood(self, θ):
            return 0.0

    specs = prior_models.get_prior_model_specs(
        config, n, None, None, jnp.ones(n), jnp.zeros(n)
    )
    tr = handlers.trace(
        handlers.seed(bayesian_inference.numpyro_model, jax.random.PRNGKey(0))
    ).get_trace(Zero(), specs, n)
    return {
        k: int(jnp.size(v["value"]))
        for k, v in tr.items()
        if v["type"] == "sample" and not v["is_observed"]
    }


def check(kf, config):
    n = kf.Npsr
    sites = sampled_sites(config, n)
    eps = config.getfloat("PriorModel", "orf_epsilon_value")

    def loglike(p):
        return kf.get_likelihood(
            Parameters(
                log10_gamma_a=p[1],
                γa=10.0 ** p[1],
                ha=10.0 ** p[0],
                σp=10.0 ** p[2 : 2 + n],
                γp=10.0 ** p[2 + n :],
                EFAC=jnp.ones(n),
                EQUAD=jnp.zeros(n),
                orf_epsilon=eps,
            )
        )

    vg = jax.jit(jax.value_and_grad(loglike))
    p = box_centre(config, n)
    t = time.perf_counter()
    jax.block_until_ready(vg(p))
    compile_s = time.perf_counter() - t
    times = []
    for _ in range(3):
        t = time.perf_counter()
        ll, g = jax.block_until_ready(vg(p))
        times.append(time.perf_counter() - t)
    out = {
        "device": str(jax.devices()[0]),
        "orf_epsilon": eps,
        "lnL": float(ll),
        "grad_finite": bool(jnp.all(jnp.isfinite(g))),
        "compile_s": compile_s,
        "value_and_grad_s": times,
        "sampled_sites": sites,
        "n_sampled": sum(sites.values()),
        **kf.summary(),
    }
    print(json.dumps(out, indent=2))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    config = load_config(args.config)
    print(f"JAX {jax.__version__}, devices {jax.devices()}", flush=True)
    kf = build_filter(config)
    if args.check:
        check(kf, config)
        return
    utils.check_gpu_availability()
    install(kf)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = workflow.run_inference(
        config_path=args.config, use_gw=True, timestamp=timestamp
    )
    print(f"Done: {output_dir}")


if __name__ == "__main__":
    main()

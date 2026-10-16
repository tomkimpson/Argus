"""One fixed-ε rung of the Discovery reference ladder (spec D3, tasks 1.4/1.5).

Samples the NG15 narrowband posterior at a fixed point ε of the correlation path
Γ(ε) = (1 − ε) I + ε Γ_HD with numpyro NUTS (settings from evidence_procedure_v1.json),
then evaluates the path-sampling integrand ∂lnL/∂ε at every draw by autodiff. The
output ``samples.nc`` carries ``orf_epsilon`` and ``d_loglik_d_eps`` as posterior sites,
which is exactly what the frozen estimator reads:

    python workflows/ng15_sgwb_demo/scripts/lnb_path_sampling.py \\
        --from-runs workflows/ng15_nb/outputs/ng15nb_pl_eps*/samples.nc --out ...

Priors are exact uniforms on the configured boxes (numpyro Uniform sites; NUTS works in
their logit-transformed space with the Jacobian included). Discovery's own
``makemodel_transformed`` adds a Normal(0, 10) on the transformed coordinates, which
is not a uniform prior, so it is not used.

    conda activate discovery
    python workflows/ng15_nb/reference/run_rung.py --config workflows/ng15_nb/reference/configs/pl.ini --eps 0.0
    # cost probe, no sampling:
    python workflows/ng15_nb/reference/run_rung.py --config ... --eps 0.5 --benchmark
"""

import argparse
import configparser
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import arviz as az  # noqa: E402
import discovery as ds  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import numpyro  # noqa: E402
import numpyro.distributions as dist  # noqa: E402
from numpyro.diagnostics import summary  # noqa: E402
from numpyro.infer import MCMC, NUTS  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import models  # noqa: E402

INTEGRAND_SITE = "d_loglik_d_eps"
EPS_SITE = "orf_epsilon"
BATCH = 100


def load_pulsars(data_dir, n_pulsars):
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "workflows/ng15_nb/scripts/build_manifest.py"), str(data_dir), "--check"],
        check=True,
    )
    manifest = json.loads((REPO_ROOT / "data/ng15_nb/MANIFEST.json").read_text())
    entries = manifest["feathers"][:n_pulsars]
    return [ds.Pulsar.read_feather(str(Path(data_dir) / "data" / e["file"])) for e in entries]


class Sampler:
    """Maps sampled sites (Argus's basis) onto the Discovery likelihood's parameters."""

    def __init__(self, cfg, psrs, eps):
        self.cfg, self.psrs, self.eps = cfg, psrs, float(eps)
        self.kernel = cfg.get("Model", "kernel")
        gw_gamma = cfg.get("Model", "gw_gamma", fallback="variable")
        self.gw_gamma = None if gw_gamma == "variable" else float(gw_gamma)
        self.logl = models.make_model(
            psrs,
            kernel=self.kernel,
            rn_components=cfg.getint("Model", "rn_components"),
            gw_components=cfg.getint("Model", "gw_components"),
            gw_gamma=self.gw_gamma,
        ).logL
        self.names = [p.name for p in psrs]
        pm = cfg["PriorModel"]
        box = lambda key: (float(pm[f"{key}_min"]), float(pm[f"{key}_max"]))  # noqa: E731
        if self.kernel == "pl":
            self.rn_sites = {"rn_log10_A": ("log10_A", box("rn_log10_A")), "rn_gamma": ("gamma", box("rn_gamma"))}
            self.gw_sites = {"gw_log10_A": box("gw_log10_A")}
            if self.gw_gamma is None:
                self.gw_sites["gw_gamma"] = box("gw_gamma")
        else:
            f0_cat = json.loads((REPO_ROOT / cfg.get("Data", "f0_catalog")).read_text())["f0_hz"]
            self.log10_f0 = jnp.array([np.log10(f0_cat[n]) for n in self.names])
            self.rn_sites = {
                "log10_gamma_p": ("log10_gamma_p", box("log10_gamma_p")),
                "log10_sigma_p": ("log10_sigma_r", box("log10_sigma_p")),
            }
            self.gw_sites = {"gw_log10_pivot_psd": box("log10_pivot_psd"), "gw_log10_gamma_a": box("log10_gamma_a")}
        expected = set(self.logl.params)
        got = set(self.to_params(self.midpoint()).keys())
        if got != expected:
            raise RuntimeError(f"parameter mismatch: missing {expected - got}, extra {got - expected}")

    def midpoint(self):
        n = len(self.psrs)
        out = {s: jnp.full(n, 0.5 * (lo + hi)) for s, (_, (lo, hi)) in self.rn_sites.items()}
        out.update({s: 0.5 * (lo + hi) for s, (lo, hi) in self.gw_sites.items()})
        return out

    def to_params(self, sites, eps=None):
        """Sampled sites -> the Discovery parameter dict (ε defaults to the rung's)."""
        p = {models.EPS: self.eps if eps is None else eps}
        for site, (arg, _) in self.rn_sites.items():
            vals = sites[site]
            if site == "log10_sigma_p":
                vals = vals - self.log10_f0
            for i, name in enumerate(self.names):
                p[f"{name}_{models.RN_NAME}_{arg}"] = vals[i]
        for site in self.gw_sites:
            p[site] = sites[site]
        return p

    def model(self):
        n = len(self.psrs)
        sites = {s: numpyro.sample(s, dist.Uniform(lo, hi).expand([n])) for s, (_, (lo, hi)) in self.rn_sites.items()}
        sites.update({s: numpyro.sample(s, dist.Uniform(lo, hi)) for s, (lo, hi) in self.gw_sites.items()})
        numpyro.factor("loglik", self.logl(self.to_params(sites)))

    def integrand(self, sites):
        """lnL and ∂lnL/∂ε at one draw, at the rung's ε."""
        f = lambda e: self.logl(self.to_params(sites, eps=e))  # noqa: E731
        return jax.value_and_grad(f)(self.eps)


def benchmark(sampler, reps=20):
    """Seconds per lnL and per lnL+gradient (all sampled sites) on this device."""
    x0 = sampler.midpoint()
    val = jax.jit(lambda s: sampler.logl(sampler.to_params(s)))
    vg = jax.jit(jax.value_and_grad(lambda s: sampler.logl(sampler.to_params(s))))
    t0 = time.perf_counter()
    jax.block_until_ready(vg(x0))
    compile_s = time.perf_counter() - t0
    out = {"compile_value_and_grad_s": compile_s}
    for label, fn in (("value_s", val), ("value_and_grad_s", vg)):
        jax.block_until_ready(fn(x0))
        t0 = time.perf_counter()
        for _ in range(reps):
            jax.block_until_ready(fn(x0))
        out[label] = (time.perf_counter() - t0) / reps
    out["lnL_midpoint"] = float(val(x0))
    out["device"] = str(jax.devices()[0])
    return out


def evaluate_integrand(sampler, samples):
    """(chains, draws) arrays of lnL and ∂lnL/∂ε, in batches of BATCH draws."""
    shape = next(iter(samples.values())).shape[:2]
    flat = {k: v.reshape((-1,) + v.shape[2:]) for k, v in samples.items()}
    n = shape[0] * shape[1]
    fn = jax.jit(jax.vmap(sampler.integrand))
    vals, grads = [], []
    for start in range(0, n, BATCH):
        v, g = fn({k: x[start : start + BATCH] for k, x in flat.items()})
        vals.append(np.asarray(v))
        grads.append(np.asarray(g))
    return np.concatenate(vals).reshape(shape), np.concatenate(grads).reshape(shape)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--eps", type=float, required=True)
    ap.add_argument("--gw-components", type=int, help="override [Model] gw_components; pass --output-id too, or the run overwrites the config's output dir")
    ap.add_argument("--gw-gamma", help="override [Model] gw_gamma, e.g. 4.333333333333333")
    ap.add_argument("--output-id", help="override [Output] output_id")
    ap.add_argument("--n-pulsars", type=int, help="override [Data] n_pulsars (probes only)")
    ap.add_argument("--num-warmup", type=int)
    ap.add_argument("--num-samples", type=int)
    ap.add_argument("--out-root", default=str(REPO_ROOT / "workflows/ng15_nb/outputs"))
    ap.add_argument("--benchmark", action="store_true", help="time lnL and its gradient, then exit")
    args = ap.parse_args()

    cfg = configparser.ConfigParser()
    cfg.read(args.config)
    if args.gw_components is not None:
        cfg["Model"]["gw_components"] = str(args.gw_components)
    if args.gw_gamma is not None:
        cfg["Model"]["gw_gamma"] = args.gw_gamma
    if args.n_pulsars is not None:
        cfg["Data"]["n_pulsars"] = str(args.n_pulsars)
    for key in ("num_warmup", "num_samples"):
        if getattr(args, key) is not None:
            cfg["NUTS"][key] = str(getattr(args, key))
    cfg["PriorModel"]["orf_epsilon_value"] = repr(args.eps)
    output_id = args.output_id or cfg.get("Output", "output_id")
    output_id = f"{output_id}_eps{round(args.eps * 100):03d}"
    cfg["Output"]["output_id"] = output_id

    psrs = load_pulsars(cfg.get("Data", "data_path"), cfg.getint("Data", "n_pulsars"))
    sampler = Sampler(cfg, psrs, args.eps)
    print(f"{output_id}: {len(psrs)} pulsars, kernel {sampler.kernel}, {len(sampler.logl.params)} likelihood params, "
          f"device {jax.devices()[0]}", flush=True)

    if args.benchmark:
        print(json.dumps(benchmark(sampler), indent=2))
        return

    out_dir = Path(args.out_root) / output_id
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "resolved_config.ini", "w") as fh:
        cfg.write(fh)

    nuts_cfg = cfg["NUTS"]
    gw_block = tuple(sampler.gw_sites)
    kernel = NUTS(
        sampler.model,
        target_accept_prob=float(nuts_cfg["target_accept_prob"]),
        max_tree_depth=int(nuts_cfg["max_tree_depth"]),
        dense_mass=[gw_block] if len(gw_block) > 1 else False,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=int(nuts_cfg["num_warmup"]),
        num_samples=int(nuts_cfg["num_samples"]),
        num_chains=int(nuts_cfg["num_chains"]),
        chain_method="vectorized",
        progress_bar=False,
    )
    t0 = time.perf_counter()
    mcmc.run(jax.random.PRNGKey(int(nuts_cfg["seed"])), extra_fields=("diverging", "num_steps"))
    sample_s = time.perf_counter() - t0

    samples = {k: np.asarray(v) for k, v in mcmc.get_samples(group_by_chain=True).items()}
    t0 = time.perf_counter()
    loglik, integrand = evaluate_integrand(sampler, samples)
    integrand_s = time.perf_counter() - t0

    idata = az.from_numpyro(mcmc)
    dims = ("chain", "draw")
    idata.posterior[INTEGRAND_SITE] = (dims, integrand)
    idata.posterior["loglik"] = (dims, loglik)
    idata.posterior[EPS_SITE] = (dims, np.full(integrand.shape, args.eps))
    if sampler.kernel == "ou":
        sys.path.insert(0, str(HERE))
        import ou_psd

        idata.posterior["gw_log10_ha"] = (
            dims,
            ou_psd.log10_ha_from_pivot(samples["gw_log10_pivot_psd"], samples["gw_log10_gamma_a"]),
        )
    idata.posterior.attrs.update({"pulsars": json.dumps(sampler.names), "config": output_id})
    idata.to_netcdf(out_dir / "samples.nc")

    extra = mcmc.get_extra_fields(group_by_chain=True)
    diag = summary(samples, group_by_chain=True)
    report = {
        "output_id": output_id,
        "eps": args.eps,
        "n_pulsars": len(psrs),
        "device": str(jax.devices()[0]),
        "sample_wall_s": sample_s,
        "integrand_wall_s": integrand_s,
        "divergences": int(np.sum(extra["diverging"])),
        "divergence_fraction": float(np.mean(extra["diverging"])),
        "mean_num_steps": float(np.mean(extra["num_steps"])),
        "max_r_hat": float(max(np.max(d["r_hat"]) for d in diag.values())),
        "min_n_eff": float(min(np.min(d["n_eff"]) for d in diag.values())),
        "integrand_mean": float(np.mean(integrand)),
        "integrand_sd": float(np.std(integrand)),
        "integrand_chain_means": [float(x) for x in np.mean(integrand, axis=1)],
    }
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    main()

"""G(θ) = ∂lnL/∂C at C = I on the sparse array filter (detection-statistic spec, task 1.1).

At ε = 0 the correlation statistic of `spec/detection-statistic.md` §3 is

    S(θ; Γ) = ∂lnL/∂ε at ε = 0 along C(ε) = (1 − ε) I + ε Γ  =  Σᵢⱼ Gᵢⱼ(θ) (Γ − I)ᵢⱼ,

so one reverse-mode gradient with respect to the 67 × 67 correlation matrix scores any
number of ORFs by a dot product. The seam is `inp["orf"]` of
`union.array_loglike_sparse`, which passes it to the filter as the ORF to use as-is
(`Parameters.orf_epsilon` is None there). The MDC2 script's `_with_orf`, which swaps
`kalman_filter.hellings_downs_matrix`, does nothing here: the sparse filter never reads
that attribute.

The likelihood is the production one, `array/sparse_filter.SparseArrayFilter`, built from
the gitignored ingest caches. No library code is changed (spec D4).

Pieces, imported by the job driver (`opening_job.py`), the readout and the tests:
- `build_filter`: the 67-pulsar filter in manifest order, with D2's epoch cells;
- `load_draws`: the v3 probe's 4 × 100 ε = 0 draws, read with h5py and checked against
  the SHA-256 recorded in `results/stat/inputs.json`;
- `make_fns`: the jitted G, lnL at C and the forward-mode (JVP) ε-derivative through the
  harness's own `get_likelihood`, which applies `correlation_path` itself;
- `score`: the full-matrix contraction (imported from the MDC2 script, spec D4).
"""

import copy
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
WF = REPO_ROOT / "workflows/ng15_nb"
sys.path.insert(0, str(WF / "array"))
sys.path.insert(0, str(WF / "compression"))
sys.path.insert(0, str(WF / "single_psr"))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

INPUTS_JSON = WF / "results/stat/inputs.json"
DEFAULT_INGEST = Path(
    os.environ.get("NG15_INGEST_DIR", REPO_ROOT / "workflows/ng15_nb/data/ingest")
)
SCRIPTS = REPO_ROOT / "workflows/ng15_sgwb_demo/scripts"


def load_script(name):
    """Import a script from `ng15_sgwb_demo/scripts` by path (they are not a package)."""
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build_filter(ingest_dir=DEFAULT_INGEST, names=None, epoch_cells=None):
    """`SparseArrayFilter` over `names` (default: the 67 in manifest order)."""
    import common
    from sparse_filter import SparseArrayFilter

    if names is None:
        names = common.pulsars()
    if epoch_cells is None:
        epoch_cells = json.loads(common.EPOCH_RULE.read_text())["needs_epoch_cells"]
    return SparseArrayFilter(ingest_dir, names, epoch_cells)


def load_draws(nc=None, key="v3_probe_eps000"):
    """The probe's draws as p = [log10 h_a, log10 γa, log10 σp (N), log10 γp (N)].

    Returns (p (n_chain, n_draw, 2 + 2N), extras) where extras carries the stored
    log-likelihood, the per-pulsar σp and γp, the path and its verified hash.
    """
    import h5py

    rec = json.loads(INPUTS_JSON.read_text())["inputs"][key]
    nc = Path(nc or rec["path"])
    digest = sha256(nc)
    if digest != rec["sha256"]:
        raise SystemExit(f"{nc}: sha256 {digest} != recorded {rec['sha256']}")
    with h5py.File(nc, "r") as f:
        post = f["posterior"]
        eps = post["orf_epsilon"][...]
        if not np.all(eps == 0.0):
            raise SystemExit(f"{nc} is not an ε = 0 run (ε in {np.unique(eps)})")
        ha = post["log10_ha"][...]
        ga = post["log10_gamma_a"][...]
        sp = post["log10_σp"][...]
        gp = post["log10_γp"][...]
        stored = f["log_likelihood"]["likelihood"][...]
    p = np.concatenate([ha[..., None], ga[..., None], sp, gp], axis=-1)
    extras = {
        "path": str(nc),
        "sha256": digest,
        "stored_lnL": stored,
        "log10_sigma_p": sp,
        "log10_gamma_p": gp,
    }
    return p, extras


def parameters(p, n, eps=None):
    """`Parameters` for `SparseArrayFilter.get_likelihood` (white noise is pinned there)."""
    from argus.bayesian_inference import Parameters

    return Parameters(
        log10_gamma_a=p[1],
        γa=10.0 ** p[1],
        ha=10.0 ** p[0],
        σp=10.0 ** p[2 : 2 + n],
        γp=10.0 ** p[2 + n : 2 + 2 * n],
        EFAC=jnp.ones(n),
        EQUAD=jnp.zeros(n),
        orf_epsilon=eps,
    )


def with_orf(kf, orf):
    """A shallow copy of the filter whose `inp["orf"]` (the seam it reads) is `orf`."""
    swapped = copy.copy(kf)
    swapped.inp = dict(kf.inp, orf=jnp.asarray(orf))
    return swapped


def make_fns(kf):
    """Jitted per-point functions on the filter `kf`.

    lnl_at(C, p)          lnL with the ORF C used as-is, constant included;
    grad_orf(p)           G = ∂lnL/∂C at C = I (vmapped over a batch of p);
    harness_lnl(p)        `kf.get_likelihood` at ε = 0 (V2's other side);
    eps_jvp(orf, p)       ∂lnL/∂ε at ε = 0 by forward mode through `get_likelihood`
                          with `orf` as the Γ of C(ε) (V1's independent side);
    """
    import union

    n = kf.Npsr
    eye = jnp.eye(n, dtype=jnp.float64)

    def lnl_at(C, p):
        return union.array_loglike_sparse(p, dict(kf.inp, orf=C)) + kf.inp["const"]

    def grad_one(p):
        return jax.grad(lnl_at)(eye, p)

    def harness_one(p):
        return kf.get_likelihood(parameters(p, n, eps=0.0))

    def eps_jvp_one(orf, p):
        k = with_orf(kf, orf)

        def f(e):
            return k.get_likelihood(parameters(p, n, eps=e))

        return jax.jvp(f, (jnp.float64(0.0),), (jnp.float64(1.0),))[1]

    def eps_value_one(orf, p, e):
        return with_orf(kf, orf).get_likelihood(parameters(p, n, eps=e))

    return {
        "lnl_at": jax.jit(lnl_at),
        "grad_orf": jax.jit(jax.vmap(grad_one)),
        "harness_lnl": jax.jit(harness_one),
        "eps_jvp": jax.jit(eps_jvp_one),
        "eps_value": jax.jit(eps_value_one),
    }


def batched_G(grad_orf, p_flat, batch):
    """G for every row of p_flat, in fixed chunks of `batch` (the last may be shorter)."""
    out = []
    for start in range(0, len(p_flat), batch):
        chunk = jnp.asarray(p_flat[start : start + batch])
        out.append(np.asarray(jax.block_until_ready(grad_orf(chunk))))
    return np.concatenate(out)


def score(G, orfs):
    """S for every (G, ORF) pair: Σ over the FULL matrix of G ∘ (Γ − I)."""
    return load_script("scramble_ensemble_eps0").score(np.asarray(G), orfs)


def peak_bytes():
    try:
        return int(jax.devices()[0].memory_stats().get("peak_bytes_in_use", -1))
    except Exception:  # noqa: BLE001 (CPU backends carry no memory stats)
        return -1

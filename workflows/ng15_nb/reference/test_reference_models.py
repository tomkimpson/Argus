"""The ε-ORF reference models reduce to Discovery's own NANOGrav models (spec task 1.3).

Runs in the ``discovery`` env (envs/discovery.yml) against the feathers pinned in
data/ng15_nb/MANIFEST.json; skipped anywhere else (Argus CI has neither):

    conda activate discovery
    NG15_DISCOVERY_DIR=<DISCOVERY_DIR> python -m pytest workflows/ng15_nb/reference -q

- ε = 1 must reproduce ``makemodel_hd`` to 1e-8 (relative; the spec's gate).
- ε = 0 must reproduce ``makemodel_curn``. That model folds the common power law into the
  first 2×14 of each pulsar's 30 red-noise modes rather than carrying a separate GP, so
  it checks the mode bookkeeping that ε = 1 does not touch.
- The autodiff path-sampling integrand ∂lnL/∂ε must match a central finite difference.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ds = pytest.importorskip("discovery")
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

from discovery.models import nanograv  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import models  # noqa: E402

DISCOVERY_DIR = Path(os.environ.get("NG15_DISCOVERY_DIR", "/fred/oz022/tkimpson/external/discovery"))
N_PSR = 5
N_POINTS = 4


@pytest.fixture(scope="module")
def psrs():
    manifest = json.loads((REPO_ROOT / "data/ng15_nb/MANIFEST.json").read_text())
    files = [DISCOVERY_DIR / "data" / e["file"] for e in manifest["feathers"][:N_PSR]]
    if not all(f.exists() for f in files):
        pytest.skip(f"NG15 feathers not found under {DISCOVERY_DIR}")
    return [ds.Pulsar.read_feather(str(f)) for f in files]


def _draw(params, rng):
    """A point inside NANOGrav's prior box (rn log10_A U[-20,-11], gw/crn U[-18,-11], γ U[0,7])."""
    out = {}
    for p in params:
        if p.endswith("log10_A"):
            out[p] = rng.uniform(-20, -11) if "red_noise" in p else rng.uniform(-15.5, -13.5)
        elif p.endswith("gamma"):
            out[p] = rng.uniform(1, 6)
        elif p == models.EPS:
            out[p] = rng.uniform(0, 1)
        else:
            raise KeyError(p)
    return out


def _rel(a, b):
    return abs(float(a) - float(b)) / abs(float(b))


def test_eps1_is_discovery_hd(psrs):
    ours = jax.jit(models.make_model(psrs, kernel="pl").logL)
    ref = jax.jit(nanograv.makemodel_hd(psrs).logL)
    rng = np.random.default_rng(1)
    for _ in range(N_POINTS):
        p = _draw(ref.params, rng)
        assert _rel(ours({**p, models.EPS: 1.0}), ref(p)) <= 1e-8


def test_eps0_is_discovery_curn(psrs):
    ours = jax.jit(models.make_model(psrs, kernel="pl").logL)
    ref = jax.jit(nanograv.makemodel_curn(psrs).logL)
    rng = np.random.default_rng(2)
    for _ in range(N_POINTS):
        p = _draw(ref.params, rng)
        mine = {k.replace("crn_", "gw_"): v for k, v in p.items()}
        assert _rel(ours({**mine, models.EPS: 0.0}), ref(p)) <= 1e-8


@pytest.mark.parametrize("kernel", ["pl", "ou"])
def test_eps_gradient_matches_finite_difference(psrs, kernel):
    logl = models.make_model(psrs, kernel=kernel).logL
    rng = np.random.default_rng(3)
    p = {}
    for name in logl.params:
        if name == models.EPS:
            p[name] = 0.4
        elif kernel == "pl":
            p.update(_draw([name], rng))
        elif name.endswith("log10_gamma_p"):
            p[name] = rng.uniform(-11, -8)
        elif name.endswith("log10_sigma_r"):
            p[name] = rng.uniform(-17, -15)
        elif name.endswith("log10_pivot_psd"):
            p[name] = -6.0
        elif name.endswith("log10_gamma_a"):
            p[name] = -9.0
        else:
            raise KeyError(name)
    grad = jax.jit(jax.grad(lambda e: logl({**p, models.EPS: e})))(0.4)
    h = 1e-4
    f = jax.jit(lambda e: logl({**p, models.EPS: e}))
    fd = (f(0.4 + h) - f(0.4 - h)) / (2 * h)
    assert np.isfinite(float(grad))
    assert float(grad) == pytest.approx(float(fd), rel=1e-5, abs=1e-6)


def test_ou_model_parameters(psrs):
    params = models.make_model(psrs, kernel="ou").logL.params
    assert set(params) == {models.EPS, "gw_log10_pivot_psd", "gw_log10_gamma_a"} | {
        f"{psr.name}_red_noise_{a}" for psr in psrs for a in ("log10_gamma_p", "log10_sigma_r")
    }

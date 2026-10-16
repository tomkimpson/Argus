"""Known-answer toys for the uniform-prior CURN-vs-noise readout (spec task 1.5).

``workflows/ng15_nb/reference/lnb_curn_vs_noise.py`` maps a U(lo, hi) amplitude onto
the N(0, 1) latent that ``lnb_gw_vs_noise.analyse`` expects. With a step likelihood
L(s) = 1 below m and K above, Z_model / Z_noise = q + K (1 − q) with q = (m − lo)/(hi − lo),
so the region estimator must return ln(q + K (1 − q)) for every threshold below m.
"""

import importlib.util
import math
import os

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LO, HI = -18.0, -11.0


@pytest.fixture(scope="module")
def mod():
    path = os.path.join(REPO_ROOT, "workflows", "ng15_nb", "reference", "lnb_curn_vs_noise.py")
    spec = importlib.util.spec_from_file_location("lnb_curn_vs_noise", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _step_posterior(rng, m, k, shape):
    """Exact draws from U(LO, HI) × L, L = 1 below m and k above."""
    q = (m - LO) / (HI - LO)
    p_below = q / (q + k * (1 - q))
    below = rng.random(shape) < p_below
    return np.where(below, rng.uniform(LO, m, shape), rng.uniform(m, HI, shape))


def test_to_latent_is_standard_normal_under_the_prior(mod):
    from scipy.stats import kstest

    x = np.random.default_rng(0).uniform(LO, HI, 20000)
    assert kstest(mod.to_latent(x, LO, HI), "norm").pvalue > 0.01
    with pytest.raises(ValueError):
        mod.to_latent(np.array([LO - 0.1]), LO, HI)


@pytest.mark.parametrize("k", [1.0, 4.0])
def test_step_likelihood_known_answer(mod, k):
    rng = np.random.default_rng(1)
    m = -14.0
    s = _step_posterior(rng, m, k, (4, 20000))
    gamma = rng.uniform(0, 7, s.shape)  # likelihood-flat nuisance, as in the region
    res = mod.analyse(mod.to_latent(s, LO, HI), mod.to_latent(gamma, 0.0, 7.0))
    q = (m - LO) / (HI - LO)
    want = math.log(q + k * (1 - q))
    assert res["reliable"], res["failed_diagnostics"]
    assert res["ln_bayes_factor"] == pytest.approx(want, abs=4 * res["uncert"] + 1e-3)

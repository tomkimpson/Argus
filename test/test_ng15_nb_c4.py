"""Known-answer toys for the Phase 3 per-pulsar checks (spec C4, task 3.1).

``workflows/ng15_nb/single_psr/c4_compare.py`` reads "within 1σ" as
z = |m_A − m_D| / √(σ_A² + σ_D²) ≤ 1 with σ half the 16–84% width, at both
frequencies; ``common.edges`` decides red-noise detection and rails from the posterior
mass in each box's outer 2%.
"""

import importlib.util
import math
import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SP = os.path.join(REPO_ROOT, "workflows", "ng15_nb", "single_psr")


def _load(name):
    sys.path.insert(0, SP)
    spec = importlib.util.spec_from_file_location(name, os.path.join(SP, f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def c4():
    return _load("c4_compare")


@pytest.fixture(scope="module")
def common():
    return _load("common")


def q(median, sigma):
    return {"median": median, "q16": median - sigma, "q84": median + sigma}


def test_z_score_known_answer(c4):
    # Δ = 5, σ = 3 and 4: z = 5 / 5 = 1 exactly.
    assert c4.z_score(q(0.0, 3.0), q(5.0, 4.0)) == pytest.approx(1.0, rel=1e-15)
    assert c4.z_score(q(1.0, 0.5), q(1.0, 0.1)) == 0.0


def test_gaussian_quantiles_give_the_normal_width(common):
    # For a normal posterior the half 16–84% width is Φ⁻¹(0.84) σ = 0.99446 σ.
    k = 0.994457883
    rng = np.random.default_rng(0)
    a = common.quantiles(rng.normal(0.0, 1.0, 400_000))
    d = common.quantiles(rng.normal(1.0, 1.0, 400_000))
    assert 0.5 * (a["q84"] - a["q16"]) == pytest.approx(k, abs=5e-3)
    z = _load("c4_compare").z_score(a, d)
    assert z == pytest.approx(1.0 / (math.sqrt(2.0) * k), abs=5e-3)


def test_compare_needs_both_frequencies(c4):
    rec = lambda psd: {  # noqa: E731
        "pulsar": "J0000+0000",
        "shallow_spectrum": False,
        "lower_edge_pileup": False,
        "log10_psd": psd,
    }
    a = rec({"1/T": q(0.0, 1.0), "1/(5 yr)": q(0.0, 1.0)})
    d_ok = rec({"1/T": q(1.4, 1.0), "1/(5 yr)": q(-1.4, 1.0)})  # z = 0.99 at both
    d_bad = rec({"1/T": q(1.4, 1.0), "1/(5 yr)": q(1.5, 1.0)})  # z = 1.06 at one
    assert c4.compare(a, d_ok)["within_1sigma"]
    assert not c4.compare(a, d_bad)["within_1sigma"]


def test_edges_detection_and_rails(common):
    rng = np.random.default_rng(1)
    n = 20_000
    lo_g, hi_g = common.BOX["log10_gamma_p"]
    lo_s, hi_s = common.BOX["log10_sigma_p"]
    # Detected red noise, γp piled on its floor (unconstrained corner): not a rail.
    floor = {
        "log10_gamma_p": rng.uniform(lo_g, lo_g + 1.0, n),
        "log10_sigma_p": rng.normal(-15.0, 0.1, n),
    }
    e = common.edges(floor)
    assert e["red_noise_detected"] and e["rails"] == []
    assert e["edge_mass"]["log10_gamma_p"]["floor"] > 0.05
    # Detected red noise, γp piled on its ceiling (an f⁻² spectrum): flagged, not a rail
    # (spec §12, 2026-10-20).
    ceil = dict(floor, log10_gamma_p=rng.uniform(hi_g - 0.2, hi_g, n))
    e = common.edges(ceil)
    assert e["rails"] == [] and e["at_f2_limit"]
    # Detected red noise, σp railing on its ceiling: a rail.
    loud = dict(floor, log10_sigma_p=rng.uniform(hi_s - 0.2, hi_s, n))
    assert common.edges(loud)["rails"] == ["log10_sigma_p ceiling"]
    # No detection: σp flat down to its floor; reported, never a rail.
    flat = dict(floor, log10_sigma_p=rng.uniform(lo_s, -16.0, n))
    e = common.edges(flat)
    assert e["lower_edge_pileup"] and not e["red_noise_detected"] and e["rails"] == []
    assert not e["at_f2_limit"]

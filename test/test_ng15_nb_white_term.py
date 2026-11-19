"""The flat-PSD term of `run_discovery.py --free-white` is inert when it is off.

Spec §12 2026-11-13, step 1: the Discovery single-pulsar check adds one sampled flat
one-sided PSD 10^log10_wpsd to the OU spin-noise prior on the same Fourier modes. The check
is only a test of the excess's shape if, with the term switched off, the likelihood is the
one the 200-mode baseline (`results/single_psr/discovery_gp-4_sp-10_m200/`) used. At
log10_wpsd = -40 the two log-likelihoods must agree to rounding; at the box floor they must
agree to well under a nat, so "off" is reachable inside the prior.

Needs the discovery env and the NG15 feathers; skipped otherwise.
"""

import importlib.util
import os
import sys

import numpy as np
import pytest

pytest.importorskip("discovery")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SP = os.path.join(REPO_ROOT, "workflows", "ng15_nb", "single_psr")
DISCOVERY_DIR = os.environ.get(
    "NG15_DISCOVERY_DIR", "/fred/oz022/tkimpson/external/discovery"
)
if not os.path.isdir(DISCOVERY_DIR):
    pytest.skip("NG15 Discovery data not present", allow_module_level=True)


def _run_discovery():
    sys.path.insert(0, SP)
    spec = importlib.util.spec_from_file_location(
        "run_discovery", os.path.join(SP, "run_discovery.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def loglikes():
    rd = _run_discovery()
    rd.RN_COMPONENTS = 200
    psr = rd.load("J1713+0747")
    span_s = rd.common.array_span_s(rd.DISCOVERY_DIR)
    return (
        rd.make_loglike(psr, span_s, free_white=False),
        rd.make_loglike(psr, span_s, free_white=True),
        rd.WQ_BOX,
    )


@pytest.mark.parametrize(
    "gamma_p, sigma_p", [(-8.0, -14.0), (-5.0, -13.7), (-11.0, -20.0)]
)
def test_white_term_off_matches_fixed_noise(loglikes, gamma_p, sigma_p):
    fixed, free, _ = loglikes
    a = float(fixed(gamma_p, sigma_p))
    b = float(free(gamma_p, sigma_p, -40.0))
    assert np.isfinite(a)
    assert abs(a - b) < 1e-6 * max(1.0, abs(a))


def test_white_term_floor_is_off(loglikes):
    fixed, free, box = loglikes
    a = float(fixed(-8.0, -14.0))
    b = float(free(-8.0, -14.0, box[0]))
    assert abs(a - b) < 0.1


def test_white_term_on_changes_likelihood(loglikes):
    fixed, free, box = loglikes
    a = float(fixed(-8.0, -14.0))
    b = float(free(-8.0, -14.0, box[1]))
    assert (
        b < a - 10.0
    )  # a 1e-6 s²/Hz flat PSD (≈ 8 µs white) on J1713 is strongly disfavoured

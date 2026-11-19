"""`SparseArrayFilter` is the Phase 3 likelihood with the harness's ε applied (spec 5.1).

Needs the gitignored ingest caches, so it is skipped where they are absent (Argus CI):

    conda activate Argus
    PYTHONPATH=$PWD/python python -m pytest workflows/ng15_nb/array -q

On a 3-pulsar subset (one on epoch cells), at random points inside the production boxes,
`get_likelihood(θ)` must equal `union.array_loglike_sparse(p, inp) + const` exactly, with
the reference's ORF replaced by C(ε) = (1 − ε)I + εΓ_HD, at ε = 0, 0.5 and 1. EFAC and
EQUAD must not move it, since white noise is fixed inside the compression.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "compression"))

INGEST = Path(
    os.environ.get("NG15_INGEST_DIR", REPO_ROOT / "workflows/ng15_nb/data/ingest")
)
NAMES = ["B1855+09", "J1012+5307", "J1909-3744"]
EPOCH = {"J1012+5307"}

pytestmark = pytest.mark.skipif(
    not all((INGEST / f"{n}.npz").exists() for n in NAMES),
    reason="NG15 narrowband ingest caches not present",
)


@pytest.fixture(scope="module")
def kf():
    from sparse_filter import SparseArrayFilter

    return SparseArrayFilter(INGEST, NAMES, EPOCH)


def theta(p, n, eps, efac=None, equad=None):
    from argus.bayesian_inference import Parameters

    return Parameters(
        log10_gamma_a=p[1],
        γa=10.0 ** p[1],
        ha=10.0 ** p[0],
        σp=10.0 ** p[2 : 2 + n],
        γp=10.0 ** p[2 + n : 2 + 2 * n],
        EFAC=jnp.ones(n) if efac is None else efac,
        EQUAD=jnp.zeros(n) if equad is None else equad,
        orf_epsilon=eps,
    )


def points(n, k=3, seed=0):
    rng = np.random.default_rng(seed)
    return [
        np.concatenate(
            [
                [rng.uniform(-15.5, -13.5), rng.uniform(-10.0, -7.0)],
                rng.uniform(-20.0, -13.0, n),
                rng.uniform(-11.0, -7.0, n),
            ]
        )
        for _ in range(k)
    ]


@pytest.mark.parametrize("eps", [0.0, 0.5, 1.0])
def test_equals_phase3_likelihood(kf, eps):
    import union
    from argus import gravitational_waves

    ref_inp = dict(kf.inp, orf=gravitational_waves.correlation_path(kf.inp["orf"], eps))
    for p in points(kf.Npsr):
        got = float(kf.get_likelihood(theta(jnp.asarray(p), kf.Npsr, eps)))
        want = float(
            union.array_loglike_sparse(jnp.asarray(p), ref_inp) + kf.inp["const"]
        )
        assert np.isfinite(got)
        assert got == want, (eps, got, want, got - want)


def test_white_noise_parameters_ignored(kf):
    p = jnp.asarray(points(kf.Npsr, k=1)[0])
    base = float(kf.get_likelihood(theta(p, kf.Npsr, 0.0)))
    other = float(
        kf.get_likelihood(
            theta(
                p,
                kf.Npsr,
                0.0,
                efac=jnp.full(kf.Npsr, 1.7),
                equad=jnp.full(kf.Npsr, 1e-6),
            )
        )
    )
    assert base == other


def test_hd_moves_the_likelihood(kf):
    """ε reaches the ORF: the CURN and HD values differ at a GW-loud point."""
    p = jnp.asarray(points(kf.Npsr, k=1)[0]).at[0].set(-13.5)
    l0 = float(kf.get_likelihood(theta(p, kf.Npsr, 0.0)))
    l1 = float(kf.get_likelihood(theta(p, kf.Npsr, 1.0)))
    assert l0 != l1


def test_resume_fingerprint(kf):
    """The harness's checkpoint fingerprint reads the filter's data, ORF and timing prior."""
    from argus.bayesian_inference import _data_fingerprint

    fp = _data_fingerprint(kf)
    assert fp["n_pulsars"] == kf.Npsr
    assert fp["n_epochs"] == kf.n_obs
    assert fp["m_sum"] == kf.M_sum
    assert fp["timing_prior"] == "diffuse"

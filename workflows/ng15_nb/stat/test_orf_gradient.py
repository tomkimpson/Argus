"""V1 and V2 of `spec/detection-statistic.md` §3 on a 3-pulsar subset (task 1.2's test).

Needs the gitignored ingest caches, so it is skipped where they are absent (Argus CI):

    conda activate Argus
    export NG15_INGEST_DIR=/fred/oz022/tkimpson/Argus/workflows/ng15_nb/data/ingest
    PYTHONPATH=$PWD/python python -m pytest workflows/ng15_nb/stat -q

Without NG15_INGEST_DIR a worktree has no caches, and every test skips silently.

- V2: lnL with the ORF fed as-is at C = I equals `SparseArrayFilter.get_likelihood` at
  ε = 0 to ≤ 1e-8 nats.
- V1: the contraction Σ G (Γ − I) equals ∂lnL/∂ε at ε = 0 computed by forward mode
  through the harness path (`get_likelihood` applying `correlation_path` itself) to
  ≤ 1e-6 relative, for Γ_HD and for a scrambled ORF; a central finite difference agrees
  to ≤ 1e-4 relative.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import orf_gradient as og  # noqa: E402

NAMES = ["B1855+09", "J1012+5307", "J1909-3744"]
EPOCH = {"J1012+5307"}

pytestmark = pytest.mark.skipif(
    not all((og.DEFAULT_INGEST / f"{n}.npz").exists() for n in NAMES),
    reason="NG15 narrowband ingest caches not present",
)


@pytest.fixture(scope="module")
def setup():
    kf = og.build_filter(names=NAMES, epoch_cells=EPOCH)
    return kf, og.make_fns(kf)


def points(n, k=3, seed=0):
    """Random points in the production box, GW-loud enough that S is not ~0."""
    rng = np.random.default_rng(seed)
    return np.array(
        [
            np.concatenate(
                [
                    [rng.uniform(-14.5, -13.5), rng.uniform(-10.0, -7.0)],
                    rng.uniform(-20.0, -13.0, n),
                    rng.uniform(-11.0, -7.0, n),
                ]
            )
            for _ in range(k)
        ]
    )


def scrambled_orf(n, seed=1):
    sky = og.load_script("sky_scrambles")
    ra, dec = sky.random_sky_positions(n, np.random.default_rng(seed))
    return sky.build_orf(ra, dec)


def test_v2_value_path_unchanged(setup):
    kf, f = setup
    eye = np.eye(kf.Npsr)
    for p in points(kf.Npsr):
        got = float(f["lnl_at"](eye, p))
        want = float(f["harness_lnl"](p))
        assert np.isfinite(got)
        assert abs(got - want) <= 1e-8, (got, want, got - want)


@pytest.mark.parametrize("which", ["hd", "scramble"])
def test_v1_contraction_exact(setup, which):
    kf, f = setup
    orf = np.asarray(kf.inp["orf"]) if which == "hd" else scrambled_orf(kf.Npsr)
    assert np.all(np.diag(orf) == 1.0)
    P = points(kf.Npsr)
    G = og.batched_G(f["grad_orf"], P, batch=len(P))
    assert np.all(np.isfinite(G))
    contracted = og.score(G, [orf])[:, 0]
    direct = np.array([float(f["eps_jvp"](orf, p)) for p in P])
    scale = np.max(np.abs(direct))
    assert scale > 0
    rel = np.abs(contracted - direct) / np.maximum(np.abs(direct), scale)
    assert rel.max() <= 1e-6, (contracted, direct, rel)
    h = 1e-4
    fd = np.array(
        [
            (float(f["eps_value"](orf, p, h)) - float(f["eps_value"](orf, p, -h)))
            / (2 * h)
            for p in P
        ]
    )
    rel_fd = np.abs(contracted - fd) / np.maximum(np.abs(fd), scale)
    assert rel_fd.max() <= 1e-4, (contracted, fd, rel_fd)


def test_g_is_batch_invariant(setup):
    """Same draws, different batch sizes: G agrees (task 1.4's rerun check, in small)."""
    kf, f = setup
    P = points(kf.Npsr, k=4, seed=3)
    a = og.batched_G(f["grad_orf"], P, batch=4)
    b = og.batched_G(f["grad_orf"], P, batch=1)
    assert np.max(np.abs(a - b)) <= 1e-8 * max(1.0, np.max(np.abs(a)))

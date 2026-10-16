"""Exact epoch compression on a synthetic pulsar (spec D2, task 2.2).

The synthetic pulsar carries every ingest edge case D2 names:

- a window with two ECORR epochs;
- an ECORR epoch straddling two windows (they must be merged);
- a single-frequency window (D ∝ 1, so it can carry no achromatic information);
- a global DM column, which is exactly the sum of the DMX columns because the windows
  tile all TOAs (the rank rule must drop it);
- two backends, a backend JUMP and a quadratic spin-down.

With the state held constant within each cell, A1 holds exactly, so the compressed
likelihood must equal the dense flat-prior likelihood on the raw TOAs (DMX and timing
columns marginalised together) up to the tracked constant `const`, for any state
covariance. Tolerance 1e-10 relative to |lnL|.

The rank rule re-expresses β in a whitened singular basis; the dense side uses the same
basis (`comp["basis"]`), so that change carries no constant.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "workflows" / "ng15_nb" / "compression"))
import compress  # noqa: E402

DAY = 86400.0


def synthetic_pulsar(seed=0):
    """A per-TOA cache in the ingest format, with every D2 edge case."""
    rng = np.random.default_rng(seed)
    t_start = compress.T0_DEFAULT + 3.0 * DAY
    toas, freqs, backend, window, epoch = [], [], [], [], []
    n_ep = 0

    def observe(t, b, w, nfreq=4, f_single=None):
        nonlocal n_ep
        f = np.full(nfreq, f_single) if f_single else rng.uniform(700, 2000, nfreq)
        toas.extend(t + rng.uniform(0, 0.5, nfreq))  # subbands within 1 s: one epoch
        freqs.extend(f)
        backend.extend([b] * nfreq)
        window.extend([w] * nfreq)
        epoch.extend([n_ep] * nfreq)
        n_ep += 1

    w = 0
    for i in range(14):
        t = t_start + 11.0 * i * DAY
        b = "A" if i % 3 else "B"
        if i == 4:  # two ECORR epochs inside one window
            observe(t, b, w)
            observe(t + 0.6 * DAY, b, w)
        elif i == 7:  # single-frequency window
            observe(t, b, w, nfreq=3, f_single=1400.0)
        elif i == 9:  # one ECORR epoch straddling windows w and w+1
            observe(t, b, w)
            observe(t + 0.3 * DAY, b, w + 1)
            toas.append(toas[-1] + 0.2)  # same backend, < 1 s later: same epoch,
            freqs.append(900.0)  # but assigned to window w (straddle)
            backend.append(b)
            window.append(w)
            epoch.append(n_ep - 1)
            w += 1
        else:
            observe(t, b, w)
        w += 1

    toas, freqs = np.array(toas), np.array(freqs)
    backend, window, epoch = np.array(backend), np.array(window), np.array(epoch)
    n = toas.size
    # Epoch ids per backend (ingest's convention: contiguous, backend-major).
    _, epoch = np.unique(
        [f"{b}:{e:04d}" for b, e in zip(backend, epoch)], return_inverse=True
    )

    nw = window.max() + 1
    D = np.zeros((n, nw))
    D[np.arange(n), window] = (1400.0 / freqs) ** 2
    x = (toas - toas.mean()) / (toas.max() - toas.min())
    Mt = np.column_stack(
        [
            np.ones(n),  # offset
            x,  # F0-like
            x**2,  # F1-like
            (1400.0 / freqs) ** 2,  # DM: the sum of the DMX columns
            (backend == "B").astype(float),  # JUMP
        ]
    )
    M = np.column_stack([Mt, D])
    names = np.array(
        ["Offset", "F0", "F1", "DM", "JUMP_B"] + [f"DMX_{i:04d}" for i in range(nw)]
    )
    is_dmx = np.array([False] * Mt.shape[1] + [True] * nw)

    err = rng.uniform(0.3, 1.5, n) * 1e-6
    white = err**2 * rng.uniform(0.8, 1.3, n)
    ecorr_var = rng.uniform(0.2, 1.0, epoch.max() + 1) * 1e-12
    return {
        "name": "SYN",
        "f0": 300.0,
        "toas": toas,
        "residuals": rng.standard_normal(n) * 1e-6,
        "toaerrs": err,
        "freqs": freqs,
        "backend": backend,
        "white_var": white,
        "ecorr_epoch": epoch,
        "ecorr_var": ecorr_var,
        "dmx_window": window,
        "M": M,
        "M_names": names,
        "is_dmx": is_dmx,
    }


def _dense(cache, comp, K_cells):
    """Dense flat-prior lnL on the raw TOAs with a state constant per cell."""
    n = cache["toas"].size
    U = np.zeros((n, comp["cell"].max() + 1))
    U[np.arange(n), comp["cell"]] = 1.0
    N = np.diag(cache["white_var"])
    for e in np.unique(cache["ecorr_epoch"]):
        s = cache["ecorr_epoch"] == e
        N[np.ix_(s, s)] += cache["ecorr_var"][e]
    C = U @ K_cells @ U.T + N
    timing = cache["M"][:, ~cache["is_dmx"]] @ comp["basis"]
    X = np.column_stack([cache["M"][:, cache["is_dmx"]], timing])
    return compress.gls_diffuse(cache["residuals"], C, X)


def _K(times, amp, tau_days):
    dt = np.abs(times[:, None] - times[None, :]) / DAY
    return amp**2 * np.exp(-dt / tau_days) + 1e-30 * np.eye(times.size)


@pytest.mark.parametrize("delta_days", [None, 30.0])
@pytest.mark.parametrize("amp,tau", [(3e-7, 50.0), (2e-6, 400.0), (1e-9, 5.0)])
def test_compressed_equals_dense(delta_days, amp, tau):
    cache = synthetic_pulsar()
    comp = compress.compress_pulsar(cache, delta_days=delta_days)
    K = _K(comp["times"], amp, tau)
    ll_dense = _dense(cache, comp, K)
    ll_comp = compress.compressed_dense_loglike(comp, K) + comp["const"]
    assert abs(ll_comp - ll_dense) <= 1e-10 * abs(ll_dense), (ll_comp, ll_dense)


def test_edge_cases_are_handled():
    cache = synthetic_pulsar()
    comp = compress.compress_pulsar(cache)
    n_windows = cache["dmx_window"].max() + 1
    # The straddling ECORR epoch merges two windows into one cell.
    assert comp["cell"].max() + 1 == n_windows - 1
    # The single-frequency window emits no scalar.
    single = np.unique(comp["cell"][cache["freqs"] == 1400.0])
    assert single.size == 1 and not comp["has_scalar"][single[0]]
    assert comp["has_scalar"].sum() == n_windows - 2
    # DM is degenerate with DMX: exactly one direction is dropped, and DM dominates it.
    assert comp["dropped"].tolist() == ["DM"]
    assert comp["basis"].shape == (5, 4)
    assert comp["singular_ratio2"].min() < 1e-25
    # Two ECORR epochs inside one window share a cell.
    cells_multi = [
        c
        for c in range(comp["cell"].max() + 1)
        if np.unique(cache["ecorr_epoch"][comp["cell"] == c]).size > 1
    ]
    assert len(cells_multi) >= 2  # the two-epoch window and the merged pair


def test_grid_cells_are_stamped_at_their_centres():
    cache = synthetic_pulsar()
    comp = compress.compress_pulsar(cache, delta_days=30.0)
    k = (comp["times"] - compress.T0_DEFAULT) / (30.0 * DAY) - 0.5
    np.testing.assert_allclose(k, np.round(k), atol=1e-12)
    assert np.all(np.diff(comp["times"]) > 0)


# --------------------------------------------------------------------------- #
# Dense oracle vs the filter (task 2.4)
# --------------------------------------------------------------------------- #
import types  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import oracle  # noqa: E402

# (log10 ha, log10 γa, log10 σp, log10 γp) with µs-level signal and γ·span ≳ 1, so the
# state is not absorbed by the F0/F1 columns and the gradients are O(0.1–1).
POINTS = [
    (-12.0, -6.0, -13.0, -6.0),
    (-11.5, -6.5, -13.5, -5.5),
    (-12.5, -5.5, -12.5, -6.5),
]


def _jitter_free_filter_module():
    """The library filter with its two 1e-9 PD jitters set to zero (test-only copy)."""
    src = (REPO / "python" / "argus" / "jax_kalman_filter.py").read_text()
    edits = [
        ("return 1e-9 * (jnp.sum", "return 0.0 * (jnp.sum"),
        ("return 1e-9 * jnp.sum", "return 0.0 * jnp.sum"),
        ("jitter = 1e-9 * (jnp.trace", "jitter = 0.0 * (jnp.trace"),
    ]
    for old, new in edits:
        assert src.count(old) == 1, f"jitter site moved: {old!r}"
        src = src.replace(old, new)
    mod = types.ModuleType("argus._jk_nojitter")
    mod.__package__ = "argus"
    exec(compile(src, "jk_nojitter", "exec"), mod.__dict__)
    return mod


def _collapsed_synthetic():
    """The synthetic pulsar with each cell's TOAs moved onto one time, so A1 is exact
    for the continuous-time model too."""
    cache = synthetic_pulsar()
    cell = compress.compress_pulsar(cache)["cell"]
    t = cache["toas"].copy()
    for k in np.unique(cell):
        t[cell == k] = t[cell == k].min()
    return dict(cache, toas=t)


@pytest.fixture(scope="module")
def collapsed():
    cache = _collapsed_synthetic()
    comp = compress.compress_pulsar(cache)
    return comp, oracle.oracle_inputs(cache, comp), oracle.filter_inputs(comp)


@pytest.mark.parametrize("p", POINTS)
def test_oracle_equals_jitter_free_filter(collapsed, p, monkeypatch):
    """With A1 exact, the only filter–oracle difference is the filter's PD jitter."""
    comp, oi, fi = collapsed
    monkeypatch.setattr(oracle, "jk", _jitter_free_filter_module())
    p = jnp.asarray(p)
    lo = float(oracle.oracle_loglike(p, oi))
    lf = float(oracle.filter_loglike(p, fi)) + comp["const"]
    assert abs(lf - lo) <= 1e-10 * abs(lo), (lf, lo)
    go = np.asarray(jax.grad(oracle.oracle_loglike)(p, oi))
    gf = np.asarray(jax.grad(oracle.filter_loglike)(p, fi))
    assert np.max(np.abs(go)) > 0.1  # the parameters matter at these points
    np.testing.assert_allclose(gf, go, rtol=1e-6, atol=1e-6 * np.max(np.abs(go)))


@pytest.mark.parametrize("p", POINTS)
def test_oracle_matches_stock_filter(collapsed, p):
    """The production filter (with jitter) agrees well inside C1's tolerances."""
    comp, oi, fi = collapsed
    p = jnp.asarray(p)
    lo = float(oracle.oracle_loglike(p, oi))
    lf = float(oracle.filter_loglike(p, fi)) + comp["const"]
    assert abs(lf - lo) <= 1e-4, (lf, lo)
    go = np.asarray(jax.grad(oracle.oracle_loglike)(p, oi))
    gf = np.asarray(jax.grad(oracle.filter_loglike)(p, fi))
    assert np.max(np.abs(gf - go)) <= 1e-4 * np.max(np.abs(go))


def test_single_step_grid():
    """All TOAs at one time and one grid cell: a one-step filter equals the oracle."""
    cache = synthetic_pulsar()
    cache = dict(cache, toas=np.full_like(cache["toas"], cache["toas"].min()))
    comp = compress.compress_pulsar(cache, delta_days=1e5)
    assert comp["times"].size == 1
    oi, fi = oracle.oracle_inputs(cache, comp), oracle.filter_inputs(comp)
    for p in POINTS:
        p = jnp.asarray(p)
        lo = float(oracle.oracle_loglike(p, oi))
        lf = float(oracle.filter_loglike(p, fi)) + comp["const"]
        assert abs(lf - lo) <= 1e-10 * abs(lo), (lf, lo)

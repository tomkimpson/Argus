"""Exact epoch compression on a synthetic pulsar (spec D2, task 2.2).

The synthetic pulsar carries every ingest edge case D2 names:

- a window with two ECORR epochs;
- an ECORR epoch straddling two windows (they must be merged);
- a single-frequency window (D ∝ 1, so it can carry no achromatic information);
- a global DM column, which is exactly the sum of the DMX columns because the windows
  tile all TOAs (the rank rule must drop it);
- two backends, a backend JUMP and a quadratic spin-down.

With the state held constant within each cell (one scalar per cell), or linear in time
within each cell (two scalars, position and velocity), A1 holds exactly, so the compressed
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
    """Dense flat-prior lnL on the raw TOAs with a state constant per cell, or (for a
    (2 n_cells)² K_cells) linear per cell: s(t) = s_w + ṡ_w (t − t_w)."""
    n = cache["toas"].size
    nc = comp["cell"].max() + 1
    rows = np.arange(n)
    if K_cells.shape[0] == 2 * nc:
        U = np.zeros((n, 2 * nc))
        U[rows, 2 * comp["cell"]] = 1.0
        U[rows, 2 * comp["cell"] + 1] = cache["toas"] - comp["times"][comp["cell"]]
    else:
        U = np.zeros((n, nc))
        U[rows, comp["cell"]] = 1.0
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


def _K2(times, amp, tau_days):
    """(position, velocity) covariance interleaved per cell: independent exponential
    kernels, the velocity one scaled to a rate of amp per tau."""
    Kp = _K(times, amp, tau_days)
    K = np.zeros((2 * times.size,) * 2)
    K[0::2, 0::2] = Kp
    K[1::2, 1::2] = Kp / (tau_days * DAY) ** 2
    return K


@pytest.mark.parametrize("velocity", [False, True])
@pytest.mark.parametrize("delta_days", [None, 30.0])
@pytest.mark.parametrize("amp,tau", [(3e-7, 50.0), (2e-6, 400.0), (1e-9, 5.0)])
def test_compressed_equals_dense(velocity, delta_days, amp, tau):
    cache = synthetic_pulsar()
    comp = compress.compress_pulsar(cache, delta_days=delta_days, velocity=velocity)
    K = (_K2 if velocity else _K)(comp["times"], amp, tau)
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
    # Every cell with a position scalar spans enough time to fix a slope, except none:
    # the single-frequency cell has neither scalar.
    assert not comp["has_velocity"][single[0]]
    assert np.array_equal(comp["has_velocity"], comp["has_scalar"])
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
    """The library filter with its innovation-S PD jitters set to zero (test-only copy).

    The diffuse-Λ jitter was removed from the library in PR #129; only the `_jitter_scale`
    sites remain."""
    src = (REPO / "python" / "argus" / "jax_kalman_filter.py").read_text()
    edits = [
        ("return 1e-9 * (jnp.sum", "return 0.0 * (jnp.sum"),
        ("return 1e-9 * jnp.sum", "return 0.0 * jnp.sum"),
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
    """The production filter agrees with the oracle far inside C1's tolerances.

    Since PR #129 removed the diffuse-Λ jitter, only the innovation-S jitter separates
    them: measured 2.4e-9 nat and 1.3e-8 relative gradient error at these points."""
    comp, oi, fi = collapsed
    p = jnp.asarray(p)
    lo = float(oracle.oracle_loglike(p, oi))
    lf = float(oracle.filter_loglike(p, fi)) + comp["const"]
    assert abs(lf - lo) <= 1e-7, (lf, lo)
    go = np.asarray(jax.grad(oracle.oracle_loglike)(p, oi))
    gf = np.asarray(jax.grad(oracle.filter_loglike)(p, fi))
    assert np.max(np.abs(gf - go)) <= 1e-6 * np.max(np.abs(go))


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


def _iou_joint_cov(t, gamma, sigma2):
    """Exact (position, velocity) covariance of an integrated OU at times t (from the
    origin, stationary velocity), interleaved per time: ∂ of the oracle's position kernel."""

    def k(s, u):
        return oracle._iou_cov(jnp.array([s, u]), gamma, sigma2)[0, 1]

    k_sv = jax.grad(k, argnums=1)
    k_vv = jax.grad(k_sv, argnums=0)
    S, U = jnp.meshgrid(t, t, indexing="ij")
    f = lambda g: jax.vmap(jax.vmap(g))(S, U)  # noqa: E731
    xx = oracle._iou_cov(t, gamma, sigma2)
    xv, vv = f(k_sv), f(k_vv)
    # On the diagonal the kernel's kink at s = u makes autodiff one-sided, so use
    # Cov[x(t), v(t)] = ½ d Var x(t)/dt and the stationary Var v = σ²/2γ.
    dvar = jax.vmap(jax.grad(lambda s: 0.5 * k(s, s)))(t)
    xv = xv.at[jnp.diag_indices(t.size)].set(dvar)
    vv = vv.at[jnp.diag_indices(t.size)].set(sigma2 / (2.0 * gamma))
    K = jnp.zeros((2 * t.size,) * 2)
    K = K.at[0::2, 0::2].set(xx).at[0::2, 1::2].set(xv)
    K = K.at[1::2, 0::2].set(xv.T).at[1::2, 1::2].set(vv)
    return np.asarray(K)


@pytest.mark.parametrize("delta_days", [None, 30.0])
@pytest.mark.parametrize("p", POINTS)
def test_two_step_filter_equals_compressed_dense(p, delta_days, monkeypatch):
    """Two filter steps per cell at dt = 0 reproduce the two-scalar compressed likelihood
    under the exact joint (s, ṡ) covariance at the cell stamps. At Δ = 30 d the stamps are
    grid centres, so this also checks the position row's (t_w − t_c) shift."""
    cache = synthetic_pulsar()
    comp = compress.compress_pulsar(cache, delta_days=delta_days)
    fi = oracle.filter_inputs(comp)
    assert int(np.sum(np.asarray(fi["mask_matrices"]) == 0)) == 0  # no lone positions
    monkeypatch.setattr(oracle, "jk", _jitter_free_filter_module())
    ha, ga, sp, gp = oracle.physical(jnp.asarray(p))
    has = comp["has_scalar"]
    t = jnp.asarray(comp["times"] - comp["times"][has][0])
    K = _iou_joint_cov(t, ga, ha**2 * ga / 12.0) + _iou_joint_cov(t, gp, sp**2) / comp[
        "f0"
    ] ** 2
    ll_dense = compress.compressed_dense_loglike(comp, K)
    ll_filt = float(oracle.filter_loglike(jnp.asarray(p), fi))
    assert abs(ll_filt - ll_dense) <= 1e-9 * abs(ll_dense), (ll_filt, ll_dense)


def test_masked_velocity_step_offset():
    """A cell without a velocity scalar is a masked step; `offset` cancels its log 2π."""
    cache = synthetic_pulsar()
    comp = compress.compress_pulsar(cache)
    i = int(np.nonzero(comp["has_velocity"])[0][3])
    comp_m = dict(comp, has_velocity=comp["has_velocity"].copy())
    comp_m["has_velocity"][i] = False
    fi = oracle.filter_inputs(comp_m)
    assert int(np.sum(np.asarray(fi["mask_matrices"]) == 0)) == 1
    assert fi["offset"] == pytest.approx(0.5 * np.log(2 * np.pi))
    # Same likelihood as dropping that velocity row from the dense compressed form.
    p = jnp.asarray(POINTS[0])
    ha, ga, sp, gp = oracle.physical(p)
    has = comp["has_scalar"]
    t = jnp.asarray(comp["times"] - comp["times"][has][0])
    K = _iou_joint_cov(t, ga, ha**2 * ga / 12.0) + _iou_joint_cov(t, gp, sp**2) / comp[
        "f0"
    ] ** 2
    ll_dense = compress.compressed_dense_loglike(comp_m, K)
    ll_filt = float(oracle.filter_loglike(p, fi))
    # The removed scalar's own −½[log 2πσ_v²] is part of `const`, not of either side.
    assert abs(ll_filt - ll_dense) <= 1e-6 * abs(ll_dense), (ll_filt, ll_dense)


# --------------------------------------------------------------------------- #
# Union grid (task 2.6)
# --------------------------------------------------------------------------- #
import union  # noqa: E402


@pytest.mark.parametrize("delta_days", [None, 30.0])
def test_union_grid_equals_sum_of_pulsars(delta_days):
    """With an identity ORF the pulsars are independent, so the masked union-grid filter
    equals the sum of the single-pulsar filters (each on its own cells). At Δ = 30 d the
    two pulsars share grid times."""
    caches = [synthetic_pulsar(seed=0), synthetic_pulsar(seed=1)]
    caches[1] = dict(caches[1], f0=211.0, toas=caches[1]["toas"] + 2.3 * DAY)
    comps = [compress.compress_pulsar(c, delta_days=delta_days) for c in caches]
    inp = union.build_union_inputs(comps, np.eye(2))
    n_cells = sum(int(c["has_scalar"].sum()) for c in comps)
    if delta_days is None:
        assert inp["n_union_times"] == n_cells
    else:
        assert inp["n_union_times"] < n_cells
    p1 = (-12.0, -6.0, -13.0, -6.0)
    p2 = (-12.0, -6.0, -12.5, -6.5)
    pa = jnp.asarray([p1[0], p1[1], p1[2], p2[2], p1[3], p2[3]])
    ll_array = float(union.array_loglike(pa, inp))
    ll_sum = sum(
        float(oracle.filter_loglike(jnp.asarray(p), oracle.filter_inputs(c)))
        for p, c in zip((p1, p2), comps)
    )
    assert abs(ll_array - ll_sum) <= 1e-8 * abs(ll_sum), (ll_array, ll_sum)

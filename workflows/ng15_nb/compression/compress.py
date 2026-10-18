"""Exact epoch compression with local DMX marginalisation (spec D2, task 2.2).

Within a compression cell (one DMX window, or every window in a grid cell of width Δ) the
TOAs are modelled as

    z = 1 s + D d + M β + n,   n ~ N(0, N),   N = diag(white) + ECORR blocks (fixed),

with s the achromatic state (taken constant in the cell: assumption A1), d the cell's DMX
parameters (flat prior, marginalised here) and β the global timing parameters. Whitening
by N's Cholesky factor and projecting out D, the data split exactly into

- a scalar ŝ = s + mᵀβ + e, e ~ N(0, σ²), which the filter takes as one observation;
- an orthogonal remainder that depends on β alone. Its quadratic form adds a static
  (A₀, b₀, c₀) to the marginal filter's accumulators (`accum0`).

With `velocity=True` (the default since the 2026-10-18 amendment) a second scalar per cell
measures the state's rate: v̂ = ṡ + m'ᵀβ + e', along the projected direction of
τ = t − t_w, where t_w is ŝ's weighted mean time. That choice of t_w makes the τ direction
orthogonal to ŝ's, so e' is independent of e. One scalar per cell failed C1 because the
within-cell drift τ·ṡ went into the β-only remainder (spec §12, 2026-10-18).

The result is exact for a state constant per cell (one scalar) or linear in time per cell,
s(t) = s_w + ṡ_w (t − t_w) (two scalars): `const` is the parameter-independent offset such
that

    dense_gls(z; C, [D, M_kept]) = filter_or_gls(ŝ, σ², m, accum0) + const,

where dense_gls is the flat-prior marginal −½[zᵀC⁻¹z − bᵀA⁻¹b + logdet(2πC) + logdet A]
(the filter's own convention, see `_run_kalman_filter_marginal`), and the right side is
evaluated in the same convention.

Edge-case rules (spec D2):
- ECORR epochs straddling two windows: the windows are merged into one cell first.
- Several ECORR epochs inside one cell: handled by the full N metric.
- A cell observed at a single radio frequency has D ∝ 1, so projecting out D removes the
  achromatic direction too (a = 0). It emits no scalar; its β-only remainder is still
  kept in (A₀, b₀, c₀), which is exactly what the dense likelihood sees. The same rule
  drops the velocity scalar of a cell whose projected τ direction vanishes (too few TOAs
  or frequencies to fix a slope beyond DMX and the mean).
- Epoch cells (`per_epoch=True`): each ECORR epoch is its own cell, and the DMX columns
  stay in β as ordinary timing columns instead of being projected out per window. A1 then
  only needs the state constant within one epoch (TOAs within ~1 s), at the price of the
  DMX columns joining M_sum. This is for pulsars whose red noise decorrelates within a DMX
  window (J1012+5307), where the window compression fails C1 (spec §12, 2026-10-18).
- Rank rule: after projection, SVD of the stacked projected design (columns normalised by
  their unprojected norms). Directions whose squared singular-value ratio falls below
  `rank_tol` (numerically in the DMX span) are dropped, and β is re-expressed in the
  remaining whitened singular basis. On NG15 the smallest ratio is 3.7e-13 (J1853+1303),
  so at the default 1e-14 nothing is dropped; an exact degeneracy (a global DM column
  with DMX tiling every TOA) sits near 1e-30.
"""

import numpy as np
from scipy.linalg import cho_factor, solve_triangular

LN2PI = np.log(2.0 * np.pi)
DAY = 86400.0
T0_DEFAULT = 53000.0 * DAY  # grid origin, MJD 53000 in the TOAs' seconds convention


# --------------------------------------------------------------------------- #
# Cells
# --------------------------------------------------------------------------- #
def _merge_windows_sharing_epochs(window, epoch):
    """Union-find: windows that share an ECORR epoch become one window."""
    parent = {w: w for w in np.unique(window)}

    def find(w):
        while parent[w] != w:
            parent[w] = parent[parent[w]]
            w = parent[w]
        return w

    for e in np.unique(epoch):
        ws = np.unique(window[epoch == e])
        for w in ws[1:]:
            parent[find(w)] = find(ws[0])
    roots = np.array([find(w) for w in window])
    _, merged = np.unique(roots, return_inverse=True)
    return merged


def assign_cells(cache, delta_days=None, t0=T0_DEFAULT):
    """Cell id per TOA, and the grid index of each cell (None for DMX-window cells).

    TOAs without a DMX window (none in NG15) get a window per ECORR epoch. Windows that
    share an ECORR epoch are merged. With `delta_days`, each merged window joins the grid
    cell holding its mean TOA time.
    """
    window = cache["dmx_window"].copy()
    epoch = cache["ecorr_epoch"]
    loose = window < 0
    if loose.any():
        _, ep_id = np.unique(epoch[loose], return_inverse=True)
        window[loose] = window.max() + 1 + ep_id
    merged = _merge_windows_sharing_epochs(window, epoch)
    if delta_days is None:
        return merged, None
    t_mean = np.bincount(merged, weights=cache["toas"]) / np.bincount(merged)
    k = np.floor((t_mean - t0) / (delta_days * DAY)).astype(np.int64)
    grid_ids, cell = np.unique(k[merged], return_inverse=True)
    return cell, grid_ids


def _noise_cholesky(white_var, epoch, ecorr_var):
    """Lower Cholesky factor of N = diag(white) + Σ_e ecorr_e 1_e 1_eᵀ."""
    N = np.diag(white_var)
    for e in np.unique(epoch):
        s = epoch == e
        N[np.ix_(s, s)] += ecorr_var[e]
    return np.linalg.cholesky(N)


# --------------------------------------------------------------------------- #
# Compression
# --------------------------------------------------------------------------- #
def compress_pulsar(
    cache,
    delta_days=None,
    t0=T0_DEFAULT,
    rank_tol=1e-14,
    velocity=True,
    per_epoch=False,
):
    """Compress one pulsar's TOAs. See the module docstring for the model.

    Args:
        cache: ingest dict (ingest.ingest / ingest.load).
        delta_days: grid cell width; None compresses each (merged) DMX window alone.
        t0: grid origin (s). Cell k is [t0 + kΔ, t0 + (k+1)Δ), stamped at its centre.
        rank_tol: rank-rule threshold on the squared singular-value ratio.
        velocity: emit the second (rate) scalar per cell. Grid cells are still stamped
            at their centres t_c, so pulsars share filter steps; the position scalar
            measures s(t_w) = s(t_c) + ṡ(t_c)(t_w − t_c), which the filter's position row
            carries as a −(t_w − t_c) coefficient on the rate states (exact for a state
            linear in the cell).
        per_epoch: one cell per ECORR epoch, with DMX kept in β (forces velocity off:
            an epoch spans ~1 s).

    Returns a dict: times (cell timestamps, s), s_hat, sigma2, m (cells × kept, scaled),
    v_hat, sigma2_v, m_v, has_velocity (the rate scalar; zero/inf/False if velocity=False),
    accum0 = (A0, b0, c0), const, basis (non-DMX timing columns -> the β' the filter
    sees), dropped (the column dominating each dropped direction), singular_ratio2,
    cell (id per TOA), t_weighted (per-cell weighted mean TOA time), has_scalar per cell.
    """
    if per_epoch:
        # Cells in time order (epoch ids are backend-major, not chronological).
        _, ep = np.unique(cache["ecorr_epoch"], return_inverse=True)
        t_ep = np.bincount(ep, weights=cache["toas"]) / np.bincount(ep)
        rank = np.empty_like(ep[: t_ep.size])
        rank[np.argsort(t_ep, kind="stable")] = np.arange(t_ep.size)
        cell, grid_ids = rank[ep], None
        is_dmx = np.zeros_like(cache["is_dmx"])
        velocity = False
    else:
        cell, grid_ids = assign_cells(cache, delta_days, t0)
        is_dmx = cache["is_dmx"]
    Mt = cache["M"][:, ~is_dmx]
    Mt_names = cache["M_names"][~is_dmx]
    D_all = cache["M"][:, is_dmx]

    # Global unit-norm prescale of the timing columns (conditioning only).
    pre = 1.0 / np.sqrt(np.sum(Mt**2, axis=0))
    Mt = Mt * pre

    n_cells = cell.max() + 1
    K = Mt.shape[1]
    a = np.zeros(n_cells)
    s_hat = np.zeros(n_cells)
    m = np.zeros((n_cells, K))
    b_v = np.zeros(n_cells)
    v_hat = np.zeros(n_cells)
    m_v = np.zeros((n_cells, K))
    t_w = np.zeros(n_cells)
    rem_M, rem_z = [], []  # β-only remainder rows (whitened, projected)
    unproj_norm2 = np.zeros(K)
    const = 0.0

    # Each DMX column must live in exactly one cell (true by construction of the cells).
    col_cell = np.full(D_all.shape[1], -1)
    for j in range(D_all.shape[1]):
        cs = np.unique(cell[D_all[:, j] != 0.0])
        if cs.size != 1:
            raise ValueError(f"DMX column {j} spans cells {cs}")
        col_cell[j] = cs[0]

    z_all, t_all = cache["residuals"], cache["toas"]
    for c in range(n_cells):
        idx = np.nonzero(cell == c)[0]
        n = idx.size
        L = _noise_cholesky(
            cache["white_var"][idx], cache["ecorr_epoch"][idx], cache["ecorr_var"]
        )
        D = D_all[np.ix_(idx, np.nonzero(col_cell == c)[0])]
        w = lambda X: solve_triangular(L, X, lower=True)  # noqa: E731
        zt, onet, Dt, Mtt = w(z_all[idx]), w(np.ones(n)), w(D), w(Mt[idx])
        unproj_norm2 += np.sum(Mtt**2, axis=0)
        k = D.shape[1]
        logdetN = 2.0 * np.sum(np.log(np.diag(L)))

        if k:
            QD, RD = np.linalg.qr(Dt)
            logdetDWD = 2.0 * np.sum(np.log(np.abs(np.diag(RD))))
            proj = lambda X: X - QD @ (QD.T @ X)  # noqa: E731
        else:
            QD, logdetDWD = np.zeros((n, 0)), 0.0
            proj = lambda X: X  # noqa: E731
        p1 = proj(onet)
        a_c = p1 @ p1
        if a_c > 1e-12 * (onet @ onet):
            a[c] = a_c
            u = p1 / a_c  # ŝ = uᵀ z̃ in the whitened metric
            s_hat[c] = u @ zt
            m[c] = Mtt.T @ u
            # Weighted mean time: ŝ's weights on the raw TOAs are L⁻ᵀ u.
            t_w[c] = solve_triangular(L.T, u, lower=False) @ t_all[idx]
            e = p1 / np.sqrt(a_c)
            Qb = np.column_stack([QD, e])
            const += -0.5 * ((n - 1) * LN2PI + logdetN + logdetDWD - np.log(1.0 / a_c))
            if velocity:
                # τ = t − t_w is ⟂ e analytically (t_w is ŝ's weighted mean); the explicit
                # Gram–Schmidt step removes the roundoff.
                tau_t = w(t_all[idx] - t_w[c])
                pt = proj(tau_t)
                pt = pt - e * (e @ pt)
                b_c = pt @ pt
                if b_c > 1e-16 * (tau_t @ tau_t):
                    b_v[c] = b_c
                    v = pt / b_c  # v̂ = vᵀ z̃: unit response to ṡ, zero to s and D
                    v_hat[c] = v @ zt
                    m_v[c] = Mtt.T @ v
                    Qb = np.column_stack([Qb, pt / np.sqrt(b_c)])
                    const += -0.5 * (-LN2PI + np.log(b_c))
        else:
            Qb = QD
            t_w[c] = np.mean(t_all[idx])
            const += -0.5 * (n * LN2PI + logdetN + logdetDWD)
        rem_M.append(Mtt - Qb @ (Qb.T @ Mtt))
        rem_z.append(zt - Qb @ (Qb.T @ zt))

    rem_M = np.vstack(rem_M)
    rem_z = np.concatenate(rem_z)
    has = a > 0
    has_v = b_v > 0

    # Rank rule: SVD of the stacked projected design [remainder; √a m], columns
    # normalised by their unprojected norms. Directions whose squared singular-value
    # ratio is below rank_tol lie (numerically) in the DMX span and are dropped; β is
    # re-expressed in the remaining whitened singular basis, β = T β', so that the
    # state-free information is exactly the identity (well conditioned for the filter's
    # relative jitter). Under the flat prior only the column space matters, so the
    # basis change is exact: the dense oracle uses the same basis (`basis`).
    norm = np.sqrt(unproj_norm2)
    stacked = (
        np.vstack(
            [
                rem_M,
                np.sqrt(a[has])[:, None] * m[has],
                np.sqrt(b_v[has_v])[:, None] * m_v[has_v],
            ]
        )
        / norm
    )
    _, S, Vt = np.linalg.svd(stacked, full_matrices=False)
    ratio2 = (S / S[0]) ** 2
    keep = ratio2 >= rank_tol
    T = (Vt[keep].T / norm[:, None]) / S[keep]  # prescaled columns -> β'
    # Name each dropped direction after the column that dominates it.
    dropped = [Mt_names[int(np.argmax(np.abs(v)))] for v in Vt[~keep]]
    rem_M, m, m_v = rem_M @ T, m @ T, m_v @ T
    basis = pre[:, None] * T  # original timing columns -> β'

    A0 = rem_M.T @ rem_M
    b0 = (rem_M.T @ rem_z)[:, None]
    c0 = float(rem_z @ rem_z)

    if grid_ids is None:
        times = t_w
    else:
        times = t0 + (grid_ids + 0.5) * delta_days * DAY

    sigma2 = np.where(has, 1.0 / np.where(has, a, 1.0), np.inf)
    sigma2_v = np.where(has_v, 1.0 / np.where(has_v, b_v, 1.0), np.inf)
    return {
        "name": cache["name"],
        "f0": cache["f0"],
        "times": times,
        "t_weighted": t_w,
        "s_hat": s_hat,
        "sigma2": sigma2,
        "has_scalar": has,
        "m": m,
        "v_hat": v_hat,
        "sigma2_v": sigma2_v,
        "has_velocity": has_v,
        "m_v": m_v,
        "accum0": (A0, b0, c0),
        "const": float(const),
        "timing_names": Mt_names,
        "basis": basis,
        "dropped": np.array(dropped, dtype="U"),
        "singular_ratio2": ratio2,
        "cell": cell,
        "delta_days": delta_days,
        "timing_cols": ~is_dmx,  # the M columns `basis` maps; the rest are local DMX
    }


# --------------------------------------------------------------------------- #
# Dense reference likelihoods (numpy; for tests)
# --------------------------------------------------------------------------- #
def gls_diffuse(z, C, X):
    """−½[zᵀC⁻¹z − bᵀA⁻¹b + logdet(2πC) + logdet A], A = XᵀC⁻¹X, b = XᵀC⁻¹z."""
    cf = cho_factor(C, lower=True)
    Lc = cf[0]
    zt = solve_triangular(Lc, z, lower=True)
    Xt = solve_triangular(Lc, X, lower=True)
    Qx, Rx = np.linalg.qr(Xt)
    proj = zt - Qx @ (Qx.T @ zt)
    return -0.5 * (
        proj @ proj
        + z.size * LN2PI
        + 2.0 * np.sum(np.log(np.diag(Lc)))
        + 2.0 * np.sum(np.log(np.abs(np.diag(Rx))))
    )


def compressed_dense_loglike(comp, K_cells):
    """The compressed likelihood for a state with covariance K_cells over the cells.

    K_cells is the covariance of the cell positions s_w (n_cells²), or, for a two-scalar
    compression, of (s_w, ṡ_w) interleaved per cell ((2 n_cells)²). Same convention as the
    filter: the scalars ~ N(Xβ, K + diag σ²), plus the remainder's (A₀, b₀, c₀), β
    marginalised under the flat prior. Add comp["const"] to compare with `gls_diffuse` on
    the raw TOAs.
    """
    nc = comp["s_hat"].size
    if K_cells.shape[0] == 2 * nc:
        # K_cells is about the cell stamps `times`; ŝ measures s(t_w) = s + ṡ (t_w − t).
        J = np.eye(2 * nc)
        J[0::2, 1::2] = np.diag(comp["t_weighted"] - comp["times"])
        K_cells = J @ K_cells @ J.T
        has = np.column_stack([comp["has_scalar"], comp["has_velocity"]]).ravel()
        sig = np.column_stack([comp["sigma2"], comp["sigma2_v"]]).ravel()
        z = np.column_stack([comp["s_hat"], comp["v_hat"]]).ravel()
        X = np.stack([comp["m"], comp["m_v"]], axis=1).reshape(2 * nc, -1)
    else:
        assert not comp["has_velocity"].any()
        has, sig, z, X = comp["has_scalar"], comp["sigma2"], comp["s_hat"], comp["m"]
    C = K_cells[np.ix_(has, has)] + np.diag(sig[has])
    z, X = z[has], X[has]
    A0, b0, c0 = comp["accum0"]
    cf = cho_factor(C, lower=True)
    Lc = cf[0]
    zt = solve_triangular(Lc, z, lower=True)
    Xt = solve_triangular(Lc, X, lower=True)
    A = Xt.T @ Xt + A0
    b = Xt.T @ zt + b0[:, 0]
    c = zt @ zt + c0
    sign, logdetA = np.linalg.slogdet(A)
    assert sign > 0
    return -0.5 * (
        c
        - b @ np.linalg.solve(A, b)
        + z.size * LN2PI
        + 2.0 * np.sum(np.log(np.diag(Lc)))
        + logdetA
    )

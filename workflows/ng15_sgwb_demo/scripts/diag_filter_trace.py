"""Per-epoch trace of the marginalized Kalman filter at the NUTS-frozen positions.

diag_warmup_collapse.py showed that the chains that freeze in warmup sit where the
potential energy is +inf (logL = -inf) with a finite gradient. The marginalized filter can
return -inf in only one way: an epoch whose symmetrised, jittered innovation covariance S
has a non-positive slogdet sign, so `_update_marginal` sets dL = +inf. This script steps the
filter epoch by epoch, replaying `_run_kalman_filter_marginal` exactly with the library's own
predict helpers and a verbatim copy of `_update_marginal`'s arithmetic, and logs per epoch:

  * S: min/max eigenvalue before jitter, the condition number, the jitter, jitter / min
    observed diag(S), the slogdet sign and dL;
  * the updated covariance P: min eigenvalue and asymmetry ||P - P'|| / ||P||, since a
    Joseph update without re-symmetrisation can drift P off the PD cone and S inherits that.

It then names the first epoch where the sign goes non-positive and checks that the replayed
total equals KF.get_likelihood at the same point (replay fidelity).

Read-only. Usage (CPU; run as a SLURM job):
    JAX_PLATFORMS=cpu python scripts/diag_filter_trace.py CONFIG CHECKPOINT CHAIN [CHAIN ...]
"""

import argparse
import os
import pickle
import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diag_warmup_collapse import build  # noqa: E402  (also puts ../../python on sys.path)

from argus import jax_kalman_filter as jkf  # noqa: E402


def capture_theta(KF, pe_fn, z):
    """Run the model's potential at z once and grab the Parameters it hands the filter."""
    seen = {}
    orig = KF.get_likelihood

    def spy(theta):
        seen["theta"] = theta
        return orig(theta)

    KF.get_likelihood = spy
    try:
        pe = float(pe_fn(z))
    finally:
        KF.get_likelihood = orig
    return seen["theta"], pe


@jax.jit
def update_traced(xp, Pp, Xi_pred, H_dyn, H_eps, R, z, mask):
    """`_update_marginal` verbatim, plus the diagnostics it does not expose."""
    H_dyn = mask[:, None] * H_dyn
    H_eps = mask[:, None] * H_eps
    R = (mask[:, None] * mask[None, :]) * R + jnp.diag(1.0 - mask)
    y0 = mask[:, None] * (z[:, None] - H_dyn @ xp)
    S_raw = H_dyn @ Pp @ H_dyn.T + R
    Psi = H_eps + H_dyn @ Xi_pred

    n = S_raw.shape[0]
    S_sym = 0.5 * (S_raw + S_raw.T)
    jit_ = jkf._jitter_scale(S_sym, mask)
    S = S_sym + jit_ * jnp.eye(n)
    sign, logdet = jnp.linalg.slogdet(2.0 * jnp.pi * S)
    Sinv = jnp.linalg.solve(S, jnp.eye(n))

    K = Pp @ H_dyn.T @ Sinv
    x = xp + K @ y0
    I_KH = jnp.eye(len(xp)) - K @ H_dyn
    P = I_KH @ Pp @ I_KH.T + K @ R @ K.T
    Xi = I_KH @ Xi_pred - K @ H_eps

    Sinv_y0 = Sinv @ y0
    dA = Psi.T @ (Sinv @ Psi)
    db = Psi.T @ Sinv_y0
    dc = (y0.T @ Sinv_y0)[0, 0]
    dL = jnp.where(sign > 0, logdet, jnp.inf)

    # Diagnostics. Eigenvalues of the observed block only (absent slots are unit variance).
    ev_S = jnp.linalg.eigvalsh(S_sym)
    obs_diag = jnp.where(mask > 0, jnp.diag(S_sym), jnp.inf)
    ev_Pp = jnp.linalg.eigvalsh(0.5 * (Pp + Pp.T))
    ev_P = jnp.linalg.eigvalsh(0.5 * (P + P.T))
    diag = dict(
        sign=sign, logdet=logdet, dL=dL, dc=dc, jitter=jit_,
        S_min_ev=ev_S[0], S_max_ev=ev_S[-1], S_min_obs_diag=jnp.min(obs_diag),
        S_asym=jnp.linalg.norm(S_raw - S_raw.T) / jnp.linalg.norm(S_raw),
        Pp_min_ev=ev_Pp[0], Pp_max_ev=ev_Pp[-1],
        P_min_ev=ev_P[0], P_asym=jnp.linalg.norm(P - P.T) / jnp.linalg.norm(P),
        n_obs=jnp.sum(mask),
    )
    return x, P, Xi, dA, db, dc, dL, diag


def replay(KF, θ, symmetrise=False):
    """Step `_run_kalman_filter_marginal` (informative or diffuse) one epoch at a time."""
    Npsr, M_sum, dim_x = KF.Npsr, KF.M_sum, 2 * KF.Npsr
    Γ = jkf._effective_orf(KF.hellings_downs_matrix, θ.orf_epsilon)
    σa2 = jkf._compute_sigma_matrix(θ.ha**2, θ.γa, Γ)
    x, P = jkf._initialize_dynamic_kalman_filter(Npsr, σa2, θ.γa, θ.σp**2, θ.γp)
    Rs = jkf.precompute_R_matrices(KF.jax_data_errors, θ.EFAC, θ.EQUAD)
    T = KF.jax_data.shape[0]
    (Fg, Fs), (Qg, Qs) = jkf._precompute_transition_matrices(
        θ.γa, θ.γp, σa2, θ.σp**2, KF.jax_t_diffs[jnp.arange(T - 1)], Npsr, M_sum)
    n_dyn = 4 * Npsr
    H = KF.jax_H_matrices
    Xi = jnp.zeros((n_dyn, M_sum))
    A = jnp.zeros((M_sum, M_sum)); b = jnp.zeros((M_sum, 1)); c = 0.0; L = 0.0
    rows = []
    for t in range(T):
        if t > 0:
            F, Q = (Fg[t - 1], Fs[t - 1]), (Qg[t - 1], Qs[t - 1])
            xp = jkf.compute_predicted_state(F, x, dim_x, dim_x)
            Pp = jkf._predict_dynamic_cov(P, F, Q, dim_x)
            Xi = jkf._predict_xi(Xi, Fg[t - 1], Fs[t - 1], dim_x)
        else:
            xp, Pp = x, P
        x, P, Xi, dA, db, dc, dL, d = update_traced(
            xp, Pp, Xi, H[t, :, :n_dyn], H[t, :, n_dyn:], Rs[t], KF.jax_data[t],
            KF.jax_mask_matrices[t])
        if symmetrise:  # HYPOTHESIS TEST ONLY: not what the library does
            P = 0.5 * (P + P.T)
        A, b, c, L = A + dA, b + db, c + dc, L + dL
        rows.append({k: float(v) for k, v in d.items()})
        rows[-1]["_Pp"] = Pp

    if KF.timing_prior == "diffuse":
        n = A.shape[0]
        Lam = 0.5 * (A + A.T); Lam = Lam + 1e-9 * (jnp.trace(Lam) / n) * jnp.eye(n)
        sgn, ldL = jnp.linalg.slogdet(Lam)
        ll = -0.5 * (c + L - (b.T @ jnp.linalg.solve(Lam, b))[0, 0] + ldL)
        ll = jnp.where(sgn > 0, ll, -jnp.inf); extra = f"Λ sign {float(sgn):+.0f}"
    else:
        Lam = KF.P_eps_inv + A
        sgn, ldL = jnp.linalg.slogdet(Lam)
        _, ldP = jnp.linalg.slogdet(KF.P_eps_inv)
        ll = -0.5 * (c + L - (b.T @ jnp.linalg.solve(Lam, b))[0, 0] + ldL - ldP)
        extra = f"Λ sign {float(sgn):+.0f} (unguarded on this path)"
    return rows, float(ll), extra


def attribute(Pp, names, label):
    """Which states carry Pp's most negative eigenvector, and how big their variances are.
    Layout: GW block [r_0, a_0, r_1, a_1, ...] then spin block [phi_0, f_0, phi_1, f_1, ...]."""
    Pp = np.asarray(0.5 * (Pp + Pp.T))
    n = Pp.shape[0] // 4
    lab = [f"GW {'ra'[i % 2]} {names[i // 2]}" for i in range(2 * n)]
    lab += [f"spin {('phi', 'f')[i % 2]} {names[i // 2]}" for i in range(2 * n)]
    w, V = np.linalg.eigh(Pp)
    v = V[:, 0]
    print(f"   {label}: Pp eig min {w[0]:.3e}, 2nd {w[1]:.3e}, max {w[-1]:.3e}; "
          f"top states of the negative eigenvector (weight, diag var):")
    for i in np.argsort(-np.abs(v))[:6]:
        print(f"      {lab[i]:<26s} {v[i]: .3f}  var {Pp[i, i]:.3e}")
    # scale-free view: correlation-matrix eigenvalues (removes the units spread)
    d = np.sqrt(np.clip(np.diag(Pp), 1e-300, None))
    wc = np.linalg.eigvalsh(Pp / np.outer(d, d))
    print(f"      diag var range [{np.diag(Pp).min():.2e}, {np.diag(Pp).max():.2e}]  "
          f"corr-matrix eig min {wc[0]:.3e}")


def report(rows, ll_replay, ll_lib, extra, names):
    bad = [t for t, r in enumerate(rows) if not r["sign"] > 0 or not np.isfinite(r["dL"])]
    print(f"   replay logL = {ll_replay:.6f}   library logL = {ll_lib:.6f}   {extra}")
    print(f"   epochs with sign<=0 or non-finite dL: {len(bad)} / {len(rows)}"
          + (f"   FIRST = {bad[0]}   all = {bad[:40]}" if bad else ""))
    minPP = min(range(len(rows)), key=lambda t: rows[t]["Pp_min_ev"])
    print(f"   worst Pp min-eig at epoch {minPP}: {rows[minPP]['Pp_min_ev']:.3e} "
          f"(max {rows[minPP]['Pp_max_ev']:.3e})")
    if bad:
        attribute(rows[bad[0]]["_Pp"], names, f"first bad epoch {bad[0]}")
    print("   time series  ep: Pp_min/Pp_max   P_asym")
    print("     " + "  ".join(f"{t}:{rows[t]['Pp_min_ev'] / rows[t]['Pp_max_ev']:.0e}/{rows[t]['P_asym']:.0e}"
                              for t in range(0, len(rows), 15)))
    hdr = ("   ep  nobs sign     dL            S_min_ev    S_max_ev   cond(S)   "
           "jitter   jit/minDiag  S_asym    Pp_min_ev   P_min_ev    P_asym")
    print(hdr)
    show = sorted(set(list(range(3)) + bad[:8] + [b - 1 for b in bad[:3] if b > 0]
                      + [minPP] + list(range(len(rows) - 2, len(rows)))))
    for t in show:
        r = rows[t]
        cond = r["S_max_ev"] / r["S_min_ev"] if r["S_min_ev"] > 0 else np.inf
        flag = " <==" if t in bad else ""
        print(f"   {t:3d} {int(r['n_obs']):4d} {r['sign']:+3.0f} {r['dL']:13.4e} "
              f"{r['S_min_ev']: .3e} {r['S_max_ev']: .3e} {cond:9.2e} {r['jitter']:.2e} "
              f"{r['jitter'] / r['S_min_obs_diag']:10.2e} {r['S_asym']:.1e} "
              f"{r['Pp_min_ev']: .3e} {r['P_min_ev']: .3e} {r['P_asym']:.1e}{flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("checkpoint")
    ap.add_argument("chains", type=int, nargs="+")
    ap.add_argument("--symmetrise", action="store_true",
                    help="hypothesis test: symmetrise P after every update in the replay")
    args = ap.parse_args()

    print(f"===== {args.config} =====  backend: {jax.default_backend()}")
    kernel, seed, n_chains, n_warm, names, pe_fn, KF = build(args.config)
    print(f"   filter: use_marginal={KF.use_marginal} timing_prior={KF.timing_prior} "
          f"Npsr={KF.Npsr} M_sum={KF.M_sum} epochs={KF.jax_data.shape[0]}")
    s = pickle.load(open(args.checkpoint, "rb"))["sampler_state"]
    for c in args.chains:
        z = jax.tree.map(lambda a: jnp.asarray(a)[c], s.z)
        θ, pe = capture_theta(KF, pe_fn, z)
        ll_lib = float(KF.get_likelihood(θ))
        step = float(np.asarray(s.adapt_state.step_size)[c])
        print(f"\n-- chain {c}: step {step:.2e}  PE {pe:.3f}  "
              f"log10 ha {float(jnp.log10(θ.ha)):.3f} log10 γa {float(jnp.log10(θ.γa)):.3f}")
        lsp, lgp = np.log10(np.asarray(θ.σp)), np.log10(np.asarray(θ.γp))
        for i in np.argsort(lgp)[:3]:
            print(f"   lowest γp: {names[i]:<12s} log10 σp {lsp[i]:.2f} log10 γp {lgp[i]:.2f}")
        rows, ll_rep, extra = replay(KF, θ, args.symmetrise)
        if args.symmetrise:
            extra += "  [REPLAY SYMMETRISES P]"
        report(rows, ll_rep, ll_lib, extra, names)


if __name__ == "__main__":
    main()

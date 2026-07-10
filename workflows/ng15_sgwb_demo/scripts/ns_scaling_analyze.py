#!/usr/bin/env python
"""NS cost-scaling study — synthesis, extrapolation, and verdict.

Aggregates every stage of the study into one tidy table, fits the two cost axes, builds a
combined cost model T(N, D) ~ n_steps(D) * t_per_step(N, D), extrapolates to full-PTA scale
(NG15: N~67, D~274), and prints a GO / CONDITIONAL / KILL verdict.

Inputs (whatever exists — the script degrades gracefully):
  * Stage 1a  — outputs/scaling/dimension_scaling.csv (pure-dimension, likelihood-free):
                gives n_steps(D), wall(D), and the logZ-accuracy-vs-D trend.
  * Stages 1b/2/3 — outputs/<output_id>/<output_id>_evidence.json, each now carrying the
                Stage-0 metadata (runtime_s, n_steps, ndim, n_live, num_delete, n_pulsars,
                log_Z_mean, log_Z_uncert). Stage 1b (D=2, varying N) gives t_per_step(N);
                Stage 2 (coupled) validates the combined model; Stage 3 gives knob constants.

Outputs: outputs/scaling/scaling_combined.csv, PNG plots, and a printed verdict block.

Run:
    python workflows/ng15_sgwb_demo/scripts/ns_scaling_analyze.py
"""
import argparse
import csv
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORKFLOW = os.path.dirname(HERE)
OUTPUTS = os.path.join(WORKFLOW, "outputs")
SCALING = os.path.join(OUTPUTS, "scaling")

# Full-PTA extrapolation target (NG15-scale, all noise free): D = 4N + 6.
NG15_N = 67
NG15_D = 4 * NG15_N + 6
# Accuracy-preserving inner-step policy (matches gen_scaling_configs.INNER_MULT).
INNER_MULT = 6
# Tractability bar agreed with the user: a full-PTA evidence run should finish within about
# a week on one A100 (a day or two is ideal for dev iteration).
WEEK_S = 7 * 24 * 3600.0
DAY_S = 24 * 3600.0


def _fit_powerlaw(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = (x > 0) & (y > 0)
    if ok.sum() < 2:
        return float("nan"), float("nan")
    b, loga = np.polyfit(np.log(x[ok]), np.log(y[ok]), 1)
    return float(np.exp(loga)), float(b)


def load_stage1a(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("d", "n_steps", "num_inner_steps", "num_live", "num_delete"):
            if k in r and r[k] != "":
                r[k] = int(float(r[k]))
        for k in ("logZ_true", "logZ_est", "logZ_err", "abs_err", "wall_s"):
            if k in r and r[k] != "":
                r[k] = float(r[k])
    return rows


def load_evidence_jsons(outputs):
    """Load every *_evidence.json under outputs/, keeping those with scaling metadata."""
    rows = []
    for path in glob.glob(os.path.join(outputs, "**", "*_evidence.json"), recursive=True):
        try:
            with open(path) as f:
                d = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if "n_steps" not in d or "ndim" not in d:
            continue  # pre-instrumentation evidence.json (no timing) — skip
        d["_path"] = path
        d["output_id"] = os.path.basename(path).replace("_evidence.json", "")
        rows.append(d)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage1a", default=os.path.join(SCALING, "dimension_scaling.csv"))
    ap.add_argument("--outputs", default=OUTPUTS)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()
    os.makedirs(SCALING, exist_ok=True)

    s1a = load_stage1a(args.stage1a)
    ev = load_evidence_jsons(args.outputs)

    print("=" * 88)
    print("NS cost-scaling synthesis")
    print("=" * 88)

    # --- Axis 1: pure-dimension scaling (Stage 1a) ---------------------------------------
    n_steps_of_D = (float("nan"), float("nan"))
    if s1a:
        dims = [r["d"] for r in s1a]
        a_s, b_s = _fit_powerlaw(dims, [r["n_steps"] for r in s1a])
        n_steps_of_D = (a_s, b_s)
        print(f"\n[Stage 1a] pure-dimension (likelihood-free), {len(s1a)} points d={dims}")
        print(f"  n_steps(D) ~ {a_s:.3g} * D^{b_s:.2f}")
        # Accuracy-vs-D: flag the smallest D where |err| exceeds 3*logZ_err.
        biased = [r["d"] for r in s1a
                  if r["abs_err"] > max(3 * r["logZ_err"], 0.5)]
        if biased:
            print(f"  logZ ACCURACY breaks (|err|>3σ) from D={min(biased)} upward "
                  f"at num_live={s1a[0]['num_live']}, num_delete={s1a[0]['num_delete']} "
                  f"-> evidence bias grows with dimension (see Stage 3 for the tuning fix).")
        else:
            print("  logZ accuracy holds across all tested D (no bias detected).")

    # --- Axis 2: per (NS-step * inner-step) cost vs N (Stage 1b, D=2) ---------------------
    # Isolate the Kalman-eval cost c(N) = wall / (n_steps * num_inner_steps). Dividing out
    # BOTH n_steps and num_inner_steps leaves the cost of one inner slice iteration, which is
    # what actually carries the O(N) likelihood work — so c(N) is directly reusable at any D.
    s1b = sorted([r for r in ev if r.get("ndim") == 2], key=lambda r: r["n_pulsars"])
    c_of_N = (float("nan"), float("nan"))
    if s1b:
        Ns = [r["n_pulsars"] for r in s1b]
        cs = [r["runtime_s"] / max(1, r["n_steps"] * r.get("num_inner_steps", 1))
              for r in s1b]
        a_c, b_c = _fit_powerlaw(Ns, cs)
        c_of_N = (a_c, b_c)
        print(f"\n[Stage 1b] Kalman cost at D=2, N={Ns}")
        for r, c in zip(s1b, cs):
            print(f"  N={r['n_pulsars']:>3}  {r['n_steps']:>6} steps x "
                  f"{r.get('num_inner_steps','?')} inner  {r['runtime_s']:>8.1f}s  "
                  f"{c*1e6:>8.2f} us/(step·inner)")
        print(f"  c(N) ~ {a_c:.3g} * N^{b_c:.2f} s   (per NS-step·inner-step)")

    # --- Coupled points (Stage 2) --------------------------------------------------------
    s2 = sorted([r for r in ev if r.get("ndim", 0) > 2], key=lambda r: r["ndim"])
    if s2:
        print(f"\n[Stage 2] coupled real-model points:")
        for r in s2:
            print(f"  N={r['n_pulsars']:>3} D={r['ndim']:>3}  {r['n_steps']:>6} steps  "
                  f"{r['runtime_s']:>9.1f}s  logZ={r['log_Z_mean']:.2f}"
                  f"+/-{r['log_Z_uncert']:.2f}")

    # --- Combined cost model + extrapolation ---------------------------------------------
    # T(N,D) = n_steps(D) * num_inner_steps(D) * c(N), with num_inner_steps = INNER_MULT*D
    # (the accuracy-preserving policy from Stage 1a). All three factors measured separately.
    a_s, b_s = n_steps_of_D
    a_c, b_c = c_of_N
    print("\n" + "=" * 88)
    if not (np.isnan(a_s) or np.isnan(a_c)):
        def T(N, D):
            n_steps = a_s * D ** b_s
            nis = max(5, INNER_MULT * D)
            c = a_c * N ** b_c
            return n_steps * nis * c
        for (N, D, tag) in [(6, 30, "T3.4 few-pulsar"),
                            (16, 70, "mid"),
                            (NG15_N, NG15_D, "NG15 full-PTA")]:
            t = T(N, D)
            print(f"  extrapolated T(N={N}, D={D}) [{tag}] "
                  f"= {t:.3g}s = {t/3600:.2f} h = {t/DAY_S:.2f} d")
        t_full = T(NG15_N, NG15_D)
        if t_full <= 2 * DAY_S:
            verdict = "GO (full-PTA evidence run projected under ~2 days/A100)"
        elif t_full <= WEEK_S:
            verdict = "GO/CONDITIONAL (projected within the ~1-week/A100 bar)"
        else:
            verdict = ("CONDITIONAL/KILL (projected beyond ~1 week/A100 — restrict "
                       "dimension/pulsars or use NUTS+stepping-stone for evidence)")
        print(f"\n  RUNTIME VERDICT: {verdict}")
    else:
        print("  (need both Stage 1a and Stage 1b results to build the cost model)")
    print("  NOTE: accuracy is a separate gate from runtime — a fast run with biased logZ")
    print("        is useless for a Bayes factor. See the Stage 1a accuracy line above.")
    print("=" * 88)

    # --- Persist combined table ----------------------------------------------------------
    combined = os.path.join(SCALING, "scaling_combined.csv")
    fields = ["source", "output_id", "N", "D", "n_steps", "runtime_s",
              "num_live", "num_delete", "log_Z_mean", "log_Z_uncert"]
    with open(combined, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in s1a:
            w.writerow({"source": "1a_analytic", "output_id": f"gauss_d{r['d']}",
                        "N": "", "D": r["d"], "n_steps": r["n_steps"],
                        "runtime_s": r["wall_s"], "num_live": r["num_live"],
                        "num_delete": r["num_delete"], "log_Z_mean": r["logZ_est"],
                        "log_Z_uncert": r["logZ_err"]})
        for r in ev:
            w.writerow({"source": "gpu_run", "output_id": r["output_id"],
                        "N": r.get("n_pulsars", ""), "D": r.get("ndim", ""),
                        "n_steps": r.get("n_steps", ""), "runtime_s": r.get("runtime_s", ""),
                        "num_live": r.get("n_live", ""), "num_delete": r.get("num_delete", ""),
                        "log_Z_mean": r.get("log_Z_mean", ""),
                        "log_Z_uncert": r.get("log_Z_uncert", "")})
    print(f"combined table -> {combined}")

    if not args.no_plots:
        _make_plots(s1a, s1b, s2)


def _make_plots(s1a, s1b, s2):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; skipping plots")
        return

    if s1a:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        d = [r["d"] for r in s1a]
        axes[0].loglog(d, [r["n_steps"] for r in s1a], "o-")
        axes[0].set(xlabel="dimension D", ylabel="NS steps", title="Stage 1a: n_steps(D)")
        axes[1].loglog(d, [r["wall_s"] for r in s1a], "o-", color="C1")
        axes[1].set(xlabel="dimension D", ylabel="wall (s)", title="Stage 1a: wall(D)")
        axes[2].semilogx(d, [r["abs_err"] for r in s1a], "o-", color="C3", label="|logZ err|")
        axes[2].semilogx(d, [3 * r["logZ_err"] for r in s1a], "s--", color="k", label="3σ")
        axes[2].set(xlabel="dimension D", ylabel="logZ error",
                    title="Stage 1a: evidence accuracy")
        axes[2].legend()
        fig.tight_layout()
        p = os.path.join(SCALING, "stage1a_dimension.png")
        fig.savefig(p, dpi=110)
        print(f"plot -> {p}")

    if s1b:
        fig, ax = plt.subplots(figsize=(6, 4))
        N = [r["n_pulsars"] for r in s1b]
        ax.loglog(N, [r["runtime_s"] / max(1, r["n_steps"]) * 1e3 for r in s1b], "o-")
        ax.set(xlabel="N pulsars", ylabel="ms / NS step (D=2)",
               title="Stage 1b: Kalman likelihood cost vs N")
        fig.tight_layout()
        p = os.path.join(SCALING, "stage1b_pulsars.png")
        fig.savefig(p, dpi=110)
        print(f"plot -> {p}")


if __name__ == "__main__":
    main()

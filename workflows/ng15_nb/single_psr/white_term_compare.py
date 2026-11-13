#!/usr/bin/env python
"""Step 1 of spec §12 2026-11-13: does a free white term carry the >30/T excess? (task 5.2a)

    python workflows/ng15_nb/single_psr/white_term_compare.py \
        [--base discovery_gp-4_sp-10_m200] [--test discovery_gp-4_sp-10_m200_wq]

Compares, pulsar by pulsar, the Discovery single-pulsar runs at 200 modes and the v2 box
(γp ≤ −4, σp ≤ −10) without (`--base`) and with (`--test`, `run_discovery.py --free-white`)
a sampled flat-PSD term log10_wpsd, from their result JSONs. Writes
results/single_psr/white_term_check.{json,md}.

A pulsar passes when both hold:
- γp closes off the −4 ceiling: top-2 % mass ≤ 0.02 and the median below the baseline's.
  If the red noise is no longer detected with the term on (σp at its floor, the
  `red_noise_detected` flag of `common.verdict`), γp reverts to its uniform prior and
  its top-2 % mass is ≈ 0.02 by construction, so that case counts as closed: the term
  has taken the whole excess;
- the flat term is on and inside its box: < 1 % of log10_wpsd draws in the bottom 2 % of
  its box (the edge rule `common.verdict` uses to call red noise detected) and ≤ 5 % in
  the top 2 % (`common.EDGE_MASS_RAIL`), so the box does not decide the answer.
Step 1 passes (the excess is white, so lever B's term has the right shape) when J1713+0747
passes and so do most of the pulsars. It cannot test the owner indifference between a
pulsar and the common process; only an array run sees that.

Also reported: the shift of the median PSD at 1/T and 1/(5 yr), which should stay small if
the term only takes the high-frequency excess, and r̂/ESS/divergences of the new runs.
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402

KEY = "J1713+0747"
WQ = "log10_wpsd"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default="discovery_gp-4_sp-10_m200")
    ap.add_argument("--test", default="discovery_gp-4_sp-10_m200_wq")
    ap.add_argument("--ceiling-mass", type=float, default=0.02)
    ap.add_argument("--wq-floor-mass", type=float, default=0.01)
    args = ap.parse_args()

    base_dir, test_dir = common.RESULTS / args.base, common.RESULTS / args.test
    rows, missing = [], []
    for b in sorted(base_dir.glob("*.json")):
        t = test_dir / b.name
        if not t.exists():
            missing.append(b.stem)
            continue
        B, T = json.loads(b.read_text()), json.loads(t.read_text())
        gp_b, gp_t = B["site_quantiles"]["log10_gamma_p"], T["site_quantiles"]["log10_gamma_p"]
        ceil_b = B["edge_mass"]["log10_gamma_p"]["ceiling"]
        ceil_t = T["edge_mass"]["log10_gamma_p"]["ceiling"]
        wq_floor = T["edge_mass"][WQ]["floor"]
        rn_on = bool(T["red_noise_detected"])
        closes = (not rn_on) or (ceil_t <= args.ceiling_mass and gp_t["median"] < gp_b["median"])
        wq_ceil = T["edge_mass"][WQ]["ceiling"]
        wq_on = wq_floor < args.wq_floor_mass and wq_ceil <= common.EDGE_MASS_RAIL
        rows.append({
            "pulsar": b.stem,
            "gamma_p_median": [gp_b["median"], gp_t["median"]],
            "gamma_p_ceiling_mass": [ceil_b, ceil_t],
            "sigma_p_median": [B["site_quantiles"]["log10_sigma_p"]["median"],
                               T["site_quantiles"]["log10_sigma_p"]["median"]],
            "wpsd_q16_50_84": [T["site_quantiles"][WQ][k] for k in ("q16", "median", "q84")],
            "wq_floor_mass": wq_floor,
            "wq_ceiling_mass": T["edge_mass"][WQ]["ceiling"],
            "red_noise_detected": [bool(B["red_noise_detected"]), rn_on],
            "psd_shift_dex": {f: T["log10_psd"][f]["median"] - B["log10_psd"][f]["median"]
                              for f in B["log10_psd"]},
            "max_r_hat": T["diagnostics"]["max_r_hat"],
            "min_ess_bulk": T["diagnostics"]["min_ess_bulk"],
            "divergences": T["diagnostics"]["divergences"],
            "gamma_p_closes": bool(closes),
            "white_term_on": bool(wq_on),
            "pass": bool(closes and wq_on),
        })
    n_pass = sum(r["pass"] for r in rows)
    key = next((r for r in rows if r["pulsar"] == KEY), None)
    verdict = ("INCOMPLETE" if missing or key is None else
               "PASS" if key["pass"] and n_pass > len(rows) / 2 else "FAIL")
    report = {"base": args.base, "test": args.test, "sha": common.git_sha(),
              "rule": {"gamma_p_ceiling_mass_max": args.ceiling_mass,
                       "wq_floor_mass_max": args.wq_floor_mass,
                       "step": f"{KEY} passes and a majority of pulsars pass"},
              "n_pulsars": len(rows), "n_pass": n_pass, "missing": missing,
              "verdict": verdict, "pulsars": rows}
    out = common.RESULTS / "white_term_check.json"
    out.write_text(json.dumps(report, indent=2) + "\n")

    lines = [
        "# Step 1: does a free white term carry the >30/T excess? (spec §12, 2026-11-13)",
        "",
        f"Base `{args.base}`, test `{args.test}` (Discovery, 200 modes, γp ≤ −4, σp ≤ −10;"
        f" test adds log10_wpsd on {T['box'][WQ] if rows else '?'}, one-sided s²/Hz). Commit `{report['sha']}`.",
        "",
        f"**Verdict: {verdict}** — {n_pass}/{len(rows)} pulsars pass"
        + (f"; missing {', '.join(missing)}" if missing else "") + ".",
        "",
        "| pulsar | γp median base→wq | γp ceiling mass base→wq | log10_wpsd median [q16, q84] |"
        " wq floor / ceiling mass | ΔPSD 1/T | ΔPSD 1/(5 yr) | r̂ | pass |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        q = r["wpsd_q16_50_84"]
        s = r["psd_shift_dex"]
        lines.append(
            f"| {r['pulsar']} | {r['gamma_p_median'][0]:.2f} → {r['gamma_p_median'][1]:.2f}"
            f" | {r['gamma_p_ceiling_mass'][0]:.3f} → {r['gamma_p_ceiling_mass'][1]:.3f}"
            f" | {q[1]:.2f} [{q[0]:.2f}, {q[2]:.2f}] | {r['wq_floor_mass']:.3f} / {r['wq_ceiling_mass']:.3f}"
            f" | {s['1/T']:+.2f} | {s['1/(5 yr)']:+.2f} | {r['max_r_hat']:.3f}"
            f" | {'yes' if r['pass'] else 'no'} |")
    (common.RESULTS / "white_term_check.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

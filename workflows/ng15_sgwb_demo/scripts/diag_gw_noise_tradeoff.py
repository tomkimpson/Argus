#!/usr/bin/env python
"""Does per-pulsar red noise stand in for the common process? (post-processing only)

Splits a ridge run's draws by the GW pivot log-PSD s (low: s < -10, high: s > -8) and
compares the per-pulsar red-noise posteriors between the two. MDC2 1b has NO injected
red noise, so any pulsar whose sigma_p rises when the GW is switched down is absorbing
the common signal. Also counts low<->high transitions per chain: if the chains hop
between the modes freely, the bimodal pivot posterior is real and not a mixing artefact.

    python scripts/diag_gw_noise_tradeoff.py --run outputs/mdc2_d1_flat_uprior_eps000 \
        --out outputs/diag_gw_noise_tradeoff_mdc2_d1_flat_uprior_eps000.json
"""

import argparse
import glob
import json
import os

import numpy as np

LOW, HIGH = -10.0, -8.0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    import arviz as az

    nc = sorted(glob.glob(os.path.join(args.run, "*_results.nc")))
    if len(nc) != 1:
        raise SystemExit(f"{args.run}: expected one *_results.nc, found {nc}")
    post = az.from_netcdf(nc[0]).posterior
    s = post["log10_pivot_psd"].values
    sp, gp = post["log10_σp"].values, post["log10_γp"].values
    ga = post["log10_gamma_a"].values
    lo, hi = s < LOW, s > HIGH

    shift = np.median(sp[lo], axis=0) - np.median(sp[hi], axis=0)
    order = np.argsort(shift)[::-1]
    pulsars = [
        {
            "index": int(i),
            "sigma_p_low": float(np.median(sp[lo][:, i])),
            "sigma_p_high": float(np.median(sp[hi][:, i])),
            "gamma_p_low": float(np.median(gp[lo][:, i])),
            "gamma_p_high": float(np.median(gp[hi][:, i])),
            "sigma_p_shift": float(shift[i]),
        }
        for i in order
    ]
    chains = []
    for c in range(s.shape[0]):
        mode = np.where(s[c] < LOW, 0, np.where(s[c] > HIGH, 2, 1))
        chains.append({
            "frac_low": float((mode == 0).mean()),
            "frac_high": float((mode == 2).mean()),
            "low_high_transitions": int(np.sum(np.abs(np.diff(mode)) == 2)),
        })
    res = {
        "run": os.path.basename(args.run.rstrip("/")),
        "n_low": int(lo.sum()),
        "n_high": int(hi.sum()),
        "gamma_a_median_low": float(np.median(ga[lo])),
        "gamma_a_median_high": float(np.median(ga[hi])),
        "chains": chains,
        "pulsars_by_sigma_p_shift": pulsars,
    }
    print(f"{res['run']}: {res['n_low']} low-s / {res['n_high']} high-s draws")
    for k, c in enumerate(chains):
        print(f"  chain {k}: low {c['frac_low']:.2f} high {c['frac_high']:.2f} "
              f"transitions {c['low_high_transitions']}")
    print("  psr  sigma_p low -> high   (shift)   gamma_p low / high")
    for r in pulsars[:8]:
        print(f"  {r['index']:3d}  {r['sigma_p_low']:6.2f} -> {r['sigma_p_high']:6.2f}  "
              f"({r['sigma_p_shift']:+.2f})   {r['gamma_p_low']:.2f} / {r['gamma_p_high']:.2f}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""NS cost-scaling study — config generator for the GPU stages (1b, 2, 3).

Clones the T2.6 validation config ``configs/mdc2_blackjax_ns.ini`` and stamps out one
derived ``.ini`` per grid point, varying the two cost axes and the sampler knobs:

  * N_pulsars   — via ``[Data] excluded_psrs`` (subset MDC2 dataset_2b down to N).
  * dimension D — via the noise-fix toggles in ``[PriorModel]``:
        dmode=fixed      red+white FIXED   -> D = 2                 (T2.6 floor)
        dmode=red_free   red FREE, white FIXED -> D = 2 + 4 + 2N    (hierarchical)
        dmode=white_free white FREE, red FIXED -> D = 2 + 2N
        dmode=all_free   both FREE         -> D = 4N + 6            (full model)
    Red noise is freed by blanking ``spin_injections_path``; white noise by blanking
    ``noise_params_path`` (see prior_models.py: non-empty path => that group is fixed).
  * NS knobs    — ``[NestedSampler] num_live_points / num_delete / num_inner_steps``.
    num_inner_steps is set to max(5, INNER_MULT*D). The Stage-1a diagnostic found the
    engine's default (2*D) is too few for *accurate evidence* at high D (logZ bias grows
    with dimension); 6*D recovers accuracy and, per-eval, is cheaper than compensating with
    more live points. Stage 3 refines this frontier.

Dimension arithmetic mirrors ``parameter_sampling.count_free_parameters``:
    global   = 2 GW (log10_ha, log10_gamma_a) + 4 red-noise hyperparameters (when red free)
    per-psr  = gamma_p + sigma_p (red) + efac + equad (white)

Writes derived configs into ``outputs/derived_configs/`` and prints one config path per
line (consumed by slurm_scripts/ns_scaling_run.sh). Pure config generation — no compute.

Usage:
    python gen_scaling_configs.py --stage 1b        # D=2, N in {2,4,8,16,32}
    python gen_scaling_configs.py --stage 2         # red_free, N in {6,16,32}
    python gen_scaling_configs.py --stage 3         # red_free N=16, knob sweep
"""

import argparse
import configparser
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
WORKFLOW = os.path.dirname(HERE)
TEMPLATE = os.path.join(WORKFLOW, "configs", "mdc2_blackjax_ns.ini")
DERIVED_DIR = os.path.join(WORKFLOW, "outputs", "derived_configs")
OUTPUT_DIR = os.path.join(WORKFLOW, "outputs")

# Always drop this pulsar (the T2.6 validation excluded it); N is chosen from the rest.
ALWAYS_EXCLUDE = "J1640+2224"

# num_inner_steps = INNER_MULT * D (Stage-1a diagnostic: 2*D biases logZ, 6*D is accurate).
INNER_MULT = 6


def _load_template():
    cp = configparser.ConfigParser(interpolation=None)
    cp.optionxform = str  # preserve key case
    cp.read(TEMPLATE)
    return cp


def _pulsar_list(data_path):
    """Sorted pulsar names (from .par files) in the dataset directory."""
    names = sorted(f[:-4] for f in os.listdir(data_path) if f.endswith(".par"))
    return names


def dimension(n, dmode):
    if dmode == "fixed":
        return 2
    if dmode == "red_free":
        return 2 + 4 + 2 * n
    if dmode == "white_free":
        return 2 + 2 * n
    if dmode == "all_free":
        return 4 * n + 6
    raise ValueError(f"unknown dmode {dmode!r}")


def make_config(
    n, dmode, num_live, num_delete, seed, num_inner_steps=None, output_id=None
):
    """Return (output_id, ConfigParser) for one grid point."""
    cp = _load_template()
    data_path = cp.get("Data", "data_path")
    all_psrs = _pulsar_list(data_path)
    candidates = [p for p in all_psrs if p != ALWAYS_EXCLUDE]
    if n > len(candidates):
        raise ValueError(
            f"requested N={n} but only {len(candidates)} pulsars available"
        )
    keep = candidates[:n]
    excluded = [ALWAYS_EXCLUDE] + [p for p in candidates if p not in keep]
    # Comma-only (no space): utils.get_noise_parameters splits excluded_psrs on "," WITHOUT
    # stripping (unlike workflow.py's data loader), so a ", " separator leaves leading spaces
    # that fail the substring match -> the fixed noise arrays would keep all 32 pulsars while
    # the data is subset to N, giving a (32,) vs (N,) broadcast error. Comma-only sidesteps it.
    cp.set("Data", "excluded_psrs", ",".join(excluded))

    D = dimension(n, dmode)
    if num_inner_steps is None:
        num_inner_steps = max(5, INNER_MULT * D)

    # Noise-fix toggles (blank path => that group becomes free/sampled).
    if dmode in ("red_free", "all_free"):
        cp.set("PriorModel", "spin_injections_path", "")
    if dmode in ("white_free", "all_free"):
        cp.set("PriorModel", "noise_params_path", "")

    cp.set("NestedSampler", "num_live_points", str(num_live))
    cp.set("NestedSampler", "num_delete", str(num_delete))
    cp.set("NestedSampler", "num_inner_steps", str(num_inner_steps))
    cp.set("NestedSampler", "seed", str(seed))

    if output_id is None:
        output_id = (
            f"ns_scal_{dmode}_N{n:02d}_D{D:03d}" f"_nl{num_live}_nd{num_delete}_s{seed}"
        )
    cp.set("Output", "output_dir", OUTPUT_DIR + os.sep)
    cp.set("Output", "output_id", output_id)
    return output_id, cp, D, num_inner_steps


def _points_for_stage(stage):
    """Return list of dicts of make_config kwargs for a named stage."""
    if stage == "1b":  # likelihood-cost vs N, dimension held at D=2
        return [
            dict(n=n, dmode="fixed", num_live=500, num_delete=25, seed=42)
            for n in (2, 4, 8, 16, 32)
        ]
    if stage == "2":  # realistic coupled proxy: red noise free, climb N (and thus D)
        return [
            dict(n=n, dmode="red_free", num_live=500, num_delete=25, seed=42)
            for n in (6, 16, 32)
        ]
    if (
        stage == "3"
    ):  # sampler-knob frontier at one representative config (N=16, red_free)
        pts = []
        for num_delete in (25, 50, 100):
            for num_live in (250, 500, 1000):
                pts.append(
                    dict(
                        n=16,
                        dmode="red_free",
                        num_live=num_live,
                        num_delete=num_delete,
                        seed=42,
                    )
                )
        return pts
    raise ValueError(f"unknown stage {stage!r}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["1b", "2", "3"], required=True)
    p.add_argument("--out-dir", default=DERIVED_DIR)
    p.add_argument(
        "--manifest",
        default=None,
        help="CSV manifest path (default: <out-dir>/manifest_stage<stage>.csv)",
    )
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    manifest = args.manifest or os.path.join(
        args.out_dir, f"manifest_stage{args.stage}.csv"
    )

    rows = []
    paths = []
    for kw in _points_for_stage(args.stage):
        output_id, cp, D, nis = make_config(**kw)
        path = os.path.join(args.out_dir, f"{output_id}.ini")
        with open(path, "w") as f:
            cp.write(f)
        paths.append(path)
        rows.append(
            {
                "output_id": output_id,
                "config_path": path,
                "N": kw["n"],
                "dmode": kw["dmode"],
                "D": D,
                "num_live": kw["num_live"],
                "num_delete": kw["num_delete"],
                "num_inner_steps": nis,
                "seed": kw["seed"],
            }
        )

    with open(manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # stderr: human summary; stdout: config paths (one per line) for the SLURM runner.
    import sys

    print(
        f"# stage {args.stage}: {len(paths)} configs -> {args.out_dir}", file=sys.stderr
    )
    print(f"# manifest: {manifest}", file=sys.stderr)
    for r in rows:
        print(
            f"#   {r['output_id']}  (N={r['N']} D={r['D']} "
            f"nl={r['num_live']} nd={r['num_delete']} nis={r['num_inner_steps']})",
            file=sys.stderr,
        )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()

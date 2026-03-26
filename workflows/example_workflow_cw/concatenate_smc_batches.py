"""Concatenate multiple SMC batch runs into a single ArviZ InferenceData.

Usage:
    python concatenate_smc_batches.py outputs/cw_smc_heavy_seed42/no_gw outputs/cw_smc_heavy_seed43/no_gw ...

Or with a glob:
    python concatenate_smc_batches.py outputs/cw_smc_heavy_seed*/no_gw
"""

import argparse
import glob
import json
import os
import sys

import arviz as az
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Concatenate SMC batch results")
    parser.add_argument("dirs", nargs="+", help="Output directories from each batch run")
    parser.add_argument("--output", "-o", default="outputs/cw_smc_heavy_combined",
                        help="Output directory for combined results")
    args = parser.parse_args()

    # Find result files
    result_files = []
    evidence_files = []
    for d in sorted(args.dirs):
        nc_files = glob.glob(os.path.join(d, "*_results.nc"))
        ev_files = glob.glob(os.path.join(d, "*_evidence.json"))
        if nc_files:
            result_files.append(nc_files[0])
        if ev_files:
            evidence_files.append(ev_files[0])

    if not result_files:
        print("No result files found!")
        sys.exit(1)

    print(f"Found {len(result_files)} batch results:")
    for f in result_files:
        print(f"  {f}")

    # Load all InferenceData objects
    datasets = [az.from_netcdf(f) for f in result_files]

    # Concatenate posteriors along the chain dimension
    # Each batch is treated as a separate "chain"
    combined = az.concat(datasets, dim="chain")

    # Print summary
    for name, var in combined.posterior.data_vars.items():
        if var.ndim <= 2:
            print(f"  {name}: shape={var.shape}")
    total_samples = combined.posterior.dims["chain"] * combined.posterior.dims["draw"]
    print(f"\nTotal: {combined.posterior.dims['chain']} chains × "
          f"{combined.posterior.dims['draw']} draws = {total_samples} samples")

    # Save combined results
    os.makedirs(args.output, exist_ok=True)
    output_path = os.path.join(args.output, "cw_smc_heavy_combined_results.nc")
    combined.to_netcdf(output_path)
    print(f"\nSaved combined results to {output_path}")

    # Combine evidence estimates
    if evidence_files:
        log_z_values = []
        for f in evidence_files:
            with open(f) as fh:
                ev = json.load(fh)
                log_z_values.append(ev["log_Z_mean"])

        combined_evidence = {
            "log_Z_mean": float(np.mean(log_z_values)),
            "log_Z_std": float(np.std(log_z_values)),
            "log_Z_individual": log_z_values,
            "num_batches": len(log_z_values),
        }
        ev_path = os.path.join(args.output, "cw_smc_heavy_combined_evidence.json")
        with open(ev_path, "w") as fh:
            json.dump(combined_evidence, fh, indent=2)
        print(f"Combined evidence: log_Z = {combined_evidence['log_Z_mean']:.2f} "
              f"+/- {combined_evidence['log_Z_std']:.2f}")
        print(f"Saved to {ev_path}")


if __name__ == "__main__":
    main()

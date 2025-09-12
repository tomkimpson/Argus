# CLI Example Full Inference Workflow

This workflow demonstrates how to use the Argus CLI for **full parameter inference** Bayesian analysis with Slurm job management.

## Overview

This example shows the **modern way** to run complete Argus analysis using the installed package and CLI, performing inference on **all parameters** rather than fixing noise and pulsar parameters via injection files.

**Key Difference from `cli_example`:**
- **`cli_example`**: Samples only GW background parameters (`ha`, `gamma_a`) with noise/pulsar parameters fixed via injection files
- **`cli_example_full`**: Performs full parameter inference including pulsar red noise, EFAC/EQUAD, and all GW parameters

## Prerequisites

1. **Install Argus package:**
   ```bash
   # From PyPI (when published)
   pip install argus-pta
   
   # Or from source  
   cd /path/to/Argus
   pip install .
   ```

2. **Verify installation:**
   ```bash
   argus --version
   argus --help
   ```

## Workflow Steps

### 1. Create Configuration File
```bash
# Generate a template configuration file
argus init -o my_analysis.ini

# Edit the configuration with your specific parameters
# (data paths, inference settings, etc.)
# IMPORTANT: Comment out or remove noise_params_path and spin_injections_path
# for full parameter inference
```

### 2. Submit Slurm Job
```bash
# Submit the job to Slurm
sbatch slurm_scripts/cli_slurm_run.sh
```

### 3. Monitor Progress
```bash
# Check job status
squeue -u $USER

# Monitor output
tail -f outputs/logfiles/cli_full_run_output.txt
```

## Files in this workflow

- **`slurm_scripts/cli_slurm_run.sh`** - Slurm submission script using CLI (2 hour runtime)
- **`configs/cli_config.ini`** - Example full inference configuration file  
- **`submit_job.sh`** - Convenience script to submit the job
- **`README.md`** - This documentation

## Configuration

The configuration file format is the same as the development workflow, but with key modifications for full inference:

**Critical Configuration Changes:**
- **Commented out** `noise_params_path` - enables EFAC/EQUAD sampling
- **Commented out** `spin_injections_path` - enables pulsar red noise parameter sampling
- **Longer runtime** - full inference requires more computation time

Example config sections:
- `[Data]` - Data paths and pulsar selection
- `[NUTS]` - MCMC sampling parameters (may need more samples for convergence)
- `[PriorModel]` - Parameter priors and bounds for **all parameters**
- `[Output]` - Results directory and naming

## Parameter Inference Details

This workflow will sample:

### GW Background Parameters
- `log10_ha` - GW strain amplitude
- `log10_gamma_a` - GW spectral index

### Pulsar Red Noise Parameters (per pulsar)
- `log10_gamma_p` - Red noise spectral index
- `log10_sigma_p` - Red noise amplitude (via log-ratio parameterization)

### Noise Parameters (per pulsar)
- `efac` - EFAC scaling factor
- `log10_equad` - EQUAD white noise

### Hierarchical Parameters
- `log10_gamma_p_mean`, `log10_gamma_p_std` - Hierarchical priors for gamma_p
- `log10_ratio_mean`, `log10_ratio_std` - Log-ratio parameterization priors

## Advanced Usage

### Multiple Configs
```bash
# Run multiple analyses
argus run config1.ini
argus run config2.ini
```

### Custom Output Locations
Edit the config file `[Output]` section to specify custom output directories.

### GPU Usage
The CLI automatically detects and configures GPU usage when available through Slurm's `--gres=gpu` allocation.

### Runtime Considerations
Full parameter inference typically requires:
- **More samples** for convergence (especially with hierarchical modeling)
- **Longer runtime** (2+ hours vs. 10 minutes for fixed parameters)
- **More memory** for storing all parameter chains
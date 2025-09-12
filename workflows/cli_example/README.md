# CLI Example Workflow

This workflow demonstrates how to use the Argus CLI for Bayesian inference analysis with Slurm job management.

## Overview

This example shows the **modern way** to run Argus analysis using the installed package and CLI, rather than the development approach with sys.path manipulation.

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
tail -f outputs/logfiles/cli_run_output.txt
```

## Files in this workflow

- **`slurm_scripts/cli_slurm_run.sh`** - Slurm submission script using CLI
- **`configs/cli_config.ini`** - Example configuration file  
- **`submit_job.sh`** - Convenience script to submit the job
- **`README.md`** - This documentation

## Configuration

The configuration file format is the same as the development workflow, but the CLI provides additional validation and error handling.

Example config sections:
- `[Data]` - Data paths and pulsar selection
- `[Inference]` - MCMC sampling parameters  
- `[PriorModel]` - Parameter priors and bounds
- `[Output]` - Results directory and naming

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
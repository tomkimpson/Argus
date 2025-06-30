# Claude Configuration for Python Research Project

## Development Environment

### System Architecture
- **Compute**: HPC cluster with SLURM job scheduler
- **Login Node**: CPU-only access (no GPU)
- **GPU Access**: Only available through SLURM job submissions
- **Storage**: Shared filesystem accessible from login and compute nodes

### Python Environment
- **Virtual Environment**: Always use project-specific environments. Before running any local python code, activate the conda "Argus" environment

## Development Workflow

### Version Control Practices
- **CRITICAL**: Always commit changes before submitting SLURM jobs
- Use descriptive commit messages that include experiment details
- Tag commits that correspond to major experimental runs
- Include job ID in commit messages for easy tracking: "Add experiment X (job #12345)"
- Commit hyperparameter files, configs, and SLURM scripts along with code
- Track any important changes to the source code, problems that came up, solutions we tried in @latex_notes/checkpoints directory in a markdown format.

### Slurm and Config practices
- Create a new config file for each run. For example, for run indexed 020, there should be a corresponding config file with a matching index.
- When passing a config file to the slurm manager, update the slurm_run.sh file to point it to the correct config.  

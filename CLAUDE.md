# Claude Configuration for Python Research Project

## Development Environment

### System Architecture
- **Compute**: HPC cluster with SLURM job scheduler
- **Login Node**: CPU-only access (no GPU)
- **GPU Access**: Only available through SLURM job submissions
- **Storage**: Shared filesystem accessible from login and compute nodes

### Python Environment
- **Virtual Environment**: Always use project-specific environments. Before running any local python code, activate the conda "Argus" environment
- **Scientific Stack**: NumPy, SciPy, pandas, matplotlib, scikit-learn
- **Deep Learning**: PyTorch or TensorFlow (GPU versions for SLURM jobs only)

## Development Workflow

### Version Control Practices
- **CRITICAL**: Always commit changes before submitting SLURM jobs
- Use descriptive commit messages that include experiment details
- Tag commits that correspond to major experimental runs
- Include job ID in commit messages for easy tracking: "Add experiment X (job #12345)"
- Commit hyperparameter files, configs, and SLURM scripts along with code
- For any relevant changes to the source code, difficult problems that arise, etc., write them the the @latex_notes/checkpoints directory in a markdown format.

# Data Files for Example Workflow Lite

## approximate_spin_injections.pkl

This is an **example file** that demonstrates how to pass known values to Argus to set delta function priors on pulsar red noise parameters.

### Purpose

When you have prior knowledge of pulsar spin noise parameters (e.g., from previous analyses or injections in mock data), you can provide them via a pickle file. This allows the inference to fix these parameters to known values rather than sampling them from broad priors.

### Usage

The file path is specified in the configuration file:

```ini
[PriorModel]
spin_injections_path = ../data_files/approximate_spin_injections.pkl
```

When this path is provided, the workflow will:
- Load the `gamma_p` (spectral index) and `sigma_p` (amplitude) values from the file
- Fix the pulsar red noise parameters to these values (delta function priors)
- Focus sampling on other uncertain parameters

### File Format

The pickle file should contain a pandas DataFrame with columns:
- `psr`: Pulsar name
- `optimal_gamma`: Red noise spectral index values
- `optimal_sigma`: Red noise amplitude values

### Creating Your Own

To use your own known parameter values:
1. Create a pandas DataFrame with the required columns
2. Save it using `df.to_pickle('your_file.pkl')`
3. Update the config file to point to your file
4. Ensure pulsar names match those in your data directory

### Note

If you don't have prior knowledge of these parameters, simply comment out or remove the `spin_injections_path` line in your config file. The workflow will then sample these parameters from the specified prior ranges.

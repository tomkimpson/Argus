# Jupyter Notebooks

Interactive notebooks for hands-on exploration of Argus functionality.

## Available Notebooks

### Data Exploration
- [`01.explore_how_minnow_works.ipynb`](../../../notebooks/01.explore_how_minnow_works.ipynb) - Understanding the Minnow framework
- [`02.run_on_mock_data_challenge.ipynb`](../../../notebooks/02.run_on_mock_data_challenge.ipynb) - IPTA mock data analysis
- [`02e.loading_mock_data_with_argus.ipynb`](../../../notebooks/02e.loading_mock_data_with_argus.ipynb) - Data loading with Argus

### Parameter Estimation
- [`09.estimate_psr_noise_params.ipynb`](../../../notebooks/09.estimate_psr_noise_params.ipynb) - Pulsar noise characterization
- [`10.estimate_timing_ephemeris_params.ipynb`](../../../notebooks/10.estimate_timing_ephemeris_params.ipynb) - Timing model parameters
- [`11.inspect_parameter_estimation_results.ipynb`](../../../notebooks/11.inspect_parameter_estimation_results.ipynb) - Result analysis

### Analysis Methods
- [`07.explore_inference_convergence.ipynb`](../../../notebooks/07.explore_inference_convergence.ipynb) - Convergence diagnostics
- [`08.explore_different_likelihood_values.ipynb`](../../../notebooks/08.explore_different_likelihood_values.ipynb) - Likelihood evaluation
- [`12.plot_likelihood_curves.ipynb`](../../../notebooks/12.plot_likelihood_curves.ipynb) - Visualization techniques

### Signal Processing
- [`05.PSD_for_OU_process.ipynb`](../../../notebooks/05.PSD_for_OU_process.ipynb) - Ornstein-Uhlenbeck processes
- [`06.PSD_for_OU_GW_process.ipynb`](../../../notebooks/06.PSD_for_OU_GW_process.ipynb) - Gravitational wave PSDs

## Running the Notebooks

To run these notebooks locally:

```bash
# Clone the repository
git clone https://github.com/ADACS-Australia/tkimpson_2025a.git
cd tkimpson_2025a

# Install dependencies
poetry install

# Start Jupyter
jupyter lab notebooks/
```

!!! warning "Data Requirements"
    Some notebooks require large datasets that are not included in the repository. Download instructions are provided within each notebook.

!!! tip "Cloud Computing"
    These notebooks can be run on cloud platforms like Google Colab or Binder for easier access.
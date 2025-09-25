# Jupyter Notebooks

Interactive notebooks for hands-on exploration of Argus functionality.

## Available Notebooks

### Data Exploration
*Developer notes: Nice to have interactive notebooks for data exploration and framework understanding*
- Understanding the Minnow framework
- IPTA mock data analysis
- Data loading with Argus

### Parameter Estimation
*Developer notes: Nice to have notebooks demonstrating parameter estimation workflows*
- Pulsar noise characterization
- Timing model parameters
- Result analysis

### Analysis Methods
*Developer notes: Nice to have notebooks showing analysis and diagnostic techniques*
- Convergence diagnostics
- Likelihood evaluation
- Visualization techniques

### Signal Processing
*Developer notes: Nice to have notebooks exploring signal processing methods*
- Ornstein-Uhlenbeck processes
- Gravitational wave PSDs

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
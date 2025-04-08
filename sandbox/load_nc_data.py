import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import corner

# Load the netCDF file
# Replace 'your_file.nc' with the actual path to your netCDF file
data = az.from_netcdf('outputs/inf_data.nc')

# Print information about the dataset
print("\nDataset information:")
print(data)

# Print available variables
print("\nAvailable variables:")
print(data.posterior.data_vars)

# Extract the parameters we want to plot
samples = np.column_stack([
    data.posterior['ha'].values.flatten(),
    data.posterior['γa'].values.flatten()
])

# Transform to log space
log_samples = np.log10(samples)

# Set plot limits (adjust these values based on your data)
ranges = [
    (-16, -11),  # ha limits
    (-11, -6)   # γa limits
]


# Create corner plot
fig = corner.corner(
    log_samples,
    labels=['log(ha)', 'log(γa)'],
    quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
    title_kwargs={"fontsize": 12},
    range=ranges,
    bins=30,  # Adjust number of bins as needed
    smooth=1.0,  # Adjust smoothing as needed
    plot_datapoints=True,
    fill_contours=True,
    levels=[0.68, 0.95]  # 1 and 2 sigma contours
)

plt.tight_layout()
plt.savefig('outputs/corner_plot.png')
plt.show()

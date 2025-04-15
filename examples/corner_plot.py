import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import corner
import sys
# Load the netCDF file
# Replace 'your_file.nc' with the actual path to your netCDF file

data_file = sys.argv[1]

data = az.from_netcdf(data_file)

# Print information about the dataset
print("\nDataset information:")
print(data)


# Get summary including R-hat
summary_df = az.summary(data)
print(summary_df)




rhat_values = az.rhat(data)
print(rhat_values)




print("Sample stats:")
print(data.sample_stats)

# Print available variables
print("\nAvailable variables:")
print(data.posterior.data_vars)

# Extract the parameters we want to plot
samples = np.column_stack([
    data.posterior['ha'].values.flatten(),
    data.posterior['γa'].values.flatten()
])


print("Mean of selected parameters:")
print(data.posterior['ha'].mean())
print(data.posterior['γa'].mean())

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

# Plot EFAC components
plt.figure(figsize=(10, 6))
efac_samples = data.posterior['EFAC'].values

# Select first 3 EFAC components
selected_efac_samples = np.column_stack([
    efac_samples[..., 0].flatten(),
    efac_samples[..., 1].flatten(),
    efac_samples[..., 2].flatten()
])

# Create corner plot for EFAC parameters
fig = corner.corner(
    selected_efac_samples,
    labels=['EFAC_0', 'EFAC_1', 'EFAC_2'],
    quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
    title_kwargs={"fontsize": 12},
    bins=30,
    smooth=1.0,
    plot_datapoints=True,
    fill_contours=True,
    levels=[0.68, 0.95],  # 1 and 2 sigma contours
    range=[(0.5, 2.0), (0.5, 2.0), (0.5, 2.0)]  # Set axis limits for all three parameters
)

plt.tight_layout()
plt.savefig('outputs/efac_corner_plot.png')
plt.show()
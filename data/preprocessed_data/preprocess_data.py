
import sys
import glob 
sys.path.append('../../src')

import numpy as np 

from data_loader import LoadWidebandPulsarData


directory = '../IPTA_MockDataChallenge/IPTA_Challenge1_open/Challenge_Data/Dataset2/'

# Get all .par and .tim files in the directory
par_files = sorted(glob.glob(directory + '*.par'))
tim_files = sorted(glob.glob(directory + '*.tim'))
assert len(par_files) == len(tim_files), "Mismatch between .par and .tim file counts."

#Get the data
pulsar_residuals, pulsar_metadata = LoadWidebandPulsarData.read_multiple_par_tim(par_files, tim_files)

# Also get the separation angles between all pulsars.
ra = pulsar_metadata["RA"].to_numpy(dtype=float)
dec = pulsar_metadata["DEC"].to_numpy(dtype=float)
angular_separation_matrix = LoadWidebandPulsarData.pairwise_angular_separation(ra,dec)


# Post-process the residuals
processed_pulsar_residuals = LoadWidebandPulsarData.post_process_residuals(pulsar_residuals)


#saveit 
ID = "IPTA_Challenge1_open_Dataset2"
np.save(f'{ID}_residuals.npy', processed_pulsar_residuals) #numpy array
pulsar_metadata.to_parquet(f'{ID}_metadata') #pandas df 

print("Total length of the data is ", len(processed_pulsar_residuals))# Calculate the size of the DataFrame in MB
print("Total number of pulsars is ", len(pulsar_metadata))# Calculate the size of the DataFrame in MB


# Calculate the size of the NumPy array in MB
size_in_bytes = processed_pulsar_residuals.nbytes
size_in_mb = size_in_bytes / (1024 ** 2)
print(f'Size of processed_pulsar_residuals NumPy array: {size_in_mb:.2f} MB')

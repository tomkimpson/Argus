import os 
import glob
from enterprise.pulsar import Pulsar as EnterprisePulsar
import matplotlib.pyplot as plt 
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))



data_path = "../data/IPTA_MockDataChallenge2/dataset_1b/" # https://github.com/ipta/mdc2/tree/master

directory = os.path.join(
    script_dir,
    data_path
)

# Get all .par and .tim files in the directory
par_files = sorted(glob.glob(directory + "*.par"))
tim_files = sorted(glob.glob(directory + "*.tim"))

assert len(par_files) == len(tim_files), "Mismatch between .par and .tim file counts."


#Load just one pulsar and check everything looks reasonable
year = 365.25 * 86400
psr = EnterprisePulsar(par_files[0], tim_files[0])
plt.errorbar((psr.toas - psr.toas[0])/year, psr.residuals, yerr=psr.toaerrs, fmt='o-', capsize=3)
print("Mean residual: ", np.mean(np.abs(psr.residuals)))
print("Mean uncertainty: ", np.mean(np.abs(psr.toaerrs)))


plt.xlabel("Time [years]")
plt.ylabel("Residuals [s]")
plt.title(psr.name)
plt.savefig("residuals.png")

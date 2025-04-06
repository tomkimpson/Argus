from enterprise.pulsar import Pulsar 

import glob 

import matplotlib.pyplot as plt
data_path = "../data/IPTA_MockDataChallenge2/dataset_1b/" # https://github.com/ipta/mdc2/tree/master

# Get all .par and .tim files in the directory
par_files = sorted(glob.glob(data_path + "*.par"))
tim_files = sorted(glob.glob(data_path + "*.tim"))


for i in range(5):
    pulsar_object = Pulsar(par_files[i], tim_files[i])

    plt.plot(pulsar_object.toas, pulsar_object.residuals)
plt.savefig(f"residualsls.png")


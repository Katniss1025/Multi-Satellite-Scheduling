import os
import numpy as np

root = (os.path.abspath(os.path.join(os.getcwd(), "../")))
gp_dict = np.load(root + '/data/GaussianParameters.npy', allow_pickle=True).item()

# gp_dict contains eventID, means, covs
print(gp_dict['eventID'])

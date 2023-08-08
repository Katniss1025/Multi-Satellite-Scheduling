import os
import numpy as np
from utils import get_args
import skymap.SkyMapUtils as su
from skymap.GaussianUtils import make_gaussian
from skymap.probUtils import cal_credible_level, find_credible_region
from skymap.SkyMapUtils import visualize

def skymap_sample():
    # gp_dict contains eventID, means, covs. They are obtained from gaussian estimation.
    # root = (os.path.abspath(os.path.join(os.getcwd(), "../")))
    root = os.getcwd()
    gp_dict = np.load(root + '/data/GaussianParameters.npy', allow_pickle=True).item()
    weird = ['S190910h', 'S190901ap']
    ind = []
    for i, x in enumerate(gp_dict['eventID']):
        if x not in weird:
            ind.append(i)
    # means~U[a1,b1]
    means = np.array(gp_dict['means'])[ind]
    a1 = np.min(means, axis=0)
    b1 = np.max(means, axis=0)
    mean = np.random.uniform(a1, b1)
    # covs~U[a2,b2]
    covs = np.array(gp_dict['covs'])[ind]
    a2 = np.min(covs, axis=0)
    b2 = np.max(covs, axis=0)
    cov = np.random.uniform(a2, b2)
    # symmetric cov
    cov[1, 0] = cov[0, 1]
    cov[2, 0] = cov[0, 2]
    cov[2, 1] = cov[1, 2]
    return mean, cov


def generate_skymap_by_gaussian():
    mean, cov = skymap_sample()
    # ensure the cov is semi positive definite
    while np.any(np.linalg.eig(cov)[0] < 0):
        mean, cov = skymap_sample()
    args = get_args()
    nside_std = args.nside_std
    prob = make_gaussian(mean, cov, nside_std)

    credible_prob = cal_credible_level(prob)
    area_90 = find_credible_region(nside_std, credible_prob, 0.9)
    area_50 = find_credible_region(nside_std, credible_prob, 0.5)
    # print(area_90, area_50)
    if (area_90 <= 13000) & (area_50 <= 2000):

        return prob
    return False




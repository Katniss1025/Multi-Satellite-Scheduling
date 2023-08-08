import os
import healpy as hp
import numpy as np
from scipy import stats
import SkyMapUtils as smu
import transUtils as tu
from probUtils import cal_credible_level, find_credible_region
from utils import get_args



def make_gaussian(mean, cov, nside=128):
    npix = hp.nside2npix(nside)
    xyz = np.transpose(hp.pix2vec(nside, np.arange(npix)))
    dist = stats.multivariate_normal(mean, cov)
    prob = dist.pdf(xyz)
    return prob / prob.sum()


# This one is centered at RA=45°, Dec=0° and has a standard deviation of ~1°.
# prob = make_gaussian(
#     equatorial_to_cartesian(np.random.randint(0, 360), np.random.randint(-90, 91)),
#     np.square(np.deg2rad(np.random.random()*10)))

# This one is centered at RA=45°, Dec=0°, and is elongated in the north-south direction.
# prob = make_gaussian(
#     equatorial_to_cartesian(np.random.randint(0, 360), np.random.randint(-90, 91)),
#     np.diag(np.square(np.deg2rad([np.random.randint(12), np.random.randint(12), np.random.randint(15)]))))

# This one is centered at RA=0°, Dec=0°, and is elongated in the east-west direction.
# prob = make_gaussian(
#     [1, 0, 0],
#     np.diag(np.square(np.deg2rad([1, 10, 1]))))

# This one is centered at RA=0°, Dec=0°, and has its long axis tilted about 10° to the west of north.
# prob = make_gaussian(
#     [1, 0, 0],
#     [[0.1, 0, 0],
#      [0, 0.1, -0.15],
#      [0, -0.15, 1]])

# This one is centered at RA=0°, Dec=0°, and has its long axis tilted about 10° to the east of north.
# prob = make_gaussian(
#     [1, 0, 0],
#     [[0.1, 0, 0],
#      [0, 0.1, 0.15],
#      [0, 0.15, 1]])

# This one is centered at RA=0°, Dec=0°, and has its long axis tilted about 80° to the east of north.
# prob = make_gaussian(
#     [1, 0, 0],
#     [[0.1, 0, 0],
#      [0, 1, 0.15],
#      [0, 0.15, 0.1]])

# This one is centered at RA=0°, Dec=0°, and has its long axis tilted about 80° to the west of north.
# prob = make_gaussian(
#     [1, 0, 0],
#     [[0.1, 0, 0],
#      [0, 1, -0.15],
#      [0, -0.15, 0.1]])
if __name__ == "__main__":
    ## 从57张sky maps中分别估计高斯分布的参数
    # 估计所有Sky Maps的参数
    means = []
    covs = []
    area_90 = []
    area_50 = []
    root = (os.path.abspath(os.path.join(os.getcwd(), "../")))
    eventID = np.load(root + '/data/eventID.npy')
    nside_std = 64  # get_args().nside_std
    for i in eventID:
        data, h, event, _ = smu.read_a_skymap(event=i)
        prob, npix, ra, dec, area = smu.skymap_standard(data, nside_std)
        A = np.array(tu.equatorial_to_cartesian(ra, dec))  # A[0,:], A[1,:], A[2,:]分别为各网格x,y,z坐标
        # samples = np.random.choice(npix, size=1000, p=prob)  # 按照prob随机采1000个网格作为样本
        # mean = np.mean(A[:, samples], axis=1)
        # cov = np.cov(A[:, samples])
        # means.append(mean)
        # covs.append(cov)
        credible_prob = cal_credible_level(prob)
        area = find_credible_region(nside=nside_std, credible_levels=credible_prob, credible=0.9)
        area_90.append(area)
        area = find_credible_region(nside=nside_std, credible_levels=credible_prob, credible=0.5)
        area_50.append(area)

    np.save(root + '/data/GaussianParameters.npy', {'eventID': eventID, 'means': means, 'covs': covs})
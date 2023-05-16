import healpy as hp
import numpy as np
from scipy import stats


def make_gaussian(mean, cov, nside):
    npix = hp.nside2npix(nside)
    xyz = np.transpose(hp.pix2vec(nside, np.arange(npix)))
    dist = stats.multivariate_normal(mean, cov)
    prob = dist.pdf(xyz)
    return prob / prob.sum()



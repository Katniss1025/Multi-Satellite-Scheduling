from scipy import stats
import numpy as np
from healpy.newvisufunc import projview
from matplotlib import pyplot as plt
from pylab import show
import healpy as hp
import SkyMapUtils as smu
import transUtils as tu


def estimate_gaussian(m, A):
    _A = A * m
    mu = np.array([np.sum(_A[0]), np.sum(_A[1]), np.sum(_A[2])])
    A = A.T
    B = A - mu
    cov = np.dot(B.T, B) / (npix - 1)
    return mu, cov

def make_gaussian(mean, cov, nside=128):
    npix = hp.nside2npix(nside)
    xyz = np.transpose(hp.pix2vec(nside, np.arange(npix)))
    dist = stats.multivariate_normal(mean, cov)
    prob = dist.pdf(xyz)
    return prob / prob.sum()


def equatorial_to_cartesian(ra, dec):
    x = np.cos(ra) * np.cos(dec)
    y = np.sin(ra) * np.cos(dec)
    z = np.sin(dec)
    return [x, y, z]




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
    # mean = equatorial_to_cartesian(np.random.randint(0, 360), np.random.randint(-90, 91))
    # cov = np.diag(np.square(np.deg2rad([np.random.randint(12), np.random.randint(12), np.random.randint(15)])))
    # while(np.linalg.det(cov) == 0):
    #     cov = np.diag(np.square(np.deg2rad([np.random.randint(12), np.random.randint(12), np.random.randint(20)])))
    # prob = make_gaussian(mean, cov)
    data, h, event = smu.random_select_skymap()
    m, npix, ra, dec, area = smu.skymap_standard(data)

    A = np.array(tu.equatorial_to_cartesian(ra, dec))
    mu, cov = estimate_gaussian(m, A)

    prob = make_gaussian(mu, cov)

    projview(
        m,
        coord=["E"],
        graticule=True,
        cmap=plt.cm.RdYlBu,
        cbar=False,
        graticule_labels=True,
        longitude_grid_spacing=45,
        projection_type="mollweide",  # cart, mollweide

    )
    show()

    projview(
        prob,
        coord=["E"],
        graticule=True,
        cmap=plt.cm.RdYlBu,
        cbar=False,
        graticule_labels=True,
        longitude_grid_spacing=45,
        projection_type="mollweide",  # cart, mollweide

    )
    show()

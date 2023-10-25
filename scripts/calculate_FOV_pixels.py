import numpy as np
import healpy as hp
from astropy import units as u
from astropy_healpix import HEALPix
from utils import get_args


"""
该脚本用于计算nside所有网格，在radius视场中的
"""
args = get_args()
nside = args.nside_std
radius = args.radius
npix = hp.nside2npix(nside)
pixels = np.arange(npix)
ra, dec = hp.pix2ang(nside, pixels, lonlat=True)

pixels_in_FOV = []
probs_in_FOV = np.zeros(npix)

healpix = HEALPix(nside=nside, order='nested')
for i in range(npix):
    index_nested = healpix.cone_search_lonlat(ra[i] * u.deg, dec[i] * u.deg, radius=radius * u.deg)
    index_ring = hp.nest2ring(nside, index_nested)
    pixels_in_FOV.append(index_ring)

np.savez('pixels_in_FOV.npz', data=pixels_in_FOV)

import healpy
import os
import numpy as np
from skymap.transUtils import rotate_to_origin
from skymap.probUtils import find_highest_prob_pixel
from skymap.SkyMapUtils import visualize, read_a_skymap
from skymap.DataReinforcement import data_reinforcement_by_rotate
from ligo.skymap.plot.marker import reticle

# data_reinforcement_by_rotate()

nside_std = 128
data, h, event, data_nest = read_a_skymap()
m = healpy.ud_grade(data, power=-2, nside_out=nside_std)  # 重采样
visualize(m, event)
# markers = reticle(inner=0)
npix = healpy.nside2npix(nside_std)  # 标准healpix下像素总数
pix_indices = np.arange(npix)
ra, dec = healpy.pix2ang(nside_std, pix_indices, lonlat=True)
[hra, hdec] = find_highest_prob_pixel(m, nside_std)
_ra, _dec, tag = rotate_to_origin(ra, dec, hra, hdec)  # in degree
pix_rotated = healpy.ang2pix(nside=nside_std, theta=_ra, phi=_dec, lonlat=True)
m_rotated = np.full(m.shape, np.nan)
m_rotated[pix_rotated] = m

# 对空值进行插值
# mask = np.isnan(m_rotated)
# nan_indices = np.arange(len(m_rotated))[mask]
# neighbors = healpy.get_all_neighbours(nside=nside_std, theta=_ra[nan_indices], phi=_dec[nan_indices], lonlat=True)
# m_rotated[nan_indices] = np.nanmean(m_rotated[neighbors].T, axis=1)

# m_masked = np.ma.masked_array(m_rotated, mask=mask)
# theta, phi = healpy.pix2ang(nside_std, pix_rotated)
# m_rotated = healpy.get_interp_val(m_masked,_ra, _dec, lonlat=True)

# 概率值归为1
m_rotated = m_rotated / np.sum(m_rotated)

# visualize(m, event)
visualize(m_rotated, event+tag)



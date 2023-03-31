# Get the data
from astropy.io import fits
import os
from transUtils import rotate
import matplotlib.pyplot as plt
from astropy import units as u
import numpy as np
import healpy
from astropy.coordinates import SkyCoord
from astropy_healpix import HEALPix
from astropy.coordinates import Galactic


# Read a fits file
root = (os.path.abspath(os.path.join(os.getcwd(), "../")))
hdulist = fits.open(root + '/data/SkyMap/Flat/S190421ar_Flat.fits.gz')
data = healpy.read_map(root + '/data/SkyMap/Flat/S190421ar_Flat.fits.gz')
# Set up the HEALPix projection

nside = hdulist[1].header['NSIDE']
order = hdulist[1].header['ORDERING']
hp = HEALPix(nside=nside, order=order, frame=Galactic())
hp_std = HEALPix(nside=128, order=order, frame=Galactic())
# Sample a 360x180 grid in RA/Dec

ra = np.linspace(360., 0., 361) * u.deg
dec = np.linspace(-90., 90., 181) * u.deg
ra_grid, dec_grid = np.meshgrid(ra, dec)

# Set up Astropy coordinate objects

coords = SkyCoord(ra_grid.ravel(), dec_grid.ravel(), frame=Galactic()) # .ravel()把所有的拉成一行

# Interpolate values
prob = hdulist[1].data['PROB']
pmap = hp.interpolate_bilinear_skycoord(coords, prob)
pmap = pmap.reshape((181, 361))

# Data reinforcement
# npix = healpy.nside2npix(nside)
# pix_indices = np.arange(npix)
# _ra, _dec = healpy.pix2ang(nside, pix_indices, lonlat=True)
ra_rotate, dec_rotate = rotate(ra, dec)


# Make a plot of the interpolated temperatures
plt.figure(figsize=(10, 5))
im = plt.imshow(pmap, extent=[360, 0, -90, 90], cmap=plt.cm.RdYlBu, origin='lower', aspect='auto')
plt.colorbar(im)
plt.xlabel('Right ascension')
plt.ylabel('Declination')
plt.show()



import numpy as np
import os
import openpyxl
import healpy
from healpy.newvisufunc import projview
from matplotlib import pyplot as plt
from pylab import show


def read_a_skymap(event=None, random=True):
    # root = (os.path.abspath(os.path.join(os.getcwd(), "../")))
    root = os.getcwd()
    if random == True:
        wb = openpyxl.load_workbook(root + '/data/eventID.xlsx')
        ws = wb['Sheet1']
        eventID = []
        nrow = ws.max_row
        for i in range(nrow - 1):
            cell = ws.cell(row=i + 2, column=1)
            eventID.append(cell.value)
        event = eventID[np.random.randint(len(eventID))]  # 从57条数据中随机选择一条
    data, h = healpy.read_map(root + '/data/SkyMap/Flat/'+event+'_Flat.fits.gz', h=True)
    return data, h, event


def skymap_standard(prob, nside):
    m = healpy.ud_grade(prob, power=-2, nside_out=nside)  # 重采样
    npix = healpy.nside2npix(nside)  # 标准healpix下像素总数
    pix_indices = np.arange(npix)
    area = healpy.nside2pixarea(nside=nside, degrees=True)
    ra, dec = healpy.pix2ang(nside, pix_indices, lonlat=True)
    return m, npix, ra, dec, area


def visualize(prob, title=None):
    projview(
        prob,
        coord=["E"],
        graticule=True,
        cmap=plt.cm.RdYlBu,
        cbar=False,
        graticule_labels=True,
        longitude_grid_spacing=45,
        projection_type="mollweide",  # cart, mollweide
        title=title,
    )
    show()


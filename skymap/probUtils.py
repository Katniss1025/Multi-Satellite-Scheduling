import numpy as np
import healpy as hp
from astropy import units as u
from astropy_healpix import HEALPix


def cal_credible_level(hpx):
    i = np.flipud(np.argsort(hpx))  # hpx从大到小排序并提取其index
    sorted_credible_levels = np.cumsum(hpx[i])
    credible_levels = np.empty_like(sorted_credible_levels)
    credible_levels[i] = sorted_credible_levels
    return credible_levels


def pix_in_credible_region(ipix, credible_levels, credible):
    # justify a pix whether in the credible region
    return credible_levels[ipix] <= credible


def find_credible_region(nside, credible_levels, credible):
    # return the total area of the credible region
    return np.sum(credible_levels <= credible) * hp.nside2pixarea(nside, degrees=True)


def integrated_prob_in_a_circle(ra, dec, radius, hpx, nside, lonlat=True):
    """ 计算视野里的网格集合和总概率
    Args:
        define the RA, Dec, and radius of circle in degrees:
        ra(float): 指向的赤经
        dec(float): 指向的赤纬度
        radius(float): 视场的半径，以度为单位
        hpx(array): 概率
        nside(int): SkyMap属性
    Returns:
        ipix_disc(array): 圆形视场中网格的索引
        ipix_prob(array): 圆形视场中网格对应的概率
        prob_sum(float): 圆形时长中所有网格的概率之和
    """
    healpix = HEALPix(nside=nside, order='nested')
    index_nested = healpix.cone_search_lonlat(ra * u.deg, dec * u.deg, radius=radius * u.deg)
    index_ring = hp.nest2ring(nside, index_nested)

    if lonlat:
        ra = np.radians(ra)
        dec = np.radians(dec)

    # calculate the Cartesian coordinates of the center of circle:
    xyz = [np.cos(ra)*np.cos(dec), np.cos(dec)*np.sin(ra), np.sin(dec)]

    # call hp.query_disc, which returns an array of the indices of the pixels that are inside the circle:
    ipix_disc = query_disc(nside, xyz, radius)


    # prob array
    ipix_prob = hpx[index_ring]

    # sum the probability in all of the matching pixels:
    prob_sum = ipix_prob.sum()
    return index_ring, ipix_prob, prob_sum


def find_highest_prob_pixel(hpx, nside):
    ipix_max = np.argmax(hpx)
    theta, phi = hp.pix2ang(nside, ipix_max)
    ra = np.rad2deg(phi)  # in degree
    dec = np.rad2deg(0.5 * np.pi - theta)  # in degree
    return [ra, dec]


def query_disc(nside, xyz, radius):
    # 获取球面上的所有像素编号
    pixels = np.arange(hp.nside2npix(nside))

    # 将球面上的像素转换为对应的经纬度坐标
    theta, phi = hp.pix2ang(nside, pixels)

    # 将经纬度坐标转换为三维坐标
    pix_xyz = hp.ang2vec(theta, phi)

    # 计算输入点与球面上每个像素的距离
    distances = np.arccos(np.dot(pix_xyz.T, xyz))

    # 根据距离判断是否在半径范围内，并返回满足条件的像素编号和对应的经纬度坐标
    mask = distances <= radius
    selected_pixels = pixels[mask]
    selected_theta = theta[mask]
    selected_phi = phi[mask]

    return selected_pixels, selected_theta, selected_phi
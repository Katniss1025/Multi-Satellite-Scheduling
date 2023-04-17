import numpy as np
import healpy as hp


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


def integrated_prob_in_a_circle(ra, dec, radius, hpx, nside=128):
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

    # convert to spherical polar coordinates and radius of circle in radians:
    theta = 0.5 * np.pi - np.deg2rad(dec)
    phi = np.deg2rad(ra)
    radius = np.deg2rad(radius)

    # calculate the Cartesian coordinates of the center of circle:
    xyz = hp.ang2vec(theta, phi)

    # call hp.query_disc, which returns an array of the indices of the pixels that are inside the circle:
    ipix_disc = hp.query_disc(nside, xyz, radius)

    # prob array
    ipix_prob = hpx[ipix_disc]

    # sum the probability in all of the matching pixels:
    prob_sum = ipix_prob.sum()
    return ipix_disc, ipix_prob, prob_sum


def find_highest_prob_pixel(hpx, nside):
    ipix_max = np.argmax(hpx)
    theta, phi = hp.pix2ang(nside, ipix_max)
    ra = np.rad2deg(phi)
    dec = np.rad2deg(0.5 * np.pi - theta)
    return [ra, dec]

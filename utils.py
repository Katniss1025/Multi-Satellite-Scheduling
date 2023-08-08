import numpy as np
import yaml
import argparse
import math
import datetime
import pdb


def read_config_file(config_parser):
    args = config_parser.parse_args()
    config_file_path = 'config_files/' + args.config_file
    with open(config_file_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
        config_parser.set_defaults(**cfg)
    args = config_parser.parse_args()
    return args


def get_args():
    # 获取yaml参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', dest='config_file', default='config_env.yml')
    parser.add_argument('--dataset-name', dest='dataset_name', default='env')
    args = read_config_file(parser)
    return args


# ------- OrbitUtils----------#
def get_utc_start_end_of_week(year, week):
    # 计算给定周数的起始日期
    start_date = datetime.datetime.strptime(f'{year}-W{week-1}-1', "%Y-W%W-%w")

    # 计算给定周数的结束日期
    end_date = start_date + datetime.timedelta(days=7)

    # 将起始日期和结束日期转换为UTC时间字符串
    utc_start = start_date.strftime("%Y-%m-%d %H:%M:%S")
    utc_end = end_date.strftime("%Y-%m-%d %H:%M:%S")

    return utc_start, utc_end


def sun_position(jd):
    """计算在某个julian date时太阳在J2000下的坐标
    Args:
        jd(float): Julian date， 儒略日
    Returns:
        alpha(float): 赤经
        delta(float): 赤纬
    """
    n = jd - 2451545.0
    L = 280.460 + 0.9856474 * n
    g = math.radians(357.528 + 0.9856003 * n)
    lambda_sun = math.radians(L + 1.915 * math.sin(g) + 0.020 * math.sin(2 * g))
    epsilon = math.radians(23.439 - 0.0000004 * n)
    alpha = math.degrees(math.atan2(math.cos(epsilon) * math.sin(lambda_sun), math.cos(lambda_sun)))
    delta = math.degrees(math.asin(math.sin(epsilon) * math.sin(lambda_sun)))
    return alpha, delta


def julian_day_to_utc(julian_day):
    # 将儒略日转换为datetime对象
    date = datetime.datetime(2000, 1, 1) + datetime.timedelta(days=julian_day - 2451545.0)

    # 格式化为UTC时间字符串
    utc_time = date.strftime("%Y-%m-%d %H:%M:%S")

    return utc_time


def utc_to_julian_day(utc_time):
    # 将UTC时间字符串转换为datetime对象
    date = datetime.datetime.strptime(utc_time, "%Y-%m-%d %H:%M:%S")

    # 计算与初始日期的时间差，并转换为儒略日
    julian_day = 2451545.0 + (date - datetime.datetime(2000, 1, 1)).total_seconds() / (24*60*60)

    return julian_day


def eq2xyz(ra, dec, r=1):
    """ 将赤道坐标转换为空间直角坐标
    Args；
        ra(float): 赤经, in degree
        dec(float): 赤纬, in degree
        r(float): 半径, 单位为米
    Returns：
        x, y, z(float)
    """
    x = r * np.cos(ra) * np.cos(dec)
    y = r * np.sin(ra) * np.cos(dec)
    z = r * np.sin(dec)
    return x, y, z


def normalize_angle(angle):
    while angle < 0:
        angle += 360
    while angle >= 360:
        angle -= 360
    return angle


def calculate_shadow_u(ra, dec, a, Omega, i, u, degree=True):
    ''' 计算遮挡
    Args:
        ra(float): 目标赤经
        dec(float): 目标赤纬
        a(float): 轨道半径，简化为圆轨道，单位为m
        Omega:(float) 轨道升交点赤经
        i(float): 轨道倾角
        u(float): 卫星初始维度幅角
        degree(bool): 默认为角度制
    '''
    flag = False  # 目标是否在整个周期内可见
    if degree:
        ra = math.radians(ra)
        dec = math.radians(dec)
        Omega = math.radians(Omega)
        i = math.radians(i)
        u = math.radians(u)
    # 1. 计算目标在J2000下的方向向量
    rt_j2000 = eq2xyz(ra, dec)
    # 2. 计算从地心惯性坐标系到节点坐标系的旋转矩阵
    L_ni = np.array([[np.cos(Omega), np.sin(Omega), 0],
                     [-np.cos(i)*np.sin(Omega), np.cos(i)*np.cos(Omega), np.sin(i)],
                     [np.sin(i)*np.sin(Omega), -np.sin(i)*np.cos(Omega), np.cos(i)]])
    # 3. 目标在节点坐标系下的方向
    rt_sat = L_ni @ rt_j2000
    # 4. 判断整个轨道周期内是否均不会被地球遮挡
    alpha = np.array([0, 0, 1])  # 轨道法向量
    cos_alpha = np.abs(np.dot(alpha, rt_sat))
    Re = 6370.856e3  # 地球半径，单位为m
    A = Re / a
    if cos_alpha > A:
        return [0, 0]
    # 5. 计算离目标最远时，卫星所在的方位(xy平面上)
    far = np.array([-rt_sat[0], -rt_sat[1]])
    # 6. 计算该方位的纬度幅角
    theta = np.arccos(np.abs(far[0]) / math.sqrt(rt_sat[0] ** 2 + rt_sat[1] ** 2))
    if (far[0] > 0) & (far[1] >= 0):
        u_far = theta
    elif (far[0] <= 0) & (far[1] > 0):
        u_far = np.pi - theta
    elif (far[0] < 0) & (far[1] <= 0):
        u_far = np.pi + theta
    elif (far[0] >= 0) & (far[1] < 0):
        u_far = 2 * np.pi - theta
    # 7. 求far向量和轨道的交点
    mid = a * far/np.linalg.norm(far)
    # 8. 求delta_u
    c, d = mid[0], mid[1]
    if (c * np.sin(u_far) - d * np.cos(u_far)) == 0:
        delta_u = np.pi/2
    else:
        delta_u = np.arctan((a - c * np.cos(u_far) - d * np.sin(u_far)) / (c * np.sin(u_far) - d * np.cos(u_far)))
    # 对delta_u进行约束
    if delta_u <= 0:
        delta_u = delta_u + np.pi/2
    elif delta_u > 90:
        delta_u = delta_u - np.pi/2
    cos_it = (np.dot(rt_sat, np.array([rt_sat[0], rt_sat[1], 0]))) / (np.linalg.norm(rt_sat) * np.linalg.norm(np.array([rt_sat[0], rt_sat[1], 0])))
    alpha_max = np.pi/2 + np.arccos(Re/a)
    # delta_u_ = -np.arccos(np.cos(alpha_max) / cos_it)
    # 9. 进出阴影区时的纬度幅角
    enter_u = normalize_angle(np.rad2deg(u_far - delta_u))
    exit_u = normalize_angle(np.rad2deg(u_far + delta_u))
    return [enter_u, exit_u]


def mean_to_true_anomaly(M, e, degree=True):
    # 将平近点角转换为真近点角
    # 输入参数：
    #   - M: 平近点角
    #   - e: 偏心率

    # 求解偏近点角
    E = np.radians(M) + e * np.sin(np.radians(M))
    cos_v = (np.cos(E) - e) / (1 - e * np.cos(E))
    sin_v = np.sqrt(1 - e ** 2) * np.sin(E)
    true_anomaly = np.arctan2(sin_v, cos_v)
    ta = np.rad2deg(true_anomaly)
    return np.mod(ta, 360)


def pix_covered_by_sun(target_pos, sun_pos, sun_avoid_angle, Omega, i, degree=True):
    ''' 针对某颗卫星计算某个目标是否被太阳遮挡
    Args:
        target_pos(list): 目标在J2000下位置
        sun_pos(list): 太阳J2000下位置
        sun_avoid_angle(float): 太阳避免角
        Omega(float): 升交点赤经
        i(float): 轨道倾角
        degree(bool): 是否使用角度制
    Returns:
        pix_covered(bool array): 如果被遮挡, 对应pix为True, 否则为False.
    '''
    if degree:
        tra, tdec = np.deg2rad(target_pos[0]), np.deg2rad(target_pos[1])
        sra, sdec = np.deg2rad(sun_pos[0]), np.deg2rad(sun_pos[1])

    # 1. 计算目标和太阳在J2000下的单位方向向量
    t_j2000 = eq2xyz(tra, tdec)
    s_j2000 = eq2xyz(sra, sdec)
    # 2. 计算从地心惯性坐标系到节点坐标系的旋转矩阵
    L_ni = np.array([[np.cos(Omega), np.sin(Omega), 0],
                     [-np.cos(i) * np.sin(Omega), np.cos(i) * np.cos(Omega), np.sin(i)],
                     [np.sin(i) * np.sin(Omega), -np.sin(i) * np.cos(Omega), np.cos(i)]])
    # 3. 目标和太阳在节点坐标系下的单位方向向量
    t_n = L_ni @ t_j2000
    s_n = L_ni @ s_j2000
    # 4. 计算卫星到目标的指向和卫星到太阳指向之间的夹角
    point1 = t_n.T  # 卫星到目标的指向
    point2 = s_n  # 卫星到太阳的指向
    cos_angle = np.dot(point1, point2) / (np.linalg.norm(point1, axis=1) * np.linalg.norm(point2))
    angle = np.arccos(cos_angle)
    # 5. 计算每个pix是否会被遮挡
    angle = np.degrees(angle)
    pix_covered = (angle <= sun_avoid_angle)
    return pix_covered


if __name__ == '__main__':
    h = 500e3   # 轨道高度，单位为m
    Re = 6371.393e3  # 地球半径，单位为m
    a = Re + h  # 轨道半径，单位为m
    Omega = 0  # 升交点赤经
    omega = 30  # 近地点幅角
    mean_ano = 0  # 平近点角
    e = 0
    theta = mean_to_true_anomaly(mean_ano, e)  # 真近点角
    u = omega + theta  # 纬度幅角
    i = 60  # 轨道倾角，角度制
    ra = 142.03125
    dec = 38.68218745348944
    flag = calculate_shadow_u(ra, dec, a, Omega, i, u, degree=True)
    flag
    # import healpy as hp
    # import random
    # nside = 64  # 设置分辨率参数
    # npix = hp.nside2npix(nside)  # 获取总像素数
    # # 生成一个范围在[0, npix-1]之间的像素索引数组
    # pix_indices = np.arange(npix)
    # # 将像素索引转换为赤经赤纬坐标
    # target_pos = hp.pix2ang(nside, pix_indices, lonlat=True)
    # utc_start, utc_end = get_utc_start_end_of_week(2020, 20)
    # sun_pos_start = sun_position(utc_to_julian_day(utc_start))
    # sun_pos_end = sun_position(utc_to_julian_day(utc_end))
    # sun_avoid_angle = 70
    # pix_covered_start = pix_covered_by_sun(target_pos, sun_pos_start, sun_avoid_angle, Omega, i, degree=True)
    # pix_covered_end = pix_covered_by_sun(target_pos, sun_pos_end, sun_avoid_angle, Omega, i, degree=True)
    # from skymap.SkyMapUtils import visualize_selected_pixel
    # m = np.ones(npix)
    # pixel_indices = np.where(np.logical_not(pix_covered_start))[0]
    # visualize_selected_pixel(m, pixel_indices, nside)
    # pixel_indices = np.where(np.logical_not(pix_covered_end))[0]
    # visualize_selected_pixel(m, pixel_indices, nside)









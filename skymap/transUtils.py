import numpy as np
import healpy

def equatorial_to_cartesian(ra, dec):
    """
    ra(float): 赤经 (degree)
    dec(float):赤纬 (degree)
    """
    ra = np.deg2rad(ra)
    dec = np.deg2rad(dec)
    x = np.cos(ra) * np.cos(dec)
    y = np.sin(ra) * np.cos(dec)
    z = np.sin(dec)
    return [x, y, z]

# 将三维坐标转换为弧度
    ra_rad_rot = np.arctan2(y_rotated, x_rotated)
    dec_rad_rot = np.arctan2(z_rotated, np.sqrt(x_rotated ** 2 + y_rotated ** 2))

    # 将弧度转换为赤经赤纬
    ras_rotated = np.round(ra_rad_rot * 180 / np.pi, 10)
    decs_rotated = np.round(dec_rad_rot * 180 / np.pi, 10)
    # 将旋转后的赤经赤纬映射到新的矩阵中
    ras_rotated = np.clip(ras_rotated.reshape(ras.shape), -180, 180)
    ras_rotated[ras_rotated < 0] += 360  # 从(-180,180)映射到(0,360)的赤经表示
    decs_rotated = np.clip(decs_rotated.reshape(ras.shape), -90, 90)
    # 根据旋转后的赤经赤纬对概率进行插值


def calRotMat(axis, angle):
    '''
    angle: in radius
    '''
    if axis == 'x':
        rot_mat = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
    elif axis == 'y':
        rot_mat = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    elif axis == 'z':
        rot_mat = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    return rot_mat


def rotate(ras, decs):
    # 定义旋转角度和方位
    axises = ['x', 'y', 'z']  # x, y, z分别对应的数字
    # axises = ['x']
    np.random.shuffle(axises)  # 随机打乱，决定旋转轴顺序
    rotFlag = np.random.randint(0, 2, 3)  # 决定是否围绕该轴旋转
    angle = rotFlag * np.random.randint(0, 360, 3)
    # angle = np.random.randint(1, 360, 1)
    # 旋转信息
    tag = ''
    for i in np.arange(len(axises)):
        tag = tag + " " + axises[i] + ':' + str(angle[i])
    angle = angle / 180 * np.pi

    # 将赤经赤纬（角度）转换为弧度，再转换成三维坐标
    [x, y, z] = equatorial_to_cartesian(ras, decs)

    # 初始化旋转后的坐标
    x_rotated = np.zeros_like(ras, dtype=float)
    y_rotated = np.zeros_like(ras, dtype=float)
    z_rotated = np.zeros_like(ras, dtype=float)

    # 计算旋转矩阵 axis=0：绕x,axis=1：绕y，axis=2：绕z.
    rotMat = 1
    for i in len(axises) - 1 - np.arange(len(axises)):
        temp = calRotMat(axises[i], angle[i])
        rotMat = np.dot(rotMat, temp)

    # 计算旋转后的三维坐标
    for i in range(x.shape[0]):
        x_rotated[i], y_rotated[i], z_rotated[i] = np.dot(rotMat, [x[i], y[i], z[i]])

    # 将三维坐标转换为弧度
    ra_rad_rot = np.arctan2(y_rotated, x_rotated)
    dec_rad_rot = np.arctan2(z_rotated, np.sqrt(x_rotated ** 2 + y_rotated ** 2))

    # 将弧度转换为赤经赤纬
    ras_rotated = np.round(ra_rad_rot * 180 / np.pi, 10)
    decs_rotated = np.round(dec_rad_rot * 180 / np.pi, 10)
    # 将旋转后的赤经赤纬映射到新的矩阵中
    ras_rotated = np.clip(ras_rotated.reshape(ras.shape), -180, 180)
    ras_rotated[ras_rotated < 0] += 360  # 从(-180,180)映射到(0,360)的赤经表示
    decs_rotated = np.clip(decs_rotated.reshape(ras.shape), -90, 90)
    # 根据旋转后的赤经赤纬对概率进行插值

    return ras_rotated, decs_rotated, tag

def rotate_to_origin(m, nside_std):
    '''
    Args：
        ras(array):赤经数组，degree [0,360]
        decs(array):赤纬数组，degree [-90,90]
        hra(float):基准点赤经，degree
        hdec(float):基准点赤纬，degree
    Returns:
        ras_rotated(array): 旋转后的赤经 [-180,180]
        decs_rotated(array): 选钻后的赤纬 [-90,90]
        tag(str): 记录旋转信息
    '''
    npix = healpy.nside2npix(nside_std)  # 标准healpix下像素总数
    pix_indices = np.arange(npix)
    ras, decs = healpy.pix2ang(nside_std, pix_indices, lonlat=True)
    from skymap.probUtils import find_highest_prob_pixel
    [hra, hdec] = find_highest_prob_pixel(m, nside_std)

    axises = ['z', 'y']  # x, y, z分别对应的数字
    angle = [hra, -hdec]
    # 旋转信息
    tag = ''
    for i in np.arange(len(axises)):
        tag = tag + " " + axises[i] + ':' + str(angle[i])
    angle = np.deg2rad(angle)

    # 将赤经赤纬（角度）转换为弧度，再转换成三维坐标
    [x, y, z] = equatorial_to_cartesian(ras, decs)

    # 初始化旋转后的坐标
    x_rotated = np.zeros_like(ras, dtype=float)
    y_rotated = np.zeros_like(ras, dtype=float)
    z_rotated = np.zeros_like(ras, dtype=float)

    # 计算旋转矩阵
    rotMat = 1
    for i in np.arange(len(axises)):
        temp = calRotMat(axises[i], angle[i])
        rotMat = np.dot(rotMat, temp)

    # 计算旋转后的三维坐标
    for i in range(x.shape[0]):
        x_rotated[i], y_rotated[i], z_rotated[i] = np.dot(rotMat, [x[i], y[i], z[i]])
    # 将三维坐标转换为弧度
    ra_rad_rot = np.arctan2(y_rotated, x_rotated)
    dec_rad_rot = np.arctan2(z_rotated, np.sqrt(x_rotated ** 2 + y_rotated ** 2))

    # 将弧度转换为赤经赤纬
    ras_rotated = np.round(ra_rad_rot * 180 / np.pi, 10)
    decs_rotated = np.round(dec_rad_rot * 180 / np.pi, 10)
    # 将旋转后的赤经赤纬映射到新的矩阵中
    ras_rotated = np.clip(ras_rotated.reshape(ras.shape), -180, 180)
    # ras_rotated[ras_rotated < 0] += 360  # 从(-180,180)映射到(0,360)的赤经表示
    decs_rotated = np.clip(decs_rotated.reshape(ras.shape), -90, 90)

    m_rotated = healpy.get_interp_val(m, ras_rotated, decs_rotated, lonlat=True)

    # 概率值归为1
    m_rotated = m_rotated / np.sum(m_rotated)

    return m_rotated
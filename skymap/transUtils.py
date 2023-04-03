import numpy as np


def equatorial_to_cartesian(ra, dec):
    ra = ra * np.pi / 180
    dec = dec * np.pi / 180
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
    rotMat = np.ones([3, 3])
    for i in np.arange(len(axises)):
        temp = calRotMat(axises[i], angle[i])
        rotMat = temp * rotMat

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

import yaml
import argparse
import math


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
def sun_position(jd):
    """计算在某个julian date时太阳在J2000下的坐标
    Args:
        jd(float): Julian date， 儒略日
    Returns:
        alpha(float): 赤经
        delta(float): 赤纬
    """
    import math
    n = jd - 2451545.0
    L = 280.460 + 0.9856474 * n
    g = math.radians(357.528 + 0.9856003 * n)
    lambda_sun = math.radians(L + 1.915 * math.sin(g) + 0.020 * math.sin(2 * g))
    epsilon = math.radians(23.439 - 0.0000004 * n)
    alpha = math.degrees(math.atan2(math.cos(epsilon) * math.sin(lambda_sun), math.cos(lambda_sun)))
    delta = math.degrees(math.asin(math.sin(epsilon) * math.sin(lambda_sun)))
    return alpha, delta


def is_covered():




def eq2xyz(ra, dec):
    """ 将赤道坐标转换为空间直角坐标
    Args；
        ra(float): 赤经, in degree
        dec(float): 赤纬, in degree
    Returns：
        x, y, z(float)
    """
    x = math.cos(math.radians(ra)) * math.cos(math.radians(dec))
    y = math.sin(math.radians(ra)) * math.cos(math.radians(dec))
    z = math.sin(math.radians(dec))
    return x, y, z

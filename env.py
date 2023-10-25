import random
import numpy
from skymap.probUtils import integrated_prob_in_a_circle
import gym
import numpy as np
from gym import spaces
import healpy as hp
from utils import get_args
import torch
from time_dist_simulation.test import sample
from skymap.DataReinforcement import data_reinforcement_by_rotate
from skymap import SkyMapUtils as smu
from skymap.transUtils import rotate_to_origin
import time
from utils import calculate_shadow_u, mean_to_true_anomaly, sun_position, julian_day_to_utc, utc_to_julian_day, get_utc_start_end_of_week, pix_covered_by_sun
from skymap.SkyMapUtils import visualize_selected_pixel
from skymap.SkyMapUtils import visualize


def scale_action(action, action_details):
    ''' 将tanh激活的action线性映射到赤经和赤纬 [-1,1]->[0,360]/[-90,90]
    Args:
        action_tahn(tensor): 被tanh激活的action
        action_details(dict): 自定义的动作空间的信息
    Returns:
        action_scaled(tensor): 映射后的action
    '''
    action_tahn = torch.tanh(action)
    action_scaled = (action_tahn + 1) / 2 * (action_details['high'] - action_details['low']) + action_details['low']
    return action_scaled


class MultiSatelliteEnv(gym.Env):
    """ 环境类
    根据gym接口写的自定义环境
    环境中有一个网格，agent要填满网格。填空白格子得到+1奖励，重复填格子得到-1奖励
    """
    def __init__(self, num_epoch_steps, m):
        """ 初始化
        n_sat(int): 卫星数量
        n_pix(int): 像素数
        t(float): 任务时长
        action_details(dict): 动作空间, 离散: n_sat * [npix], 连续:
        state_size(dict): 状态空间 keys={'sat','task',''}  state_size = {'task':[n_pix, 2]}
        num_epoch_steps(int): 一个回合最大时间步数
        """
        super(MultiSatelliteEnv, self).__init__()
        self.m_origin = m
        self.args = get_args()
        self.nside_std = self.args.nside_std
        self.n_sat = self.args.n_sat
        self.npix = self.args.npix
        self.nside_std = self.args.nside_std
        self.max_num_steps = num_epoch_steps  # 一回合最大时间步数
        self.current_step = 0  # 当前时间步
        self.t = 0
        # self.action_details = action_details
        # Define action and observation space. They must be gym.spaces objects
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # TODO
        if self.args.simple:
            self.state_task = torch.zeros(np.round(self.n_sat*self.args.factor).astype(int)).to(device)  # 任务状态: 索引、skymap
        else:
            self.state_task = torch.zeros(self.npix).to(device)
        self.state_sat = torch.zeros(2 * self.n_sat).to(device)  # 卫星状态：目前所处平近点角、正在看的目标存在爆发源的概率
        # self.state_cover = torch.zeros(self.n_sat+1, self.npix)
        # self.state = np.concatenate([np.array(self.state_sat), np.array(self.state_task)])  # 拼接卫星、任务
        self.state = dict({'task': self.state_task, 'sat': self.state_sat, 't': self.t})

        # 定义动作空间: 离散动作空间，分配网格的索引.
        if self.args.simple:
            self.action_space = spaces.MultiDiscrete([self.n_sat*self.args.factor] * self.n_sat)
        else:
            self.action_space = spaces.MultiDiscrete([self.npix] * self.n_sat)

        # 定义观测空间 TODO
        self.observation_space = spaces.Tuple((
            spaces.MultiDiscrete([self.npix]*self.npix),
            spaces.Box(low=np.array([0] * self.n_sat * 2), high=np.repeat(np.array([360, 1]), [self.n_sat, self.n_sat]), dtype=np.float32)
        ))

    def seed(self, seed):
        np.random.seed(seed)

    def reset(self):
        """ 重置环境，新的爆发事件
        Important: the observation must be a numpy array
        """
        # 生成新的事件并计算状态
        # 归一化概率
        self.m = (self.m_origin - np.min(self.m_origin)) / (np.max(self.m_origin) - np.min(self.m_origin))
        # 随机在2020年中选一周
        utc_start, utc_end = get_utc_start_end_of_week(2020, self.args.week)
        # 可视化该事件
        #visualize(self.m_origin, title=None)
        # 以某网格为视场中心，计算视场内网格集合及概率和
        self.pixels_in_FOV = np.load('data/pixels_in_FOV.npy', allow_pickle=True)
        self.probs_in_FOV = [np.sum(self.m_origin[pixels]) for pixels in self.pixels_in_FOV]
        # 计算被太阳遮挡的网格
        sun_pos_start = sun_position(utc_to_julian_day(utc_start))  # 计算开始时刻太阳的位置
        pix_indices = np.arange(self.npix)
        target_pos = hp.pix2ang(self.nside_std, pix_indices, lonlat=True)
        self.pix_covered_by_sun = pix_covered_by_sun(target_pos, sun_pos_start, self.args.sun_avoid_angle,
                                               self.args.Omega[0], self.args.i[0], degree=True)
        # 可视化太阳遮挡情况
        pixel_indices = np.where(~self.pix_covered_by_sun)[0]
        # visualize_selected_pixel(self.m_origin, pixel_indices, self.nside_std)

        # 计算状态
        if self.args.simple:
            self.state_task = np.sort(self.m_origin[pixel_indices])[::-1][:np.round(self.args.factor*self.n_sat).astype(int)]  # 不被太阳遮挡的最大概率网格
            sorted_indices = np.argsort(-self.m_origin[pixel_indices])[:np.round(self.args.factor * self.n_sat).astype(int)]
            self.sorted_pixel_indices = np.array(pixel_indices)[sorted_indices]  # 对应的网格索引
        else:
            self.state_task = np.zeros(self.state_task.shape)
            self.state_task = self.m_origin
        self.state_sat = np.zeros(self.state_sat.shape)
        self.state_sat[0:self.n_sat] = self.args.mean_ano
        self.t = 0

        # 计算地球遮挡时候的纬度幅角
        a = self.args.Re + self.args.h  # 半径=地球半径+轨道高度
        Omega = self.args.Omega  # 升交点赤经
        omega = self.args.omega  # 近地点幅角
        mean_ano = self.args.mean_ano  # 平近点角
        e = self.args.e  # 离心率
        i = self.args.i  # 轨道倾角
        theta = mean_to_true_anomaly(mean_ano, e)  # 真近点角
        u = omega + theta  # 纬度幅角
        self.u = u
        self.a = a
        shadow_u_start = np.zeros((self.n_sat, self.npix))
        shadow_u_end = np.zeros((self.n_sat, self.npix))
        for j in range(self.npix):
            ra, dec = hp.pix2ang(nside=self.nside_std, ipix=j, lonlat=True)
            for k in range(self.n_sat):
                result = calculate_shadow_u(ra, dec, a, Omega[k], i[k], u[k], degree=True)
                shadow_u_start[k, j] = result[0]
                shadow_u_end[k, j] = result[1]
        self.shadow_u_start = shadow_u_start
        self.shadow_u_end = shadow_u_end
        # m = rotate_to_origin(m, nside_std=self.nside_std)  # 概率最高点平移到原点

        # 转换为二维图像
        # pmap = smu.interpolate_sky_map(m, self.nside_std, image=True)

        # 随任务修改 TODO
        self.state = dict({'task': self.state_task, 'sat': self.state_sat, 't': self.t})  # 新的skymap的prob

        self.current_step = 0
        info = dict({'origin_m': self.m_origin, 'm': self.m,
                     'pix_covered_by_sun': self.pix_covered_by_sun,
                     'shadow_u_start': self.shadow_u_start,
                     'shadow_u_end': shadow_u_end})
        return self.state, info

    def step(self, action, next):
        """
        action(array): 每个卫星观测nested编号的网格
        next: 是否更新状态
        """
        reward = 0
        action = self.sorted_pixel_indices[action]
        # action = hp.nest2ring(self.nside_std, action)  # TODO 转为ring结构
        action_ang = np.array(hp.pix2ang(nside=self.nside_std, ipix=action, lonlat=True))
        # action = scale_action(action, self.action_details)  # 将动作映射成赤经赤纬
        action_ang = action_ang.reshape(2, -1).T
        ipix_sat = []
        ipix_m = []
        local_reward = 0
        # 计算卫星视场内网格集合
        for i in action_ang:  # i为卫星对应观测的网格中心索引
            ra, dec = i[0], i[1]
            # 求以(ra,dec)为视场中心，以radius为半径视场内网格集合
            pix = hp.ang2pix(self.args.nside_std, ra, dec, False, True)
            ipix_disc = self.pixels_in_FOV[pix]
            prob_sum = self.probs_in_FOV[pix]
            # 效率比较低，不用这种方法
            # ipix_disc, ipix_prob, prob_sum = integrated_prob_in_a_circle(ra, dec, self.args.radius, self.m, self.nside_std)
            # ipix_total = np.append(ipix_total, [ipix_disc])
            ipix_sat.append(ipix_disc.tolist())
            ipix_m.append(prob_sum)
        ipix_m = np.array(ipix_m)

        # 随任务修改 TODO
        # ipix_total = np.unique(ipix_total).astype(np.int64)  # 去重

        # 计算当前时刻卫星所在位置
        a = self.a
        G = 6.67e-11  # 万有引力常数（单位：m^3/kg/s^2）
        M = 5.965e24  # 地球质量（单位：kg）
        v = np.sqrt(G * M / a)  # 卫星线速度m/s
        omega_v = np.degrees(v / a)  # 卫星角速度
        u_now = self.state['sat']
        # 计算太阳遮挡情况
        consflag_sun = False if np.any(self.pix_covered_by_sun[action]) else True
        # 计算到下次决策时刻，卫星被地球遮挡情况
        consflag_earth = False  # 是否满足地球可见约束
        left_u = []
        for idx, i in enumerate(action):
            start = self.shadow_u_start[idx][i]
            end = self.shadow_u_end[idx][i]
            if start > end:
                end = 360 + end
            if start <= u_now[idx] <= end:
                left_u.append(0)
            elif u_now[idx] > end:
                left_u.append(360-u_now[idx]+start)
            else:
                left_u.append(start-u_now[idx])
        consflag_earth = np.array(left_u) > 0
        # if np.all(np.array(left_u) > 0):
        #     consflag_earth = True

        # 计算奖励
        consflag = consflag_earth & consflag_sun
        t = left_u / omega_v
        unique_values_filter, unique_indices = np.unique(action[consflag], return_index=True)
        consflag_filter = consflag[unique_indices]
        local_reward = np.sum(ipix_m[unique_indices] * consflag_filter.astype(int)) * np.min(t) / 60
        # unique_indices = np.unique(action).tolist()
        # local_reward = np.sum(ipix_m[unique_indices]) * np.min(t) / 60 if np.all(consflag) else 0
        # reward = reward + local_reward
        if next:
            self.t = self.t + np.min(t)
            # 更新状态
            self.state['sat'][0:self.n_sat] = (np.min(t) * omega_v + self.state['sat'][0:self.n_sat]) % 360  # 当前纬度幅角
            self.state['sat'][self.n_sat:] = action  # 当前观测视场中心网格索引
            self.state['t'] = self.t
            # 更新步数
            self.current_step += 1

        # 判断是否达到终止条件 TODO
        if self.current_step >= self.max_num_steps:  # 终止条件
            # self.state, info = self.reset()
            return self.state, local_reward, True, consflag  # 已经终止
        return self.state, local_reward, False, consflag  # 未终止

    def render(self, mode):
        pass

    def close(self):
        pass




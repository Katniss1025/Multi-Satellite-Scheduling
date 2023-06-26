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


def scale_action(action, action_space):
    ''' 将tanh激活的action线性映射到赤经和赤纬 [-1,1]->[0,360]/[-90,90]
    Args:
        action_tahn(tensor): 被tanh激活的action
        action_space(dict): 自定义的动作空间的信息
    Returns:
        action_scaled(tensor): 映射后的action
    '''
    action_tahn = torch.tanh(action)
    action_scaled = (action_tahn + 1) / 2 * (action_space['high'] - action_space['low']) + action_space['low']
    return action_scaled


class MultiSatelliteEnv(gym.Env):
    """ 环境类
    根据gym接口写的自定义环境
    环境中有一个网格，agent要填满网格。填空白格子得到+1奖励，重复填格子得到-1奖励
    """
    def __init__(self, n_sat, n_pix, t, state_size, action_space, num_epoch_steps):
        """ 初始化
        n_sat(int): 卫星数量
        n_pix(int): 像素数
        t(float): 任务时长
        action_space(list): 动作空间, 离散: n_sat * [npix], 连续:
        state_size(dict): 状态空间 keys={'sat','task',''}  state_size = {'task':[n_pix, 2]}
        num_epoch_steps(int): 一个回合最大时间步数
        """
        super(MultiSatelliteEnv, self).__init__()
        self.max_num_steps = num_epoch_steps  # 一回合最大时间步数
        self.current_step = 0  # 当前时间步

        # Define action and observation space. They must be gym.spaces objects
        n_actions = n_sat
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # TODO
        # self.state_sat = [0] * state_size['sat']  # 卫星状态
        self.state_task = torch.zeros(181, 361).to(device)  # 任务状态
        # self.state_inr = [0] * state_size['inr']  # 中断状态
        # self.state = np.concatenate([np.array(self.state_sat), np.array(self.state_task), np.array(self.state_inr)])  # 拼接卫星、任务、中断状态
        self.state = self.state_task  # 拼接卫星、任务状态空间

        # 定义动作空间.
        # Case1: 连续动作空间，分配赤经和赤纬
        self.action_space = spaces.Box(low=action_space['low'], high=action_space['high'],
                                       shape=action_space['shape'], dtype=float)
        # Case2: 离散动作空间，分配网格的索引.

        # 定义观测空间 TODO
        self.observation_space = spaces.Box(low=0, high=1, shape=self.state.shape)  # 概率，观测时长，中断时长

    def seed(self, seed):
        random.seed(seed)

    def reset(self):
        """ 重置环境，新的爆发事件
        Important: the observation must be a numpy array
        """
        # 生成新的事件
        from skymap.DataReinforcement import data_reinforcement_by_rotate
        from skymap import SkyMapUtils as smu
        self.state = np.zeros(self.state.shape)  # [len(m), 2]
        m, m_rotated_area_90, m_rotated_area_50 = data_reinforcement_by_rotate()
        # smu.visualize(m)
        from skymap.transUtils import rotate_to_origin
        m = rotate_to_origin(m, nside_std=128)

        # 转换为二维图像
        # smu.visualize(m)
        pmap = smu.interpolate_sky_map(m, 128, image=False)

        # 随任务修改 TODO
        self.state = pmap  # 新的skymap的prob

        # 对概率这一维度进行标准化处理
        self.state = (self.state-np.min(self.state)) / (np.max(self.state)-np.min(self.state))
        m = (m-np.min(m)) / (np.max(m) - np.min(m))

        self.current_step = 0
        # if np.any(np.isnan(self.state)):
        #     pdb.set_trace()
        return self.state, m

    def step(self, action, m, action_space):
        reward = 0
        ipix_total = np.array([])
        self.state = self.state  # TODO
        action = scale_action(action, action_space)
        action = action.reshape(2, -1).T
        # 更新状态
        # 计算
        nside = get_args().nside_std
        for i in action:  # i为卫星对应观测的网格中心索引
            ra, dec = i[0], i[1]

            # 求以(ra,dec)为视场中心，以radius为半径视场内网格集合
            radius = 2.5
            ipix_disc, ipix_prob, prob_sum = integrated_prob_in_a_circle(ra, dec, radius, m, nside)
            # 求所有卫星视场内的网格集合
            ipix_total = np.append(ipix_total, ipix_disc)

        # 随任务修改 TODO
        ipix_total = np.unique(ipix_total).astype(np.int64)  # 去重
        # self.state[int(len(self.state)/2)+ipix_total] += 10  # 时长状态

        # 更新reward
        reward += np.sum(m[ipix_total] )  # 随任务修改 当前reward为 概率*时长

        # 更新步数
        self.current_step += 1

        # 判断是否达到终止条件 TODO
        if self.current_step >= self.max_num_steps:  # 终止条件
            # self.state, m = self.reset()
            return self.state, reward, True, {}  # 已经终止
        return self.state, reward, False, {}  # 未终止

    def render(self, mode):
        pass

    def close(self):
        pass




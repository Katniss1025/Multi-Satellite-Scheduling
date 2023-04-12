import random
from skymap.probUtils import integrated_prob_in_a_circle
import gym
import numpy as np
from gym import spaces
import healpy as hp

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

        # self.state_sat = [0] * state_size['sat']  # 卫星状态
        self.state_task = np.zeros(state_size['task'])  # 任务状态
        # self.state_inr = [0] * state_size['inr']  # 中断状态
        # self.state = np.concatenate([np.array(self.state_sat), np.array(self.state_task), np.array(self.state_inr)])  # 拼接卫星、任务、中断状态
        self.state = self.state_task  # 拼接卫星、任务状态空间

        # 定义动作空间. Case1: 连续动作空间，分配赤经和赤纬, Case2: 离散动作空间，分配网格的索引.
        # self.action_space = spaces.Box(low=np.array([[0.0, -90.0]]*n_sat), high=np.array([[360.0, 90.0]]*n_sat), shape=(n_sat, 2))  # Case1
        self.action_space = spaces.MultiDiscrete(action_space)  # Case2

        # 定义观测空间
        self.observation_space = spaces.Box(low=0, high=np.array([[1, t]] * n_pix), shape=self.state.shape)  # 概率，观测时长，中断时长

    def seed(self, seed):
        random.seed(seed)

    def reset(self):
        """ 重置环境
        Important: the observation must be a numpy array
        """
        self.state = [0] * len(self.state)
        self.state[random.randrange(0, len(self.state))] = 1
        self.current_step = 0
        return self.state

    def step(self, action):
        reward = 0
        for i in action:  # i为卫星对应观测的网格中心索引
            ra, dec = hp.pix2ang(nside=128, ipix=i, lonlat=True)
            ipix_disc, ipix_prob, prob_sum = integrated_prob_in_a_circle(ra, dec, 2.5, self.state[:, 0])


            reward += self.state[:, 0] * self.state[:, 1]
        self.current_step += 1
        if self.current_step >= self.max_num_steps:
            return self.reset(), reward, True, {}
        return self.state, reward, False, {}

    def render(self, mode):
        pass

    def close(self):
        pass

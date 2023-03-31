import random
import numpy as np
import gym
from gym import spaces
import torch

class MultiSatEnv(gym.Env):
  """
  根据gym接口写的自定义环境
  环境中多个卫星，每完成一个任务给定一定的奖励
  """

  def __init__(self, n_sat, n_actions, action_space, num_epoch_steps, n_state):
    super(MultiSatEnv, self).__init__()
    self.max_num_steps = num_epoch_steps
    self.current_step = 0

    # Define action and observation space. They must be gym.spaces objects
    self.state = torch.zeros(n_sat, n_state)
    self.action_space = spaces.MultiDiscrete([n_actions]*n_sat)
    self.observation_space = spaces.Box(low=0, high=1, shape=(n_sat*n_state,), dtype=np.float32)

  def seed(self, seed):
    random.seed(seed)

  def reset(self):
    """
    Important: the observation must be a numpy array
    """
    self.state = torch.zeros(self.state.shape)
    # self.state[random.randrange(0, len(self.state))] = 1
    self.current_step = 0
    return self.state

  def step(self, action, task):
    reward = 0
    self.actions = np.where(action == 1)
    for actionID in self.actions[0]:
      if self.state[actionID] == 0:
        reward += 1
        self.state[actionID] = 1
      elif self.state[actionID] == 1:
        reward -= 1
    self.current_step += 1
    if self.current_step >= self.max_num_steps:
        return self.reset(), reward, True, {}
    return self.state, reward, False, {}

  def render(self, mode):
    pass

  def close(self):
    pass
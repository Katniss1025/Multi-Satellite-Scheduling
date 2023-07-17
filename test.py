from env import MultiSatelliteEnv
import argparse
import utils
import numpy as np


def get_args():
    # 获取yaml参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', dest='config_file', default='config_env.yml')
    parser.add_argument('--dataset-name', dest='dataset_name', default='env')
    args = utils.read_config_file(parser)
    return args


if __name__ == '__main__':
    # 定义参数
    args = get_args()
    n_sat = args.n_sat
    n_pix = 196608
    t = args.t
    state_size = args.state_size

    action_details = {'low': np.concatenate((n_sat * [0.], n_sat * [-90.])),
                    'high': np.concatenate((n_sat * [360.], n_sat * [90.])),
                    'shape': (n_sat * 2,)}

    num_epoch_steps = args.num_epoch_steps

    #
    m = np.array([0] * n_pix)
    env = MultiSatelliteEnv(n_sat, n_pix, t, state_size, action_details, num_epoch_steps, m)
    state, info = env.reset()
    action = env.action_space.sample()
    state, reward, flag, info = env.step(action)

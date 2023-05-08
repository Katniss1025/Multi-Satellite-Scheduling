from env import MultiSatelliteEnv
import argparse
import utils

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
    action_space = n_sat * [n_pix]
    num_epoch_steps = args.num_epoch_steps

    #
    env = MultiSatelliteEnv(n_sat, n_pix, t, state_size, action_space, num_epoch_steps)
    state = env.reset()
    action = env.action_space.sample()
    state, reward, flag, info = env.step(action)
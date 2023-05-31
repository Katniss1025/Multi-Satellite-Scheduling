import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.normal import Normal
import gymnasium as gym
import argparse
import utils
import time


def get_args():
    # 获取yaml参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', dest='config_file', default='config_env.yml')
    parser.add_argument('--dataset-name', dest='dataset_name', default='env')
    args = utils.read_config_file(parser)
    return args


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, env, action_space, num_nn, critic_std, actor_std):
        super().__init__()
        # 卷积层
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=2, kernel_size=3, stride=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3, stride=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=0),
            # nn.ReLU(),
            nn.Flatten(),  # 将卷积层输出的多维数据展平为向量
        )

        self.action_space = action_space
        self.critic = nn.Sequential(  # critic网络，2个线性层，输入尺寸为352，输出尺寸为1
            layer_init(nn.Linear(8, num_nn)),  # np.array(env.observation_space.shape).prod()
            nn.Tanh(),
            layer_init(nn.Linear(num_nn, num_nn)),
            nn.Tanh(),
            layer_init(nn.Linear(num_nn, 1), std=critic_std),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(8, num_nn)),  # np.array(env.observation_space.shape).prod()
            nn.Tanh(),
            layer_init(nn.Linear(num_nn, num_nn)),
            nn.Tanh(),
            layer_init(nn.Linear(num_nn, np.prod(env.action_space.shape)), std=actor_std),  # 输出尺寸为动作空间大小
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(env.action_space.shape)))


    def get_value(self, x):
        ''' 估计状态价值
        Args:
            x: 状态
        Returns:
            critic估计状态价值
        '''
        return self.critic(self.network(x))

    def get_action_and_value(self, x, action=None):
        ''' 选取动作或计算状态价值
        Args:
            x: 状态
            action: 在更新网络参数时输入action
        Returns:
            action: 在采样阶段输出动作
            对数概率
            交叉熵
            状态价值
        '''
        feature = self.network(x)
        action_mean = self.actor_mean(feature)
        action_logstd = self.actor_logstd.expand_as(action_mean)  # 将actor_logstd扩张到action_mean形状
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
            action = scale_action(action, action_space)
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(feature)


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

        #
        # logits = self.actor(self.network(x))  # 尺寸为(1,nsat*动作空间尺寸)
        # split_logits = torch.split(logits, self.action_space, dim=1)  # 拆分logits为块，每块儿为action_space大小
        # multi_categoricals = [Categorical(logits=logits) for logits in split_logits]  # 分类，决定采取哪个动作
        # if action is None:
        #     action = torch.stack([categorical.sample() for categorical in multi_categoricals]).T  # 对每个卫星输出动作的采样结果
        # logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action.T, multi_categoricals)])  # 输出采样的得到动作的对数概率
        # entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals]).T  # 计算交叉熵
        # return action, logprob.sum(0), entropy.sum(0), self.critic(self.network(x))


def train(env, name, action_space, target_kl, minibatch_size, gamma, ent_coef, vf_coef, num_nn, critic_std, actor_std,
          learning_rate, num_epoch_steps, seed, anneal_rate):
    print('开始训练')
    # 配置超参数，这部分超参数固定在算法中
    total_timesteps = 100000  # How many steps you interact with the env
    num_env_steps = 128  # How many steps you interact with the env before an update
    num_update_steps = 4  # How many times you update the neural networks after interation
    gae_lambda = 0.95  # Parameter in advantage estimation
    max_grad_norm = 0.5  # max norm of the gradient vector
    clip_coef = 0.2  # Parameter to clip the (p_new/p_old) ratio

    writer = SummaryWriter('runs/' + name)  # 创建一个基于Tensorboard的writer对象，用于记录训练过程中的数据
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模型和优化器
    agent = Agent(env, action_space, num_nn, critic_std, actor_std).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)  # 定义优化器，优化优化智能体策略，即网络的参数

    # seeding
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  # 使用固定随机数种子，从而保证实验结果可重现

    # Initialize storage for a round
    channels = env.observation_space.shape[2]
    h = env.observation_space.shape[0]
    w = env.observation_space.shape[1]
    obs = torch.zeros((num_env_steps, channels, h, w)).to(device)
    actions = torch.zeros(num_env_steps, np.prod(action_space['shape'])).to(device)
    logprobs = torch.zeros(num_env_steps).to(device)
    rewards = torch.zeros(num_env_steps).to(device)
    dones = torch.zeros(num_env_steps).to(device)
    values = torch.zeros(num_env_steps).to(device)
    next_obs, _ = env.reset()
    next_obs = torch.Tensor(next_obs).permute(2, 0, 1)  # torch.Tensor(env.reset()).to(device)
    next_done = torch.zeros(1).to(device)

    global_step = 0  # 定义全局步数
    cumu_rewards = 0  # 定义累计奖励
    num_rounds = total_timesteps // num_env_steps  # 训练回合数
    for round in range(1, num_rounds+1):  # round不同于回合。 训练过程不以回合存储，以固定时间步存储。
        if anneal_rate:
            frac = 1.0 - (round - 1.0) / num_rounds
            lrnow = frac * learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # action logic
        for step in range(num_env_steps):
            global_step += 1
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():  # 禁止梯度计算，减少内存占用和计算时间
                action, logprob, _, value = agent.get_action_and_value(next_obs.unsqueeze(0))  # 采样历史数据
            action = action.flatten()
            values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # execute the game and log data.
            next_obs, reward, done, _, info = env.step(action.cpu().numpy())  # 执行动作，状态转移，计算奖励
            cumu_rewards += reward  # 累计奖励
            if done == True:  # 回合终止
                writer.add_scalar("cumulative rewards", cumu_rewards, global_step) # 在Tensorboard中记录累计奖励
                print("global step:", global_step, "cumulative rewards:", cumu_rewards)
                cumu_rewards = 0  # 清空累积奖励
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).permute(2, 0, 1).to(device), torch.Tensor([done]).to(device)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs.unsqueeze(0)).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(num_env_steps)):  # 反向遍历
                if t == num_env_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs  # 从[128,181,361]变换为[128,1,181,361]
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + env.action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(num_env_steps)
        clipfracs = []

        for update in range(num_update_steps):  # 反复进行更新
            np.random.shuffle(b_inds)

            clipfracs = []
            # approx_kl = []
            np.random.shuffle(b_inds)  # 打乱每次更新用的样本顺序
            for start in range(0, num_env_steps, minibatch_size):  # 基于小批量的更新
                end = start + minibatch_size  # 小批量数据的结束位置
                mb_inds = b_inds[start:end]  # 当前小批量的样本索引

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                # if args.norm_adv:
                #     mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = - mb_advantages * ratio
                pg_loss2 = - mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # Total loss
                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                # Update the neural networks
                optimizer.zero_grad()  # 梯度清空
                loss.backward()  # 反向传播
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)  # 梯度剪裁，防止梯度爆炸
                optimizer.step()  # 更新参数

            # Annealing the learning rate, if KL is too high
            # TODO

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # 记录指标并关闭writer
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("value_loss", v_loss.item(), global_step)
        writer.add_scalar("policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("entropy", entropy_loss.item(), global_step)
        writer.add_scalar("approx_kl", np.mean(approx_kl.item()), global_step)
        writer.add_scalar("clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("explained_variance", explained_var, global_step)
        writer.add_scalar("mean_value", values.mean().item(), global_step)
    writer.close()
    print('结束训练')


if __name__ == "__main__":

    target_kl = [0.02]
    minibatch_size = [32]  # The batch size to update the neural network
    gamma = [0.9]
    ent_coef = [0.001]  # Weight of the entropy loss in the total loss
    vf_coef = [0.5]  # Weight of the value loss in the total loss
    num_nn = [128]
    critic_std = [1]
    actor_std = [0.01]
    learning_rate = [5e-4]
    env_seed = 12315  # 环境中使用的随机数种子

    args = get_args()
    num_nn = [args.num_nn]
    n_sat = args.n_sat
    n_pix = args.npix
    t = args.t
    state_size = args.state_size


    num_epoch_steps = args.num_epoch_steps
    seed = args.seed
    anneal_rate = True

    env = gym.make("ALE/AirRaid-v5")
    # env.seed(env_seed)

    action_space = {'low': env.action_space.low,
                    'high': env.action_space.high,
                    'shape': env.action_space.shape}

    # 对部分超参数进行网格搜索
    for tk in target_kl:
        for bs in minibatch_size:
            for ga in gamma:
                for ef in ent_coef:
                    for vf in vf_coef:
                        for num in num_nn:
                            for cstd in critic_std:
                                for astd in actor_std:
                                    for lr in learning_rate:
                                        name = 'tk' + str(tk) + '_bs' + str(bs) + '_ga' + str(ga) + '_ef' + str(
                                            ef) + '_vf' + str(vf) + '_num' + str(num) + '_cs' + str(cstd) + '_as' + str(
                                            astd) + '_lr' + str(lr) + time.strftime('%Y%m%d_%H:%M:%S', time.localtime(int(round(time.time()*1000))/1000))
                                        train(env, name, action_space, tk, bs, ga, ef, vf, num, cstd, astd, lr,
                                              num_epoch_steps, seed, anneal_rate)

from math import log
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from Methods.Transfomer import Transformer
from sklearn.preprocessing import MinMaxScaler
from numba import cuda


class MyDataset(Dataset):
    def __init__(self, path):
        data = np.load(path, allow_pickle=True)
        task_num = data['state_batch'].shape[0]
        step_num = data['state_batch'].shape[1]
        self.length = task_num * step_num  # 样本数量=任务数*步数, 把每一次决策当做一个样本

        # 直接把数据集存入内存, 对小数据集, 这样读取速度比较快
        # 存储(动作，状态)对
        self.task_states = torch.zeros(task_num*step_num, data['state_batch'][0,0]['task'].shape[0])
        self.sat_states = torch.zeros(task_num*step_num, data['state_batch'][0,0]['sat'].shape[0])
        self.actions = torch.zeros(task_num*step_num, data['action_batch'][0,0].shape[0])
        self.rewards = torch.zeros(task_num*step_num)
        cnt = 0
        for i in range(task_num):
            for j in range(step_num):
                self.task_states[cnt] = torch.from_numpy(data['state_batch'][i, j]['task'])
                self.sat_states[cnt] = torch.from_numpy(data['state_batch'][i, j]['sat'])
                self.actions[cnt] = torch.from_numpy(data['action_batch'][i, j])
                self.rewards[cnt] = data['reward_batch'][i, j]
                cnt += 1
        # 对天区概率图，可以使用排序结果，或者使用原图
        # self.task_states = self.task_states.argsort(dim=1, descending=True) / self.task_states.shape[1]
        # 天区概率图进行归一化处理
        # scaler = MinMaxScaler()
        # self.task_states = scaler.fit_transform(self.task_states.T).T
        # 计算均值和方差
        self.task_std, self.task_mean = torch.std_mean(self.task_states)
        self.sat_std, self.sat_mean = torch.std_mean(self.sat_states)
        self.action_std, self.action_mean = torch.std_mean(self.actions)
        self.reward_std, self.reward_mean = torch.std_mean(self.rewards)

    def __getitem__(self, idx):
        # Z-score标准化。注意在模拟器中需要反标准化
        task_state = (self.task_states[idx] - self.task_mean) / self.task_std
        sat_state = (self.sat_states[idx] - self.sat_mean) / self.sat_std
        action = (self.actions[idx] - self.action_mean) / self.action_std
        reward = (self.rewards[idx] - self.reward_mean) / self.reward_std
        reward = torch.tensor([reward])  # 将其尺寸从(1,)转为(1,1)
        return task_state, sat_state, action, reward

    def __len__(self):
        return self.length


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc4 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        return x


class MyModel(nn.Module):
    def __init__(
            self,
            sat_state_dim=12,
            d_model=128,
            nhead=2,
            num_encoder_layers=4,
            num_actions=6,
            max_seq_len=800
        ):
        super().__init__()
        self.num_actions = num_actions
        self.pixel_tokenizer = nn.Linear(1, d_model)
        self.sat_state_tokenizer = nn.Linear(sat_state_dim, d_model)
        self.action_tokenizer = nn.Linear(num_actions, d_model)
        self.dummy_token = nn.Parameter(torch.randn(1, d_model))
        self.to_actions = nn.Linear(d_model, num_actions)
        self.transformer = Transformer(
            token_dim=d_model,
            ff_dim=d_model,
            head_dim=d_model//nhead,
            n_heads=nhead,
            n_blocks=num_encoder_layers,
            max_T=max_seq_len,
            drop_p=0.1,
            causal=False,
        )

    def forward(self, task_states, sat_states):
        # 尺寸变换: (batch_size, pixel_num) -> (batch_size, pixel_num, d_model)
        batch_size, pixel_num = task_states.shape
        task_states = task_states.view(batch_size, pixel_num, 1)
        pixel_tokens = self.pixel_tokenizer(task_states)

        # 尺寸变换: (batch_size, state_dim) -> (batch_size, 1, d_model)
        sat_states = sat_states.view(batch_size, 1, -1)
        sat_tokens = self.sat_state_tokenizer(sat_states)

        # 尺寸变换: (1, d_model) -> (batch_size, 1, d_model)
        dummy_token = self.dummy_token.expand((batch_size,) + self.dummy_token.shape)

        input_tokens = torch.cat((pixel_tokens, sat_tokens, dummy_token), dim=1)
        output_tokens = self.transformer(input_tokens)
        pred_actions = self.to_actions(output_tokens[:, -1])
        return pred_actions


def train(epoch, train_loader, model, optimizer, scheduler, criterion):
    record_loss = 0
    for batch_id, (task_states, sat_states, actions, rewards) in enumerate(train_loader):
        model.train()  # 将模型调到训练模式

        # 将数据转移到GPU
        task_states = task_states.to(device)
        sat_states = sat_states.to(device)
        actions = actions.to(device)

        pred_actions = model(task_states, sat_states) # 使用transformer
        # pred_actions  = model(torch.cat((task_states, sat_states), dim=1)) # 使用MLP
        action_loss = criterion(pred_actions, actions)
        optimizer.zero_grad()  # 历史梯度清零
        action_loss.backward()
        optimizer.step()
        record_loss += action_loss.item()

        if batch_id % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_id * len(actions),
                len(train_loader.dataset),
                100. * batch_id / len(train_loader),
                action_loss.item()))
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, record_loss / len(train_loader)))
    scheduler.step()


if __name__ == '__main__':
    device = torch.device('cuda:0')  # 选择GPU 0
    epoch_num = 1000
    dataset = MyDataset('data.npz')
    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0,  # 没有使用多进程同时加载, 但你可以尝试
    )
    # 使用Transformer
    # model = MyModel().to(device)

    # 使用MLP
    model = torch.nn.Sequential(
        nn.Linear(768+12, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 6),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4, fused=True)
    # 学习率会在训练过程中逐渐衰减
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch_num)
    criterion = torch.nn.L1Loss().to(device)  # L1损失对异常点不敏感
    for epoch in range(epoch_num):
        train(epoch, loader, model, optimizer, scheduler, criterion)

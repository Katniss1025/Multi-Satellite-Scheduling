import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision import datasets
from torchvision import transforms
from torchsummary import summary
from IPython.display import Image

import h5py
from torch.utils.data import Dataset
from skymap.SkyMapUtils import interpolate_sky_map, plot_2d_image
import pdb
import os
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bs = 32


def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.normal(m.weight, mean=0, std=0.5)
    if type(m) == nn.Linear:
        torch.nn.init.uniform(m.weight, a=-0.1, b=0.1)
        m.bias.data.fill_(0.01)


class MyDataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path

        # 打开HD5文件
        with h5py.File(file_path, 'r') as hf:
            self.loaded_array = hf['data'][:]

        # 读取矩阵数量
        self.num_matrices = len(self.loaded_array)

    def __len__(self):
        return self.num_matrices

    def __getitem__(self, index):
        # 读取指定索引的概率
        m = self.loaded_array[index]  # 概率
        # 插值为二维概率
        # pmap = interpolate_sky_map(m, 128, image=False)
        # # 归一化
        # pmap = (pmap - np.min(pmap)) / (np.max(pmap) - np.min(pmap))
        # 转换为PyTorch张量
        tensor = torch.from_numpy(m).unsqueeze(0).to(device)

        return tensor


class Flatten(nn.Module):
    ''' 展平
    '''
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    ''' 逆展平
    '''
    def forward(self, input_, size=1024):
        return input_.view(input_.size(0), int(size/4), 2, 2)


class VAE(nn.Module):
    def __init__(self, image_channels=1, h_dim=1024, z_dim=32):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=5, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=5, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=5, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((2, 2)),
            nn.ReLU(),
            Flatten()
        )

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((181, 361)),
            # nn.Sigmoid(),
            nn.Tanh(),
        )

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size())
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar


def save_image(tensor_batch, filepath):
    batch_size = tensor_batch.shape[0]
    plt.close()
    fig, axes = plt.subplots(int(batch_size/8), 8, sharex=True, sharey=True, figsize=(15, 8))
    pmaps = tensor_batch.clone().detach().cpu()  # 克隆张量并移动到CPU上
    k = 0
    for i in range(int(batch_size/8)):
        for j in range(8):
            pmap = pmaps[k].squeeze(0)
            axes[i, j].imshow(pmap, extent=[360, 0, -90, 90], cmap=plt.cm.RdYlBu, origin='lower', aspect='auto')
            k = k + 1
    plt.savefig('real_image.png')
    plt.show()


def loss_fn(recon_x, x, mu, logvar):
    # BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    BCE = F.mse_loss(recon_x, x, size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD, BCE, KLD


if __name__ == '__main__':
    root = (os.path.abspath(os.path.join(os.getcwd(), "../")))
    dataset = MyDataset(root+'/data/skymaps20000.h5')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    # len(dataset), len(dataloader)
    # Fixed input for debugging
    fixed_x = next(iter(dataloader))  # 获取下一个批次的数据

    image_channels = fixed_x.size(1)
    vae = VAE(image_channels=image_channels).to(device)
    vae.apply(init_weights)
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-4)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    epochs = 1
    for epoch in range(epochs):
        for idx, images in enumerate(dataloader):
            images_log = np.log10(images + 1e-100)  # 取对数, 原数据中大多取值很小，在e-N级别
            images_norm = (images_log - images_log.mean()) / (images_log.max() - images_log.min())  # 标准化为[0,1]
            recon_images, mu, logvar = vae(images_norm.float())
            loss, bce, kld = loss_fn(recon_images.float(), images.float(), mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            to_print = "Epoch[{}/{}] Batch[{}/{}] Loss: {:.3f} {:.3f} {:.3f}".format(epoch + 1, epochs,
                                                                                      idx, len(dataloader),
                                                                         loss.data.item() / bs, bce.data.item() / bs,
                                                                        kld.data.item() / bs)
            print(to_print)
        # scheduler.step()
    # notify to android when finished training
    # notify(to_print, priority=1)

    torch.save(vae.state_dict(), 'vae.torch')

    # test
    images = next(iter(dataloader))  # 获取下一个批次的数据
    save_image(images, 'real_image.png')
    pred_images, _, _ = vae(images.float())
    save_image(fixed_x, 'pred_image.png')








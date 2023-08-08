from skymap.DataReinforcement import data_reinforcement_by_rotate
from skymap.skymap_sample import generate_skymap_by_gaussian
import numpy as np
from utils import get_args
from tqdm import tqdm
import h5py
from skymap.SkyMapUtils import visualize


args = get_args()
skymap_num = 10000
rot_num = skymap_num * 1  # 通过旋转产生的skymap数量
ge_num = skymap_num * 0  # 通过高斯估计产生的skymap数量
current_iteration = 0

# 初始化进度条
progress_bar = tqdm(total=skymap_num)

skymaps = np.zeros((skymap_num, args.npix))

while current_iteration < ge_num:
    prob = generate_skymap_by_gaussian()
    if prob is False:
        continue
    else:
        skymaps[current_iteration, :] = prob
        # print(prob.sum())
        # visualize(prob)
        current_iteration = current_iteration + 1
        # 更新进度条
        progress_bar.update(1)

while current_iteration < skymap_num:
    m, m_rotated_area_90, m_rotated_area_50 = data_reinforcement_by_rotate()  # 通过旋转生成新的事件
    # pmap = smu.interpolate_sky_map(m, 128, image=False)
    skymaps[current_iteration, :] = m
    # print(m.sum())
    current_iteration = current_iteration + 1
    # 更新进度条
    progress_bar.update(1)

# 结束进度条
progress_bar.close()

# skymaps保存到本地
def write_large_array(filename, large_array):
    with h5py.File(filename, 'w') as hf:
        data = hf.create_dataset('data', shape=large_array.shape,
                                 dtype=large_array.dtype)

        # 使用 tqdm 显示进度条
        with tqdm(total=large_array.shape[0], ncols=80) as pbar:
            for i in range(large_array.shape[0]):
                # 写入数据到 HDF5 数据集
                data[i] = large_array[i]
                # 更新进度条
                pbar.update(1)

write_large_array('data/skymaps_by_rotation'+str(skymap_num)+'.h5', skymaps)

# 从 HDF5 文件中读取数组
with h5py.File('data/skymaps_by_rotation'+str(skymap_num)+'.h5', 'r') as hf:
    loaded_array = hf['data'][:]

loaded_array





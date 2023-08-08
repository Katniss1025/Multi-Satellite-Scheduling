import random
from utils import get_args
import numpy as np
from env import MultiSatelliteEnv
from utils import calculate_shadow_u
from tqdm import tqdm
from skymap.SkyMapUtils import visualize_selected_pixel
import time
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import h5py
from emailReminder import send_email
from datetime import datetime


def ABC(args, env, modified):
    zero_map_flag = False
    NP = 200
    FoodNumber = int(NP / 2)
    limit = 50
    maxCycle = 200
    runtime = 1
    D = args.n_sat
    ub = np.ones(nsat) * (npix-1)
    lb = np.zeros(nsat)
    Foods = np.zeros((FoodNumber, D))  # size:(FoodNumber, D)
    if modified:
        p = 0.1
        index = [i for i in range(1, maxCycle+1)]
        r = [(ind/maxCycle) ** (-1.5) - 1 for ind in index]
        T = int(p * NP)

    for run in range(runtime):
        # 随机生成初始解
        ObjVal = np.zeros(FoodNumber)
        abc_ObjVals = np.zeros(maxCycle)
        trial = np.zeros(FoodNumber)
        flag = np.zeros(FoodNumber, dtype=bool)
        GlobalFlags = np.zeros(maxCycle, dtype=bool)

        for i in range(FoodNumber):
            consflag = False
            while not consflag:
                action = env.action_space.sample()
                state, local_reward, terminated, consflag = env.step(action, next=False)  # 不更新状态，仅获取动作价值
            Foods[i] = action
            ObjVal[i] = local_reward
            flag[i] = consflag
        Foods = Foods.astype(int)

        # 记录当前最优解
        BestInd = np.argmax(ObjVal)
        GlobalMax = ObjVal[BestInd]
        GlobalParams = Foods[BestInd]

        pbar = tqdm(total=maxCycle, desc='Process Bar')
        iter = 0
        while iter < maxCycle:
            # employed bee phase
            for i in range(FoodNumber):
                Param2Change = np.random.randint(D)
                neighbor = np.random.randint(FoodNumber)
                while i == neighbor:
                    neighbor = np.random.randint(FoodNumber)
                sol = Foods[i].copy()
                if not modified:
                    value = round(Foods[i, Param2Change] + 2*(np.random.rand()-0.5)*(Foods[i, Param2Change] - Foods[neighbor, Param2Change]))
                else:
                    sorted_indices = np.argsort(ObjVal)[::-1]
                    max_T_indices = sorted_indices[0:T]  # ObjVal最高的T个下标
                    eli_ind = np.random.randint(T)
                    value = round(Foods[i, Param2Change] + 2*(np.random.rand()-0.5)*(Foods[max_T_indices[eli_ind], Param2Change] - Foods[i, Param2Change]))
                sol[Param2Change] = value
                # shift onto the boundaries
                ind = np.where(sol < lb)
                sol[ind] = lb[ind]
                ind = np.where(sol > ub)
                sol[ind] = ub[ind]
                sol = sol.astype(int)
                # evaluate new solution
                state, ObjvalSol, terminated, consflag = env.step(sol, next=False)
                # greedy selection
                if ObjvalSol >= ObjVal[i]:
                    Foods[i] = sol
                    flag[i] = consflag
                    ObjVal[i] = ObjvalSol
                    trial[i] = 0
                else:
                    trial[i] += 1

            # calculate probabilities
            if max(ObjVal) != 0:
                prob = 0.9 * (ObjVal/max(ObjVal)) + 0.1
            else:
                prob = np.zeros_like(ObjVal) + 0.1
                zero_map_flag = True
            # onlooker bee phase
            i = 0
            t = 0
            while t < FoodNumber:
                if np.random.rand() < prob[i]:
                    t += 1
                    Param2Change = np.random.randint(D)
                    neighbor = np.random.randint(FoodNumber)
                    while neighbor == i:
                        neighbor = np.random.randint(FoodNumber)
                    if not modified:
                        sol = Foods[i].copy()
                        value = round(Foods[i, Param2Change] + 2 * (np.random.rand() - 0.5) * (
                                    Foods[i, Param2Change] - Foods[neighbor, Param2Change]))
                    else:
                        dist = np.linalg.norm(Foods - Foods[i], axis=1)
                        mdm = np.sum(dist) / (FoodNumber - 1)
                        ind = np.where(dist <= r[iter] * mdm)[0]
                        bestInd = np.argmax(ObjVal[ind])
                        bestInd = ind[bestInd]
                        sol = Foods[bestInd].copy()
                        value = round(sol[Param2Change] + 2 * (np.random.rand()-0.5) * (sol[Param2Change]-Foods[neighbor, Param2Change]))

                    sol[Param2Change] = value
                    # shift onto the boundaries
                    ind = np.where(sol < lb)
                    sol[ind] = lb[ind]
                    ind = np.where(sol > ub)
                    sol[ind] = ub[ind]
                    sol = sol.astype(int)
                    # evaluate new solution
                    state, ObjvalSol, terminated, consflag = env.step(sol, next=False)
                    # greedy selection
                    x = i if not modified else bestInd
                    if ObjvalSol >= ObjVal[x]:
                        Foods[x] = sol
                        flag[x] = consflag
                        ObjVal[x] = ObjvalSol
                        trial[x] = 0
                    else:
                        trial[x] += 1
                i += 1
                if i == FoodNumber:
                    i = 0

            # Memorize the best food source ever
            ind = np.argmax(ObjVal)
            if ObjVal[ind] >= GlobalMax:
                GlobalMax = ObjVal[ind]
                GlobalFlag = flag[ind]
                GlobalParams = Foods[ind]

            # scout bee phase
            ind = np.where(trial >= limit)
            for i in ind:
                consflag = False
                while not consflag:
                    action = env.action_space.sample()
                    state, local_reward, terminated, consflag = env.step(action, next=False)  # 不更新状态，仅获取动作价值
                Foods[i] = action
                ObjVal[i] = local_reward
                flag[i] = consflag
                trial[i] = 0
            # print('iter={}, ObjVal={}, flag={}'.format(iter, GlobalMax, GlobalFlag))
            abc_ObjVals[iter] = GlobalMax
            GlobalFlags[iter] = GlobalFlag
            iter += 1
            pbar.set_postfix({'ObjVal': GlobalMax, 'Algorithm': 'DirectABC' if modified else 'BasicABC'})
            pbar.update(1)
            pbar.refresh()
        print('最优解为:{}'.format(GlobalParams))
        pbar.close()
        info = {'zero_map': zero_map_flag}
        return GlobalParams, abc_ObjVals, info


def write_large_array(action_batch, state_batch, reward_batch, start, end):
    filename = f"data/DirectABC_output/DirectABC_{start}_{end}.npz"
    np.savez(filename, action_batch=action_batch, state_batch=state_batch, reward_batch=reward_batch)


if __name__ == '__main__':
    # 获取当前时间
    current_time = datetime.now()
    # 格式化当前时间为字符串
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    # 定义参数
    args = get_args()
    sender = args.sender  # 发送方的电子邮件地址
    receiver = args.receiver  # 接收方的电子邮件地址
    subject = args.subject  # 邮件主题
    nsat = args.n_sat
    npix = args.npix
    nside_std = args.nside_std
    num_epoch_steps = args.num_epoch_steps
    m = np.array([0] * npix)

    # data = np.load("data/DirectABC_0_30.npz", allow_pickle=True)
    # 读取数据集
    with h5py.File('data/skymaps_by_rotation10000.h5', 'r') as hf:
        loaded_array = hf['data'][:]

    action_batch = []
    state_batch = []
    reward_batch = []
    zero_maps = []
    iteration = 0
    start = 640
    end = start + 32 * 20
    for idx, i in enumerate(loaded_array[start:end]):
        env = MultiSatelliteEnv(num_epoch_steps, i)
        # 用搜索算法求解
        action_episode = []
        state_episode = []
        reward_episode = []
        terminated = False
        # 随机生成动作
        # for i in range(5):
        #     consflag = False
        #     while not consflag:
        #         action = env.action_space.sample()
        #         state, local_reward, terminated, consflag, info = env.step(action, next=False)
        #     actions.append(action)
        #     rewards.append(local_reward)
        # 使用蜂群算法生成动作
        action_num = 0
        state, info = env.reset()
        while not terminated:
            mabc_action, mabc_ObjVals, info = ABC(args, env, modified=True)
            # abc_action, abc_ObjVals, info = ABC(args, env, modified=False)
            action_episode.append(mabc_action)
            state_episode.append(state)
            # visualize_selected_pixel(origin_m, abc_action, nside=nside_std)
            state, local_reward, terminated, consflag = env.step(mabc_action, next=True)
            reward_episode.append(local_reward)
            print('任务{}:执行第{}个动作后，当前运行时间为{}秒'.format(idx+start, action_num+1, state['t']))
            action_num += 1
        if info['zero_map']:
            zero_maps.append(start+idx)
            np.savez('data/DirectABC_output/zero_maps.npz', zero_maps=np.array(zero_maps))
            print('zero map: {}'.format(zero_maps))
        action_batch.append(action_episode)
        state_batch.append(state_episode)
        reward_batch.append(reward_episode)
        iteration += 1
        if iteration % 32 == 0:
            startEvent = start + (int(iteration/32)-1) * 32
            endEvent = startEvent + 32 - 1
            write_large_array(action_batch, state_batch, reward_batch, startEvent, endEvent)
    print('zero map: {}'.format(zero_maps))
    message = "您于{}开始的阿里云任务{}-{}已完成，请注意查看结果。".format(formatted_time, start, end-1)  # 邮件内容
    send_email(sender, receiver, subject, message)












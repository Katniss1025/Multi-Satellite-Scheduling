import random
from utils import get_args
import numpy as np
from env import MultiSatelliteEnv
from utils import calculate_shadow_u
from tqdm import tqdm
from skymap.SkyMapUtils import visualize_selected_pixel
import time
import h5py
from emailReminder import send_email
from datetime import datetime
from multiprocessing import Pool
import multiprocessing
import healpy as hp
from skymap.SkyMapUtils import read_a_skymap
from utils import sun_position, utc_to_julian_day, pix_covered_by_sun, get_utc_start_end_of_week
import pandas as pd


def greedy_init(args, env, state, info):
    '''
    Returns:
        action(list): nested排序的网格索引
    '''
    nsat = args.n_sat
    npix = args.npix
    m = state['task']
    if not args.simple:
        pix_indices = np.arange(npix)  # ring
    else:
        pix_indices = env.sorted_pixel_indices
    # 计算被太阳遮挡的网格
    covered_by_sun = info['pix_covered_by_sun']
    covered_by_sun = [0 if x else 1 for x in covered_by_sun]
    shadow_u_start = info['shadow_u_start']
    shadow_u_end = info['shadow_u_end']

    left = np.zeros_like(shadow_u_end)
    u_now = state['sat'][:nsat]
    for sat in range(nsat):
        for pix in pix_indices:
            start = shadow_u_start[sat][pix]
            end = shadow_u_end[sat][pix]
            if start > end:
                end = 360 + end
            if start <= u_now[sat] <= end:
                left[sat][pix] = 0
            elif u_now[sat] > end:
                left[sat][pix] = 360 - u_now[sat] + start
            else:
                left[sat][pix] = start - u_now[sat]
    value = left[:, pix_indices] * m * np.array(covered_by_sun)[pix_indices]
    action = []
    for sat in range(nsat):
        for i in range(1, npix+1):
            ind = np.argsort(value[sat])[-i]
            if ind not in action:
                action.append(ind)
                break
    # nested
    # action = hp.ring2nest(args.nside_std, action)
    return action


def ABC(args, env, state, info):
    zero_map_flag = False
    NP = args.NP
    FoodNumber = int(NP / 2)
    limit = args.limit
    maxCycle = args.maxCycle
    runtime = args.runtime
    D = args.n_sat
    nsat = args.n_sat
    npix = args.npix
    tqdm_show = args.qtdm_show
    if args.simple:
        ub = np.ones(nsat) * (nsat * args.factor - 1)
        lb = np.zeros(nsat)
    else:
        ub = np.ones(nsat) * (npix-1)
        lb = np.zeros(nsat)
    Foods = np.zeros((FoodNumber, D))  # size:(FoodNumber, D)
    modified = args.modified
    runParameters = np.zeros((runtime, D), dtype=int)
    runObjVals = np.zeros(runtime)
    # greedy_init = args.greedy_init
    simple = args.simple

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
        flag = np.zeros((FoodNumber, nsat), dtype=bool)
        GlobalFlags = np.zeros((maxCycle,nsat), dtype=bool)

        # 如果使用贪心算法，就令一半的解为贪心解，作为初始解
        if args.greedy_init:
            greedy_sol = greedy_init(args, env, state, info)
            _, local_reward, _, consflag = env.step(greedy_sol, next=False)
            Foods[:int(FoodNumber/2)] = greedy_sol
            ObjVal[:int(FoodNumber/2)] = local_reward
        start = int(FoodNumber/2) if args.greedy_init else 0
        for i in range(start, FoodNumber):
            consflag = False
            cnt = 0
            best_reward = 0
            while not np.all(consflag):
                action = env.action_space.sample()
                state, local_reward, terminated, consflag = env.step(action, next=False)  # 不更新状态，仅获取动作价值
                if local_reward > best_reward:
                    best_action = action
                    best_reward = local_reward
                cnt += 1
                if cnt > 10000:  # 找不到满足约束的解
                    print('找不到满足约束的解')
                    break
            Foods[i] = best_action
            ObjVal[i] = best_reward
            flag[i] = consflag
        Foods = Foods.astype(int)

        # shuffle
        indices = np.random.permutation(Foods.shape[0])
        Foods = Foods[indices]
        ObjVal = ObjVal[indices]

        # 记录当前最优解
        BestInd = np.argmax(ObjVal)
        GlobalMax = ObjVal[BestInd]
        GlobalParams = Foods[BestInd]

        if tqdm_show:
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
                consflag = np.zeros(nsat, dtype=bool)
                while not np.all(consflag):
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
            if tqdm_show:
                pbar.set_postfix({'ObjVal': GlobalMax, 'Algorithm': 'DirectABC' if modified else 'BasicABC'})
                pbar.update(1)
                pbar.refresh()
        # print('最优解为:{}'.format(GlobalParams))
        if tqdm_show:
            pbar.close()
        abc_info = {'zero_map': zero_map_flag}
        runParameters[run] = GlobalParams
        runObjVals[run] = GlobalMax

    ind = np.argmax(runObjVals)
    bestSol = runParameters[ind]
    bestObj = runObjVals[ind]
    return bestSol, bestObj, abc_info


def write_large_array(action_batch, state_batch, reward_batch, start, end):
    filename = f"data/DirectABC_output/DirectABC_{start}_{end}.npz"
    np.savez(filename, action_batch=action_batch, state_batch=state_batch, reward_batch=reward_batch)


def run_task(loaded_array, data, args):
    # 数据处理和蜂群算法运行代码
    # 返回结果
    nsat = args.n_sat
    npix = args.npix
    nside_std = args.nside_std
    num_epoch_steps = args.num_epoch_steps
    m = np.array([0] * npix)
    action_batch = []
    state_batch = []
    reward_batch = []
    zero_maps = []
    iteration = 0
    start = data[0]
    end = data[-1]
    print(data[0], data[-1])
    for idx, i in enumerate(loaded_array[start:end+1]):
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
            mabc_action, mabc_Val, abc_info = ABC(args, env, state, info)
            # nested
            # mabc_ring_action = hp.nest2ring(args.nside_std, mabc_action)  # nested to ring
            # abc_action, abc_ObjVals, info = ABC(args, env, modified=False)
            # from skymap.SkyMapUtils import visualize_selected_pixel
            # visualize_selected_pixel(loaded_array[0], mabc_ring_action, args.nside_std)
            action_episode.append(mabc_action)
            state_episode.append(state)
            # visualize_selected_pixel(origin_m, abc_action, nside=nside_std)
            state, local_reward, terminated, consflag = env.step(mabc_action, next=True)
            reward_episode.append(local_reward)
            print('--------------------------------------------------------------------------')
            # 输出的动作均是ring排列下的网格索引
            print('进程{}正在规划任务{}:\n执行第{}个动作{}后，获得奖励{},当前运行时间为{}秒'.format(multiprocessing.current_process().name, idx+start, action_num+1, mabc_action, local_reward, state['t']))
            action_num += 1
        if abc_info['zero_map']:
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
    # 并行线程数量
    num_processes = args.num_processes
    # data = np.load("data/DirectABC_0_30.npz", allow_pickle=True)

    # 读取数据集
    with h5py.File('data/skymaps_by_rotation10000.h5', 'r') as hf:
        loaded_array = hf['data'][:]

    # 将数据划分成多个任务
    batch_size = 32
    start_task = 8358
    end_task = start_task + batch_size * num_processes
    tasks = [list(range(i, i+batch_size)) for i in range(start_task, end_task, batch_size)]
    # ranges = [(1408, 1439), (1568, 1659), (1728, 1759), (1888, 1919)]
    # tasks = [[i for i in range(start, end + 1)] for start, end in ranges]

    # 创建进程池
    pool = Pool(num_processes)

    # 启动并行任务
    results = pool.starmap(run_task, [(loaded_array, task, args) for task in tasks])

    # 关闭进程池
    pool.close()
    pool.join()

    # print('zero map: {}'.format(zero_maps))
    message = "您于{}开始的本地任务{}-{}已完成，请注意查看结果。".format(formatted_time, start_task, end_task-1)  # 邮件内容
    send_email(sender, receiver, subject, message)












import torch
import wandb
import matplotlib
import numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt
 

def get_ids(regex):
    api = wandb.Api()
    runs = api.runs(path="kaiyan3/CS498finale", filters={"display_name": {"$regex": regex}})
    r = []
    for run in runs:
        print(run.id)
        r.append(run.id)
    return r



def exponential_smooth(array, factor): # adopted from wandb https://docs.wandb.ai/v/zh-hans/dashboard/features/standard-panels/line-plot/smoothing
    res = np.zeros_like(array)
    last = 0
    for i in range(array.shape[0]):
        last = last * factor + (1 - factor) * array[i]
        debias_weight = 1 - factor ** (i + 1)
        res[i] = last / debias_weight
    return res
"""
# VICREG
name = "m3a52vp5"

def plot_ours(run_id, rew_name, color='r', legend="ours", alpha=1):
    api = wandb.Api()
    run = api.run("kaiyan3/CS498finale/"+run_id)
    history = run.scan_history(keys=[rew_name]) # "main_step"
    X1 = []
    for row in history:
        X1.append(row[rew_name]),

    MSX = np.array([i * 100 for i in range(len(X1) // 100)])
    print(MSX)
    plt.plot(MSX / 2, exponential_smooth(np.array(X1), 0.9)[MSX.astype('int')], color=color, label=legend, alpha=alpha)
    #plt.plot(main_step_at_X1, exponential_smooth(np.array(R), 0.6), color=color, label=legend)
"""
"""
# # SIMCLR & VINN, for contrastive curve
#name = "dfdv1p0o" # "2kbptug2" # "dfdv1p0o"


# for WBC loss
name = "5jxp7m4z" # "kvb1i8bg" # kvb1i8bg for VINN, 2kbptug2 for simclr, 5jxp7m4z for vicreg, yioyw3so for BCcrop, 5o1vigru for BCtranslate

def plot_ours(run_id, rew_name, color='r', legend="ours", alpha=1):
    api = wandb.Api()
    run = api.run("kaiyan3/CS498finale/"+run_id)
    history = run.scan_history(keys=[rew_name]) # "main_step"
    X1 = []
    for row in history:
        X1.append(row[rew_name]),

    MSX = np.array([i * 500 for i in range(len(X1))]) # 25 for contrastive curve
    print(MSX)
    plt.plot(MSX, exponential_smooth(np.array(X1), 0.9), color=color, label=legend, alpha=alpha)
    #plt.plot(main_step_at_X1, exponential_smooth(np.array(R), 0.6), color=color, label=legend)


plt.cla()    
plt.yscale('log')
name = "yioyw3so"
plot_ours(name, 'WBC train loss', color='r', legend="crop") # train_loss
name = "5o1vigru"
plot_ours(name, 'WBC train loss', color='b', legend='translate') # train_loss
plt.legend()
plt.savefig('da-bcloss.png')


plt.cla()    
plot_ours(name, 'covloss')
plt.savefig('vicreg-covloss.png')

plt.cla()    
plot_ours(name, 'stdloss')
plt.savefig('vicreg-stdloss.png')

plt.cla()    
plot_ours(name, 'simloss')
plt.savefig('vicreg-simloss.png')

"""
import math

def get_reward(run_ids, color='red', legend="threshold", alpha=0.5):
    rewards = []
    api = wandb.Api()
    for run_id in run_ids:
        run = api.run("kaiyan3/CS498finale/"+run_id)
        history = run.scan_history(keys=["avg_reward"]) # "main_step"
        for row in history:
            rewards.append(row["avg_reward"])
    print(run_ids, rewards)
    rewards = np.array(rewards)
    mean_rewards = rewards.mean()
    std_rewards = rewards.std(ddof=0)
    #print("mean:", mean_rewards)
    #print("std:", std_rewards)
    #exit(0)
    return math.floor(mean_rewards+0.5), math.floor(std_rewards+0.5)
print("st0:")
st0 = input()
print("st1:")
st1 = input()
print("st2:")
st2 = input()
print(get_reward(get_ids("walker.*"+st0+".*NN_"+st1+"core_"+st2)))    
print(get_reward(get_ids("cartpole.*"+st0+".*NN_"+st1+"core_"+st2)))
print(get_reward(get_ids("reacher.*"+st0+".*NN_"+st1+"core_"+st2)))

import stable_baselines3
import dmc2gym
from tqdm import tqdm
from utils import *
import os
import gym
import torch
import argparse, subprocess
from datetime import datetime
import numpy as np
from stable_baselines3 import SAC, PPO
from hyperparams import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain_name", type=str)
    parser.add_argument("--task_name", type=str)
    parser.add_argument("--seed", type=int, default=1) # use batchnorm
    parser.add_argument("--action_repeat", type=int, default=4)
    parser.add_argument("--train_agent", type=int, default=0) # 0 = generate dataset, 1 = train agent, 2 = both
    parser.add_argument("--algo_type", type=str, default="sac")
    # parser.add_argument("--frame_stack", type=int, default=)
    args = parser.parse_args()
    return args

args = get_args()

def get_git_diff():
    tmp = subprocess.run(['git', 'diff', '--exit-code'], capture_output=True)
    tmp2 = subprocess.run(['git', 'diff', '--cached', '--exit-code'], capture_output=True)
    return tmp.stdout.decode('ascii').strip() + tmp2.stdout.decode('ascii').strip()


def git_commit(runtime):
    tmp = subprocess.run(['git', 'commit', '-a', '-m', runtime], capture_output=True)
    return tmp.stdout.decode('ascii').strip()

runtime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
if len(get_git_diff()) > 0:
    git_commit(runtime)
if args.train_agent in [1, 2]:
    env = dmc2gym.make(domain_name=args.domain_name,
                       task_name=args.task_name,
                       seed=args.seed,
                       visualize_reward=False,
                       from_pixels=False,
                       frame_skip=args.action_repeat
                       )
    NAME = args.domain_name+"/"+args.task_name+"/"+str(runtime).replace("/", "-").replace(" ", "-").replace(":", "-")+"seed"+str(args.seed)+"action_repeat"+str(args.action_repeat) + "-" + args.algo_type
    if args.algo_type == "sac": model = SAC("MlpPolicy", env, verbose=1, tensorboard_log="./proprio_expert/"+NAME)
    else: model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./proprio_expert"+NAME)
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        # print("reward:", reward, "state:", obs)
        if done:
            obs = env.reset()

    print("before learning...")
    callback = RegularSaveCallback(check_freq=50000, log_dir="expert_model/"+NAME)  
    model.learn(total_timesteps=1000000, log_interval=1, callback=callback)

    name = args.domain_name+"-"+args.task_name+"-seed"+str(args.seed)+"-actionrepeat"+str(args.action_repeat)

    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        # print("reward:", reward)
        if done:
            obs = env.reset()
    # env = utils.FrameStack(env, k=args.frame_stack)

    model.save("expert_model/sac_proprioceptive_"+name) 
if args.train_agent in [0, 2]:

    env = dmc2gym.make(domain_name=args.domain_name,
                       task_name=args.task_name,
                       seed=args.seed,
                       visualize_reward=False,
                       from_pixels=True,
                       height=IMAGE_SIZE,
                       width=IMAGE_SIZE,
                       frame_skip=args.action_repeat
                       )
    num_traj = 50
    
    obss, actions, next_obss, rewards, dones, steps = [], [], [], [], [], []
    if args.train_agent == 0:
        model = SAC.load("expert_model/sac_proprioceptive_"+args.domain_name+"-"+args.task_name+"-seed"+str(args.seed)+"-actionrepeat"+str(args.action_repeat)) if args.algo_type == "sac" else PPO.load("expert_model/sac_proprioceptive_"+args.domain_name+"-"+args.task_name+"-seed"+str(args.seed)+"-actionrepeat"+str(args.action_repeat))
    for i in tqdm(range(num_traj)):
        done = False
        last_obs = env.reset()
        step = 0
        while not done:
            state = env.current_state
            action, _states = model.predict(state, deterministic=True)
            epsilon = 1e-6
            action = np.clip(action, env.action_space.low + epsilon, env.action_space.high - epsilon)
            # print("action:", action)
            obs, reward, done, info = env.step(action)
            obss.append(np.expand_dims(last_obs, 0))
            rewards.append(np.expand_dims(np.array([reward]), 0))
            next_obss.append(np.expand_dims(obs, 0))
            actions.append(np.expand_dims(action, 0))
            dones.append(np.expand_dims(np.array([done]), 0))
            steps.append(np.expand_dims(np.array([step]), 0)) 
            
            step += 1
            last_obs = obs
        
    obss = np.concatenate(obss, axis=0)
    print("rewards:", rewards.sum())
    rewards = np.concatenate(rewards, axis=0)
    next_obss = np.concatenate(next_obss, axis=0)
    actions = np.concatenate(actions, axis=0)
    dones = np.concatenate(dones, axis=0)
    steps = np.concatenate(steps, axis=0)
    print("shape:", actions.shape)
    import os
    #if not os.path.exists("data/"+args.domain_name+"-"+args.task_name):
    #    os.makedirs("data/"+args.domain_name+"-"+args.task_name)
    #torch.save([obss, rewards, next_obss, actions, dones, steps], "data/"+args.domain_name+"-"+args.task_name+"/data.pt")
    
    print(obss.shape, rewards.shape, next_obss.shape, actions.shape, dones.shape, steps.shape)    
        
    
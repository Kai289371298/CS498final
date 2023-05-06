import torch
import gym
import dmc2gym
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn import manifold
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain_name", type=str)
    parser.add_argument("--task_name", type=str)
    parser.add_argument("--encode_method", type=str)
    
    args = parser.parse_args()
    return args

args = get_args()
 
os.environ['MUJOCO_GL'] = 'egl'
domain_name = args.domain_name
task_name = args.task_name
encode_method = args.encode_method

seed = 0

NAME = str(seed)+"_"+domain_name+"_"+task_name+"_"+encode_method

LOAD_TRAJ = 0

device = torch.device('cuda:0')
if LOAD_TRAJ == 0:
    if encode_method not in ["dacrop", "datran"]:
        encoder = torch.load("model/finetune/"+NAME+".pt")
        actor = torch.load("model/finetune/actor/"+domain_name+"-"+task_name+"-"+encode_method+"-seed"+str(seed)+"-actionrepeat4-archresnet18")
    elif encode_method == "dacrop":
        actor, encoder = torch.load("model/finetune/da_forreport_crop.pt")
    else:
        actor, encoder = torch.load("model/finetune/da_forreport.pt")
    
    
    env = dmc2gym.make(domain_name=domain_name,
                       task_name=task_name,
                       seed=seed,
                       visualize_reward=False,
                       from_pixels=True,
                       height=84,
                       width=84,
                       frame_skip=4
                       )
    
    done = False
    
    obss, embeds = [], [] 
    
    obs = env.reset()
    step = 0
    encoder.eval()
    tot_reward = 0
    while not done:
        with torch.no_grad():
            
            obss.append(torch.from_numpy(obs / 255.0).float().to(device).unsqueeze(0))
            
            embed = encoder(torch.from_numpy(obs / 255.0).float().to(device).unsqueeze(0))
            
            embeds.append(embed)
            
            action = actor(embed).view(-1) 
        # print("shappe:", action.shape, env.action_space)
        obs, reward, done, info = env.step(action.cpu().numpy())
        tot_reward += reward
        step += 1
        print("step:", step, "reward:", tot_reward)
    obss   = torch.cat(obss, dim=0)
    embeds = torch.cat(embeds, dim=0)
    
################################################################################
# for nearest neighbour

expert_state,_, __, ___, ____, step = torch.load("data/"+domain_name+"-"+task_name+"/data.pt")
expert_state = torch.from_numpy(expert_state / 255.0).float().to(device)
expert_embeds = []

with torch.no_grad():
    for i in tqdm(range(expert_state.shape[0] // 32 + 1)):
        expert_embeds.append(encoder(expert_state[i*32:(i+1)*32]))

    expert_embeds = torch.cat(expert_embeds, dim=0)

    dists_pixel = []
    dists_embed = []
    maxx_coeff = []
    top5_coeff = []
    top10_coeff = []
    
    for i in range(embeds.shape[0]):
        dist_pixel = torch.norm(obss[i].view(1, -1).repeat(expert_state.shape[0], 1) - expert_state.view(expert_state.shape[0], -1), 2, dim=1)
        dist_embed = torch.norm(embeds[i].view(1, -1).repeat(expert_state.shape[0], 1) - expert_embeds, 2, dim=1)
        # print(i, dist_pixel, dist_embed)
        
        idx = torch.argsort(dist_embed)[:10]
        print("dist:", i, torch.exp(-dist_embed[idx]), torch.exp(-dist_embed).sum())        
        dists_pixel.append(dist_pixel[idx][0].cpu())
        dists_embed.append(dist_embed[idx][0].cpu())
        
        d = -dist_embed 
        dmax = d.max()
        
        maxx_coeff.append(torch.exp(d.max() - dmax) / torch.exp(d - dmax).sum()) 
        top5_coeff.append((torch.exp(d[idx[:5]] - dmax) / torch.exp(d - dmax).sum()).sum())
        top10_coeff.append((torch.exp(d[idx]- dmax) / torch.exp(d - dmax).sum()).sum())
        
        print(i, dists_pixel[-1], dists_embed[-1], maxx_coeff[-1], top5_coeff[-1], top10_coeff[-1])
    
    maxx_coeff = [x.cpu() for x in maxx_coeff]
    top5_coeff = [x.cpu() for x in top5_coeff]
    top10_coeff = [x.cpu() for x in top10_coeff]
    
    plt.cla()
    plt.plot(np.arange(embeds.shape[0]), maxx_coeff, 'red', label='top1')
    plt.plot(np.arange(embeds.shape[0]), top5_coeff, 'orange', label='top5')
    plt.plot(np.arange(embeds.shape[0]), top10_coeff, 'yellow', label='top10')
    plt.legend()
    plt.savefig("saved_traj/pic/top1510-"+NAME+".png")
    plt.cla()
    
    plt.plot(np.arange(embeds.shape[0]), dists_pixel, 'blue', label='pixelwise')
    plt.plot(np.arange(embeds.shape[0]), dists_embed, 'red', label='embedding')
    plt.legend()
    plt.savefig("saved_traj/pic/distNN-"+NAME+".png")
    
################################################################################
    
"""
################################################################################
# for representation analysis

    e, e2 = [], []
    
    for i in range(1, embeds.shape[0]):
        e.append(torch.norm(embeds[i] - embeds[i-1] + 1e-10, 2))
        e2.append(torch.norm(obss[i] - obss[i-1] + 1e-10, 2))
        print("distance:", torch.norm(embeds[i] - embeds[i-1] + 1e-10, 2))
    
    e = torch.tensor(e)   
    e2= torch.tensor(e2)

    plt.plot([i for i in range(e.shape[0])], e, color='r', label="embedding") 
    plt.plot([i for i in range(e2.shape[0])], e2, color='b', label="pixelwise")  
    
    plt.yscale('log')
    plt.legend()
    plt.savefig("saved_traj/pic/"+NAME+".png")
    
    torch.save([obss, embeds], "saved_traj/"+NAME+".pt")
    
    expert_state,_, __, ___, ____, step = torch.load("data/"+domain_name+"-"+task_name+"/data.pt")
    
    expert_state,_, __, ___, ____, step = torch.load("data/"+domain_name+"-"+task_name+"/data.pt")
    step = torch.from_numpy(step).to(device)
    
    initial_idx = torch.nonzero(step == 0).int()[:, 0].cpu()
    # print(initial_idx)
    
    expert1, expert2, expert3 = expert_state[:initial_idx[1]], expert_state[initial_idx[1]:initial_idx[2]], expert_state[initial_idx[2]:initial_idx[3]]
    
    tsne = manifold.TSNE(n_components=2, init='pca')
    
    X = torch.cat([obss, expert1, expert2, expert3], dim=0).to(device).float()
    with torch.no_grad():
        embed_X = encoder(X).cpu()
    
    
    idx_ours, idx_expert1, idx_expert2, idx_expert3 = np.arange(obss.shape[0]), np.arange(obss.shape[0], obss.shape[0] + initial_idx[1]), np.arange(obss.shape[0] + initial_idx[1], obss.shape[0] + initial_idx[2]), np.arange(obss.shape[0] + initial_idx[2], obss.shape[0] + initial_idx[3]), 
    
    print(idx_ours, idx_expert1, idx_expert2, idx_expert3)
    
    print(embed_X.shape)
    
    ours_embed, expert1_embed, expert2_embed, expert3_embed = embed_X[idx_ours], embed_X[idx_expert1], embed_X[idx_expert2], embed_X[idx_expert3]
    plt.cla()
    
    plt.scatter(ours_embed[:, 0], ours_embed[:, 1], color='r', marker='x', label='agent')
    
    plt.scatter(expert1_embed[:, 0], expert1_embed[:, 1], color='b', marker='o', label='expert1')
    plt.scatter(expert2_embed[:, 0], expert2_embed[:, 1], color='b', marker='*', label='expert2')
    plt.scatter(expert3_embed[:, 0], expert3_embed[:, 1], color='b', marker='+', label='expert3')
    #res = tsne.fit_transform(embed_X)
    plt.legend()
    plt.savefig("saved_traj/pic/tsne-"+ NAME+".png")
    
else:
    obss, embeds = torch.load("saved_traj/"+NAME+".pt")
################################################################################
"""
import torch
import numpy as np
import os
os.environ['MUJOCO_GL'] = 'egl'
import gym
from net import *
from hyperparams import *
from utils import *
from byol_embedding import getarch, trainBYOL, trainVICREG, trainSIMCLR
import dmc2gym, subprocess
from torchvision import models
import torchvision.transforms as T
from datetime import datetime
import wandb
import argparse
import random
import imageio

device = torch.device('cuda:0')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain_name", type=str)
    parser.add_argument("--task_name", type=str)
    parser.add_argument("--seed", type=int, default=1) # use batchnorm
    parser.add_argument("--action_repeat", type=int, default=4)
    parser.add_argument("--encode_method", type=str) # 0 = generate dataset, 1 = train agent, 2 = both
    parser.add_argument("--actor_method", type=str)
    parser.add_argument("--pretrain", type=int, default=1)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--embed_epoch", type=int, default=100)
    parser.add_argument("--log_interval", type=int, default=500)
    parser.add_argument("--load", type=int, default=0)
    parser.add_argument("--lr_bc", type=float, default=0.001)
    parser.add_argument("--lr_enc", type=float, default=0.001)
    parser.add_argument("--bc_epoch", type=int, default=5000)
    parser.add_argument("--arch", type=str, default='resnet50')
    parser.add_argument("--aug", type=str, default="default")
    parser.add_argument("--debug", type=int, default=0)
    args = parser.parse_args()
    return args

def evaluate():
    pass
    
def get_git_diff():
    tmp = subprocess.run(['git', 'diff', '--exit-code'], capture_output=True)
    tmp2 = subprocess.run(['git', 'diff', '--cached', '--exit-code'], capture_output=True)
    return tmp.stdout.decode('ascii').strip() + tmp2.stdout.decode('ascii').strip()


def git_commit(runtime):
    tmp = subprocess.run(['git', 'commit', '-a', '-m', runtime], capture_output=True)
    return tmp.stdout.decode('ascii').strip()\

def evaluation_BC(env, actor, encoder, debug=None):
    avg_reward, T, tot_reward = 0, 10, 0
    if debug is not None:
        video = []
    for i in range(T):
        done = False
        obs = env.reset()
        step = 0
        encoder.eval()
        if debug is not None:
            video.append(env.render(mode='rgb_array'))
        while not done:
            with torch.no_grad():
                embed = encoder(torch.from_numpy(obs / 255.0).float().to(device).unsqueeze(0))
                action = actor(embed).view(-1) 
            # print("shappe:", action.shape, env.action_space)
            obs, reward, done, info = env.step(action.cpu().numpy())
            if debug is not None:
                video.append(env.render(mode='rgb_array'))
            tot_reward += reward
            step += 1
        encoder.train()
    
    wandb.log({"avg_reward": tot_reward / T, "avg_steps": step / T})
    if debug is not None:
        #imageio.imsave("video/"+debug+'.jpg', video[0])
        #exit(0)
        imageio.mimsave("video/"+debug+'.mp4', video, fps=25)

def BC(env, actor, encoder, obs, reward, next_obs, action, done, step, args):

    o, a = obs.float(), torch.from_numpy(action).float() 
    
    
    if args.encode_method in ["vinn", "simclr", "vicreg"]: # the embedding will not change anymore
        optimizer = torch.optim.Adam(actor.parameters(), lr=args.lr_bc)
        e = []    
        WBC_step = 0
        print("calculating embeddings...")
        for i in tqdm(range((obs.shape[0] // 32) + 1)): # assert batch size is not 1, and obs.shape[0] % args.batchsize not in {0, 1}
            with torch.no_grad():
                embed = encoder(o[i*32:(i+1)*32].to(device))
                e.append(embed)
        e = torch.cat(e, dim=0)
        print("shape:", e.shape)
        print("embeddings calculate complete!")
        train_loader = RepeatedDataset([e, a], batch_size=args.batch_size)
        for _ in tqdm(range(args.bc_epoch)):
            # for batch_idx, sample_batched in enumerate(train_loader): 
        
            if  _ == 0 or _ % args.eval_interval == args.eval_interval - 1:
                evaluation_BC(env, actor, encoder, debug=(None if args.debug == 0 else args.domain_name+"-"+args.task_name+"-"+args.encode_method+"-seed"+str(args.seed)+"-actionrepeat"+str(args.action_repeat)+"-arch"+args.arch+"-step"+str(WBC_step)))  
                torch.save(actor, "model/finetune/actor/tmp/"+args.domain_name+"-"+args.task_name+"-"+args.encode_method+"-seed"+str(args.seed)+"-actionrepeat"+str(args.action_repeat)+"-arch"+args.arch+"-step"+str(WBC_step)+".pt")
        
            debug_actions, debug_coeffs = [], []
            # print("size:", states_TA.shape, len(train_loader))
            for __ in tqdm(range(len(train_loader))):
                embed, action = train_loader.getitem() 
                pred_action = actor(embed.to(device))
                action = torch.clamp(action.to(device), min=torch.from_numpy(env.action_space.low + 1e-6).to(device), max=torch.from_numpy(env.action_space.high - 1e-6).to(device))
                loss = ((action - pred_action) ** 2).mean() # simple MSE loss
                # print("action:", action[0], "pred_action:", pred_action[0], "output_low:", actor.output_low, "output_high:", actor.output_high)
                optimizer.zero_grad()
                loss.backward()
                g = 0
                for param in actor.parameters():
                    g += torch.norm(param.grad, 2)
                if WBC_step % args.log_interval == 0: 
                    wandb.log({"WBC gradient norm": g, "WBC train loss": loss, 'WBC_steps': WBC_step})    
                WBC_step += 1
                optimizer.step()   
            
        torch.save(actor, "model/finetune/actor/"+args.domain_name+"-"+args.task_name+"-"+args.encode_method+"-seed"+str(args.seed)+"-actionrepeat"+str(args.action_repeat)+"-arch"+args.arch)
    elif args.encode_method == "da":
      actor, encoder = actor.to(device), encoder.to(device)
      optimizer = torch.optim.Adam([*encoder.parameters(), *actor.parameters()], lr=args.lr_bc)
      assert args.aug != "default", "Error!"
      WBC_step = 0
      train_loader = RepeatedDataset([o, a], batch_size=args.batch_size)
      for _ in tqdm(range(args.bc_epoch)): 
            # for batch_idx, sample_batched in enumerate(train_loader): 
        
            if  _ == 0 or _ % args.eval_interval == args.eval_interval - 1:
                evaluation_BC(env, actor, encoder, debug=(None if args.debug == 0 else args.domain_name+"-"+args.task_name+"-"+args.encode_method+"-seed"+str(args.seed)+"-actionrepeat"+str(args.action_repeat)+"-arch"+args.arch+"-step"+str(WBC_step)))  
        
            debug_actions, debug_coeffs = [], []
            # print("size:", states_TA.shape, len(train_loader))
            for __ in tqdm(range(len(train_loader))): 

                state, action = train_loader.getitem()
                action = action.to(device)
                if args.aug == "translate":
                    Aug = T.RandomAffine(degrees=0, translate=(0.2, 0.2))
                elif args.aug == "crop":
                    Aug = T.RandomResizedCrop(size=(IMAGE_SIZE - 8, IMAGE_SIZE - 8))
                
                print("state:", state.shape)
                state_aug = Aug(state)
                print("state_aug:", state_aug.shape)
                embed = encoder(state_aug.to(device))
                print("embed-shape:", embed.shape)
                pred_action = actor(embed)
                action = torch.clamp(action.to(device), min=torch.from_numpy(env.action_space.low + 1e-6).to(device), max=torch.from_numpy(env.action_space.high - 1e-6).to(device))
                loss = ((action - pred_action) ** 2).mean() # simple MSE loss
                
                if args.encode_method == "da" and _ % 10 == 9:
                    torch.save([actor, encoder], "model/finetune/da_forreport_crop.pt")
                
                optimizer.zero_grad()
                loss.backward()
                g = 0
                for param in actor.parameters():
                    g += torch.norm(param.grad, 2)
                if WBC_step % args.log_interval == 0: 
                    wandb.log({"WBC gradient norm": g, "WBC train loss": loss, 'WBC_steps': WBC_step})    
                WBC_step += 1
                optimizer.step()   

def evaluation_NN(env, encoder, state, action, args):
    
    o, a = state.to(device).float(), torch.from_numpy(action).to(device).float()
    
    assert args.actor_method.find("hard") != -1 or args.actor_method.find("soft") != -1, "Error!"
    if args.actor_method.find("euccore") != -1:
        dist = Euccoredist
    elif args.actor_method.find("euc") != -1:
        dist = Eucdist
    elif args.actor_method.find("mancore") != -1:
        dist = Mancoredist
    elif args.actor_method.find("man") != -1:
        dist = Mandist
    else:
        raise NotImplementedError("Error!")
    
    e = []    
    WBC_step = 0
    print("calculating embeddings...")
    for i in tqdm(range((state.shape[0] // 32) + 1)): # assert batch size is not 1, and obs.shape[0] % args.batchsize not in {0, 1}
        with torch.no_grad():
            embed = encoder(o[i*32:(i+1)*32].to(device))
            e.append(embed)
    e = torch.cat(e, dim=0)    
    
    avg_reward, T, tot_reward = 0, 10, 0
    for i in range(T):
        done = False
        obs = env.reset()
        step = 0
        encoder.eval()
        while not done:
            with torch.no_grad():
                embed = encoder(torch.from_numpy(obs / 255.0).float().to(device).unsqueeze(0))
                action = NN(embed, e, a, dist, args.actor_method.find("hard") != -1).view(-1) 
                # print("action:", action)
            obs, reward, done, info = env.step(action.cpu().numpy())
            tot_reward += reward
            step += 1
        encoder.train()
    wandb.log({"avg_reward": tot_reward / T, "avg_steps": step / T})
    
    # args.actor_method
    

def NN(embed, data, action, dist, use_hard): # find nearest neighbour
    target = embed.repeat(data.shape[0], 1)
    # print(target.shape, data.shape, dist, target, data)
    d = dist(target, data)
    if use_hard:
        agmx = d.argmax()
        return action[agmx]
    else:
        # print("d:", d)
        d /= d.sum()
        # print(action, d.max())
        # exit(0)
        return (action * d.view(-1, 1)).sum(dim=0).detach()

if __name__ == "__main__":
    runtime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    if len(get_git_diff()) > 0:
        git_commit(runtime)
    
    args = get_args()
    
    wandb.login(key="XXXXXXXX") # keys commented out for privacy
    
    NAME = str(args.seed)+"_"+args.domain_name+"_"+args.task_name+"_"+args.encode_method
    
    wandb.init(entity="kaiyan3",project="CS498finale", name=str(runtime)+"_"+NAME+"_"+args.actor_method + "-" + ("" if args.encode_method != "da" else args.aug)) 
    
    seed = args.seed
    
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) # when using multiple GPUs torch.cuda.manual_seed(seed) 
    np.random.seed(seed) 
    random.seed(seed) 
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False
    
    env = dmc2gym.make(domain_name=args.domain_name,
                       task_name=args.task_name,
                       seed=args.seed,
                       visualize_reward=False,
                       from_pixels=True,
                       height=IMAGE_SIZE,
                       width=IMAGE_SIZE,
                       frame_skip=args.action_repeat
                       )
    state, reward, next_state, action, done, step = torch.load("data/"+args.domain_name+"-"+args.task_name+"/data.pt")
    #print("shape::", action.shape, reward.shape)
    #exit(0)
    state, next_state = torch.from_numpy(state).float() / 255.0, torch.from_numpy(next_state).float() / 255.0
    
    
    if args.load == 0:
        if args.encode_method != "predfuture":
            encoder = getarch(args)
            actor = Actor(encoder.fc.out_features if args.encode_method in ["vinn", 'da'] else encoder.fc.in_features, env.action_space.shape[0], env.action_space.low, env.action_space.high).float().to(device)
            if args.encode_method == "vinn": # BYOL
                params = {"epochs": args.embed_epoch, "img_size": (84, 84), "hidden_layer": 'avgpool', 'batch_size': 256}
                trainBYOL(args, params, encoder, state)
                
                print(encoder)
            elif args.encode_method == "simclr": # BYOL variant
                params = {"epochs": args.embed_epoch, "img_size": (84, 84), "hidden_layer": 'avgpool', 'batch_size': 256}
                trainSIMCLR(args, params, encoder, state)
            elif args.encode_method == "vicreg": # BYOL variant
                params = {"epochs": args.embed_epoch, "img_size": (84, 84), "hidden_layer": 'avgpool', 'batch_size': 256}
                trainVICREG(args, params, encoder, state)
        else: 
            raise NotImplementedError("Error!")
        torch.save(encoder, "model/finetune/"+NAME+".pt")
    else:  
        encoder = torch.load("model/finetune/"+NAME+".pt")
        if args.arch == "resnet18": 
            if args.encode_method in ["vinn"]:
                actor = Actor(encoder.fc.out_features, env.action_space.shape[0], env.action_space.low, env.action_space.high).float().to(device)
            else:
                actor = Actor(512, env.action_space.shape[0], env.action_space.low, env.action_space.high).float().to(device)
    
    if args.actor_method == "BC": BC(env, actor, encoder, state, reward, next_state, action, done, step, args)
    elif args.actor_method.find("NN") != -1: evaluation_NN(env, encoder, state, action, args) 
    else: raise notImplementedError("Error!")
    

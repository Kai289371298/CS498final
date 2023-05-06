from byol_pytorch import BYOL
from torchvision import models
import torchvision.transforms as T
import torch
from net import *
from torch.utils.data import DataLoader 
from tqdm import tqdm
import math
device = torch.device('cuda:0') 
import wandb

def getarch(args):
    if args.pretrain == 1:
        if args.arch == "resnet50": model = models.resnet50(pretrained=True).float()
        elif args.arch == "resnet18": model = models.resnet18(pretrained=True).float()
    else:
        if args.arch == "resnet50": model = models.resnet50(pretrained=False).float()
        elif args.arch == "resnet18": model = models.resnet18(pretrained=False).float()
    return model

def trainBYOL(args, params, model, data):
    hidden_layer = params['hidden_layer']
    
    Aug = T.Compose([T.RandomResizedCrop(params['img_size'], scale=(0.6,1.0)),
                            T.RandomApply(torch.nn.ModuleList([T.ColorJitter(.8,.8,.8,.2)]), p=.3),
                            T.RandomGrayscale(p=0.2),
                            T.RandomApply(torch.nn.ModuleList([T.GaussianBlur((3, 3), (1.0, 2.0))]), p=0.2),
                            T.Normalize(
                            mean=torch.tensor([0.485, 0.456, 0.406]),
                            std=torch.tensor([0.229, 0.224, 0.225]))])
    
    
    print(params['img_size'])
    
    model = model.to(device)
    learner = BYOL(model, params['img_size'][0], hidden_layer, augment_fn=Aug)
    optimizer = torch.optim.Adam(learner.parameters(), lr=args.lr_enc)    
    dataLoader = DataLoader(data, batch_size=params['batch_size'], shuffle=True)
    epochs = params['epochs']

    

    print(model)

    for epoch in tqdm(range(epochs)):
        epoch_loss = 0
        for i, data in enumerate(dataLoader, 0):
                            
            # print("data.shape:", data.shape)  
            
            loss = learner(data.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            learner.update_moving_average()

            epoch_loss += loss.item() # *data.shape[0]

        # print(epoch_loss / len(dataLoader)) 
        
        wandb.log({'train_loss': epoch_loss / len(dataLoader)})

        if(epoch % 20 == 0):
            torch.save({'model_state_dict': model.state_dict()
                    }, 'model/finetune/BYOL_'+str(args.seed)+"_"+args.domain_name+"_"+args.task_name+"_"+args.encode_method+"_"+args.actor_method+"-"+str(epoch)+'.pt')        


def crossent_loss(out_1, out_2, temperature, eps=1e-6):
    """
    assume out_1 and out_2 are normalized
    out_1: [batch_size, dim]
    out_2: [batch_size, dim]
    """
    # gather representations in case of distributed training
    # out_1_dist: [batch_size * world_size, dim]
    # out_2_dist: [batch_size * world_size, dim]
    
    out_1_dist = out_1
    out_2_dist = out_2

    # out: [2 * batch_size, dim]
    # out_dist: [2 * batch_size * world_size, dim]
    out = torch.cat([out_1, out_2], dim=0)
    out_dist = torch.cat([out_1_dist, out_2_dist], dim=0)

    # cov and sim: [2 * batch_size, 2 * batch_size * world_size]
    # neg: [2 * batch_size]
    cov = torch.mm(out, out_dist.t().contiguous())
    sim = torch.exp(cov / temperature)
    neg = sim.sum(dim=-1)

    # from each row, subtract e^(1/temp) to remove similarity measure for x1.x1
    row_sub = torch.tensor(neg.shape).fill_(math.e ** (1 / temperature)).to(neg.device)
    neg = torch.clamp(neg - row_sub, min=eps)  # clamp for numerical stability

    # Positive similarity, pos becomes [2 * batch_size]
    pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    pos = torch.cat([pos, pos], dim=0)

    loss = -torch.log(pos / (neg + eps)).mean()

    return loss


def trainSIMCLR(args, params, model, data):
    
    Aug = T.Compose([T.RandomResizedCrop(params['img_size'], scale=(0.6,1.0)),
                            T.RandomApply(torch.nn.ModuleList([T.ColorJitter(.8,.8,.8,.2)]), p=.3),
                            T.RandomGrayscale(p=0.2),
                            T.RandomApply(torch.nn.ModuleList([T.GaussianBlur((3, 3), (1.0, 2.0))]), p=0.2),
                            T.Normalize(
                            mean=torch.tensor([0.485, 0.456, 0.406]),
                            std=torch.tensor([0.229, 0.224, 0.225]))])
    
    model = model.to(device)
    #print(model.fc.in_features, model.fc.out_features)
    #exit(0)
    proj = SIMCLR_projector(model.fc.in_features).to(device)
    
    epochs = params['epochs']
    model.fc = NullEncoder().to(device)
    
    data_loader = DataLoader(data, batch_size=params['batch_size'], shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_enc)
   
    for epoch in tqdm(range(epochs)):
        epoch_loss = 0
        for i, data in enumerate(data_loader, 0):
            print(Aug(data.to(device)).shape)
            h1, h2 = model(Aug(data.to(device))), model(Aug(data.to(device)))
            print("shapae:", h1.shape, h2.shape, "BS:", params['batch_size'])
            z1, z2 = proj(h1), proj(h2)
            loss = crossent_loss(z1, z2, 0.5)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        wandb.log({'train_loss': epoch_loss / len(data_loader)})
        
        if(epoch % 20 == 0):
            torch.save({'model_state_dict': model.state_dict()
                    }, 'model/finetune/SIMCLR_'+str(args.seed)+"_"+args.domain_name+"_"+args.task_name+"_"+args.encode_method+"_"+args.actor_method+"-"+str(epoch)+'.pt')   
        
        
def trainVICREG(args, params, model, data):
    
    Aug = T.Compose([T.RandomResizedCrop(params['img_size'], scale=(0.6,1.0)),
                            T.RandomApply(torch.nn.ModuleList([T.ColorJitter(.8,.8,.8,.2)]), p=.3),
                            T.RandomGrayscale(p=0.2),
                            T.RandomApply(torch.nn.ModuleList([T.GaussianBlur((3, 3), (1.0, 2.0))]), p=0.2),
                            T.Normalize(
                            mean=torch.tensor([0.485, 0.456, 0.406]),
                            std=torch.tensor([0.229, 0.224, 0.225]))])
    
    model = model.to(device)
    proj = VICREG_projector(model.fc.in_features).to(device)
    epochs = params['epochs']
    model.fc = NullEncoder().to(device)
    
    data_loader = DataLoader(data, batch_size=params['batch_size'], shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_enc)
    mseLoss = nn.MSELoss()
    for epoch in tqdm(range(epochs)):
        epoch_loss = 0
        for i, data in enumerate(data_loader, 0):
            h1, h2 = model(Aug(data.to(device))), model(Aug(data.to(device)))
            print("shape:", h1.shape, h2.shape)
            z_a, z_b = proj(h1), proj(h2)
            sim_loss = mseLoss(z_a, z_b)

            std_z_a = torch.sqrt(z_a.var(dim=0) + 1e-04)
            std_z_b = torch.sqrt(z_b.var(dim=0) + 1e-04)
            std_loss = torch.mean(F.relu(1 - std_z_a))
            std_loss = std_loss + torch.mean(F.relu(1 - std_z_b))

            N, D = z_a.size()

            z_a = z_a - z_a.mean(dim=0)
            z_b = z_b - z_b.mean(dim=0)
            cov_z_a = (z_a.T @ z_a) / (N - 1)
            cov_z_b = (z_b.T @ z_b) / (N - 1)
            diag = torch.eye(D, device=z_a.device)
            cov_loss = cov_z_a[~diag.bool()].pow_(2).sum() / D + cov_z_b[~diag.bool()].pow_(2).sum() / D

            loss = 25.0*sim_loss + 25.0*std_loss + cov_loss
            # wandb.log({"simloss": sim_loss, "stdloss": std_loss, "covloss": cov_loss, "totloss1": 25*sim_loss, "totloss2": 25*std_loss, "loss": loss})
            epoch_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        wandb.log({'train_loss': epoch_loss / len(data_loader)})
        
        if(epoch % 20 == 0):
            torch.save({'model_state_dict': model.state_dict()
                    }, 'model/finetune/VICREG_'+str(args.seed)+"_"+args.domain_name+"_"+args.task_name+"_"+args.encode_method+"_"+args.actor_method+"-"+str(epoch)+'.pt')   
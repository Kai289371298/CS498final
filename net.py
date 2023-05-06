import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda:0')

class NullEncoder(nn.Module):
    def __init__(self):
        super(NullEncoder, self).__init__()

    def forward(self, x):
        return x

class Actor(nn.Module):
    def __init__(self, input_dim, output_dim, output_low, output_high):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, 1024)
        self.fc3 = nn.Linear(1024, output_dim)
        self.output_low, self.output_high = torch.from_numpy(output_low).to(device), torch.from_numpy(output_high).to(device)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.tanh(self.fc3(x))
        return self.output_low + (self.output_high - self.output_low) * (x + 1) / 2

class SIMCLR_projector(nn.Module):
    def __init__(self, input_dim):
        super(SIMCLR_projector, self).__init__()

        self.f1 = nn.Linear(input_dim, 2048, bias=True)
        self.f2 = nn.Linear(2048, 128, bias=False)
        self.batch_norm = nn.BatchNorm1d(2048)

    def forward(self, x):
        x = self.f1(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        x = self.f2(x)
        return(F.normalize(x, dim=1))

class VICREG_projector(nn.Module):
    def __init__(self, input_dim):
        super(VICREG_projector, self).__init__()

        self.f1 = nn.Linear(input_dim, 8192, bias=True)
        self.f2 = nn.Linear(8192, 8192, bias=True)
        self.batch_norm = nn.BatchNorm1d(8192)

    def forward(self, x):
        x = self.f1(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        x = self.f2(x)
        return(x)
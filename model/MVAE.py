import numpy as np
from torch_geometric.data import DataLoader
from tqdm import tqdm
from torch_geometric.nn import GCNConv,GATConv
import copy
import random
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling, add_self_loops
from torch_scatter import scatter_mean


class Encoder(nn.Module):
    def __init__(self, n_total_features, n_latent,out_feats, p_drop=0.):
        super(Encoder, self).__init__()
        self.bilstm = nn.LSTM(n_total_features, n_latent,num_layers=2, bidirectional=True)
        self.ReLU = nn.ReLU()
        self.mean_fc = nn.Linear(n_latent*2,out_feats)
        self.std_fc = nn.Linear(n_latent*2,out_feats)
    def forward(self, data):
        x  = data.x
        x.unsqueeze_(0)
        x = self.ReLU(self.bilstm(x)[0])
        x.squeeze_(0)
        self.mean = self.mean_fc(x)
        self.log_std = self.std_fc(x)
        self.z = self.mean + torch.randn_like(self.log_std) * torch.exp(self.log_std)
        return z

class Decoder(nn.Module):
    def __init__(self, in_feats,hid_feats,out_feats):
        super(Decoder, self).__init__()
        self.bilstm1 = nn.LSTM(out_feats*2, out_feats,num_layers=2, bidirectional=True)
        self.ReLU1 = nn.ReLU()
        self.fc1 = nn.Linear(out_feats*2,in_feats)
    def forward(self, z):
        z.unsqueeze_(0)
        z = self.ReLU1(self.bilstm1(z)[0])
        z = self.fc1(z)
        return z.squeeze_(0)

class Net(nn.Module):
    def __init__(self,in_feats,hid_feats,out_feats):
        super(Net, self).__init__()
        self.encoder = Encoder(in_feats,hid_feats,out_feats)
        self.decoder = Decoder(in_feats,hid_feats,out_feats)
        self.fc = nn.Linear(out_feats,4)
   
    def loss(self):
        #### recon_loss
        re_loss = F.mse_loss(self.z,self.re_z)
        ### kl loss
        kl = -0.5 / self.num_nodes * torch.mean(
            torch.sum(1 + self.log_std - self.mean**2 - self.log_std.exp(), dim=1))
        return re_loss + kl
        
    def forward(self, data):
        self.z = self.encoder(data)
        self.re_z = self.decoder(self.z)

        x= scatter_mean(self.z, data.batch, dim=0)
        x=self.fc(x)
        x = F.log_softmax(x,dim=1)
        return x
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
class Net(nn.Module):
    def __init__(self, n_total_features, n_latent,out_feats, p_drop=0.):
        super(Net, self).__init__()
        self.n_total_features = n_total_features
        self.conv1 = GCNConv(self.n_total_features, 2*n_latent)
        self.act1=nn.Sequential(nn.ReLU(),
                              nn.Dropout(p_drop))
        #self.conv2 = GCNConv(n_latent, n_latent)
        #self.act2 = nn.Sequential(nn.ReLU(),
        #                      nn.Dropout(p_drop))
        self.conv3 = GCNConv(2*n_latent, out_feats)
        #self.conv4 = GCNConv(2*n_latent, out_feats)
        self.fc=nn.Linear(out_feats,4)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        self.edge_index = edge_index
        x = self.act1(self.conv1(x, edge_index))
        x = self.conv3(x,edge_index)
        #x = self.act2(self.conv2(x, edge_index))
        #self.mean = self.conv3(x, edge_index)
        #self.log_std = self.conv4(x, edge_index)
        # reparametrize
        #self.z = self.mean + torch.randn_like(self.log_std) * torch.exp(self.log_std)
        x= scatter_mean(x, data.batch, dim=0)
        x=self.fc(x)
        x = F.log_softmax(x,dim=1)
        return x
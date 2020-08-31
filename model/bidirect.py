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
class EncoderTD(nn.Module):
    def __init__(self, n_total_features, n_latent,out_feats, p_drop=0.):
        super(EncoderTD, self).__init__()
        self.n_total_features = n_total_features
        self.conv1 = GCNConv(self.n_total_features, 2*n_latent)
        self.act1=nn.Sequential(nn.ReLU(),
                              nn.Dropout(p_drop))
        #self.conv2 = GCNConv(n_latent, n_latent)
        #self.act2 = nn.Sequential(nn.ReLU(),
        #                      nn.Dropout(p_drop))
        self.conv3 = GCNConv(2*n_latent, out_feats)
        self.conv4 = GCNConv(2*n_latent, out_feats)

    def loss(self):
        #### recon_loss
        EPS = 1e-15
        value = (self.z[self.edge_index[0]] * self.z[self.edge_index[1]]).sum(dim=1)
        value = torch.sigmoid(value)
        pos_loss = -torch.log(value + EPS).mean()

        # add self-loops
        pos_edge_index, _ = add_self_loops(self.edge_index)
        # negative
        neg_edge_index = negative_sampling(pos_edge_index, self.z.size(0))
        value_ = (self.z[neg_edge_index[0]] * self.z[neg_edge_index[1]]).sum(dim=1)
        value_ = torch.sigmoid(value_)
        neg_loss = -torch.log(1 - value_ + EPS).mean()
        
        ### kl loss
        kl = -0.5 / self.z.size(0) * torch.mean(
            torch.sum(1 + self.log_std - self.mean**2 - self.log_std.exp(), dim=1))
        return pos_loss + neg_loss + kl

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        self.edge_index = edge_index
        x = self.act1(self.conv1(x, edge_index))
        #x = self.act2(self.conv2(x, edge_index))
        self.mean = self.conv3(x, edge_index)
        self.log_std = self.conv4(x, edge_index)
        # reparametrize
        self.z = self.mean + torch.randn_like(self.log_std) * torch.exp(self.log_std)
        x= scatter_mean(self.z, data.batch, dim=0)
        return x

class EncoderBU(nn.Module):
    def __init__(self, n_total_features, n_latent,out_feats, p_drop=0.):
        super(EncoderBU, self).__init__()
        self.n_total_features = n_total_features
        self.conv1 = GCNConv(self.n_total_features, 2*n_latent)
        self.act1=nn.Sequential(nn.ReLU(),
                              nn.Dropout(p_drop))
        #self.conv2 = GCNConv(n_latent, n_latent)
        #self.act2 = nn.Sequential(nn.ReLU(),
        #                      nn.Dropout(p_drop))
        self.conv3 = GCNConv(2*n_latent, out_feats)
        self.conv4 = GCNConv(2*n_latent, out_feats)

    def loss(self):
        #### recon_loss
        EPS = 1e-15
        value = (self.z[self.edge_index[0]] * self.z[self.edge_index[1]]).sum(dim=1)
        value = torch.sigmoid(value)
        pos_loss = -torch.log(value + EPS).mean()

        # add self-loops
        pos_edge_index, _ = add_self_loops(self.edge_index)
        # negative
        neg_edge_index = negative_sampling(pos_edge_index, self.z.size(0))
        value_ = (self.z[neg_edge_index[0]] * self.z[neg_edge_index[1]]).sum(dim=1)
        value_ = torch.sigmoid(value_)
        neg_loss = -torch.log(1 - value_ + EPS).mean()
        
        ### kl loss
        kl = -0.5 / self.z.size(0) * torch.mean(
            torch.sum(1 + self.log_std - self.mean**2 - self.log_std.exp(), dim=1))
        return pos_loss + neg_loss + kl

    def forward(self, data):
        x, edge_index = data.x, data.BU_edge_index
        self.edge_index = edge_index
        x = self.act1(self.conv1(x, edge_index))
        #x = self.act2(self.conv2(x, edge_index))
        self.mean = self.conv3(x, edge_index)
        self.log_std = self.conv4(x, edge_index)
        # reparametrize
        self.z = self.mean + torch.randn_like(self.log_std) * torch.exp(self.log_std)
        x= scatter_mean(self.z, data.batch, dim=0)
        return x

class Net(nn.Module):
    def __init__(self,in_feats,hid_feats,out_feats):
        super(Net, self).__init__()
        self.encoderTD = EncoderTD(in_feats, hid_feats,out_feats)
        self.encoderBU = EncoderBU(in_feats, hid_feats,out_feats)
        self.fc=nn.Linear(out_feats*2,4)

    def loss(self):
        return self.encoderTD.loss() + self.encoderBU.loss()

    def forward(self, data):
        TD_x = self.encoderTD(data)
        BU_x = self.encoderBU(data)

        x = torch.cat((TD_x,BU_x), 1)
        x=self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x
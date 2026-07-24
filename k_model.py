# Define Model (XLSR-ORIG)
# by HH
import random
import numpy as np
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from typing import Callable, Optional, Union
#----------------------------------------------------------------------------------------------------
# Conv Layers
class ConvLayers(nn.Module):
    def __init__(self, input_dim=1024, proj_dim=256, conv_channels=32):
        super(ConvLayers, self).__init__()

        # 1. Conv1D projection: (B, T, 1024) → (B, T, 256)
        self.proj = nn.Conv1d(in_channels=input_dim,
                              out_channels=proj_dim,
                              kernel_size=1)

        # 2. Conv2D blocks: apply 3 times
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=conv_channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(conv_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(conv_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(conv_channels),
            nn.ReLU()
        )

    def forward(self, x):
        # x: (B, T, 1024)
        x = x.transpose(1, 2)          # (B, 1024, T)
        x = self.proj(x)               # (B, 256, T)
        x = x.transpose(1, 2)          # (B, T, 256)

        # Conv2D expects 4D: (B, C, H, W), so add channel dim
        x = x.unsqueeze(1)             # (B, 1, T, 256)
        x = self.convs(x)              # (B, 32, T, 256)
        x = x.permute(0, 2, 3, 1)      # (B, T, 256, 32) 
        return x
#-----------------------------------------------------------------------------------------------------
# SE-Re2blocks
class SELayer(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
    """
    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(SELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output tensor
        """
        batch_size, num_channels, H, W = input_tensor.size()
        # Average along each channel
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return output_tensor

class SERe2blocks(nn.Module):
    def __init__(self, input_dim=32, conv_channels=32, scale=4, reduction_ratio=2):
        super().__init__()
        assert conv_channels % scale == 0
        self.scale = scale
        self.width = conv_channels // scale

        # First 1x1 conv: input_dim → conv_channels
        self.conv1 = nn.Conv2d(input_dim, conv_channels, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(conv_channels)
        self.relu  = nn.ReLU(inplace=True)

        # 3x3 conv: 
        self.conv3 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.width, self.width, 3, padding=1, bias=False),
                nn.BatchNorm2d(self.width),
                nn.ReLU(inplace=True)
            ) for _ in range(scale - 1)
        ])

        # Last 1x1 conv: conv_channels → input_dim 
        self.conv_out = nn.Conv2d(conv_channels, input_dim, kernel_size=1, bias=False)
        self.bn3      = nn.BatchNorm2d(input_dim)
        self.se       = SELayer(num_channels=input_dim, reduction_ratio=reduction_ratio)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)   # (B, T, F, C) → (B, C, T, F)

        out = self.conv1(x)
        out = self.bn1(out)

        # Res2Net's 3x3 conv
        spl = torch.chunk(out, self.scale, dim=1)
        y = [spl[0]]
        for i in range(1, self.scale):
            yi = self.conv3[i-1](spl[i] + y[i-1])
            y.append(yi)

        out = torch.cat(y, dim=1)
        # Last 1x1 conv 
        out = self.conv_out(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.se(out)

        out = out.permute(0, 2, 3, 1)   # (B, C, T, F) → (B, T, F, C)
        
        return out
#----------------------------------------------------------------------------------------------------
# BLDL
class BiLSTM(nn.Module):
    def __init__(self, input_size, num_layers=2, hidden_size=256):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, bidirectional=True)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))  # out shape: (batch, seq_len, hidden_size*2)
        return out
    
class BLDL(nn.Module):
    def __init__(self, input_size=512, hidden_size=256, num_layers=2):
        super(BLDL, self).__init__()
        self.bilstm = BiLSTM(
                            input_size=input_size, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers)
        self.gap = nn.AdaptiveAvgPool1d(1)  # (B, hidden*2, T) → (B, hidden*2, 1)
        self.fc1 = nn.Linear(hidden_size * 2, 128)  
        self.fc2 = nn.Linear(128, 2)  

    def forward(self, x):
        """
        x: Tensor of shape (B, T, F, C)  e.g., (1, 25, 16, 32)
        """
        B, T, F, C = x.shape
        x_flat = x.reshape(B, T, F * C)  # (B, T, 512)
        bilstm_out = self.bilstm(x_flat)  # (B, T, hidden*2)
        
        out = bilstm_out + x_flat   # (B, T, 512)
        out = out.permute(0, 2, 1)  # (B, 512, T)
        out = self.gap(out)         # (B, 512, 1)
        out = out.squeeze(-1)       # (B, 512)
        out = self.fc1(out)         # FC1
        out = self.fc2(out)         # FC2
        return out
#----------------------------------------------------------------------------------------------------
# STJ-GAT
"""
AASIST
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()

        # attention map
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_weight = self._init_new_params(out_dim, 1)

        # project
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)

        # batch norm
        self.bn = nn.BatchNorm1d(out_dim)

        # dropout for inputs
        self.input_drop = nn.Dropout(p=0.2)

        # activate
        self.act = nn.SELU(inplace=True)

        # temperature
        self.temp = 1.
        if "temperature" in kwargs:
            self.temp = kwargs["temperature"]

    def forward(self, x):
        '''
        x   :(#bs, #node, #dim)
        '''
        # apply input dropout
        x = self.input_drop(x)

        # derive attention map
        att_map = self._derive_att_map(x)

        # projection
        x = self._project(x, att_map)

        # apply batch norm
        x = self._apply_BN(x)
        x = self.act(x)
        return x

    def _pairwise_mul_nodes(self, x):
        '''
        Calculates pairwise multiplication of nodes.
        - for attention map
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, #dim)
        '''

        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x.transpose(1, 2)

        return x * x_mirror

    def _derive_att_map(self, x):
        '''
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, 1)
        '''
        att_map = self._pairwise_mul_nodes(x)
        # size: (#bs, #node, #node, #dim_out)
        att_map = torch.tanh(self.att_proj(att_map))
        # size: (#bs, #node, #node, 1)
        att_map = torch.matmul(att_map, self.att_weight)

        # apply temperature
        att_map = att_map / self.temp

        att_map = F.softmax(att_map, dim=-2)

        return att_map

    def _project(self, x, att_map):
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)

        return x1 + x2

    def _apply_BN(self, x):
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        x = x.view(org_size)

        return x

    def _init_new_params(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out
    
class GraphPool(nn.Module):
    def __init__(self, k: float, in_dim: int, p: Union[float, int]):
        super().__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()
        self.in_dim = in_dim

    def forward(self, h):
        Z = self.drop(h)
        weights = self.proj(Z)
        scores = self.sigmoid(weights)
        new_h = self.top_k_graph(scores, h, self.k)

        return new_h

    def top_k_graph(self, scores, h, k):
        """
        args
        =====
        scores: attention-based weights (#bs, #node, 1)
        h: graph data (#bs, #node, #dim)
        k: ratio of remaining nodes, (float)

        returns
        =====
        h: graph pool applied data (#bs, #node', #dim)
        """
        _, n_nodes, n_feat = h.size()
        n_nodes = max(int(n_nodes * k), 1)
        _, idx = torch.topk(scores, n_nodes, dim=1)
        idx = idx.expand(-1, -1, n_feat)

        h = h * scores
        h = torch.gather(h, 1, idx)

        return h

class STJGAT(nn.Module):
    def __init__(self, in_channels=32, out_dim=32, k=0.5, dropout=0.2):
        super().__init__()
        # Mo (Attention map)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)
        self.activation = nn.SELU(inplace=True)
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, 1, kernel_size=1, padding=0)  

        # GAT for FC and TC
        self.gat_fc = GraphAttentionLayer(in_dim=in_channels, out_dim=out_dim)
        self.gat_tc = GraphAttentionLayer(in_dim=in_channels, out_dim=out_dim)

        # GPooling for FC and TC
        self.gpool_fc = GraphPool(k=k, in_dim=out_dim, p=dropout)
        self.gpool_tc = GraphPool(k=k, in_dim=out_dim, p=dropout)
        
        # Cross-attention weight matrices
        self.W1 = nn.Linear(out_dim, out_dim, bias=False)
        self.W2 = nn.Linear(out_dim, out_dim, bias=False)
        
        # FC for Classifier
        self.fc = nn.Linear(out_dim * 2, 2)

    def forward(self, x):
        """
        x: (B, T, F, C)  — SE-Re2blocks output
        return:
        FC_graph: (B, F', out_dim)
        TC_graph: (B, T', out_dim)
        """
        B, T, F_, C = x.shape
        
        # Mo
        x_perm = x.permute(0, 3, 1, 2)  # (B, C, T, F)
        out = self.conv1(x_perm)
        out = self.activation(out)
        out = self.bn(out)
        out = self.conv2(out)  # (B, 1, T, F)
        out = out.squeeze(1)   # (B, T, F)
        
        M0 = F.softmax(out, dim=1)  # (B, T, F)
        M0_exp = M0.unsqueeze(-1)  # (B, T, F, 1)

        FCatt = torch.sum(M0_exp * x, dim=1)  # (B, F, C)
        TCatt = torch.sum(M0_exp * x, dim=2)  # (B, T, C)

        # GAT
        FC_gat = self.gat_fc(FCatt)  # (B, F, out_dim)
        TC_gat = self.gat_tc(TCatt)  # (B, T, out_dim)

        # Gpooling
        FC_graph = self.gpool_fc(FC_gat)  # (B, F', out_dim)
        TC_graph = self.gpool_tc(TC_gat)  # (B, T', out_dim)
        
        # Cross Attention
        # FC_graph: (B, NF, d), TC_graph: (B, NT, d)
        xF = FC_graph
        xT = TC_graph

        # 1. W1, W2 projection
        xF_proj = self.W1(xF)  # (B, NF, d)
        xT_proj = self.W2(xT)  # (B, NT, d)

        # 2. M1 = softmax(xF_proj @ xT^T)
        M1 = torch.softmax(torch.matmul(xF_proj, xT.transpose(1, 2)), dim=-1)  # (B, NF, NT)

        # 3. M2 = softmax(xT_proj @ xF^T)
        M2 = torch.softmax(torch.matmul(xT_proj, xF.transpose(1, 2)), dim=-1)  # (B, NT, NF)

        # 4. Attention-based fusion
        xF_fused = xF + torch.matmul(M1, xT)  # (B, NF, d)
        xT_fused = xT + torch.matmul(M2, xF)  # (B, NT, d)

        # 5. Node-wise max pooling 
        node_F = torch.max(xF_fused, dim=1)[0]  # (B, d)
        node_T = torch.max(xT_fused, dim=1)[0]  # (B, d)

        # 6. Final output
        fused_node = torch.cat([node_F, node_T], dim=-1)  # (B, 2d)
        out = self.fc(fused_node)  # (B, 2) → binary classification

        return out
#---------------------------------------------------------------------------------------
# Model
class Permute(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)
    
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.convlayers = ConvLayers()  # (B, 1, 201, 256) → (B, 32, 201, 256)

        def pool_block(pool_kernel, pool_stride):
            return nn.Sequential(
                Permute(0, 3, 1, 2),  # (B, T, F, C) → (B, C, T, F)
                nn.AvgPool2d(kernel_size=pool_kernel, stride=pool_stride),
                Permute(0, 2, 3, 1),  # (B, C, T', F') → (B, T', F', C)
            )
            
        self.SEre2blocks = nn.Sequential(
            SERe2blocks(input_dim=32),  # Block 1
            nn.Sequential(             # Block 2: (201,256) → (100,128)
                pool_block(pool_kernel=2, pool_stride=2),
                SERe2blocks(input_dim=32)
            ),
            SERe2blocks(input_dim=32),  # Block 3
            nn.Sequential(             # Block 4: (100,128) → (50,64)
                pool_block(pool_kernel=2, pool_stride=2),
                SERe2blocks(input_dim=32)
            ),
            SERe2blocks(input_dim=32),  # Block 5
            nn.Sequential(             # Block 6: (50,64) → (25,32)
                pool_block(pool_kernel=2, pool_stride=2),
                SERe2blocks(input_dim=32)
            ),
            SERe2blocks(input_dim=32),  # Block 7
            nn.Sequential(             # Block 8: (25,32) → (25,16)
                pool_block(pool_kernel=(1, 2), pool_stride=(1, 2)),
                SERe2blocks(input_dim=32)
            ),
        )
        
        self.stjgat = STJGAT(in_channels=32, out_dim=32, k=0.5, dropout=0.2)
        self.bldl = BLDL(input_size=512, hidden_size=256, num_layers=2)

    def forward(self, x):
        x = self.convlayers(x)       # (B, 201, 256, 32)
        x = self.SEre2blocks(x)      # (B, 25, 16, 32)
        out_stj = self.stjgat(x)     # (B, 2) — logits
        out_bldl = self.bldl(x)      # (B, 2) — logits

        return out_stj, out_bldl
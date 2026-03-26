# Define Model (XLSR-ORIG)
# by HH
import math
import random
import pywt
import numpy as np
from typing import Union

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import WavLMModel

from typing import Callable, Optional, Union
#----------------------------------------------------------------------------------------------------
# Time Branch
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

        self.conv1 = nn.Conv2d(input_dim, conv_channels, kernel_size=1, bias=False)  
        self.bn1   = nn.BatchNorm2d(conv_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv3 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.width, self.width, 3, padding=1, bias=False),
                nn.BatchNorm2d(self.width),
                nn.ReLU(inplace=True)
            ) for _ in range(scale - 1)
        ])
        self.bn3   = nn.BatchNorm2d(input_dim)
        self.se    = SELayer(num_channels=input_dim, reduction_ratio=reduction_ratio)

    def forward(self, x):
        x = x.permute(0,3,1,2)
        out = self.conv1(x)
        out = self.bn1(out)

        spl = torch.chunk(out, self.scale, dim=1)
        y = [spl[0]]
        for i in range(1, self.scale):
            yi = self.conv3[i-1](spl[i] + y[i-1])
            y.append(yi)

        out = torch.cat(y, dim=1)
        out = self.conv1(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.se(out)
        out = out.permute(0,2,3,1)    # (B, T, 256, 32)
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
        self.fc_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            
            nn.Linear(64, 2) 
        )
        
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
        out = self.fc_head(out)     # (B, 2)

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

class STJGAT(nn.Module):
    def __init__(self, in_channels=32, out_dim=32, dropout=0.2):
        super().__init__()
        
        # Mo (Attention map)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)
        self.activation = nn.SELU(inplace=True)
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, 1, kernel_size=1, padding=0)  

        # GAT for TC
        self.gat_tc = GraphAttentionLayer(in_dim=in_channels, out_dim=out_dim)
        self.fc_proj = nn.Linear(in_features=25 * out_dim, out_features=512)
        self.fc_head = nn.Sequential(
            nn.Linear(512, 256),  
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),  
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),   
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            
            nn.Linear(64, 2)      
        )

    def forward(self, x):
        """
        x: (B, T, F, C)  — SE-Re2blocks output
        return: (B, 512)
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
        TCatt = torch.sum(M0_exp * x, dim=2)  # (B, T, C)

        # GAT
        TC_gat = self.gat_tc(TCatt)  # (B, T, out_dim)
        # 이 부분 PCA로 수정하기
        TC_comp = TC_gat.reshape(B, -1)
        out = self.fc_proj(TC_comp)
        out = self.fc_head(out)
   
        return out
#----------------------------------------------------------------------------------------------------
# Freq Branch
# SincNet (Original)
def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.contiguous()
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1, 
                      -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)

def sinc(band,t_right):
    y_right= torch.sin(2*math.pi*band*t_right)/(2*math.pi*band*t_right)
    y_left= flip(y_right,0)

    y=torch.cat([y_left,Variable(torch.ones(1)).cuda(),y_right])

    return y

class SincConv_fast(nn.Module):
    """Sinc-based convolution
    Parameters
    ----------
    in_channels : `int`
        Number of input channels. Must be 1.
    out_channels : `int`
        Number of filters.
    kernel_size : `int`
        Filter length.
    sample_rate : `int`, optional
        Sample rate. Defaults to 16000.
    Usage
    -----
    See `torch.nn.Conv1d`
    Reference
    ---------
    Mirco Ravanelli, Yoshua Bengio,
    "Speaker Recognition from raw waveform with SincNet".
    https://arxiv.org/abs/1808.00158
    """

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, out_channels=15, kernel_size=251, sample_rate=16000, in_channels=1,
                 stride=1, padding=0, dilation=1, bias=False, groups=1, min_low_hz=50, min_band_hz=50):

        super(SincConv_fast,self).__init__()

        if in_channels != 1:
            #msg = (f'SincConv only support one input channel '
            #       f'(here, in_channels = {in_channels:d}).')
            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size%2==0:
            self.kernel_size=self.kernel_size+1

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')

        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # initialize filterbanks such that they are equally spaced in Mel scale
        low_hz = 30
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)

        mel = np.linspace(self.to_mel(low_hz),
                          self.to_mel(high_hz),
                          self.out_channels + 1)
        hz = self.to_hz(mel)

        # filter lower frequency (out_channels, 1)
        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))

        # filter frequency band (out_channels, 1)
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))

        # Hamming window
        #self.window_ = torch.hamming_window(self.kernel_size)
        n_lin=torch.linspace(0, (self.kernel_size/2)-1, steps=int((self.kernel_size/2))) # computing only half of the window
        self.window_=0.54-0.46*torch.cos(2*math.pi*n_lin/self.kernel_size);

        # (1, kernel_size/2)
        n = (self.kernel_size - 1) / 2.0
        self.n_ = 2*math.pi*torch.arange(-n, 0).view(1, -1) / self.sample_rate # Due to symmetry, I only need half of the time axes

    def forward(self, waveforms):
        """
        Parameters
        ----------
        waveforms : `torch.Tensor` (batch_size, 1, n_samples)
            Batch of waveforms.
        Returns
        -------
        features : `torch.Tensor` (batch_size, out_channels, n_samples_out)
            Batch of sinc filters activations.
        """
        self.n_ = self.n_.to(waveforms.device)
        self.window_ = self.window_.to(waveforms.device)

        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_), self.min_low_hz, self.sample_rate/2)
        band = (high - low)[:, 0]

        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        band_pass_left = ((torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / (self.n_ / 2)) * self.window_
        band_pass_center = 2 * band.view(-1, 1)
        band_pass_right = torch.flip(band_pass_left, dims=[1])

        band_pass = torch.cat([band_pass_left, band_pass_center, band_pass_right], dim=1)
        band_pass = band_pass / (2 * band[:, None])

        filters = (band_pass).view(self.out_channels, 1, self.kernel_size)

        low_filters = filters[0:10, :, :]
        low_freq = F.conv1d(waveforms, low_filters, stride=self.stride,
                            padding=self.padding, dilation=self.dilation,
                            bias=None, groups=1) # (B, 10, 64350)

        high_filters = filters[10:15, :, :]
        high_freq = F.conv1d(waveforms, high_filters, stride=self.stride,
                             padding=self.padding, dilation=self.dilation,
                             bias=None, groups=1) # (B, 5, 64350)

        return low_freq, high_freq
# --------------------------------------------------------------------------------------------------
# LF
# Gating-Res2Net
class GatingRe2blocks(nn.Module):
    def __init__(self, out_channels=125, scale=5):
        super(GatingRe2blocks, self).__init__()
        self.scale = scale
        self.width = out_channels // scale

        self.conv1_list = nn.ModuleList([
            nn.Conv1d(2, self.width, kernel_size=1) for _ in range(scale)
        ])
        self.conv3_list = nn.ModuleList([
            nn.Conv1d(self.width, self.width, kernel_size=3, padding=1) for _ in range(scale - 1)
        ])
        self.gating_network = nn.Sequential(
            nn.Linear(out_channels, out_channels // 4), 
            nn.ReLU(inplace=True),
            nn.Linear(out_channels // 4, scale),        
            nn.Sigmoid()                                 
        )
        self.identity_proj = nn.Conv1d(10, out_channels, kernel_size=1)
        self.proj_final = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.bn_list = nn.ModuleList([nn.BatchNorm1d(self.width) for _ in range(scale)])
        self.relu = nn.ReLU(inplace=True)
        self.final_bn = nn.BatchNorm1d(out_channels)
        self.gap = nn.AdaptiveAvgPool1d(1)              
        self.ln = nn.LayerNorm(out_channels)         
        self.fc1 = nn.Linear(out_channels, 64)          # 125 -> 64
        self.leaky_relu = nn.LeakyReLU(0.2)             
        self.dropout = nn.Dropout(p=0.3)                
        self.fc2 = nn.Linear(64, 2)
        
    def forward(self, low_freq):
        identity = self.identity_proj(low_freq)
        low_grouped = torch.chunk(low_freq, self.scale, dim=1)

        y = [None] * self.scale
        y[4] = self.relu(self.bn_list[4](self.conv1_list[4](low_grouped[4])))

        for i in range(self.scale - 2, -1, -1):
            x_i = self.conv1_list[i](low_grouped[i])
            yi = self.conv3_list[i](x_i + y[i+1])
            y[i] = self.relu(self.bn_list[i](yi))

        # Gating
        ms_feat = torch.cat(y, dim=1)
        avg_pool = torch.mean(ms_feat, dim=-1)
        gate_weights = self.gating_network(avg_pool) 

        y_weighted = []
        for i in range(self.scale):
            w = gate_weights[:, i].view(-1, 1, 1)
            y_weighted.append(y[i] * w)

        ms_feat_gated = torch.cat(y_weighted, dim=1)
        ms_feat_gated = self.proj_final(ms_feat_gated)   
                   
        out = self.relu(self.final_bn(ms_feat_gated + identity))  # (B, 125, 64350)
        out_pooled = self.gap(out).squeeze(-1)          # (B, 125, 1) -> (B, 125)
        out_normed = self.ln(out_pooled)                     # (B, 2)
        
        x = self.fc1(out_normed)
        x = self.leaky_relu(x)                          
        x = self.dropout(x)
        out = self.fc2(x)                      # (B, 2)

        return out
# -------------------------------------------------------------------------------------
# V_1_F (sym2)
class WNN(nn.Module):
    def __init__(self, num_classes=2):
        super(WNN, self).__init__()
        wavelet = pywt.Wavelet('sym2')
        dec_lo = torch.tensor(wavelet.dec_lo[::-1], dtype=torch.float32).view(1, 1, -1)
        dec_hi = torch.tensor(wavelet.dec_hi[::-1], dtype=torch.float32).view(1, 1, -1)
        self.register_buffer('filter_lo', dec_lo)
        self.register_buffer('filter_hi', dec_hi)

        # feat channel : D1(8) + A2(12) + D2(12) = 32
        self.encoder = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=3, padding=1, bias=False), 
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 2, 64), 
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def sym2_op(self, x_combo):
        t = x_combo.shape[1]
        lo = self.filter_lo.repeat(t, 1, 1)
        hi = self.filter_hi.repeat(t, 1, 1)
        
        a = torch.nn.functional.conv1d(x_combo, lo, groups=t, stride=1)
        d = torch.nn.functional.conv1d(x_combo, hi, groups=t, stride=1)
        return a.transpose(1, 2), d.transpose(1, 2)

    def forward(self, x):
        """
        x: (B, 5, T) 
        """
        x_t = x.transpose(1, 2) # (B, T, 5)

        # --- Level-1 ---
        c1_1 = x_t[:, :, [0, 1, 2, 3]]  # (0, 1, 2, 3)
        c1_2 = x_t[:, :, [1, 2, 3, 4]]  # (1, 2, 3, 4)
        
        a1_list, d1_list = [], []
        for c in [c1_1, c1_2]:
            a, d = self.sym2_op(c)
            a1_list.append(a); d1_list.append(d)
        
        # Concat: (B, 4+4, T) -> (B, 8, T)
        # Index: (0, 1, 2, 3, 1, 2, 3, 4) 
        D1 = torch.cat(d1_list, dim=1) 
        A1 = torch.cat(a1_list, dim=1) 
        
        # --- Level-2 ---
        A1_t = A1.transpose(1, 2) # (B, T, 8)
        # C2_1: (0, 1, 2, 3) -> A1_t의 0~3
        # C2_2: (2, 3, 1, 2) -> A1_t의 2, 3 (C1_1) + 4, 5 (C1_2)
        # C2_3: (1, 2, 3, 4) -> A1_t의 4, 5, 6, 7 (C1_2)
        c2_1 = A1_t[:, :, [0, 1, 2, 3]]           
        c2_2 = A1_t[:, :, [2, 3, 4, 5]] 
        c2_3 = A1_t[:, :, [4, 5, 6, 7]] 
        
        a2_list, d2_list = [], []
        for c in [c2_1, c2_2, c2_3]:
            a, d = self.sym2_op(c)
            a2_list.append(a); d2_list.append(d)
            
        # Concat : (B, 4+4+4, T) -> (B, 12, T)
        A2 = torch.cat(a2_list, dim=1) 
        D2 = torch.cat(d2_list, dim=1) 
        feat = torch.cat([D1, A2, D2], dim=1) # (B, 32, T)
        
        x_enc = self.encoder(feat)
        stat = torch.cat([x_enc.mean(dim=-1), x_enc.std(dim=-1)], dim=1) # (B, 128)
        out = self.classifier(stat) # (B, 2)
        
        return out
#----------------------------------------------------------------------------------------------------
# Model
class Permute(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)
    
class AudioModel(nn.Module):
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
        
        self.stjgat = STJGAT(in_channels=32, out_dim=32, dropout=0.2)
        self.bldl = BLDL(input_size=512, hidden_size=256, num_layers=2)
        
    def forward(self, audio_x):
        audio_x = self.convlayers(audio_x)       # (B, 201, 256, 32)
        audio_x = self.SEre2blocks(audio_x)      # (B, 25, 16, 32)
        out_stj = self.stjgat(audio_x)     
        out_bldl = self.bldl(audio_x)    
        
        return out_stj, out_bldl

class FusionModel(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.wavlm_model = WavLMModel.from_pretrained("microsoft/wavlm-large").to(device)
        self.time_model = AudioModel().to(device)
        self.sinc_layer = SincConv_fast(out_channels=15).to(device) 
        self.low_branch = GatingRe2blocks().to(device)              
        self.high_branch = WNN().to(device)

        for param in self.wavlm_model.parameters():
            param.requires_grad = False

    def forward(self, audio_input):
        wavlm_features = self.wavlm_model(audio_input).last_hidden_state
        out_stj, out_bldl = self.time_model(wavlm_features)
        raw_audio = audio_input.unsqueeze(1) if audio_input.dim() == 2 else audio_input
        
        low_freq, high_freq = self.sinc_layer(raw_audio)
        out_low = self.low_branch(low_freq)
        out_high = self.high_branch(high_freq)
        
        return out_stj, out_bldl, out_low, out_high
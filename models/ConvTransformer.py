from doctest import OutputChecker
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from torchvision.models import resnet18

import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from torchvision import transforms

import torch.nn as nn
import torch

class MultiHeadConvAttn(nn.Module):
    def _init_(self, dim, n_heads, attn_drop_rate, fc_drop_rate, conv_bias, kernels):
        super(MultiHeadConvAttn, self)._init_()
        self.q_fc = nn.Linear(dim, dim, bias=False)
        self.k_fc = nn.Linear(dim, dim, bias=False)
        self.conv_bias = conv_bias
        if conv_bias:
            assert n_heads % len(kernels) == 0
            assert all(kernel % 2 == 1 for kernel in kernels)
            assert dim % len(kernels) == 0
            self.k_conv = nn.ModuleList(
                [
                    nn.Conv1d(dim, dim // len(kernels), kernel, padding=kernel // 2)
                    for kernel in kernels
                ]
            )
            self.gate = nn.Parameter(torch.ones(n_heads))
        self.v_fc = nn.Linear(dim, dim, bias=False)
        self.out_fc = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.fc_drop = nn.Dropout(fc_drop_rate)
        self.n_heads = n_heads

    def forward(self, q, k, v, mask=None):
        q = self.q_fc(q)
        q = q.view(q.shape[0], q.shape[1], self.n_heads, -1).permute(0, 2, 1, 3)
        if self.conv_bias:
            k_conv = k.permute(0, 2, 1)
            k_conv = torch.cat([conv(k_conv) for conv in self.k_conv], dim=1).permute(0, 2, 1)
            k_conv = k_conv.view(k_conv.shape[0], k_conv.shape[1], self.n_heads, -1).permute(
                0, 2, 1, 3
            )
        k = self.k_fc(k)
        k = k.view(k.shape[0], k.shape[1], self.n_heads, -1).permute(0, 2, 1, 3)
        v = self.v_fc(v)
        v = v.view(v.shape[0], v.shape[1], self.n_heads, -1).permute(0, 2, 1, 3)

        if self.conv_bias:
            gate = torch.sigmoid(self.gate.view(1, -1, 1, 1))
            attn = (1 - gate) * torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(q.shape[-1]) + gate * torch.matmul(q, k_conv.transpose(-2, -1)) / np.sqrt(q.shape[-1])
        else:
            attn = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(q.shape[-1])
        if mask is not None:
            mask = mask[:, None, :, :].repeat(1, self.n_heads, 1, 1)
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = self.attn_drop(torch.softmax(attn, dim=-1))

        out = torch.matmul(attn, v).permute(0, 2, 1, 3).contiguous()
        out = out.view(q.shape[0], q.shape[2], -1)
        out = self.fc_drop(self.out_fc(out))
        return out

class MultiHeadAttn(nn.Module):
    def __init__(self, dim, n_heads, attn_dropout_rate, fc_dropout_rate, vis=False):
        super(MultiHeadAttn, self).__init__()
        self.q_fc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1)
        self.k_fc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1)
        self.v_fc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1)
        self.out_fc = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(attn_dropout_rate)
        self.fc_dropout = nn.Dropout(fc_dropout_rate)
        self.n_heads = n_heads
        self.vis = vis

    def forward(self, q, k, v, mask=None):
        B, T, C, H, W = q.shape
        q = self.q_fc(q.reshape(-1, self.dim, 4, 4))
        q = q.view(B, q.shape[1], self.n_heads, -1).permute(0, 2, 1, 3)
        k = self.k_fc(k)
        k = k.view(k.shape[0], k.shape[1], self.n_heads, -1).permute(0, 2, 1, 3)
        v = self.v_fc(v)
        v = v.view(v.shape[0], v.shape[1], self.n_heads, -1).permute(0, 2, 1, 3)

        attn = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(q.shape[-1])
        if mask is not None:
            mask = mask[:, None, :, :].repeat(1, self.n_heads, 1, 1)
            attn = attn.masked_fill(mask == 0, -1e9)
        attn_weights = torch.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn_weights)
        weights = attn_weights if self.vis else None

        out = torch.matmul(attn, v).permute(0, 2, 1, 3).contiguous()
        out = out.view(q.shape[0], q.shape[2], -1)
        out = self.fc_dropout(self.out_fc(out))
        return out, weights

class FeedForward(nn.Module):
    def __init__(self, dim, hid_dim, dropout_rate):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.activ = nn.ReLU()

    def forward(self, x):
        x = self.activ(self.fc1(x))
        x = self.dropout(self.fc2(x))
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        def get_position_angle_vec(position):
            return [np.ones((4,4)) * (position / np.power(10000, 2 * (hid_j // 2) / d_hid)) for hid_j in range(d_hid)]
        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].to(x.device)

class EncoderBlock(nn.Module):
    def __init__(self, dim, n_heads, hid_dim, dropout_rate, vis):
        super(EncoderBlock, self).__init__()
        self.self_attn = MultiHeadAttn(dim, n_heads, 0.1, dropout_rate, vis)
        self.dec_enc_attn = MultiHeadAttn(dim, n_heads, 0.1, dropout_rate)
        self.ff = FeedForward(dim, hid_dim, dropout_rate)
        self.norm1 = nn.LayerNorm([dim,4,4])
        self.norm2 = nn.LayerNorm([dim,4,4])

    def forward(self, x, mask=None):
        residue = x
        x = self.norm1(x)
        x, weights = self.self_attn(x, x, x, mask)
        x = x + residue

        residue = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + residue

        return x, weights

class Encoder(nn.Module):
    def __init__(self, max_seq_len, n_layers, n_heads, hid_dim, dropout_rate, vis, pool):
        super(Encoder, self).__init__()
        dim = hid_dim
        self.dim = dim
        self.resnet = resnet18(pretrained=True)
        self.resnet.fc = nn.Identity()

        self.vis = vis
        self.pool = pool

        if self.pool != 'pool':
            max_seq_len += 1
        self.max_seq_len = max_seq_len
        self.pos_embed = PositionalEncoding(dim, max_seq_len)

        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim, 4, 4))

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(EncoderBlock(dim, n_heads, hid_dim, dropout_rate, vis))
        self.norm = nn.LayerNorm([dim,4,4])
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x_3d):
        b, tm, n, _, _ = x_3d.shape

        attn_weights = []
        cnn_embed_seq = []
        for t in range(x_3d.size(1)):
            x = self.resnet(x_3d[:, t, :, :, :]).reshape(b, 32, 4, 4)
            cnn_embed_seq.append(x)
        
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=1)

        print("CNN Embed Seq Shape : " , cnn_embed_seq.shape)

        if self.pool != 'pool':
            cls_temporal_tokens = repeat(self.temporal_token, '() n d h w -> b n d h w', b=b)
            cnn_embed_seq = torch.cat((cls_temporal_tokens, cnn_embed_seq), dim=1)
        
        x = self.dropout(self.pos_embed(cnn_embed_seq))
        out = self.norm(x.reshape(-1, self.dim, 4, 4)).reshape(b, tm+1, self.dim, 4, 4)

        for layer in self.layers:
            out, weights = layer(out)
            if self.vis:
                attn_weights.append(weights)
        return out, attn_weights

class CNNTrans(nn.Module):
    def __init__(self, pool='pool', vis=False):
        super(CNNTrans, self).__init__()
        self.encoder = Encoder(100, 4, 4, 32, 0.1, vis, pool)
        self.out_fc = nn.Linear(128, 2)
        self.pool = pool
        self.vis = vis

    def forward(self, x):
        out, weights = self.encoder(x)
        out = out.mean(dim = 1) if self.pool == 'pool' else out[:, 0]
        out = self.out_fc(out)
        if self.vis:
            return out, weights
        else:
            return out

def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000

if __name__ == "__main__":
    img = torch.randn((2, 10, 3, 224, 224))#.cuda()
    model = CNNTrans(pool='mean')#.cuda()
    print(count_parameters(model))

    out = model(img)
    print(out.shape)
# from typing import Type, Tuple, Optional, Set, List, Union

import math
import numpy as np
import torch
import torch.nn as nn
from timm.models.layers import drop_path, trunc_normal_, Mlp, DropPath

from graph.sign_27 import Graph
from model.TC import TemporalConvLayer
from model.gcn_drop import TCN_GCN_drop

def partition1(input, partition_size):
    B, C, T, V = input.shape
    partitions = input.view(B, C, T // partition_size[0], partition_size[0], V // partition_size[1], partition_size[1])
    partitions = partitions.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, partition_size[0], partition_size[1], C)
    return partitions

def reverse1(partitions, original_size, partition_size):
    T, V = original_size
    B = int(partitions.shape[0] / (T * V / partition_size[0] / partition_size[1]))
    output = partitions.view(B, T // partition_size[0], V // partition_size[1], partition_size[0], partition_size[1], -1)
    output = output.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, -1, T, V)
    return output

def partition2(input, partition_size):
    B, C, T, V = input.shape
    partitions = input.view(B, C, partition_size[0], T // partition_size[0], V // partition_size[1], partition_size[1])
    partitions = partitions.permute(0, 3, 4, 2, 5, 1).contiguous().view(-1, partition_size[0], partition_size[1], C)
    return partitions

def reverse2(partitions, original_size, partition_size):
    T, V = original_size
    B = int(partitions.shape[0] / (T * V / partition_size[0] / partition_size[1]))
    output = partitions.view(B, T // partition_size[0], V // partition_size[1], partition_size[0], partition_size[1], -1)
    output = output.permute(0, 5, 3, 1, 2, 4).contiguous().view(B, -1, T, V)
    return output


def get_relative_position_index_1d(T):
    coords = torch.stack(torch.meshgrid([torch.arange(T)]))
    coords_flatten = torch.flatten(coords, 1)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += T - 1
    return relative_coords.sum(-1)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, in_channels, num_heads=32, partition_size=(1, 1), attn_drop=0.):
        super(MultiHeadSelfAttention, self).__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.partition_size = partition_size
        self.scale = num_heads ** -0.5
        self.attn_area = partition_size[0] * partition_size[1]
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.softmax = nn.Softmax(dim=-1)

        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * partition_size[0] - 1), num_heads))
        self.register_buffer("relative_position_index", get_relative_position_index_1d(partition_size[0]))
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.ones = torch.ones(partition_size[1], partition_size[1], num_heads)

    def _get_relative_positional_bias(self):  
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.partition_size[0], self.partition_size[0], -1)
            relative_position_bias = relative_position_bias.unsqueeze(1).unsqueeze(3).repeat(1, self.partition_size[1], 1, self.partition_size[1], 1, 1).view(self.attn_area, self.attn_area, -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            return relative_position_bias.unsqueeze(0)

    def forward(self, input):
        B_, N, C = input.shape
        qkv = input.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # 是否可以增加一个Linear
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)             # 计算了32个组之间的attention

        # if self.rel:
        attn = attn + self._get_relative_positional_bias()
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        output = (attn @ v).transpose(1, 2).reshape(B_, N, -1)
        return output


class GroupAttn(nn.Module):
    def __init__(self, in_channels, type, num_heads):
        super(GroupAttn, self).__init__()
        self.qkv = nn.Linear(in_channels, in_channels * 3)
        self.proj = nn.Linear(in_features=in_channels, out_features=3 * in_channels, bias=True)
        self.num_heads = num_heads
        self.type = type

    def forward(self, input):
        B, C, T, V = input.shape
        G = 3
        x = input.view(B, C, T, G, V // G).permute(0, 2, 3, 4, 1)  # B T G V//G C    N=V//G
        qkv = self.qkv(x).reshape(B*T*G, V//G, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # 3 B_ head N C//heads
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * (self.num_heads ** -0.5)
        attn = attn.softmax(dim=-1)

        output = (attn @ v).transpose(1, 2).reshape(B, T, V, C)
        output = self.proj(output).permute(0, 3, 1, 2)
        return output


class VSformer(nn.Module):
    def __init__(self, in_channels, out_channels, num_points=50, kernel_size=7, num_heads=32,
                 type=(8, 4), attn_drop=0., drop=0., drop_path=0., mlp_ratio=4.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, downsample=False):
        super(VSformer, self).__init__()

        if downsample:
            self.downsample = PatchMergingTconv(in_channels, out_channels, kernel_size=kernel_size)
        else:
            self.downsample = None

        self.partition_function = [partition1, partition2]
        self.reverse_function = [reverse1, reverse2]
        self.type = type

        self.norm_1 = norm_layer(out_channels)
        self.mapping = nn.Linear(in_features=out_channels, out_features=out_channels, bias=True)
        self.gconv = nn.Parameter(torch.zeros(num_heads // (2 * 2), num_points, num_points))
        trunc_normal_(self.gconv, std=.02)
        self.tconv = nn.Conv2d(out_channels // (2 * 2), out_channels // (2 * 2), kernel_size=(kernel_size, 1),
                               padding=((kernel_size - 1) // 2, 0), groups=num_heads // (2 * 2))

        # Attention layers
        attention = []
        pair_attn = []
        for i in range(2):
            attention.append(
                # MultiHeadSelfAttention(in_channels=out_channels // (2 * 2),
                #                        # rel_type=self.rel_type[i],
                #                        num_heads=num_heads // (2 * 2),
                #                        partition_size=type, attn_drop=attn_drop)
                TemporalConvLayer(in_channels=in_channels // (len(self.partition_function) * 2) * 3,
                              out_channels=in_channels // (len(self.partition_function) * 2),
                              kernel_size=5, padding='causal', num_nodes=24)
            )
            pair_attn.append(
                GroupAttn(in_channels=out_channels // (2 * 2),
                          type=type[1],
                          num_heads=4)
            )
        self.pair_attn = nn.ModuleList(pair_attn)
        self.attention = nn.ModuleList(attention)

        self.proj = nn.Linear(in_features=out_channels, out_features=out_channels, bias=True)
        self.proj_drop = nn.Dropout(p=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm_2 = norm_layer(out_channels)
        self.mlp = Mlp(in_features=out_channels, hidden_features=int(mlp_ratio * out_channels),
                       act_layer=act_layer, drop=drop)

    def forward(self, input):
        if self.downsample is not None:
            input = self.downsample(input)

        B, C, T, V = input.shape

        input = input.permute(0, 2, 3, 1).contiguous()
        skip = input

        f = self.mapping(self.norm_1(input)).permute(0, 3, 1, 2).contiguous()

        f_conv, f_attn = torch.split(f, [C // 2, C // 2], dim=1)
        y = []

        # GCN
        split_f_conv = torch.chunk(f_conv, 2, dim=1)
        y_gconv = []
        split_f_gconv = torch.chunk(split_f_conv[0], self.gconv.shape[0], dim=1)
        for i in range(self.gconv.shape[0]):
            z = torch.einsum('n c t u, v u -> n c t v', split_f_gconv[i], self.gconv[i])
            y_gconv.append(z)
        y.append(torch.cat(y_gconv, dim=1))  # N C T V

        # TCN
        y.append(self.tconv(split_f_conv[1]))

        split_f_attn = torch.chunk(f_attn, 2, dim=1)

        for i in range(2):
            split = split_f_attn[i]
            output_attn = self.pair_attn[i](split)
            # C = output_attn.shape[1]
            # input_partitioned = self.partition_function[i](output_attn, self.type)
            # input_partitioned = input_partitioned.view(-1, self.type[0] * self.type[1], C)
            # y.append(self.reverse_function[i](self.attention[i](input_partitioned), (T, V), self.type))

            output_attn = self.attention[i](output_attn)
            y.append(output_attn)

        output = self.proj(torch.cat(y, dim=1).permute(0, 2, 3, 1).contiguous())
        output = self.proj_drop(output)
        output = skip + self.drop_path(output)

        # Feed Forward
        output = output + self.drop_path(self.mlp(self.norm_2(output)))
        output = output.permute(0, 3, 1, 2).contiguous()

        return output


# Down sampling
class PatchMergingTconv(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=7, stride=2, dilation=1):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.reduction = nn.Conv2d(dim_in, dim_out, kernel_size=(kernel_size, 1), padding=(pad, 0), stride=(stride, 1),
                                   dilation=(dilation, 1))
        self.bn = nn.BatchNorm2d(dim_out)

    def forward(self, x):
        x = self.bn(self.reduction(x))
        return x


class VSNet(nn.Module):
    def __init__(self, in_channels=3, channels=(96, 192, 192, 192), num_classes=60,
                 embed_dim=96, num_people=2, num_points=50, kernel_size=7, num_heads=32,
                 type=(8, 4), attn_drop=0., mlp_ratio=4.):

        super(VSNet, self).__init__()
        self.num_classes: int = num_classes
        self.embed_dim = embed_dim

        graph1 = Graph(labeling_mode='spatial', graph='wlasl_44')
        graph2 = Graph(labeling_mode='spatial', graph='wlasl_36')
        graph3 = Graph(labeling_mode='spatial', graph='wlasl_24')

        self.l1 = TCN_GCN_drop(in_channels, 64, graph1.A, [[6,6,8,8,8,8],[0,0,2,2,2,2]], groups=8, num_point=graph1.num_node,
                               block_size=41, residual=False)
        self.l2 = TCN_GCN_drop(64, embed_dim, graph2.A, [[6,6,6,6,6,6],[2,2,2,2,2,2]], groups=8, num_point=graph2.num_node, block_size=41)
        self.l3 = TCN_GCN_drop(embed_dim, embed_dim, graph3.A, None, groups=8, num_point=graph3.num_node, block_size=41)


        self.joint_person_embedding = nn.Parameter(torch.zeros(embed_dim, num_points * num_people))
        trunc_normal_(self.joint_person_embedding, std=.02)

        drop_path = torch.linspace(0.0, 0.2, 8).tolist()

        assert embed_dim == channels[0]

        act_layer = nn.GELU
        norm_layer = nn.LayerNorm
        self.l4 = VSformer(in_channels=embed_dim, out_channels=channels[0], num_points=num_points, kernel_size=kernel_size,
                           num_heads=num_heads, type=type, attn_drop=attn_drop, drop=0., drop_path=drop_path[0],
                           mlp_ratio=mlp_ratio, act_layer=act_layer, norm_layer=norm_layer, downsample=False)

        self.l5 = VSformer(in_channels=channels[0], out_channels=channels[0], num_points=num_points, kernel_size=kernel_size,
                           num_heads=num_heads, type=type, attn_drop=attn_drop, drop=0., drop_path=drop_path[1],
                           mlp_ratio=mlp_ratio, act_layer=act_layer, norm_layer=norm_layer, downsample=False)

        self.l6 = VSformer(in_channels=channels[0], out_channels=channels[1], num_points=num_points, kernel_size=kernel_size,
                           num_heads=num_heads, type=type, attn_drop=attn_drop, drop=0., drop_path=drop_path[2],
                           mlp_ratio=mlp_ratio, act_layer=act_layer, norm_layer=norm_layer, downsample=True)

        self.l7 = VSformer(in_channels=channels[1], out_channels=channels[1], num_points=num_points, kernel_size=kernel_size,
                           num_heads=num_heads, type=type, attn_drop=attn_drop, drop=0., drop_path=drop_path[3],
                           mlp_ratio=mlp_ratio, act_layer=act_layer, norm_layer=norm_layer, downsample=False)

        self.l8 = VSformer(in_channels=channels[1], out_channels=channels[2], num_points=num_points, kernel_size=kernel_size,
                           num_heads=num_heads, type=type, attn_drop=attn_drop, drop=0., drop_path=drop_path[4],
                           mlp_ratio=mlp_ratio, act_layer=act_layer, norm_layer=norm_layer, downsample=True)

        self.l9 = VSformer(in_channels=channels[2], out_channels=channels[2], num_points=num_points, kernel_size=kernel_size,
                           num_heads=num_heads, type=type, attn_drop=attn_drop, drop=0., drop_path=drop_path[5],
                           mlp_ratio=mlp_ratio, act_layer=act_layer, norm_layer=norm_layer, downsample=False)

        self.l10 = VSformer(in_channels=channels[2], out_channels=channels[3], num_points=num_points, kernel_size=kernel_size,
                           num_heads=num_heads, type=type, attn_drop=attn_drop, drop=0., drop_path=drop_path[6],
                           mlp_ratio=mlp_ratio, act_layer=act_layer, norm_layer=norm_layer, downsample=True)

        self.l11 = VSformer(in_channels=channels[3], out_channels=channels[3], num_points=num_points, kernel_size=kernel_size,
                           num_heads=num_heads, type=type, attn_drop=attn_drop, drop=0., drop_path=drop_path[7],
                           mlp_ratio=mlp_ratio, act_layer=act_layer, norm_layer=norm_layer, downsample=False)

        self.fc = nn.Linear(channels[-1], num_classes)

    def pos_emb(self, output, abs_pos):
        B, C, T, V = output.shape

        te = torch.zeros(B, T, self.embed_dim).to(output.device)  # B, T, C
        div_term = torch.exp(
            (torch.arange(0, self.embed_dim, 2, dtype=torch.float) * -(math.log(10000.0) / self.embed_dim))).to(
            output.device)
        te[:, :, 0::2] = torch.sin(abs_pos.unsqueeze(-1).float() * div_term)
        te[:, :, 1::2] = torch.cos(abs_pos.unsqueeze(-1).float() * div_term)
        return output + torch.einsum('b t c, c v -> b c t v', te, self.joint_person_embedding)

    def forward(self, input, index_t, keep_prob):
        B, C, T, V, M = input.shape

        output = input.permute(0, 1, 2, 4, 3).contiguous().view(B, C, T, -1)  # [B, C, T, M * V]
        output = self.l1(output, 1.)
        output = self.l2(output, keep_prob)
        output = self.l3(output, keep_prob)

        output = self.pos_emb(output, index_t)

        output = self.l4(output)
        output = self.l5(output)
        output = self.l6(output)
        output = self.l7(output)
        output = self.l8(output)
        output = self.l9(output)
        output = self.l10(output)
        output = self.l11(output)

        output = output.mean(dim=(2, 3))
        return self.fc(output)


def VSNet_(**kwargs):
    return VSNet(
        channels=(96, 192, 192, 192),
        embed_dim=96,
        **kwargs
    )

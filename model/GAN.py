import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer4D(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.0, alpha=0.2, concat=True):
        """
        图注意力层，支持输入形状 [B, C, T, V] 的张量。

        Args:
            in_features (int): 输入特征维度 C。
            out_features (int): 输出特征维度。
            dropout (float): Dropout 概率。
            alpha (float): LeakyReLU 的负斜率。
            concat (bool): 是否在输出时使用 ELU 激活并拼接多头注意力。
        """
        super(GraphAttentionLayer4D, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        # 权重矩阵
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        # 注意力权重向量
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        # LeakyReLU 激活
        self.leakyrelu = nn.LeakyReLU(self.alpha)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        前向传播

        Args:
            x (torch.Tensor): 输入特征张量，形状为 [B, C, T, V]。
            adj (torch.Tensor): 邻接矩阵，形状为 [B, V, V]。

        Returns:
            torch.Tensor: 输出特征张量，形状为 [B, out_features, T, V]。
        """
        B, C, T, V = x.shape  # 获取输入形状

        # Step 1: 线性变换
        x = x.permute(0, 3, 2, 1)  # 调整为 [B, V, T, C]
        Wh = torch.matmul(x, self.W)  # 线性变换，输出 [B, V, T, out_features]

        # Step 2: 构造注意力输入
        Wh1 = Wh.unsqueeze(2).repeat(1, 1, V, 1, 1)  # [B, V, V, T, out_features]
        Wh2 = Wh.unsqueeze(1).repeat(1, V, 1, 1, 1)  # [B, V, V, T, out_features]
        a_input = torch.cat([Wh1, Wh2], dim=-1)  # [B, V, V, T, 2*out_features]

        # Step 3: 计算注意力分数
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))  # [B, V, V, T]

        # Step 4: 注意力归一化
        attention = F.softmax(e, dim=2)  # 在邻居维度 V 上进行 softmax
        attention = self.dropout(attention)  # Dropout 防止过拟合

        # Step 5: 注意力加权特征聚合
        h_prime = torch.einsum("bvvt,bvto->bvto", attention, Wh)  # [B, V, T, out_features]

        # Step 6: 输出变换
        h_prime = h_prime.permute(0, 3, 2, 1)  # 调整为 [B, out_features, T, V]
        if self.concat:
            return F.elu(h_prime)  # 非线性激活
        else:
            return h_prime

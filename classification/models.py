from thop import profile
from thop import clever_format
from einops import rearrange, repeat
import math
from pointnet2_ops import pointnet2_utils
from torch import einsum
import torch.nn as nn
import torch
import torch.nn.functional as F
from utils.tools import *


class standard_transformer(nn.Module):
    def __init__(self, small_dim=1024, large_dim=512, k=16, class_num=40):
        super(standard_transformer, self).__init__()
        self.embedding = Embedding_Layer(in_channel=3, embedding_channel=64)
        self.sample_1 = Multi_Grouping(channel=64, groups=512, kneighbors=k, use_xyz=False, normalize="center")
        self.sample_2 = Multi_Grouping(channel=128, groups=256, kneighbors=k, use_xyz=False, normalize="center")
        self.sample_3 = Multi_Grouping(channel=256, groups=128, kneighbors=k, use_xyz=False, normalize="center")
        self.sample_4 = Multi_Grouping(channel=512, groups=64, kneighbors=k, use_xyz=False, normalize="center")
        self.class_token_small = nn.Parameter(torch.randn(1, 1, small_dim))
        self.class_token_large = nn.Parameter(torch.randn(1, 1, large_dim))
        self.net = AttBlock(large_dim, small_dim, 8)
        self.mlp_head_large = nn.Sequential(
            nn.Linear(large_dim, large_dim // 2),
            nn.BatchNorm1d(large_dim // 2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.5),
            nn.Linear(large_dim // 2, class_num)
        )

        self.mlp_head_small = nn.Sequential(
            nn.Linear(small_dim, small_dim // 2),
            nn.BatchNorm1d(small_dim // 2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.5),
            nn.Linear(small_dim // 2, small_dim // 4),
            nn.BatchNorm1d(small_dim // 4),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.5),
            nn.Linear(small_dim // 4, class_num)
        )

    def forward(self, x):
        xyz = x.permute(0, 2, 1)  # B, N, C
        batch_size, _, _ = x.size()
        x = self.embedding(x).permute(0, 2, 1)  # B, N, C

        xyz, x = self.sample_1(xyz, x)
        xyz, x = self.sample_2(xyz, x)
        xyz, x1 = self.sample_3(xyz, x)
        xyz, x2 = self.sample_4(xyz, x1)

        class_token_small = repeat(self.class_token_small, '() n d -> b n d', b=batch_size)
        class_token_large = repeat(self.class_token_large, '() n d -> b n d', b=batch_size)
        x1 = torch.cat([class_token_large, x1], dim=1)
        x2 = torch.cat([class_token_small, x2], dim=1)
        x1, x2 = self.net(x1, x2)
        x1 = self.mlp_head_large(x1)
        x2 = self.mlp_head_small(x2)
        x = x1 + x2
        # x = self.classifier(x)
        return x


class PointCAT(nn.Module):
    # noinspection PyDefaultArgument
    def __init__(self, small_dim=1024, large_dim=512, k=16, class_num=40):
        super(PointCAT, self).__init__()
        self.embedding = Embedding_Layer(in_channel=3, embedding_channel=64)
        self.sample_1 = Multi_Grouping(channel=64, groups=512, kneighbors=k, use_xyz=False, normalize="center")
        self.sample_2 = Multi_Grouping(channel=128, groups=256, kneighbors=k, use_xyz=False, normalize="center")
        self.sample_3 = Multi_Grouping(channel=256, groups=128, kneighbors=k, use_xyz=False, normalize="center")
        self.sample_4 = Multi_Grouping(channel=512, groups=64, kneighbors=k, use_xyz=False, normalize="center")

        self.class_token_small = nn.Parameter(torch.randn(1, 1, small_dim))
        self.class_token_large = nn.Parameter(torch.randn(1, 1, large_dim))

        self.cross_attn_layer = CrossAttBlock(large_dim=large_dim, small_dim=small_dim, cross_attn_depth=2,
                                              cross_attn_heads=8, channels=large_dim)

        self.mlp_head_large = nn.Sequential(
            nn.Linear(large_dim, large_dim // 2),
            nn.BatchNorm1d(large_dim // 2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.5),
            nn.Linear(large_dim // 2, class_num)
        )

        self.mlp_head_small = nn.Sequential(
            nn.Linear(small_dim, small_dim // 2),
            nn.BatchNorm1d(small_dim // 2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.5),
            nn.Linear(small_dim // 2, small_dim // 4),
            nn.BatchNorm1d(small_dim // 4),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.5),
            nn.Linear(small_dim // 4, class_num)
        )

    def forward(self, x):
        xyz = x.permute(0, 2, 1)  # B, N, C
        batch_size, _, _ = x.size()
        x = self.embedding(x).permute(0, 2, 1)  # B, N, C

        xyz, x = self.sample_1(xyz, x)
        xyz, x = self.sample_2(xyz, x)
        xyz, x1 = self.sample_3(xyz, x)
        xyz, x2 = self.sample_4(xyz, x1)

        class_token_small = repeat(self.class_token_small, '() n d -> b n d', b=batch_size)
        class_token_large = repeat(self.class_token_large, '() n d -> b n d', b=batch_size)

        x1 = torch.cat([class_token_large, x1], dim=1)
        x2 = torch.cat([class_token_small, x2], dim=1)

        x1, x2 = self.cross_attn_layer(x1, x2)
        x1 = self.mlp_head_large(x1)
        x2 = self.mlp_head_small(x2)
        x = x1 + x2
        # x = self.classifier(x)
        return x


class Multi_Grouping(nn.Module):
    def __init__(self, channel, groups, kneighbors, use_xyz, normalize="center", double_feature=True):
        super(Multi_Grouping, self).__init__()
        self.double_feature = double_feature
        self.grouper = LocalGrouper(channel=channel, groups=groups, kneighbors=kneighbors, use_xyz=use_xyz,
                                    normalize=normalize, double_feature=double_feature)
        if self.double_feature is True:
            self.net = Local_Aggregation(2 * channel, 2 * channel)
        else:
            self.net = Local_Aggregation(channel, channel)

    def forward(self, xyz, x):  # B, N, C
        batch_size, _, c = x.shape
        new_xyz, new_feature = self.grouper(xyz, x)
        x = self.net(new_feature)
        return new_xyz, x


class CrossAttBlock(nn.Module):
    def __init__(self, large_dim, small_dim, cross_attn_depth, cross_attn_heads, channels):
        super(CrossAttBlock, self).__init__()
        self.cross_att1 = CrossAttEncoder(small_dim=small_dim, large_dim=large_dim, cross_attn_depth=cross_attn_depth,
                                          cross_attn_heads=cross_attn_heads, channels=channels)
        self.cross_att2 = CrossAttEncoder(small_dim=small_dim, large_dim=large_dim, cross_attn_depth=cross_attn_depth,
                                          cross_attn_heads=cross_attn_heads, channels=channels)
        self.cross_att3 = CrossAttEncoder(small_dim=small_dim, large_dim=large_dim, cross_attn_depth=cross_attn_depth,
                                          cross_attn_heads=cross_attn_heads, channels=channels)
        self.cross_att4 = CrossAttEncoder(small_dim=small_dim, large_dim=large_dim, cross_attn_depth=cross_attn_depth,
                                          cross_attn_heads=cross_attn_heads, channels=channels)

    def forward(self, xl, xs):
        batch_size, _, _ = xl.shape
        xl, xs = self.cross_att1(xl, xs)
        xl, xs = self.cross_att2(xl, xs)
        xl, xs = self.cross_att3(xl, xs)
        xl, xs = self.cross_att4(xl, xs)

        # xl = xl[:, 0]
        # xs = xs[:, 0]
        xl = F.adaptive_max_pool1d(xl.permute(0, 2, 1), 1).view(batch_size, -1)
        xs = F.adaptive_max_pool1d(xs.permute(0, 2, 1), 1).view(batch_size, -1)

        return xl, xs


class AttBlock(nn.Module):
    def __init__(self, large_dim, small_dim, cross_attn_heads):
        super(AttBlock, self).__init__()

        self.att1 = nn.Sequential()
        self.att2 = nn.Sequential()
        for _ in range(16):
            self.att1.append(
                MultiHeadAtt_Block(in_feature=large_dim, num_heads=cross_attn_heads, d_model=large_dim)
            )
            self.att2.append(
                MultiHeadAtt_Block(in_feature=small_dim, num_heads=cross_attn_heads, d_model=small_dim)
            )

    def forward(self, xl, xs):
        batch_size, _, _ = xl.shape
        xl = self.att1(xl)
        xs = self.att2(xs)
        xl = xl[:, 0]
        xs = xs[:, 0]
        # xl = F.adaptive_max_pool1d(xl.permute(0, 2, 1), 1).view(batch_size, -1)
        # xs = F.adaptive_max_pool1d(xs.permute(0, 2, 1), 1).view(batch_size, -1)

        return xl, xs


class AttOperation(nn.Module):
    def __init__(self, layer):
        super(AttOperation, self).__init__()
        self.layer = layer
        self.bn = nn.BatchNorm1d(256)
        self.act = nn.LeakyReLU(negative_slope=0.2)

        if self.layer == 1:
            self.net = nn.Sequential(
                MultiHeadAtt_Block(in_feature=64, num_heads=8, d_model=64),
                MultiHeadAtt_Block(in_feature=64, num_heads=8, d_model=64)
                # nn.BatchNorm1d(64),
            )
        if self.layer == 2:
            self.net = nn.Sequential(
                MultiHeadAtt_Block(in_feature=128, num_heads=8, d_model=128),
                MultiHeadAtt_Block(in_feature=128, num_heads=8, d_model=128)
                # nn.BatchNorm1d(128),
            )
        if self.layer == 3:
            self.net = nn.Sequential(
                MultiHeadAtt_Block(in_feature=256, num_heads=8, d_model=256),
                MultiHeadAtt_Block(in_feature=256, num_heads=8, d_model=256)
                # nn.BatchNorm1d(256),
            )
        if self.layer == 4:
            self.net = nn.Sequential(
                MultiHeadAtt_Block(in_feature=256, num_heads=16, d_model=256),
                MultiHeadAtt_Block(in_feature=256, num_heads=16, d_model=256)
            )

    def forward(self, x):
        x_r = self.net(x)
        x_r = self.act(self.bn(x - x_r))
        x = x + x_r
        return x


class ConvBNReLURes1D(nn.Module):
    def __init__(self, channel, kernel_size=1, groups=1, res_expansion=1.0, bias=True, activation='relu'):
        super(ConvBNReLURes1D, self).__init__()
        self.act = nn.ReLU()
        self.net1 = nn.Sequential(
            nn.Conv1d(in_channels=channel, out_channels=int(channel * res_expansion),
                      kernel_size=kernel_size, groups=groups, bias=bias),
            nn.BatchNorm1d(int(channel * res_expansion)),
            self.act
        )

    def forward(self, x):
        return self.act((self.net1(x)) + x)


class MultiHeadAtt_Block(nn.Module):
    def __init__(self, in_feature, num_heads, d_model):
        super(MultiHeadAtt_Block, self).__init__()
        self.heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.scale = 1 / math.sqrt(self.d_k)
        self.linear_trans_q = Pre_Linear(in_feature, num_heads, self.d_k)
        self.linear_trans_k = Pre_Linear(in_feature, num_heads, self.d_k)
        self.linear_trans_v = Pre_Linear(in_feature, num_heads, self.d_k)
        self.FeedForward = nn.Linear(in_feature, in_feature)
        self.drop_out = nn.Dropout(0.1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x = self.layer_norm(x)
        # norm = self.res_norm_layer(xyz.permute(0, 2, 1))  # B, N, C
        x = x.permute(1, 0, 2)  # N, B, C
        query = self.linear_trans_q(x)  # N, B, C -> (-1, num_heads, d_k)
        key = self.linear_trans_k(x)  # N, B, H, d_k
        value = self.linear_trans_v(x)
        N, B, H, d_k = query.shape
        scores = torch.einsum('ibhd, jbhd -> ijbh', query, key)  # QK'  N, N, B, H
        scores = scores * self.scale
        attention = self.softmax(scores)
        attention = self.drop_out(attention)
        x = torch.einsum('ijbh, jbhd -> ibhd', attention, value)  # n_points, batch, head, d_k
        x = x.reshape(N, B, -1).permute(1, 0, 2)  # B, N, C
        x = self.FeedForward(x)  # B, N, C
        return x


class Pre_Linear(nn.Module):
    def __init__(self, d_model, num_heads, d_k):
        super(Pre_Linear, self).__init__()
        self.linear = nn.Linear(d_model, num_heads * d_k)
        self.heads = num_heads
        self.d_k = d_k

    def forward(self, x):
        head_shape = x.shape[: -1]
        x = self.linear(x)
        x = x.view(*head_shape, self.heads, self.d_k)
        return x


class Cls(nn.Module):
    def __init__(self, in_channel, num_class=40):
        super(Cls, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channel, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_class)
        )

    def forward(self, x):
        return self.net(x)


class LocalGrouper(nn.Module):
    def __init__(self, channel, groups, kneighbors, use_xyz=True, normalize="center", double_feature=True, **kwargs):
        """
        Give xyz[b,p,3] and fea[b,p,d], return new_xyz[b,g,3] and new_fea[b,g,k,d]
        :param groups: groups number
        :param kneighbors: k-nerighbors
        :param kwargs: others
        """
        super(LocalGrouper, self).__init__()
        self.groups = groups
        self.kneighbors = kneighbors
        self.use_xyz = use_xyz
        self.double_feature = double_feature
        if normalize is not None:
            self.normalize = normalize.lower()
        else:
            self.normalize = None
        if self.normalize not in ["center", "anchor"]:
            print(
                f"Unrecognized normalize parameter (self.normalize), set to None. Should be one of [center, anchor].")
            self.normalize = None
        if self.normalize is not None:
            add_channel = 3 if self.use_xyz else 0
            self.affine_alpha = nn.Parameter(
                torch.ones([1, 1, 1, channel + add_channel]))
            self.affine_beta = nn.Parameter(
                torch.zeros([1, 1, 1, channel + add_channel]))

    def forward(self, xyz, points):
        B, N, C = xyz.shape
        S = self.groups
        xyz = xyz.contiguous()  # xyz [btach, points, xyz]

        # fps_idx = torch.multinomial(torch.linspace(0, N - 1, steps=N).repeat(B, 1).to(xyz.device), num_samples=self.groups, replacement=False).long()
        # fps_idx = farthest_point_sample(xyz, self.groups).long()
        fps_idx = pointnet2_utils.furthest_point_sample(
            xyz, self.groups).long()  # [B, npoint]
        new_xyz = index_points(xyz, fps_idx)  # [B, npoint, 3]
        new_points = index_points(points, fps_idx)  # [B, npoint, d]

        idx = knn_point(self.kneighbors, xyz, new_xyz)
        # idx = query_ball_point(radius, nsample, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx)  # [B, npoint, k, 3]
        grouped_points = index_points(points, idx)  # [B, npoint, k, d]
        if self.use_xyz:
            grouped_points = torch.cat(
                [grouped_points, grouped_xyz], dim=-1)  # [B, npoint, k, d+3]
        if self.normalize is not None:
            if self.normalize == "center":
                mean = torch.mean(grouped_points, dim=2, keepdim=True)
            if self.normalize == "anchor":
                mean = torch.cat([new_points, new_xyz], dim=-
                1) if self.use_xyz else new_points
                mean = mean.unsqueeze(dim=-2)  # [B, npoint, 1, d+3]
            std = torch.std((grouped_points - mean).reshape(B, -1),
                            dim=-1, keepdim=True).unsqueeze(dim=-1).unsqueeze(dim=-1)
            grouped_points = (grouped_points - mean) / (std + 1e-5)
            grouped_points = self.affine_alpha * grouped_points + self.affine_beta
        if self.double_feature:
            new_points = torch.cat([grouped_points, new_points.view(
                B, S, 1, -1).repeat(1, 1, self.kneighbors, 1)], dim=-1)
        else:
            new_points = grouped_points

        return new_xyz, new_points


class Embedding_Layer(nn.Module):
    def __init__(self, embedding_channel, in_channel=3):
        super(Embedding_Layer, self).__init__()
        self.Embedding_channel = embedding_channel
        self.act = nn.LeakyReLU(negative_slope=0.01)

        if self.Embedding_channel == 32:
            self.emd = nn.Sequential(
                nn.Conv1d(in_channel, embedding_channel, kernel_size=1, bias=False),
                nn.BatchNorm1d(embedding_channel),
            )

        if self.Embedding_channel == 64:
            self.emd = nn.Sequential(
                nn.Conv1d(in_channel, embedding_channel, kernel_size=1, bias=True),
                nn.BatchNorm1d(embedding_channel),
            )

    def forward(self, x):
        return self.act(self.emd(x))  # B, C, N


class Local_Aggregation(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Local_Aggregation, self).__init__()
        self.net1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1, bias=True),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(negative_slope=0.01)
        )

        self.net2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1, bias=True),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 128])
        x = x.permute(0, 1, 3, 2)  # 32, 512, 128, 32 B, N, D, K
        x = x.reshape(-1, d, s)  # B * N, D, K
        batch_size, _, _ = x.size()
        x = F.leaky_relu(self.net2(self.net1(x)) + x)
        x = F.adaptive_max_pool1d(x, 1)
        x = x.view(batch_size, -1)
        x = x.reshape(b, n, -1)
        return x  # B, N, D


class CrossAttEncoder(nn.Module):
    def __init__(self, small_dim=128, large_dim=256, cross_attn_depth=4, cross_attn_heads=4, channels=256):
        super(CrossAttEncoder, self).__init__()

        self.ln_ls1 = nn.LayerNorm(2 * channels)
        self.ln_ls2 = nn.LayerNorm(2 * channels)
        self.ln_sl1 = nn.LayerNorm(channels)
        self.ln_sl2 = nn.LayerNorm(channels)
        self.cross_attn_layers = nn.ModuleList()
        for _ in range(cross_attn_depth):
            self.cross_attn_layers.append(nn.ModuleList([
                nn.Conv1d(small_dim, large_dim, kernel_size=1, bias=True),
                nn.Conv1d(large_dim, small_dim, kernel_size=1, bias=True),
                PreNorm(large_dim, CrossAttention(large_dim, cross_attn_heads, large_dim)),
                nn.Conv1d(large_dim, small_dim, kernel_size=1, bias=True),
                nn.Conv1d(small_dim, large_dim, kernel_size=1, bias=True),
                PreNorm(small_dim, CrossAttention(small_dim, cross_attn_heads, small_dim))
            ]))

    def forward(self, l, s):

        for conv1_s_l, conv2_l_s, cross_attn_s, conv1_l_s, conv2_s_l, cross_attn_l in self.cross_attn_layers:
            small_class = s[:, 0]
            x_small = s[:, 1:]
            large_class = l[:, 0]
            x_large = l[:, 1:]

            # Large Branch
            cal_q = conv1_l_s(large_class.unsqueeze(-1)).permute(0, 2, 1)
            cal_q = self.ln_ls1(cal_q)
            cal_qkv = torch.cat((cal_q, x_small), dim=1)
            cal_out = cal_q + cross_attn_l(cal_qkv)  # B, N, C
            cal_out = conv2_s_l(cal_out.permute(0, 2, 1)).permute(0, 2, 1)
            cal_out = self.ln_sl1(cal_out)
            xl = torch.cat((cal_out, x_large), dim=1)

            # Small Branch
            cal_q = conv1_s_l(small_class.unsqueeze(-1)).permute(0, 2, 1)
            cal_q = self.ln_sl2(cal_q)
            cal_qkv = torch.cat((cal_q, x_large), dim=1)
            cal_out = cal_q + cross_attn_s(cal_qkv)
            cal_out = conv2_l_s(cal_out.permute(0, 2, 1)).permute(0, 2, 1)
            cal_out = self.ln_ls2(cal_out)
            xs = torch.cat((cal_out, x_small), dim=1)  # B, N, C

        return xl, xs


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class CrossAttention(nn.Module):
    def __init__(self, in_feature, num_heads, d_model):
        super(CrossAttention, self).__init__()
        self.heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.scale = 1 / math.sqrt(self.d_k)
        self.linear_trans_q = Pre_Linear(in_feature, num_heads, self.d_k)
        self.linear_trans_k = Pre_Linear(in_feature, num_heads, self.d_k)
        self.linear_trans_v = Pre_Linear(in_feature, num_heads, self.d_k)

        self.drop_out = nn.Dropout(0.1)
        self.softmax = nn.Softmax(dim=-1)
        self.conv = nn.Conv1d(in_feature, in_feature, 1)
        self.layer_norm = nn.LayerNorm(in_feature)
        self.act = nn.LeakyReLU(negative_slope=0.02)

    def forward(self, x):  # B, N, C
        x = x.permute(1, 0, 2)
        query = self.linear_trans_q(x[0, :].unsqueeze(0))  # N, B, C -> (-1, num_heads, d_k)
        key = self.linear_trans_k(x)  # N, B, H, d_k
        value = self.linear_trans_v(x)
        N, B, H, d_k = query.shape
        key = rearrange(key, 'n b h d -> b h n d', h=H)
        value = rearrange(value, 'n b h d -> b h n d', h=H)
        query = rearrange(query, 'n b h d -> b h n d', h=H)
        scores = einsum('b h i d, b h j d -> b h i j', query, key) * self.scale
        attention = self.softmax(scores)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))
        attention = self.drop_out(attention)
        x = einsum('b h i j, b h j d -> b h i d', attention, value)
        query = query.permute(0, 2, 1, 3).reshape(B, N, -1).permute(0, 2, 1)
        x = x.permute(0, 2, 1, 3).reshape(B, N, -1).permute(0, 2, 1)  # B, C, N
        x = self.conv(query - x)
        x = self.act(self.layer_norm(x.permute(0, 2, 1)))
        x = query.permute(0, 2, 1) + x
        return x


class Self_Attention(nn.Module):
    def __init__(self, channels):
        super(Self_Attention, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias

        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.LeakyReLU(negative_slope=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # b, n, c
        x_q = self.q_conv(x).permute(0, 2, 1)
        # b, c, n
        x_k = self.k_conv(x)
        x_v = self.v_conv(x)
        # b, n, n
        energy = torch.bmm(x_q, x_k)
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))
        # b, c, n
        x_r = torch.bmm(x_v, attention)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x


def compute_flops_and_params(data, net):
    Flops, params = profile(net, inputs=(data,))
    Flops, params = clever_format([Flops, params], "%.2f")
    print(Flops, params)


def compute_TEs(data, net, batch_size):
    repetitions = 100
    total_time = 0
    with torch.no_grad():
        for rep in range(repetitions):
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()
            _ = net(data)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender) / 1000
            total_time += curr_time
    Throughput = (repetitions * batch_size) / total_time
    print('FinalThroughput:', Throughput)


if __name__ == '__main__':
    batch_size = 16
    net = PointCAT().cuda()
    inputs = torch.randn(batch_size, 3, 1024).cuda()

    # compute_flops_and_params(inputs, net)
    compute_TEs(inputs, net, batch_size)





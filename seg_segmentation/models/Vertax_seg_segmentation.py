from pointnet2_ops import pointnet2_utils
from torch import einsum
import torch.nn as nn
import torch
import torch.nn.functional as F
from util.tools import *
from util.tools import fps_grouper, sample_and_group
from einops import rearrange, repeat
import math
from thop import profile
from thop import clever_format


class Vertax_elite(nn.Module):
    # noinspection PyDefaultArgument
    def __init__(self, class_num=13, channel_list=[64, 128, 256, 512], sample_list=[1024, 256, 64, 16],
                 de_inchannel_list=[1536, 768, 384, 192], de_outchannel_list=[512, 256, 128, 128], k_neighbors=16,
                 small_dim=1024, large_dim=512):
        super(Vertax_elite, self).__init__()
        self.embedding = Embedding_Layer(in_channel=6, embedding_channel=64)
        self.grouper_list = nn.ModuleList()
        self.global_feature_list = nn.ModuleList()
        self.decoder_list = nn.ModuleList()
        for i in range(len(channel_list)):
            channel = channel_list[i]
            point = sample_list[i]
            de_in = de_inchannel_list[i]
            de_out = de_outchannel_list[i]
            sampler = Multi_Grouping(channel, point, k_neighbors, use_xyz=False, normalize='center')
            global_down = nn.Sequential(
                nn.Conv1d(in_channels=channel, out_channels=64, kernel_size=1, bias=True),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(negative_slope=0.01, inplace=True)
            )
            decoder = PointNetFeaturePropagation(de_in, de_out, bias=True)
            self.grouper_list.append(sampler)
            self.global_feature_list.append(global_down)
            self.decoder_list.append(decoder)

        self.global_feature_list.append(nn.Sequential(
            nn.Conv1d(in_channels=1024, out_channels=64, kernel_size=1, bias=True),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        ))

        self.global_fuse = nn.Sequential(
            nn.Conv1d(in_channels=320, out_channels=64, kernel_size=1, bias=True),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )

        self.class_token_small = nn.Parameter(torch.randn(1, 1, small_dim))
        self.class_token_large = nn.Parameter(torch.randn(1, 1, large_dim))

        self.cross_attn_layer = CrossAttBlock(large_dim=large_dim, small_dim=small_dim, cross_attn_depth=4,
                                              cross_attn_heads=8, channels=512)

        self.classifier = nn.Sequential(
            nn.Conv1d(256, 128, 1, bias=True),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(),
            nn.Conv1d(128, class_num, 1, bias=True)
        )

    def forward(self, x):
        xyz = x[:, :3, :]
        xyz = xyz.permute(0, 2, 1)  # B, N, C
        batch_size, _, _ = x.size()
        x = x[:, :6, :]
        x = self.embedding(x).permute(0, 2, 1)  # B, N, C

        xyz_list = [xyz]
        x_list = [x]

        for i in range(4):
            xyz, x = self.grouper_list[i](xyz, x)
            xyz_list.append(xyz)
            x_list.append(x)

        class_token_small = repeat(self.class_token_small, '() n d -> b n d', b=batch_size)
        class_token_large = repeat(self.class_token_large, '() n d -> b n d', b=batch_size)

        x1 = torch.cat([class_token_large, x_list[-2]], dim=1)
        x2 = torch.cat([class_token_small, x_list[-1]], dim=1)

        x1, x2, head_token = self.cross_attn_layer(x1, x2)
        x_list[-2] = x1
        x_list[-1] = x2
        x_list.reverse()
        xyz_list.reverse()

        x = x_list[0]
        for i in range(len(self.decoder_list)):
            x = self.decoder_list[i](xyz_list[i + 1], xyz_list[i], x_list[i + 1].permute(0, 2, 1), x.permute(0, 2, 1))
            x = x.permute(0, 2, 1)
        gf_list = []
        x_list.reverse()
        for i in range(len(x_list)):
            gf_list.append(F.adaptive_max_pool1d(self.global_feature_list[i](x_list[i].permute(0, 2, 1)), 1))
        global_feature = self.global_fuse(torch.cat(gf_list, dim=1))
        x = x.permute(0, 2, 1)
        x = torch.cat([x, head_token.repeat([1, 1, x.shape[-1]]), global_feature.repeat([1, 1, x.shape[-1]])], dim=1)
        x = self.classifier(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x


class Multi_Grouping(nn.Module):
    def __init__(self, channel, groups, kneighbors, use_xyz, normalize="center"):
        super(Multi_Grouping, self).__init__()
        self.grouper = LocalGrouper(channel=channel, groups=groups, kneighbors=kneighbors, use_xyz=use_xyz,
                                    normalize=normalize)
        self.net = Local_Aggregation(2 * channel, 2 * channel)

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

        self.net = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                 nn.BatchNorm1d(1024),
                                 nn.LeakyReLU(negative_slope=0.2)
                                 )

        self.token_fuse = nn.Sequential(nn.Conv1d(1024, 64, kernel_size=1, bias=False),
                                        nn.BatchNorm1d(64),
                                        nn.LeakyReLU(negative_slope=0.2)
                                        )

    def forward(self, xl, xs):
        batch_size, _, _ = xl.shape
        xl, xs = self.cross_att1(xl, xs)
        xl, xs = self.cross_att2(xl, xs)

        l = self.net(xl.permute(0, 2, 1))
        s = xs.permute(0, 2, 1)
        l = F.adaptive_max_pool1d(l, 1)
        s = F.adaptive_max_pool1d(s, 1)
        head_token = l + s
        head_token = self.token_fuse(head_token)

        return xl[:, 1:], xs[:, 1:], head_token


class Vertax_elite_c(nn.Module):
    # noinspection PyDefaultArgument
    def __init__(self, class_num=13, channel_list=[64, 128, 256, 512], sample_list=[1024, 256, 64, 16],
                 de_inchannel_list=[1536, 768, 384, 192], de_outchannel_list=[512, 256, 128, 128], k_neighbors=16,
                 small_dim=1024, large_dim=512):
        super(Vertax_elite_c, self).__init__()
        self.embedding = Embedding_Layer(in_channel=6, embedding_channel=64)
        self.color_embedding = Embedding_Layer(in_channel=3, embedding_channel=64)
        self.grouper_list = nn.ModuleList()
        self.global_feature_list = nn.ModuleList()
        self.decoder_list = nn.ModuleList()
        for i in range(len(channel_list)):
            channel = channel_list[i]
            point = sample_list[i]
            de_in = de_inchannel_list[i]
            de_out = de_outchannel_list[i]
            sampler = Multi_Grouping(channel, point, k_neighbors, use_xyz=False, normalize='center')
            global_down = nn.Sequential(
                nn.Conv1d(in_channels=channel, out_channels=64, kernel_size=1, bias=True),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(negative_slope=0.01, inplace=True)
            )
            decoder = PointNetFeaturePropagation(de_in, de_out, bias=True)
            self.grouper_list.append(sampler)
            self.global_feature_list.append(global_down)
            self.decoder_list.append(decoder)

        self.global_feature_list.append(nn.Sequential(
            nn.Conv1d(in_channels=1024, out_channels=64, kernel_size=1, bias=True),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        ))

        self.global_fuse = nn.Sequential(
            nn.Conv1d(in_channels=320, out_channels=64, kernel_size=1, bias=True),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )

        self.class_token_small = nn.Parameter(torch.randn(1, 1, small_dim))
        self.class_token_large = nn.Parameter(torch.randn(1, 1, large_dim))

        self.cross_attn_layer = CrossAttBlock(large_dim=large_dim, small_dim=small_dim, cross_attn_depth=4,
                                              cross_attn_heads=8, channels=512)

        self.classifier = nn.Sequential(
            nn.Conv1d(256, 128, 1, bias=True),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(),
            nn.Conv1d(128, class_num, 1, bias=True)
        )

    def forward(self, x):
        xyz = x[:, :3, :]
        xyz = xyz.permute(0, 2, 1)  # B, N, C
        batch_size, _, _ = x.size()
        color = x[:, 6:]
        color = self.color_embedding(color)
        x = x[:, :6, :]
        x = self.embedding(x).permute(0, 2, 1)  # B, N, C

        xyz_list = [xyz]
        x_list = [x]

        for i in range(4):
            xyz, x = self.grouper_list[i](xyz, x)
            xyz_list.append(xyz)
            x_list.append(x)

        class_token_small = repeat(self.class_token_small, '() n d -> b n d', b=batch_size)
        class_token_large = repeat(self.class_token_large, '() n d -> b n d', b=batch_size)

        x1 = torch.cat([class_token_large, x_list[-2]], dim=1)
        x2 = torch.cat([class_token_small, x_list[-1]], dim=1)

        x1, x2, head_token = self.cross_attn_layer(x1, x2)
        x_list[-2] = x1
        x_list[-1] = x2
        x_list.reverse()
        xyz_list.reverse()

        x = x_list[0]
        for i in range(len(self.decoder_list)):
            x = self.decoder_list[i](xyz_list[i + 1], xyz_list[i], x_list[i + 1].permute(0, 2, 1), x.permute(0, 2, 1))
            x = x.permute(0, 2, 1)
        gf_list = []
        x_list.reverse()
        for i in range(len(x_list)):
            gf_list.append(F.adaptive_max_pool1d(self.global_feature_list[i](x_list[i].permute(0, 2, 1)), 1))
        global_feature = self.global_fuse(torch.cat(gf_list, dim=1))
        x = x.permute(0, 2, 1)
        x = torch.cat([x, head_token.repeat([1, 1, x.shape[-1]]), global_feature.repeat([1, 1, x.shape[-1]])], dim=1)
        x = self.classifier(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x


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


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, out_channel, bias=True):
        super(PointNetFeaturePropagation, self).__init__()
        self.fuse = nn.Sequential(
            nn.Conv1d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, bias=bias),
            nn.BatchNorm1d(out_channel),
            nn.LeakyReLU(negative_slope=0.01)
        )

        self.net1 = nn.Sequential(
            nn.Conv1d(in_channels=out_channel, out_channels=out_channel, kernel_size=1, bias=bias),
            nn.BatchNorm1d(out_channel),
            nn.LeakyReLU(negative_slope=0.01)
        )

        self.net2 = nn.Sequential(
            nn.Conv1d(in_channels=out_channel, out_channels=out_channel, kernel_size=1, bias=bias),
            nn.BatchNorm1d(out_channel)
        )

        self.net3 = nn.Sequential(
            nn.Conv1d(in_channels=out_channel, out_channels=out_channel, kernel_size=1, bias=bias),
            nn.BatchNorm1d(out_channel),
            nn.LeakyReLU(negative_slope=0.01)
        )

        self.net4 = nn.Sequential(
            nn.Conv1d(in_channels=out_channel, out_channels=out_channel, kernel_size=1, bias=bias),
            nn.BatchNorm1d(out_channel)
        )

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, N, 3]
            xyz2: sampled input points position data, [B, S, 3]
            points1: input points data, [B, D', N]
            points2: input points data, [B, D'', S]
        Return:
            new_points: upsampled points data, [B, D''', N]
        """
        # xyz1 = xyz1.permute(0, 2, 1)
        # xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        new_points = self.fuse(new_points)
        new_points = F.leaky_relu((self.net2(self.net1(new_points)) + new_points), negative_slope=0.01)
        new_points = F.leaky_relu((self.net4(self.net3(new_points)) + new_points), negative_slope=0.01)
        return new_points


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
    def __init__(self, channel, groups, kneighbors, use_xyz=True, normalize="center", **kwargs):
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

        new_points = torch.cat([grouped_points, new_points.view(
            B, S, 1, -1).repeat(1, 1, self.kneighbors, 1)], dim=-1)
        return new_xyz, new_points


class Embedding_Layer(nn.Module):
    def __init__(self, embedding_channel, in_channel=3):
        super(Embedding_Layer, self).__init__()
        self.Embedding_channel = embedding_channel
        self.act = nn.LeakyReLU(negative_slope=0.01, inplace=True)

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
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
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
        x = F.leaky_relu((self.net2(self.net1(x)) + x), inplace=True)
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

    def forward(self, xl, xs):

        for conv1_s_l, conv2_l_s, cross_attn_s, conv1_l_s, conv2_s_l, cross_attn_l in self.cross_attn_layers:
            small_class = xs[:, 0]
            x_small = xs[:, 1:]
            large_class = xl[:, 0]
            x_large = xl[:, 1:]

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


class SampleAndGroup(nn.Module):
    def __init__(self, num_points, in_channel=3, feature_channel=64, use_xyz=False, normalize='center', **kwargs):
        super(SampleAndGroup, self).__init__()
        self.in_channel = in_channel
        self.num_points = num_points
        self.use_xyz = use_xyz
        self.feature_channel = feature_channel

        if self.use_xyz is False:
            self.embedding = Embedding_Layer(in_channel=self.in_channel,
                                             embedding_channel=self.feature_channel)
        else:
            self.feature_channel -= 3
            self.embedding = Embedding_Layer(in_channel=self.in_channel,
                                             embedding_channel=self.feature_channel)

        self.normalize = normalize.lower() if normalize is not None else None

        if self.normalize not in ["center", "anchor"]:
            print(
                f"Unrecognized normalize parameter (self.normalize), set to None. Should be one of [center, anchor].")
            self.normalize = None
        if self.normalize is not None:
            self.add_channel = 3 if self.use_xyz else 0
            self.alpha = nn.Parameter(
                torch.ones([1, 1, self.feature_channel + self.add_channel]))
            self.beta = nn.Parameter(
                torch.zeros([1, 1, self.feature_channel + self.add_channel]))

    def forward(self, x):
        xyz = x.permute(0, 2, 1)
        # B, N ,C
        group_xyz = fps_grouper(xyz, self.num_points)
        B, N, _ = group_xyz.shape
        group_points = self.embedding(group_xyz.permute(0, 2, 1)).permute(0, 2, 1)  # B, N, C
        if self.use_xyz:
            group_points = torch.cat([group_points, group_xyz], dim=-1)
        if self.normalize is not None:
            if self.normalize == "center":
                mean = torch.mean(group_points, dim=1, keepdim=True)
                std = torch.std(group_points, dim=1, keepdim=True)
            group_points = (group_points - mean) / (std + 1e-5)
            group_points = self.alpha * group_points + self.beta
        return group_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, weight):
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -((one_hot * log_prb) * weight).sum(dim=1).mean()
        # total_loss = F.nll_loss(pred, target, weight=weight)

        return loss


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
        x = x.permute(0, 2, 1, 3).reshape(B, N, -1)  # B, N, C
        return x


def compute_flops_and_params(data, net):
    Flops, params = profile(net, inputs=(data,))
    Flops, params = clever_format([Flops, params], "%.2f")
    print(Flops, params)


def compute_TEs(data, net):
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
    Throughput = (repetitions * 8) / total_time
    print('FinalThroughput:', Throughput)


if __name__ == '__main__':
    from thop import profile
    from thop import clever_format
    net = Vertax_elite_c().cuda()
    inputs = torch.randn(8, 9, 4096).cuda()
    # compute_flops_and_params(inputs, net)
    compute_TEs(inputs, net)


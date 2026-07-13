'''
NOTE:
This file consolidates the SNN-specific modules from `classification/models/metaspikformer.py`
so you can drop them into your own project without distillation or the training harness.
Upstream license applies; please keep attribution per the root LICENSE.
'''

import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import (
    MultiStepParametricLIFNode,
    MultiStepLIFNode,
)
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
import torch.nn.functional as F
from functools import partial
import numpy as np

class BNAndPadLayer(nn.Module):
    def __init__(
        self,
        pad_pixels,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ):
        super(BNAndPadLayer, self).__init__()
        self.bn = nn.BatchNorm2d(
            num_features, eps, momentum, affine, track_running_stats
        )
        self.pad_pixels = pad_pixels

    def forward(self, input):
        output = self.bn(input)
        if self.pad_pixels > 0:
            if self.bn.affine:
                pad_values = (
                    self.bn.bias.detach()
                    - self.bn.running_mean
                    * self.bn.weight.detach()
                    / torch.sqrt(self.bn.running_var + self.bn.eps)
                )
            else:
                pad_values = -self.bn.running_mean / torch.sqrt(
                    self.bn.running_var + self.bn.eps
                )
            output = F.pad(output, [self.pad_pixels] * 4)
            pad_values = pad_values.view(1, -1, 1, 1)
            output[:, :, 0 : self.pad_pixels, :] = pad_values
            output[:, :, -self.pad_pixels :, :] = pad_values
            output[:, :, :, 0 : self.pad_pixels] = pad_values
            output[:, :, :, -self.pad_pixels :] = pad_values
        return output

    @property
    def weight(self):
        return self.bn.weight

    @property
    def bias(self):
        return self.bn.bias

    @property
    def running_mean(self):
        return self.bn.running_mean

    @property
    def running_var(self):
        return self.bn.running_var

    @property
    def eps(self):
        return self.bn.eps

class RepConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        bias=False,
    ):
        super().__init__()
        # hidden_channel = in_channel
        conv1x1 = nn.Conv2d(in_channel, in_channel, 1, 1, 0, bias=False, groups=1)
        bn = BNAndPadLayer(pad_pixels=1, num_features=in_channel)
        conv3x3 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, 1, 0, groups=in_channel, bias=False),
            nn.Conv2d(in_channel, out_channel, 1, 1, 0, groups=1, bias=False),
            nn.BatchNorm2d(out_channel),
        )
        self.body = nn.Sequential(conv1x1, bn, conv3x3)

    def forward(self, x):
        return self.body(x)

class SepConv(nn.Module):
    r"""
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    """

    def __init__(
        self,
        dim,
        detach_reset,
        expansion_ratio=4,
        act2_layer=nn.Identity,
        bias=False,
        kernel_size=7,
        padding=3,
    ):
        super().__init__()
        med_channels = int(expansion_ratio * dim)
        self.lif1 = MultiStepLIFNode(tau=2.0, detach_reset=detach_reset, backend="cupy")
        self.pwconv1 = nn.Conv2d(dim, med_channels, kernel_size=1, stride=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(med_channels)
        self.lif2 = MultiStepLIFNode(tau=2.0, detach_reset=detach_reset, backend="cupy")
        self.dwconv = nn.Conv2d(
            med_channels,
            med_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=med_channels,
            bias=bias,
        )  # depthwise conv
        self.pwconv2 = nn.Conv2d(med_channels, dim, kernel_size=1, stride=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(dim)

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.lif1(x)
        x = self.bn1(self.pwconv1(x.flatten(0, 1))).reshape(T, B, -1, H, W)
        x = self.lif2(x)
        x = self.dwconv(x.flatten(0, 1))
        x = self.bn2(self.pwconv2(x)).reshape(T, B, -1, H, W)
        return x

class MS_ConvBlock(nn.Module):
    def __init__(
        self,
        dim,
        detach_reset,
        mlp_ratio=4.0,
    ):
        super().__init__()

        self.Conv = SepConv(dim=dim, detach_reset=detach_reset)
        # self.Conv = MHMC(dim=dim)

        self.lif1 = MultiStepLIFNode(tau=2.0, detach_reset=detach_reset, backend="cupy")
        self.conv1 = nn.Conv2d(
            dim, dim * mlp_ratio, kernel_size=3, padding=1, groups=1, bias=False
        )
        # self.conv1 = RepConv(dim, dim*mlp_ratio)
        self.bn1 = nn.BatchNorm2d(dim * mlp_ratio)
        self.lif2 = MultiStepLIFNode(tau=2.0, detach_reset=detach_reset, backend="cupy")
        self.conv2 = nn.Conv2d(
            dim * mlp_ratio, dim, kernel_size=3, padding=1, groups=1, bias=False
        )
        # self.conv2 = RepConv(dim*mlp_ratio, dim)
        self.bn2 = nn.BatchNorm2d(dim)

    def forward(self, x):
        T, B, C, H, W = x.shape

        x = self.Conv(x) + x
        x_feat = x
        x = self.bn1(self.conv1(self.lif1(x).flatten(0, 1))).reshape(T, B, 4 * C, H, W)
        x = self.bn2(self.conv2(self.lif2(x).flatten(0, 1))).reshape(T, B, C, H, W)
        x = x_feat + x

        return x

class MS_MLP(nn.Module):
    def __init__(
        self,
        in_features,
        detach_reset,
        hidden_features=None,
        out_features=None,
        drop=0.0,
        layer=0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # self.fc1 = linear_unit(in_features, hidden_features)
        self.fc1_conv = nn.Conv1d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_lif = MultiStepLIFNode(
            tau=2.0, detach_reset=detach_reset, backend="cupy"
        )

        # self.fc2 = linear_unit(hidden_features, out_features)
        self.fc2_conv = nn.Conv1d(
            hidden_features, out_features, kernel_size=1, stride=1
        )
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_lif = MultiStepLIFNode(
            tau=2.0, detach_reset=detach_reset, backend="cupy"
        )
        # self.drop = nn.Dropout(0.1)

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        T, B, C, H, W = x.shape
        N = H * W
        x = x.flatten(3)
        x = self.fc1_lif(x)
        x = self.fc1_conv(x.flatten(0, 1))
        x = self.fc1_bn(x).reshape(T, B, self.c_hidden, N).contiguous()

        x = self.fc2_lif(x)
        x = self.fc2_conv(x.flatten(0, 1))
        x = self.fc2_bn(x).reshape(T, B, C, H, W).contiguous()

        return x

class MS_Attention_RepConv(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        sr_ratio=1,
        detach_reset=True,
    ):
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        #self.scale = 0.125
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.head_lif = MultiStepLIFNode(
            tau=2.0, detach_reset=detach_reset, backend="cupy"
        )
        self.q_conv = nn.Sequential(RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim))
        self.k_conv = nn.Sequential(RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim))
        self.v_conv = nn.Sequential(RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim))
        self.q_lif = MultiStepLIFNode(
            tau=2.0, detach_reset=detach_reset, backend="cupy"
        )
        self.k_lif = MultiStepLIFNode(
            tau=2.0, detach_reset=detach_reset, backend="cupy"
        )
        self.v_lif = MultiStepLIFNode(
            tau=2.0, detach_reset=detach_reset, backend="cupy"
        )
        self.attn_lif = MultiStepLIFNode(
            tau=2.0, v_threshold=0.5, detach_reset=detach_reset, backend="cupy"
        )
        self.proj_conv = nn.Sequential(
            RepConv(dim, dim, bias=False),
            nn.BatchNorm2d(dim),  # TODO: BatchNorm2d not needed here
        )

    def forward(self, x):
        T, B, C, H, W = x.shape
        N = H * W

        x = self.head_lif(x)

        q = self.q_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)
        k = self.k_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)
        v = self.v_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)

        q = self.q_lif(q).flatten(3)
        q = (
            q.transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        k = self.k_lif(k).flatten(3)
        k = (
            k.transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        v = self.v_lif(v).flatten(3)
        v = (
            v.transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        x = k.transpose(-2, -1) @ v
        x = (q @ x) * self.scale

        x = x.transpose(3, 4).reshape(T, B, C, N).contiguous()
        x = self.attn_lif(x).reshape(T, B, C, H, W)
        x = x.reshape(T, B, C, H, W)
        x = x.flatten(0, 1)
        x = self.proj_conv(x).reshape(T, B, C, H, W)

        return x

class MS_Attention_3D_RepConv(nn.Module):
    def __init__(
        self,
        dim,
        sim_mode,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        sr_ratio=1,
        detach_reset=True,
    ):
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        #self.scale = 0.125
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.sim_mode = sim_mode

        self.head_lif = MultiStepLIFNode(
            tau=2.0, detach_reset=detach_reset, backend="cupy"
        )
        self.q_conv = nn.Sequential(RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim))
        self.k_conv = nn.Sequential(RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim))
        self.v_conv = nn.Sequential(RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim))
        self.q_lif = MultiStepLIFNode(
            tau=2.0, detach_reset=detach_reset, backend="cupy"
        )
        self.k_lif = MultiStepLIFNode(
            tau=2.0, detach_reset=detach_reset, backend="cupy"
        )
        self.v_lif = MultiStepLIFNode(
            tau=2.0, detach_reset=detach_reset, backend="cupy"
        )

        if self.sim_mode == "dot":
            self.attn_lif = MultiStepLIFNode(
                tau=2.0,
                v_threshold=0.5,
                v_reset=0.0,
                detach_reset=detach_reset,
                backend="cupy",
            )
        elif self.sim_mode == "hamming":
            self.attn_bn = nn.BatchNorm2d(dim)
            self.attn_lif = MultiStepLIFNode(
                tau=2.0,
                v_threshold=1.0,
                v_reset=0.0,
                detach_reset=detach_reset,
                backend="cupy",
            )
        else:
            raise NotImplementedError

        self.proj_conv = nn.Sequential(
            RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        T, B, C, H, W = x.shape
        N = T * H * W

        x = self.head_lif(x)

        q = self.q_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)
        k = self.k_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)
        v = self.v_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)

        q = self.q_lif(q)
        q = (
            q.permute(1, 0, 3, 4, 2)  # [B, T, H, W, C]
            .flatten(1, 3)  # [B, THW, C]
            .reshape(B, N, self.num_heads, C // self.num_heads)  # [B, THW, M, C/M]
            .permute(0, 2, 1, 3)  # [B, M, THW, C/M]
            .contiguous()
        )

        k = self.k_lif(k)
        k = (
            k.permute(1, 0, 3, 4, 2)  # [B, T, H, W, C]
            .flatten(1, 3)  # [B, THW, C]
            .reshape(B, N, self.num_heads, C // self.num_heads)  # [B, THW, M, C/M]
            .permute(0, 2, 1, 3)  # [B, M, THW, C/M]
            .contiguous()
        )

        v = self.v_lif(v)
        v = (
            v.permute(1, 0, 3, 4, 2)  # [B, T, H, W, C]
            .flatten(1, 3)  # [B, THW, C]
            .reshape(B, N, self.num_heads, C // self.num_heads)  # [B, THW, M, C/M]
            .permute(0, 2, 1, 3)  # [B, M, THW, C/M]
            .contiguous()
        )

        if self.sim_mode == "dot":
            x = k.transpose(-2, -1) @ v
            x = (q @ x) * self.scale
        elif self.sim_mode == "hamming":
            x = (2 * k - 1).transpose(-2, -1) @ v
            x = (2 * q - 1) @ x
            # x = x / (2 * self.dim) + 0.5 * v
            x = x / (2 * self.dim)
        else:
            raise NotImplementedError

        x = (
            x.permute(0, 2, 1, 3)  # [B, THW, M, C/M]
            .reshape(B, N, C)  # [B, THW, C]
            .reshape(B, T, H, W, C)  # [B, T, H, W, C]
            .permute(1, 0, 4, 2, 3)  # [T, B, C, H, W]
            .contiguous()
        )

        if self.sim_mode == "dot":
            pass
        elif self.sim_mode == "hamming":
            x = self.attn_bn(x.flatten(0, 1)).reshape(T, B, C, H, W)
        else:
            raise NotImplementedError

        x = self.attn_lif(x)
        x = self.proj_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)

        return x

class MS_Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        detach_reset,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        #norm_layer=nn.LayerNorm,
        sr_ratio=1,
        att_mode="3D_ham",
    ):
        super().__init__()

        if att_mode == "2D":
            print("2D attention mode is used.")
            self.attn = MS_Attention_RepConv(
                dim = dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                sr_ratio=sr_ratio,
                detach_reset=detach_reset,
            )
        elif att_mode == "3D_dot":
            print("3D attention Dot-product mode is used.")
            self.attn = MS_Attention_3D_RepConv(
                dim = dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                sr_ratio=sr_ratio,
                detach_reset=detach_reset,
                sim_mode="dot",
            )
        elif att_mode == "3D_ham":
            print("3D attention Hamming distance mode is used.")
            self.attn = MS_Attention_3D_RepConv(
                dim = dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                sr_ratio=sr_ratio,
                detach_reset=detach_reset,
                sim_mode="hamming",
            )
        else:
            raise NotImplementedError

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MS_MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            drop=drop,
            detach_reset=detach_reset,
        )

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x

class MS_DownSampling(nn.Module):
    def __init__(
        self,
        detach_reset,
        in_channels=2,
        embed_dims=256,
        kernel_size=3,
        stride=2,
        padding=1,
        first_layer=True,
    ):
        super().__init__()

        self.encode_conv = nn.Conv2d(
            in_channels,
            embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.encode_bn = nn.BatchNorm2d(embed_dims)
        if not first_layer:
            self.encode_lif = MultiStepLIFNode(
                tau=2.0, detach_reset=detach_reset, backend="cupy"
            )

    def forward(self, x):
        T, B, _, _, _ = x.shape

        if hasattr(self, "encode_lif"):
            x = self.encode_lif(x)
        x = self.encode_conv(x.flatten(0, 1))
        _, _, H, W = x.shape
        x = self.encode_bn(x).reshape(T, B, -1, H, W).contiguous()

        return x

class Spiking_vit_MetaFormer(nn.Module):
    def __init__(
        self,
        detach_reset,
        #in_channels=8,
        embed_dim=[8, 8, 8, 8],
        num_heads=8,
        mlp_ratios=4,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        #norm_layer=nn.LayerNorm,
        depths=4,
        sr_ratios=1,
        att_mode="3D_ham",
    ):
        super().__init__()
        self.depths = depths

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depths)
        ]  # stochastic depth decay rule

        """self.ConvBlock1_1 = nn.ModuleList(
            [
                MS_ConvBlock(
                    dim=embed_dim[0],
                    mlp_ratio=mlp_ratios,
                    detach_reset=detach_reset,
                )
            ]
        )

        self.ConvBlock1_2 = nn.ModuleList(
            [
                MS_ConvBlock(
                    dim=embed_dim[0],
                    mlp_ratio=mlp_ratios,
                    detach_reset=detach_reset,
                )
            ]
        )

        self.ConvBlock2_1 = nn.ModuleList(
            [
                MS_ConvBlock(
                    dim=embed_dim[1],
                    mlp_ratio=mlp_ratios,
                    detach_reset=detach_reset,
                )
            ]
        )"""

        self.ConvBlock2_2 = nn.ModuleList(
            [
                MS_ConvBlock(
                    dim=embed_dim[1],
                    mlp_ratio=mlp_ratios,
                    detach_reset=detach_reset,
                )
            ]
        )

        self.block3 = nn.ModuleList(
            [
                MS_Block(
                    detach_reset=detach_reset,
                    dim=embed_dim[2],
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratios,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    #norm_layer=norm_layer,
                    sr_ratio=sr_ratios,
                    att_mode=att_mode,
                )
                for j in range(2)
            ]
        )

        """self.block4 = nn.ModuleList(
            [
                MS_Block(
                    detach_reset=detach_reset,
                    dim=embed_dim[3],
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratios,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    #norm_layer=norm_layer,
                    sr_ratio=sr_ratios,
                    att_mode=att_mode,
                )
                for j in range(1)
            ]
        )"""

        self.lif = MultiStepLIFNode(tau=2.0, detach_reset=detach_reset, backend="cupy")

        self.decoder = SpikeVideoFormerDecoder(kernel_size=3)
        """self.head = (
            nn.Linear(embed_dim[3], num_classes) if num_classes > 0 else nn.Identity()
        )

        self.kd = kd
        if self.kd:
            self.head_kd = (
                nn.Linear(embed_dim[3], num_classes)
                if num_classes > 0
                else nn.Identity()
            )
        self.apply(self._init_weights)"""

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        """for blk in self.ConvBlock1_1:
            x = blk(x)
        for blk in self.ConvBlock1_2:
            x = blk(x)

        for blk in self.ConvBlock2_1:
            x = blk(x)"""
        for blk in self.ConvBlock2_2:
            x = blk(x)

        for blk in self.block3:
            x = blk(x)

        """for blk in self.block4:
            x = blk(x)"""
        return x

    def forward(self, x):
        #x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        x1 = self.forward_features(x)

        x_lif = self.lif(x1)

        x = self.decoder(x_lif)
        x = x + x1
        return x

class SpikeDecoder(nn.Module):
    def __init__(self, in_channels=8, out_channels=8, kernel_size=3):
        super(SpikeDecoder, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=kernel_size, padding=kernel_size//2)
        self.conv3 = nn.Conv2d(64, out_channels, kernel_size=kernel_size, padding=kernel_size//2)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(8)

    def forward(self, spike_output):

        decoded_feature = spike_output

        decoded_feature = F.relu(self.bn1(self.conv1(decoded_feature)))  # (B, 32, H, W)
        decoded_feature = F.relu(self.bn2(self.conv2(decoded_feature)))  # (B, 64, H, W)
        decoded_feature = F.relu(self.bn3(self.conv3(decoded_feature)))  # (B, 8, H, W)

        #decoded_feature = torch.clamp(decoded_feature, min=0.0, max=1.0)

        return decoded_feature

class SpikeVideoFormerDecoder(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpikeVideoFormerDecoder, self).__init__()
        self.decoder = SpikeDecoder(in_channels=8, out_channels=8, kernel_size=kernel_size)

    def forward(self, spike_output):
        """
        :param spike_output: spike tensor with shape (T, B, 8, H, W)
        :return: decoded image features with shape (T, B, 8, H, W)
        """
        decoded_features = []

        for t in range(spike_output.size(0)):
            decoded_feature = self.decoder(spike_output[t])  # (B, 8, H, W)
            decoded_features.append(decoded_feature.unsqueeze(0))

        decoded_features = torch.cat(decoded_features, dim=0)

        return decoded_features

"""# ===== Convenience wrapper (no distillation) =====
from spikingjelly.clock_driven import functional as sjF

class SNNBackboneWrapper(nn.Module):
    Thin wrapper to use Spiking_vit_MetaFormer as a feature extractor or classifier.

    - Accepts either (B,C,H,W) static frames or (B,T,C,H,W) sequences.
    - If a sequence is provided, it bypasses the internal repeat and uses the true T.
    def __init__(self, base: nn.Module, time_steps: int = 4, return_features: bool = True):
        super().__init__()
        self.base = base
        self.base.T = int(time_steps)
        self.return_features = return_features
        if self.return_features:
            feat_dim = self.base.head.in_features
            self.base.head = nn.Identity()
            if hasattr(self.base, "kd"):
                self.base.kd = False
            self.out_dim = feat_dim
        else:
            self.out_dim = getattr(self.base.head, 'out_features', None)

    @torch.no_grad()
    def reset_state(self):
        sjF.reset_net(self.base)

    def forward(self, x):
        # x can be (B,C,H,W) or (B,T,C,H,W)
        if x.ndim == 4:
            return self.base(x)
        elif x.ndim == 5:
            B,T,C,H,W = x.shape
            x = x.permute(1,0,2,3,4).contiguous()   # -> (T,B,C,H,W)
            x = self.base.forward_features(x)
            x = x.flatten(3).mean(3)               # GAP on spatial
            x_lif = self.base.lif(x)               # MultiStep LIF
            out = self.base.head(x_lif).mean(0)    # average over T
            return out
        else:
            raise ValueError(f"Expected 4D or 5D input, got shape {tuple(x.shape)}")
"""

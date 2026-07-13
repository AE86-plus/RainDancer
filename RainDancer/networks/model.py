import sys

from networks.basic.base_model import BaseModel
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
import math
import torch
import torch.nn as nn
import torch.nn.init as init
from numpy import *
import torch.nn.functional as F
import itertools
#from mamba_ssm import Mamba  # pip install mamba-ssm
from . import losses
from .SSIM import SSIM

###DDP
from torch.nn.parallel import DistributedDataParallel as DDP

from networks.snn_core_extracted import Spiking_vit_MetaFormer
from spikingjelly.clock_driven import functional as sjF

def To3D(E1, group):
    [b, c, h, w] = E1.shape
    nf = int(c/group)

    E_list = []
    for i in range(0, group):
        tmp = E1[:, nf*i:nf*(i+1), :, :]
        tmp = tmp.view(b, nf, 1, h, w)
        E_list.append(tmp)

    E1_3d = torch.cat(E_list, 2)
    return E1_3d

def check_tensor_nan(x, name):
    if not torch.isfinite(x).all():
        print(f'[NaN DEBUG] {name} contains NaN/Inf! '
              f'min={x.min().item()}, max={x.max().item()}')
        return True
    #print(f'min={x.min().item()}, max={x.max().item()}')
    return False

def charbonnier(x, eps=1e-3):
    return torch.sqrt(x * x + eps * eps)

class EventBranchLoss(nn.Module):
    """
    event_bg, gt_event: (B,20,H,W), 20 = pre10 + post10
    Loss = lam_rec * L_rec_bin_weighted + lam_grad * L_grad_seg + lam_dir * L_dir_seg
    - Ignore polarity by using abs() for both pred and gt.
    """
    def __init__(
        self,
        lam_rec=1.0, lam_grad=0.5, lam_dir=0.1,
        alpha=4.0, gamma=0.5, beta=2.0,
        eps=1e-6, charbon_eps=1e-3
    ):
        super().__init__()
        self.lam_rec = lam_rec
        self.lam_grad = lam_grad
        self.lam_dir = lam_dir
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.eps = eps
        self.charbon_eps = charbon_eps

        # Sobel kernels as buffers
        kx = torch.tensor([[1, 0, -1],
                           [2, 0, -2],
                           [1, 0, -1]], dtype=torch.float32) / 8.0
        ky = torch.tensor([[1,  2,  1],
                           [0,  0,  0],
                           [-1, -2, -1]], dtype=torch.float32) / 8.0
        self.register_buffer("sobel_kx", kx.view(1, 1, 3, 3))
        self.register_buffer("sobel_ky", ky.view(1, 1, 3, 3))

    def sobel_grad(self, x):
        """
        x: (B,C,H,W) -> gx,gy: (B,C,H,W)
        """
        B, C, H, W = x.shape
        x_ = x.view(B * C, 1, H, W)
        gx = F.conv2d(x_, self.sobel_kx, padding=1)
        gy = F.conv2d(x_, self.sobel_ky, padding=1)
        gx = gx.view(B, C, H, W)
        gy = gy.view(B, C, H, W)
        return gx, gy

    def forward(self, event_bg, gt_event, return_parts=False):
        """
        event_bg: (B,20,H,W) predicted
        gt_event: (B,20,H,W) ground truth
        """
        assert event_bg.shape == gt_event.shape, "event_bg and gt_event must share the same shape"
        B, C, H, W = gt_event.shape
        assert C == 20, "channel count must be 20 (pre10+post10)"

        # ---- ignore polarity ----
        Ep = event_bg.abs()
        Eg = gt_event.abs()

        # =========================
        # 1) Bin-level sparse weighted reconstruction
        # =========================
        # A: (B,1,H,W)
        A = Eg.sum(dim=1, keepdim=True)  # sum over 20 channels

        A_mean = A.mean().clamp_min(self.eps)
        w = 1.0 + self.alpha * (A / A_mean).pow(self.gamma)   # (B,1,H,W)
        w = w.expand_as(Eg)                                   # (B,20,H,W)

        L_rec = (w * charbonnier(Ep - Eg, eps=self.charbon_eps)).sum() / w.sum().clamp_min(self.eps)

        # =========================
        # 2) Segment-aggregated gradient consistency (structure)
        # =========================
        # reshape to (B,2,10,H,W) then sum bins -> (B,2,H,W)
        Ep_seg = Ep.view(B, 2, 10, H, W).sum(dim=2)
        Eg_seg = Eg.view(B, 2, 10, H, W).sum(dim=2)

        gx_p, gy_p = self.sobel_grad(Ep_seg)
        gx_g, gy_g = self.sobel_grad(Eg_seg)

        gmag = torch.sqrt(gx_g * gx_g + gy_g * gy_g + self.eps)  # (B,2,H,W)
        gmag_mean = gmag.mean().clamp_min(self.eps)
        w_g = 1.0 + self.beta * (gmag / gmag_mean)               # (B,2,H,W)

        L_grad = (w_g * (charbonnier(gx_p - gx_g, eps=self.charbon_eps) +
                         charbonnier(gy_p - gy_g, eps=self.charbon_eps))).sum() / w_g.sum().clamp_min(self.eps)

        # =========================
        # 3) Segment-aggregated gradient direction consistency (optional but recommended)
        # =========================
        dot = gx_p * gx_g + gy_p * gy_g
        n1 = torch.sqrt(gx_p * gx_p + gy_p * gy_p + self.eps)
        n2 = torch.sqrt(gx_g * gx_g + gy_g * gy_g + self.eps)
        cos = (dot / (n1 * n2)).clamp(-1.0, 1.0)

        L_dir = (w_g * (1.0 - cos)).mean()

        loss = self.lam_rec * L_rec + self.lam_grad * L_grad + self.lam_dir * L_dir

        if return_parts:
            return loss, {"L_rec": L_rec.detach(), "L_grad": L_grad.detach(), "L_dir": L_dir.detach()}
        return loss

class Decoder(nn.Module):
    def __init__(self, isrgb, input_channels, output_channels=3):
        super(Decoder, self).__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels

        self.decode = nn.Sequential(
            nn.Conv2d(input_channels, 128, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),

            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),

            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),

            nn.Conv2d(32, output_channels, kernel_size=3, stride=1, padding=1),

            nn.Sigmoid() if isrgb == True else nn.Identity()
        )

    def forward(self, x):
        return self.decode(x)

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.PReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)

        # Flatten：[B, C, 1, 1] → [B, C]

        return x * scale

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = ConvLayer(2, 1, kernel_size, stride=1)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)
        return x * scale

class Sym_CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(Sym_CBAM, self).__init__()

        self.ChannelGate1 = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.SpatialGate1 = SpatialGate()
        self.SpatialGate2 = SpatialGate()
        self.ChannelGate2 = ChannelGate(gate_channels, reduction_ratio, pool_types)
    def forward(self, x):
        x1 = self.ChannelGate1(x)
        x2 = self.SpatialGate1(x1)
        x3 = self.SpatialGate2(x2)
        x4 = self.ChannelGate2(x3)
        return x4

class multi_Sym_CBAM(nn.Module):
    def __init__(self, gate_channels, n_blocks, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(multi_Sym_CBAM, self).__init__()
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        modules_body = []
        for _ in range(n_blocks//2):
            modules_body.append(Sym_CBAM(gate_channels))
            modules_body.append(self.act)
            modules_body.append(Sym_CBAM(gate_channels))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        out = self.body(x)
        return out

# -------------------------
# Basic blocks: Conv + GN + PReLU
# -------------------------
def _make_gn(num_channels: int, num_groups: int = 8) -> nn.GroupNorm:
    # make groups safely
    g = num_groups
    while g > 1 and (num_channels % g != 0):
        g -= 1
    return nn.GroupNorm(g, num_channels)

class ConvGNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None, groups=1, gn_groups=8):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, groups=groups, bias=False)
        self.gn = _make_gn(out_ch, gn_groups)
        self.act = nn.PReLU(out_ch)

    def forward(self, x):
        return self.act(self.gn(self.conv(x)))

class GateNet(nn.Module):
    """
    g = sigmoid( ... )  -> shape (B, C, H, W)
    """
    def __init__(self, c=96, hidden=96, gn_groups=8):
        super().__init__()
        self.fuse = ConvGNAct(in_ch=2*c, out_ch=hidden, k=1, p=0, gn_groups=gn_groups)
        self.refine = ConvGNAct(in_ch=hidden, out_ch=hidden, k=3, p=1, gn_groups=gn_groups)
        self.out = nn.Conv2d(hidden, c, kernel_size=1, padding=0, bias=True)
        nn.init.constant_(self.out.bias, -2.0)

    def forward(self, x_self, x_other_detached):
        x = torch.cat([x_self, x_other_detached], dim=1)     # (B, 2C, H, W)
        x = self.fuse(x)
        x = self.refine(x)
        g = torch.sigmoid(self.out(x))                       # (B, C, H, W) in [0,1]
        return g

# -------------------------
# Option A: Cross-modal interaction convolution (light & strong baseline)
# -------------------------
class InteractionDelta(nn.Module):
    """
    delta = Conv([self, other, self*other, |self-other|]) -> (B,C,H,W)
    """
    def __init__(self, c=96, hidden=192, gn_groups=8):
        super().__init__()
        in_ch = 4 * c
        self.pre = ConvGNAct(in_ch, hidden, k=1, p=0, gn_groups=gn_groups)
        self.mid = ConvGNAct(hidden, hidden, k=3, p=1, gn_groups=gn_groups)
        self.out = nn.Conv2d(hidden, c, kernel_size=1, padding=0, bias=True)

    def forward(self, x_self, x_other_detached):
        x = torch.cat(
            [x_self, x_other_detached, x_self * x_other_detached, (x_self - x_other_detached).abs()],
            dim=1
        )
        x = self.pre(x)
        x = self.mid(x)
        delta = self.out(x)
        return delta

# -------------------------
# Option B: Cross-modal attention (with spatial reduction to avoid O((HW)^2))
# -------------------------
class CrossAttentionDelta(nn.Module):
    """
    Improved variant:
    - Local window cross-attention at full resolution
    - Lightweight global cross-attention at reduced resolution
    - Still single-direction delta computation for event -> RGB guidance

    Args:
        c: channels
        heads: attention heads
        sr_ratio: downsampling ratio for the global branch
        gn_groups: group norm groups
        window_size: local branch window size, typically 8
    """
    def __init__(self, c=96, heads=4, sr_ratio=4, gn_groups=8, window_size=8):
        super().__init__()
        assert c % heads == 0, f"c={c} must be divisible by heads={heads}"
        self.c = c
        self.h = heads
        self.d = c // heads
        self.sr = sr_ratio
        self.ws = window_size

        # normalization
        self.norm_q = _make_gn(c, gn_groups)
        self.norm_kv = _make_gn(c, gn_groups)

        # local positional bias (important for restoration tasks)
        self.q_pos = nn.Conv2d(c, c, kernel_size=3, padding=1, groups=c, bias=False)
        self.kv_pos = nn.Conv2d(c, c, kernel_size=3, padding=1, groups=c, bias=False)

        # shared projections
        self.q_proj = nn.Conv2d(c, c, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(c, c, kernel_size=1, bias=False)
        self.v_proj = nn.Conv2d(c, c, kernel_size=1, bias=False)

        # learnable scale for cosine attention
        self.attn_scale_local = nn.Parameter(torch.tensor(4.0))
        self.attn_scale_global = nn.Parameter(torch.tensor(4.0))

        # fuse local + global
        self.gamma_global = nn.Parameter(torch.tensor(0.2))

        self.out_proj = nn.Conv2d(c, c, kernel_size=1, bias=False)
        self.out_norm = _make_gn(c, gn_groups)
        self.out_act = nn.PReLU(c)

        # local refine after fusion
        self.local_refine = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=1, groups=c, bias=False),
            _make_gn(c, gn_groups),
            nn.PReLU(c),
            nn.Conv2d(c, c, kernel_size=1, bias=False),
        )

    def _window_partition(self, x, ws):
        # x: (B,C,H,W)
        B, C, H, W = x.shape
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')

        Hp, Wp = x.shape[-2:]
        nH = Hp // ws
        nW = Wp // ws

        x = x.view(B, C, nH, ws, nW, ws)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()  # (B,nH,nW,C,ws,ws)
        windows = x.view(B * nH * nW, C, ws, ws)

        meta = (H, W, Hp, Wp, nH, nW, pad_h, pad_w)
        return windows, meta

    def _window_reverse(self, windows, meta, ws):
        # windows: (B*nH*nW, C, ws, ws)
        H, W, Hp, Wp, nH, nW, pad_h, pad_w = meta
        Bnw, C, _, _ = windows.shape
        B = Bnw // (nH * nW)

        x = windows.view(B, nH, nW, C, ws, ws)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(B, C, Hp, Wp)

        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :H, :W]

        return x

    def _attend(self, q, k, v, scale):
        # q,k,v: (B*, heads, N, d)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        return out

    def forward(self, x_self, x_other_detached):
        """
        x_self: (B,C,H,W)      -> RGB background feature
        x_other_detached: (B,C,H,W) -> Event background feature (detached outside or inside caller)
        """
        B, C, H, W = x_self.shape
        assert C == self.c

        # add local positional bias before projection
        q_in = self.norm_q(x_self + self.q_pos(x_self))
        kv_in = self.norm_kv(x_other_detached + self.kv_pos(x_other_detached))

        q_full = self.q_proj(q_in)
        k_full = self.k_proj(kv_in)
        v_full = self.v_proj(kv_in)

        # ---------------------------------------------------
        # 1) Local window cross-attention at full resolution
        # ---------------------------------------------------
        ws = min(self.ws, H, W)
        q_w, meta = self._window_partition(q_full, ws)
        k_w, _ = self._window_partition(k_full, ws)
        v_w, _ = self._window_partition(v_full, ws)

        Bwin = q_w.shape[0]
        Nl = ws * ws

        q_w = q_w.view(Bwin, self.h, self.d, Nl).permute(0, 1, 3, 2)  # (Bwin,h,N,d)
        k_w = k_w.view(Bwin, self.h, self.d, Nl).permute(0, 1, 3, 2)
        v_w = v_w.view(Bwin, self.h, self.d, Nl).permute(0, 1, 3, 2)

        out_local = self._attend(q_w, k_w, v_w, self.attn_scale_local)
        out_local = out_local.permute(0, 1, 3, 2).contiguous().view(Bwin, C, ws, ws)
        out_local = self._window_reverse(out_local, meta, ws)  # (B,C,H,W)

        # ---------------------------------------------------
        # 2) Lightweight global cross-attention (reduced res)
        # ---------------------------------------------------
        if self.sr > 1:
            q_g_in = F.avg_pool2d(q_in, kernel_size=self.sr, stride=self.sr, ceil_mode=False)
            kv_g_in = F.avg_pool2d(kv_in, kernel_size=self.sr, stride=self.sr, ceil_mode=False)
        else:
            q_g_in = q_in
            kv_g_in = kv_in

        Hg, Wg = q_g_in.shape[-2:]
        Ng = Hg * Wg

        q_g = self.q_proj(q_g_in).view(B, self.h, self.d, Ng).permute(0, 1, 3, 2)
        k_g = self.k_proj(kv_g_in).view(B, self.h, self.d, Ng).permute(0, 1, 3, 2)
        v_g = self.v_proj(kv_g_in).view(B, self.h, self.d, Ng).permute(0, 1, 3, 2)

        out_global = self._attend(q_g, k_g, v_g, self.attn_scale_global)
        out_global = out_global.permute(0, 1, 3, 2).contiguous().view(B, C, Hg, Wg)

        if self.sr > 1:
            out_global = F.interpolate(out_global, size=(H, W), mode="bilinear", align_corners=False)

        # ---------------------------------------------------
        # 3) Fuse local detail + global context
        # ---------------------------------------------------
        out = out_local + self.gamma_global * out_global
        out = self.out_act(self.out_norm(self.out_proj(out)))
        out = self.local_refine(out)

        return out

# -------------------------
# Unified bidirectional fusion block
# -------------------------
class BiModalComplementaryFusion(nn.Module):
    """
    One-way: Event Background -> RGB Background
    Updated to:
      1) rain-aware
      2) confidence-gated
      3) small residual refinement
      4) optional event detach for strict one-way guidance
    """
    def __init__(
        self,
        c=96,
        hidden=192,
        gn_groups=16,
        detach_event=True,
        use_rain_context=True
    ):
        super().__init__()
        self.detach_event = detach_event
        self.use_rain_context = use_rain_context

        self.evt_proj = ConvGNAct(c, c, k=1, p=0, gn_groups=gn_groups)

        self.delta_rgb = InteractionDelta(c=c, hidden=hidden, gn_groups=gn_groups)

        gate_in_ch = 4 * c if use_rain_context else 2 * c
        self.gate = nn.Sequential(
            nn.Conv2d(gate_in_ch, c, kernel_size=1, bias=False),
            _gn(c, gn_groups),
            nn.PReLU(c),
            nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False),
            _gn(c, gn_groups),
            nn.PReLU(c),
            nn.Conv2d(c, c, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        conf_in_ch = 2 * c if use_rain_context else c
        self.conf_head = nn.Sequential(
            nn.Conv2d(conf_in_ch, c // 2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, c // 2),
            nn.PReLU(c // 2),
            nn.Conv2d(c // 2, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        self.beta = nn.Parameter(torch.tensor(0.1))

        self.post_rgb = ConvGNAct(c, c, k=1, p=0, gn_groups=gn_groups)

    def forward(self, x_rgb_b, x_evt_b, x_rgb_r=None, x_evt_r=None):
        if self.detach_event:
            evt_b = x_evt_b.detach()
            evt_r = x_evt_r.detach() if x_evt_r is not None else None
        else:
            evt_b = x_evt_b
            evt_r = x_evt_r

        evt_ctx = self.evt_proj(evt_b)

        delta = self.delta_rgb(x_rgb_b, evt_ctx)

        # rain-aware gate
        if self.use_rain_context and (x_rgb_r is not None) and (evt_r is not None):
            gate_in = torch.cat([x_rgb_b, evt_b, x_rgb_r, evt_r], dim=1)
            conf_in = torch.cat([evt_b.abs(), evt_r.abs()], dim=1)
        else:
            gate_in = torch.cat([x_rgb_b, evt_b], dim=1)
            conf_in = evt_b.abs()

        gate = self.gate(gate_in)      # (B,C,H,W)
        conf = self.conf_head(conf_in) # (B,1,H,W)

        x_rgb_b_out = x_rgb_b + self.beta * gate * conf * delta
        x_rgb_b_out = self.post_rgb(x_rgb_b_out)

        return x_rgb_b_out

class DWBranch(nn.Module):
    def __init__(self, ch, k):
        super().__init__()
        p = k // 2
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=k, padding=p, groups=ch, bias=False),
            nn.Conv2d(ch, ch, kernel_size=1, padding=0, bias=False),
            _make_gn(ch),
            nn.PReLU(ch)
        )

    def forward(self, x):
        return self.block(x)

class CrossModalFusion_R(nn.Module):
    """
    Input:
        x1_r   : RGB rain feature   (B, C, H, W)
        x1_e_r : Event rain feature (B, C, H, W)

    Output:
        r_fusion : stage rain fusion feature (B, out_channels, H, W)

    Design:
        1) RGB rain is the primary signal
        2) event rain adds only local refinement
        3) no attention, only local multi-scale convolutions
        4) a gate controls event contribution strength
        5) output a 32-channel stage rain feature
    """
    def __init__(self, channels_x1=96, channels_x1_e=96, out_channels=32, detach_event=True):
        super(CrossModalFusion_R, self).__init__()
        assert channels_x1 == channels_x1_e
        c = channels_x1
        self.detach_event = detach_event

        self.rgb_proj = ConvGNAct(c, c, k=1, p=0)
        self.evt_proj = ConvGNAct(c, c, k=1, p=0)

        self.fuse_in = ConvGNAct(2 * c, c, k=1, p=0)

        self.branch3 = DWBranch(c, 3)
        self.branch5 = DWBranch(c, 5)
        self.branch7 = DWBranch(c, 7)

        self.ms_fuse = nn.Sequential(
            ConvGNAct(3 * c, c, k=1, p=0),
            ConvGNAct(c, c, k=3, p=1),
        )

        self.gate = nn.Sequential(
            ConvGNAct(2 * c, c, k=1, p=0),
            ConvGNAct(c, c, k=3, p=1),
            nn.Conv2d(c, c, kernel_size=1, padding=0, bias=True)
        )
        nn.init.constant_(self.gate[-1].bias, -2.0)

        self.beta = nn.Parameter(torch.tensor(0.1))

        self.refine = nn.Sequential(
            ConvGNAct(c, c, k=3, p=1),
            ConvGNAct(c, c, k=3, p=1),
        )

        self.out = nn.Sequential(
            nn.Conv2d(c, out_channels, kernel_size=3, padding=1, bias=False),
            _make_gn(out_channels),
            nn.PReLU(out_channels)
        )
        self.skip = nn.Conv2d(c, out_channels, kernel_size=1, padding=0, bias=False)

    def forward(self, x1_r, x1_e_r):
        evt = x1_e_r.detach() if self.detach_event else x1_e_r

        rgb = self.rgb_proj(x1_r)
        evt = self.evt_proj(evt)

        x_cat = torch.cat([rgb, evt], dim=1)
        base = self.fuse_in(x_cat)

        f3 = self.branch3(base)
        f5 = self.branch5(base)
        f7 = self.branch7(base)
        evt_delta = self.ms_fuse(torch.cat([f3, f5, f7], dim=1))

        gate = torch.sigmoid(self.gate(torch.cat([rgb, evt], dim=1)))
        fused = rgb + self.beta * gate * evt_delta

        fused = self.refine(fused)

        r_fusion = self.out(fused) + self.skip(x1_r)
        return r_fusion

## Coupled Representation Module (CRM)
class CRM(nn.Module):
    def __init__(self, n_feat=96, kernel_size=3, reduction=4, act=nn.PReLU(), bias=False, num_crb=3, num_rcab=3):
        super(CRM, self).__init__()
        modules_body = []
        modules_body = [CRB(n_feat, kernel_size, reduction, bias=bias, act=act, num_rcab=num_rcab) for _ in range(num_crb)]
        self.body = nn.Sequential(*modules_body)

    def forward(self, x_B, x_R):
        res = self.body([x_B, x_R])
        #res += x
        b_fusion = res[0]
        r_fusion = res[1]

        return b_fusion, r_fusion

## Coupled Representation Block (CRB)
class CRB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, num_rcab):
        super(CRB, self).__init__()
        self.down_R = st_conv(n_feat, n_feat, kernel_size, bias=bias)
        self.act1=act
        modules_body = []
        modules_body = [CAB_dsc(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(num_rcab)]
        self.body = nn.Sequential(*modules_body)

        self.lfsfb = LFSFB(n_feat, kernel_size, act, bias)
        self.CA_B = SALayer(n_feat, reduction, bias=bias)
        self.CA_R = SALayer(n_feat, reduction, bias=bias)

    def forward(self, x):
        xB = x[0]
        xR = x[1]
        res_down_R = self.act1(self.down_R(xR))
        #res_down_R = self.act1(res_down_R)
        res_R = self.body(res_down_R)
        #print(res_R.shape)
        #print(res_down_R.shape)
        xR_res = xR + self.lfsfb(res_down_R, res_R)

        res_BTOR = self.CA_B(xB)
        res_RTOB = self.CA_R(xR_res)
        x[0] = xB - res_BTOR + res_RTOB
        x[1] = xR_res - res_RTOB + res_BTOR
        return x
def st_conv(in_channels, out_channels, kernel_size, bias=False, stride = 2):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)
## Long Feature Selection and Fusion Block (LFSFB)
class LFSFB(nn.Module):
    def __init__(self, n_feat, kernel_size, act, bias):
        super(LFSFB, self).__init__()
        #self.FS = nn.Conv2d(n_feat, n_feat, kernel_size=1, stride=1, padding=0, bias=False)
        #self.act1 =act
        self.FFU = nn.ConvTranspose2d(n_feat, n_feat,  kernel_size=3, stride=2, padding=1,output_padding=1, bias= False)
        self.act2 = act

    def forward(self, x1, x2):
        #res = self.act1(self.FS(x1))
        #res = self.act1(res)
        #print(res.shape)
        res_out = self.act2(self.FFU(x2))
        #res = self.act2(res)
        #print(res.shape)
        return res_out
## Channel Attention Block (CAB)
class CAB_dsc(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB_dsc, self).__init__()
        modules_body = []
        modules_body.append(depthwise_separable_conv(n_feat, n_feat))
        modules_body.append(act)
        modules_body.append(depthwise_separable_conv(n_feat, n_feat))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)
        #self.S2FB2 = S2FB_2(n_feat, reduction, bias=bias, act=act)
    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        #res = self.S2FB2(res, x)
        res += x
        return res
## Spatial Attention Layer
class SALayer(nn.Module):
    def __init__(self, channel, reduction=4, bias=False):
        super(SALayer, self).__init__()
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.PReLU(),#nn.PReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.conv_du(x)
        return x * y
## Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=8, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.PReLU(),#nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y
class depthwise_separable_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(depthwise_separable_conv, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.depth_conv = nn.Conv2d(ch_in, ch_in, kernel_size=3, padding=1, groups=ch_in)
        self.point_conv = nn.Conv2d(ch_in, ch_out, kernel_size=1)

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x

class RGB_Extracted(nn.Module):
    """

    """
    def __init__(self):
        super().__init__()
        """self.spatial = SpatialMambaBlock(
            channels=n_feats*2,
        )"""

        self.CSA = multi_Sym_CBAM(gate_channels = 96, n_blocks = 4)

        #self.bn = nn.BatchNorm2d(96)
        self.norm = _make_gn(96, 8)
        self.act = nn.PReLU()

    def forward(self, x):
        """
        x: (B, C, H, W)
        """

        x_b = self.CSA(x)
        #x_b = self.bn(x_b)
        x_b = self.norm(x_b)
        x_b = self.act(x_b)
        x_r = x - x_b

        return x_b, x_r

# from spikingjelly.activation_based import functional as sjF

def _gn(num_channels: int, num_groups: int = 8):
    g = num_groups
    while g > 1 and num_channels % g != 0:
        g -= 1
    return nn.GroupNorm(g, num_channels)

class GNAct2d(nn.Module):
    def __init__(self, c, num_groups=8):
        super().__init__()
        self.gn = _gn(c, num_groups)
        self.act = nn.PReLU(c)
    def forward(self, x):
        return self.act(self.gn(x))

class GNAct3d(nn.Module):
    def __init__(self, c, num_groups=8):
        super().__init__()
        self.gn = _gn(c, num_groups)
        self.act = nn.PReLU(c)
    def forward(self, x):
        return self.act(self.gn(x))

class ResConv3d(nn.Module):
    def __init__(self, c, k=(3,3,3), p=(1,1,1), num_groups=8):
        super().__init__()
        self.conv = nn.Conv3d(c, c, kernel_size=k, padding=p, bias=False)
        self.na = GNAct3d(c, num_groups=num_groups)
    def forward(self, x):
        return x + self.na(self.conv(x))

class SNN_Extracted(nn.Module):
    """
    Rain-oriented multi-branch spiking event disentanglement.
    This definition intentionally overrides the earlier prototype before
    MultiStageFusion instantiates SNN_Extracted.
    """
    def __init__(self, args):
        super().__init__()

        self.base_proj = nn.Sequential(
            nn.Conv2d(20, 96, kernel_size=3, padding=1, bias=False),
            _gn(96, 8),
            nn.PReLU(96),
            nn.Conv2d(96, 96, kernel_size=3, padding=1, bias=False),
            _gn(96, 8),
            nn.PReLU(96),
        )

        self.local_motion = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False),
            GNAct3d(8, num_groups=8),
            ResConv3d(8, k=(3, 3, 3), p=(1, 1, 1), num_groups=8),
        )
        self.local_pool = nn.Conv3d(8, 1, kernel_size=1, bias=True)

        self.pre_spa = nn.Sequential(
            nn.Conv3d(1, 1, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            GNAct3d(1, num_groups=1),
        )
        self.t_down = nn.Sequential(
            nn.Conv3d(1, 1, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0), bias=False),
            GNAct3d(1, num_groups=1),
        )
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=1, stride=1, padding=0, bias=False),
            GNAct3d(8, num_groups=8),
        )
        self.s_down1 = nn.Sequential(
            nn.Conv3d(8, 8, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            GNAct3d(8, num_groups=8),
        )
        self.s_down2 = nn.Sequential(
            nn.Conv3d(8, 8, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            GNAct3d(8, num_groups=8),
        )
        self.block = Spiking_vit_MetaFormer(
            detach_reset=args["detach_reset"],
            embed_dim=[8, 8, 8, 8],
            num_heads=2,
            mlp_ratios=4,
            qkv_bias=False,
            depths=4,
            sr_ratios=1,
        )
        self.post_low = ResConv3d(8, k=(3, 3, 3), p=(1, 1, 1), num_groups=8)
        self.post_3d = nn.Sequential(
            ResConv3d(8, k=(3, 3, 3), p=(1, 1, 1), num_groups=8),
            ResConv3d(8, k=(3, 3, 3), p=(1, 1, 1), num_groups=8),
        )
        self.global_pool = nn.Conv3d(8, 1, kernel_size=1, bias=True)

        self.directional_cue = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, padding=1, bias=False),
            _gn(8, 8),
            nn.PReLU(8),
            nn.Conv2d(8, 8, kernel_size=3, padding=1, groups=8, bias=False),
            nn.Conv2d(8, 8, kernel_size=1, bias=False),
            _gn(8, 8),
            nn.PReLU(8),
        )

        self.rain_fusion = nn.Sequential(
            nn.Conv2d(24, 96, kernel_size=3, padding=1, bias=False),
            _gn(96, 8),
            nn.PReLU(96),
            nn.Conv2d(96, 96, kernel_size=3, padding=1, bias=False),
            _gn(96, 8),
            nn.PReLU(96),
        )
        self.rain_gate = nn.Sequential(
            nn.Conv2d(24, 96, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        self.CSA = multi_Sym_CBAM(gate_channels=96, n_blocks=4)
        self.bg_refine = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=3, padding=1, bias=False),
            _gn(96, 8),
            nn.PReLU(96),
        )
        self.alpha = nn.Parameter(torch.tensor(0.0))

    @torch.no_grad()
    def reset_state(self):
        sjF.reset_net(self)

    def _temporal_pool(self, x, pool_layer):
        w = torch.softmax(pool_layer(x), dim=2)
        return (x * w).sum(dim=2)

    def _global_spiking_branch(self, x_vol, H, W):
        x = self.pre_spa(x_vol)
        x = self.t_down(x)
        x = self.conv1(x)
        x = self.s_down2(self.s_down1(x))

        x = x.permute(2, 0, 1, 3, 4).contiguous()
        self.reset_state()
        x = self.block(x)
        x = x.permute(1, 2, 0, 3, 4).contiguous()

        x = self.post_low(x)
        x = F.interpolate(x, size=(x.shape[2], H, W), mode="trilinear", align_corners=False)
        x = self.post_3d(x)
        return self._temporal_pool(x, self.global_pool)

    def forward(self, x_e: torch.Tensor, x_e_r):
        B, C, H, W = x_e.shape
        assert C == 20, "x_e should have 20 channels for pre/post event bins."

        x_base = self.base_proj(x_e)
        x_vol = x_e.view(B, 1, 20, H, W)

        local_feat = self._temporal_pool(self.local_motion(x_vol), self.local_pool)
        global_feat = self._global_spiking_branch(x_vol, H, W)
        dir_input = x_e.abs().view(B, 2, 10, H, W).sum(dim=2)
        dir_feat = self.directional_cue(dir_input)

        rain_motion = torch.cat([local_feat, global_feat, dir_feat], dim=1)
        rain_pred = self.rain_fusion(rain_motion) * self.rain_gate(rain_motion)

        if x_e_r.shape[1] == 96:
            rain_pred = rain_pred + x_e_r
        rain_pred = self.CSA(rain_pred)

        a = torch.sigmoid(self.alpha)
        x_e_r = a * rain_pred

        residual_bg = x_base - x_e_r
        x_e_b = x_base + self.bg_refine(residual_bg)

        return x_e_b, x_e_r

class ConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1, norm=None, bias=True, last_bias=0):
        super(ConvLayer, self).__init__()
        padding = kernel_size // 2
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)

        if last_bias!=0:
            init.constant(self.conv2d.weight, 0)
            init.constant(self.conv2d.bias, last_bias)

        self.norm = None
        if norm == 'bn':
            #self.norm = nn.BatchNorm2d(out_channels)
            self.norm = _make_gn(out_channels, 8)
        elif norm == 'ln':

            self.norm = nn.GroupNorm(1, out_channels)

    def forward(self, x):
        out = self.conv2d(x)
        if self.norm is not None:
            out = self.norm(out)

        return out

class ResidualBlock(nn.Module):

    def __init__(self, channels, groups=1, norm=None, bias=True):
        super(ResidualBlock, self).__init__()
        self.conv1  = ConvLayer(channels, channels, kernel_size=3, stride=1, groups=groups, bias=bias, norm=norm)
        self.conv2  = ConvLayer(channels, channels, kernel_size=3, stride=1, groups=groups, bias=bias, norm=norm)
        self.relu   = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):

        input = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)

        out = out + input

        return out

class ResBlock3D(nn.Module):
    def __init__(self, c, k=(3,1,1), p=(1,0,0), norm='gn'):
        super().__init__()
        self.conv = nn.Conv3d(c, c, kernel_size=k, padding=p, bias=True)
        if norm == 'bn':
            #self.norm = nn.BatchNorm3d(c)
            g = 8 if c % 8 == 0 else 4 if c % 4 == 0 else 2 if c % 2 == 0 else 1
            self.norm = nn.GroupNorm(g, c)
        else:

            g = 8 if c % 8 == 0 else 4 if c % 4 == 0 else 2 if c % 2 == 0 else 1
            self.norm = nn.GroupNorm(g, c)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return x + self.act(self.norm(self.conv(x)))

"""class Head_E(nn.Module):

    Input: rainy_event (B, 20, H, W), where 20 = 2 * 10 bins
    Output: (B, 96, H, W)

    def __init__(self, c_mid=32, out_c=96, norm3d='gn', use_bias=True):
        super().__init__()

        self.spa = nn.Conv3d(2, c_mid, kernel_size=(1,3,3), padding=(0,1,1), groups=2, bias=use_bias)
        self.spa_norm = nn.GroupNorm(8 if c_mid % 8 == 0 else 4, c_mid)
        self.act = nn.LeakyReLU(0.2, inplace=True)

        self.tem1 = ResBlock3D(c_mid, k=(3,1,1), p=(1,0,0), norm=norm3d)
        self.tem2 = ResBlock3D(c_mid, k=(3,1,1), p=(1,0,0), norm=norm3d)

        self.pool_w = nn.Conv3d(c_mid, 1, kernel_size=1, bias=True)

        self.proj2d = nn.Conv2d(c_mid, out_c, kernel_size=3, padding=1, bias=True)
        self.post_norm = nn.GroupNorm(8 if out_c % 8 == 0 else 4, out_c)

    def forward(self, rainy_event):
        B, C, H, W = rainy_event.shape
        assert C == 20

        e = rainy_event.view(B, 2, 10, H, W)  # (B,2,T=10,H,W)
        if check_tensor_nan(e, 'e'): raise SystemExit
        x = self.act(self.spa_norm(self.spa(e)))  # (B,c_mid,10,H,W)
        if check_tensor_nan(x, 'x1'): raise SystemExit

        w = self.spa.weight
        print("spa.weight finite:", torch.isfinite(w).all().item(),
            "dtype:", w.dtype,
            "min/max:", w.min().item(), w.max().item())

        if self.spa.bias is not None:
            b = self.spa.bias
            print("spa.bias finite:", torch.isfinite(b).all().item(),
                "min/max:", b.min().item(), b.max().item())

        y = self.spa(e)
        if check_tensor_nan(y, "spa_out"): raise SystemExit

        y2 = self.spa_norm(y)
        if check_tensor_nan(y2, "spa_norm_out"): raise SystemExit

        x = self.act(y2)
        if check_tensor_nan(x, "act_out"): raise SystemExit

        x = self.tem2(self.tem1(x))               # (B,c_mid,10,H,W)
        if check_tensor_nan(x, 'x2'): raise SystemExit

        w = torch.softmax(self.pool_w(x), dim=2)  # (B,1,10,H,W)
        x2d = (x * w).sum(dim=2)                  # (B,c_mid,H,W)
        if check_tensor_nan(x2d, 'x2d'): raise SystemExit

        X1_e = self.act(self.post_norm(self.proj2d(x2d)))  # (B,96,H,W)
        if check_tensor_nan(X1_e, 'X1_e'): raise SystemExit
        return X1_e"""

class Head_F(nn.Module):

    def __init__(self, args):
        super(Head_F, self).__init__()

        num_bins = args["num_bins"]
        nf = args["nf"]
        use_bias = True
        args["norm"] = 'ln'

        self.conv1 = ConvLayer(9, nf*3, kernel_size=3, stride=1, groups=3, bias=use_bias, norm=args["norm"])
        self.res1 = ResidualBlock(nf*3, groups=3, bias=use_bias, norm=args["norm"])

        #self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv2 = nn.Conv3d(32, 32, kernel_size=(3, 3, 3), stride=(1,1,1), padding=(1,1,1), bias=True)
        #self.bn2 = nn.BatchNorm3d(32)
        self.norm2 = nn.GroupNorm(8 if 32 % 8 == 0 else 4, 32)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.res2 = ResidualBlock(96, groups=1, bias=True, norm=args["norm"])

    def forward(self, rainy_frame):

        #print('\nwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww')
        #print(rainy_frame.shape)[8,9,128,128]
        if check_tensor_nan(rainy_frame, 'rainy_frame'): raise SystemExit
        x1 = self.res1(self.relu(self.conv1(rainy_frame)))

        if check_tensor_nan(x1, 'x135'): raise SystemExit
        #print(X1.shape)[8,96,128,128]
        #print('\nwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww')

        X1_3d = To3D(x1, 3)
        x1 = self.conv2(X1_3d)
        if check_tensor_nan(x1, 'x12'): raise SystemExit
        #x1 = self.bn2(x1)
        x1 = self.norm2(x1)
        if check_tensor_nan(x1, 'x123'): raise SystemExit
        B, C, D, H, W = x1.shape
        x1 = x1.view(B, 32 * 3, H, W)
        X1 = self.res2(self.relu(x1)) #[B, 96, 128, 128]
        if check_tensor_nan(X1, 'x1234'): raise SystemExit

        return X1

class  MultiStageFusion(nn.Module):
    def __init__(self, n_stages=3):
        """
        Create n stage blocks
        """
        super(MultiStageFusion, self).__init__()
        self.n_stages = n_stages

        self.stages = nn.ModuleList()
        for i in range(n_stages):

            stage = nn.ModuleList([

                RGB_Extracted(),

                SNN_Extracted(args={  "detach_reset": True,}),

                CRM(),

                CRM(),

                CrossModalFusion_R(),

                #BiModalComplementaryFusion(c=96, mode="interaction", detach_other=True),
                BiModalComplementaryFusion(),
            ])

            self.stages.append(stage)

    def forward(self, x1, x1_e):
        r_fusions = []

        x1_e_r = torch.zeros_like(x1_e)

        for stage in self.stages:

            #x1_e = x1_e_b + x

            x1_b, x1_r = stage[0](x1)
            x1_e_b_1, x1_e_r = stage[1](x1_e, x1_e_r)
            x1_b, x1_r = stage[2](x1_b, x1_r)
            x1_e_b_2, x1_e_r = stage[3](x1_e_b_1, x1_e_r)
            x1_e_b = x1_e_b_1 + x1_e_b_2

            r_fusion = stage[4](x1_r, x1_e_r)

            x1 = stage[5](x1_b, x1_e_b, x1_r, x1_e_r)

            r_fusions.append(r_fusion)
        return x1, x1_e_b, r_fusions

class RainFusionModule(nn.Module):
    def __init__(self, n_stages=3, c_each=32, out_channels=96):

        super(RainFusionModule, self).__init__()
        in_channels = n_stages * c_each
        mid = max(32, in_channels // 2)

        self.CA = CALayer(in_channels, reduction=8, bias=False)

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 3, padding=1),
            #nn.BatchNorm3d(mid),
            _make_gn(mid, 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, out_channels, 3, padding=1)
        )

    def forward(self, r_fusions):
        F = torch.cat(r_fusions, dim=1)
        F = self.CA(F)
        r_fusion = self.net(F)
        return r_fusion

## Reconstruction and Reproduction Block (RRB)
class Recovery(nn.Module):
    def __init__(self, n_feat=96, kernel_size=3, act=nn.ReLU(), bias=False):
        super(Recovery, self).__init__()
        """self.recon_B =  conv(n_feat, n_feat, kernel_size, bias=bias)
        self.recon_R = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
                nn.Conv2d(n_feat, n_feat, 1, padding=0, bias=bias),
                nn.PReLU(),#nn.PReLU(inplace=True),
                #act,
                nn.Conv2d(n_feat, n_feat, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )"""
        self.conv = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
    def forward(self, x):
        """xB = x[0]
        xR = x[1]
        recon_B = self.recon_B(xB)
        recon_R = self.recon_R(xR)
        res = self.avg_pool(recon_B + recon_R)
        res_att = self.conv_du(res)
        re_rain = recon_B*res_att + recon_R*(1-res_att)"""

        recon_B = self.conv(x[0])
        re_rain = recon_B + self.conv1(x[1])

        return recon_B, re_rain

def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)

class RMFD(BaseModel):

    def __init__(self, args, opt, local_rank):

        BaseModel.__init__(self, opt)

        self.args = args
        self.opt = opt
        self.local_rank = local_rank

        self.nce_layers = [0,1,2]

        self.head_frame = Head_F(self.args).to(local_rank)

        self.stage = MultiStageFusion(n_stages=3).to(local_rank)

        self.rain_con = RainFusionModule().to(local_rank)

        self.recovery = Recovery().to(local_rank)

        self.decoder1 = Decoder(isrgb=True, input_channels=96, output_channels=3).to(local_rank)
        self.decoder2 = Decoder(isrgb=False, input_channels=96, output_channels=20).to(local_rank)
        self.decoder3 = Decoder(isrgb=True, input_channels=96, output_channels=3).to(local_rank)

        self.train_model_names = ["head_frame", "stage", "rain_con", "recovery", "decoder1", "decoder2", "decoder3"]
        self.ddp_model_names = ["head_frame", "stage", "rain_con", "recovery", "decoder1", "decoder2", "decoder3"]
        self.eval_model_names = ["head_frame", "stage", "rain_con", "recovery", "decoder1", "decoder2", "decoder3"]
        self.optimizer_names = ["optimizer_G"]

        self.load_ddp()

        self.optimizer_G = torch.optim.Adam(itertools.chain(self.head_frame.parameters(), self.stage.parameters(), self.rain_con.parameters(), self.recovery.parameters(), self.decoder1.parameters(), self.decoder2.parameters(), self.decoder3.parameters()),
                                            lr=opt.lr, betas=(opt.beta1, opt.beta2))

        self.optimizers = []
        self.optimizers.append(self.optimizer_G)

        self.loss_function_event = EventBranchLoss(lam_rec=1.0, lam_grad=0.5, lam_dir=0.1, alpha=4.0, gamma=0.5, beta=2.0).to(local_rank)
        self.criterion_char = losses.CharbonnierLoss().to(local_rank)
        self.criterion_edge = losses.EdgeLoss().to(local_rank)
        self.criterion_SSIM = SSIM().to(local_rank)

        #self.encoder = MultiScaleSNNEncoder(detach_reset=True)

    def load_ddp(self):

        for name in self.ddp_model_names:
            if isinstance(name, str):
                module = getattr(self, name)
                #module = torch.nn.SyncBatchNorm.convert_sync_batchnorm(module).to(self.local_rank)
                module = module.to(self.local_rank)
                ddp_module = DDP(
                    module,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                    broadcast_buffers=False,

                )
                setattr(self, name, ddp_module)

    def set_input(self, data):

        self.rainy_frame = data["Rain_frame"]
        self.rainy_event = data["Rain_event"]
        self.clean = data["clean"]
        #self.gan_clean = data["gan_clean"]
        #self.gan_event = data["gan_event"]
        self.clean_event = data["clean_event"]

        b,c,h,w = self.rainy_frame.shape

        self.target_index = (c//3)%2

        self.target_frame = self.rainy_frame[:, (self.target_index)*3:(self.target_index+1)*3, :, :]

    def set_input_test(self, data):

        self.rainy_frame = data["Rain_frame"]
        self.rainy_event = data["Rain_event"]
        self.clean = data["clean"]
        #self.gan_event = data["gan_event"]

        b,c,h,w = self.rainy_frame.shape

        self.target_index = (c//3)%2

        self.target_frame = self.rainy_frame[:, (self.target_index)*3:(self.target_index+1)*3, :, :]

    def forward(self):

        x1 = self.head_frame(self.rainy_frame)
        x1_e = self.rainy_event
        if check_tensor_nan(x1, 'x1'): raise SystemExit

        if check_tensor_nan(x1_e, 'x1_e'): raise SystemExit
        frame_bg, event_bg, r_fusions = self.stage(x1, x1_e)

        if check_tensor_nan(frame_bg, 'frame_bg'): raise SystemExit
        if check_tensor_nan(event_bg, 'event_bg'): raise SystemExit
        if check_tensor_nan(r_fusions[0], 'r_fusions[0]'): raise SystemExit
        """if check_tensor_nan(r_fusions[1], 'r_fusions[1]'): raise SystemExit
        if check_tensor_nan(r_fusions[2], 'r_fusions[2]'): raise SystemExit"""

        frame_rain = self.rain_con(r_fusions)

        if check_tensor_nan(frame_rain, 'frame_rain'): raise SystemExit

        frame_bg, I_rain = self.recovery([frame_bg, frame_rain])

        if check_tensor_nan(frame_bg, 'frame_bg1'): raise SystemExit
        if check_tensor_nan(I_rain, 'I_rain'): raise SystemExit

        frame_bg = self.decoder1(frame_bg)
        event_bg = self.decoder2(event_bg)
        I_rain = self.decoder3(I_rain)

        self.Pred_bg = frame_bg
        self.Pred_event = event_bg
        self.Pred_rl = I_rain - frame_bg

        return frame_bg, event_bg, I_rain

    def optimize_parameters(self):

        """if isinstance(self.snn_extracted, DDP):
            self.snn_extracted.module.reset_state()
        else:
            self.snn_extracted.reset_state()"""
        self.optimizer_G.zero_grad()
        frame_bg, event_bg, I_rain = self.forward()
        self.loss = self.compute_G_loss(frame_bg, event_bg, I_rain)
        self.loss.backward()

        self.optimizer_G.step()

    def compute_G_loss(self, frame_bg, event_bg, I_rain):

        with torch.no_grad():
            for name, t in [('frame_bg', frame_bg),
                            ('event_bg', event_bg),
                            ('I_rain', I_rain),
                            ('clean', self.clean),
                            ('target_frame', self.target_frame),
                            ('clean_event', self.clean_event)]:
                if not torch.isfinite(t).all():
                    print(f"[NaN DEBUG] {name} contains NaN/Inf!")

        self.loss_char0 = self.criterion_char(frame_bg, self.clean)

        if not torch.isfinite(self.loss_char0):
            print("[NaN DEBUG] loss_char0 NaN")

        self.loss_char1 = self.criterion_char(I_rain, self.target_frame)

        if not torch.isfinite(self.loss_char1):
            print("[NaN DEBUG] loss_char1 NaN")

        self.loss_edge0 = self.criterion_edge(frame_bg, self.clean)

        if not torch.isfinite(self.loss_edge0):
            print("[NaN DEBUG] loss_edge0 NaN")

        """self.loss_edge1 = self.criterion_edge(I_rain, self.target_frame)

        if not torch.isfinite(self.loss_edge1):
            print("[NaN DEBUG] loss_edge1 NaN")"""

        self.loss_SSIM0 = self.criterion_SSIM(frame_bg, self.clean)

        if not torch.isfinite(self.loss_SSIM0):
            print("[NaN DEBUG] loss_SSIM0 NaN")

        self.loss_SSIM1 = self.criterion_SSIM(I_rain, self.target_frame)

        if not torch.isfinite(self.loss_SSIM1):
            print("[NaN DEBUG] loss_SSIM1 NaN")

        #self.loss_event = self.loss_function_event(event_bg, self.gan_event)
        #self.loss_event = self.loss_function_event(event_bg, self.gan_event, return_parts=False)
        self.loss_event = self.loss_function_event(event_bg, self.clean_event, return_parts=False)

        if not torch.isfinite(self.loss_event):
            print("[NaN DEBUG] loss_event NaN")

        self.loss_sum = 0.3*(self.loss_char0 + 0.2*self.loss_char1) + (0.2*(self.loss_edge0)) - (0.15*(self.loss_SSIM0 + 0.2*self.loss_SSIM1)) + self.loss_event

        return self.loss_sum

    def train(self):

        for name in self.train_model_names:

            if isinstance(name, str):

                net = getattr(self, name)

                net.train()

    def eval(self):

        for name in self.eval_model_names:

            if isinstance(name, str):

                net = getattr(self, name)

                net.eval()

    def get_losses(self):

        loss_char0 = self.loss_char0.item()
        loss_char1 = self.loss_char1.item()
        loss_edge0 = self.loss_edge0.item()
        loss_SSIM0 = self.loss_SSIM0.item()
        loss_SSIM1 = self.loss_SSIM1.item()
        loss_event = self.loss_event.item()
        loss_sum = self.loss_sum.item()

        loss_record = { "loss_sum": loss_sum,
                        "loss_char0": loss_char0,
                        "loss_char1": loss_char1,
                        "loss_edge0": loss_edge0,
                        "loss_SSIM0": loss_SSIM0,
                        "loss_SSIM1": loss_SSIM1,
                        "loss_event": loss_event,
                        }
        return loss_record

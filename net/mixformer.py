# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Code was based on https://github.com/microsoft/Swin-Transformer
from functools import reduce, lru_cache
from operator import mul
# import numpy as np
import torch
import numpy as np
# import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, trunc_normal_,to_2tuple


# # TODO: update the urls of the pre-trained models
# MODEL_URLS = {
#     "mixformer-B0": "",
#     "mixformer-B1": "",
#     "mixformer-B2": "",
#     "mixformer-B3": "",
#     "mixformer-B4": "",
#     "mixformer-B5": "",
#     "mixformer-B6": "",
# }
#
# __all__ = list(MODEL_URLS.keys())


def window_partition(x, window_size):
    """
    Args:
        x: (B, D, H, W, C)
        window_size (tuple[int]): window size

    Returns:
        windows: (B*num_windows, window_size*window_size, C)
    """
    B,  H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    # windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, reduce(mul, window_size), C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, reduce(mul, window_size), C)

    return windows


def window_reverse(windows, window_size, H, W, C):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    x = windows.reshape(
        [-1, H // window_size[0], W // window_size[1], window_size[1], window_size[1], C])
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    return x

def window_partition2(x, window_size):
    """ Split the feature map to windows.
    B, C, H, W --> B * H // win * W // win x win*win x C

    Args:
        x: (B, C, H, W)
        window_size (tuple[int]): window size

    Returns:
        windows: (num_windows*B, window_size * window_size, C)
    """
    B, C, H, W = x.shape
    x = x.reshape(
        [B, C, H // window_size[0], window_size[0], W // window_size[1], window_size[1]])
    windows = x.permute(0, 2, 4, 3, 5, 1).contiguous().view(
    -1, window_size[0] * window_size[1], C)
    return windows


def window_reverse2(windows, window_size, H, W, C):
    """ Windows reverse to feature map.
    B * H // win * W // win x win*win x C --> B, C, H, W

    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (tuple[int]): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, C, H, W)
    """
    x = windows.reshape(
        [-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C])
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(-1, C, H, W)
    return x


class BasicBlock(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 ksize=3, stride=1, pad=1, dilation=1):
        super(BasicBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, pad, dilation),
            nn.ReLU(inplace=True)
        )

        # init_weights(self.modules)

    def forward(self, x):
        out = self.body(x)
        return out


class BasicBlockSig(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 ksize=3, stride=1, pad=1):
        super(BasicBlockSig, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, pad),
            nn.ReLU(inplace=True)
            # nn.Sigmoid()
        )

        # init_weights(self.modules)

    def forward(self, x):
        out = self.body(x)
        return out

class CALayer(nn.Module):
    def __init__(self, channel, reduction=1):
        super(CALayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.c1 = BasicBlock(channel , channel // reduction, 3, 1, 3, 3)
        self.c2 = BasicBlock(channel , channel // reduction, 3, 1, 5, 5)
        self.c3 = BasicBlock(channel , channel // reduction, 3, 1, 7, 7)
        self.c4 = BasicBlockSig((channel // reduction)*3, channel//2 , 3, 1, 1)

    def forward(self, x):
        y = self.avg_pool(x)
        c1 = self.c1(y)
        c2 = self.c2(y)
        c3 = self.c3(y)
        c_out = torch.cat([c1, c2, c3], dim=1)
        y = self.c4(c_out)
        return y


# class ResidualBlock(nn.Module):
#     def __init__(self,
#                  in_channels, out_channels):
#         super(ResidualBlock, self).__init__()
#
#         self.body = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, 3, 1, 1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, 3, 1, 1),
#         )
#
#         # init_weights(self.modules)
#
#     def forward(self, x):
#         out = self.body(x)
#         out = F.relu(out + x)
#         return out

# class Block(nn.Module):
#     def __init__(self, in_channels, out_channels, group=1):
#         super(Block, self).__init__()
#
#         self.r1 = ResidualBlock(in_channels, out_channels)
#         self.r2 = ResidualBlock(in_channels * 2, out_channels * 2)
#         self.r3 = ResidualBlock(in_channels * 4, out_channels * 4)
#         self.g =  BasicBlock(in_channels * 8, out_channels, 1, 1, 0)
#         self.ca = CALayer(in_channels)
#
#     def forward(self, x):
#         c0 = x
#
#         r1 = self.r1(c0)
#         c1 = torch.cat([c0, r1], dim=1)
#
#         r2 = self.r2(c1)
#         c2 = torch.cat([c1, r2], dim=1)
#
#         r3 = self.r3(c2)
#         c3 = torch.cat([c2, r3], dim=1)
#
#         g = self.g(c3)
#         out = self.ca(g)
#         return out

class Sobelxy(nn.Module):
    def __init__(self,channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(Sobelxy, self).__init__()
        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])
        self.convx=nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels,bias=False)
        self.convx.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.convy=nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels,bias=False)
        self.convy.weight.data.copy_(torch.from_numpy(sobel_filter.T))
        self.conv=nn.Conv2d(channels,channels,kernel_size=1,padding=0,stride=1)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        sobelx = self.convx(x)
        sobely = self.convy(x)
        x=torch.abs(sobelx) + torch.abs(sobely)
        out=self.act(self.conv(x))
        return out


class SpatialAttention(nn.Module):
    def __init__(self,in_channles,out_channels,kernel_size=3):
        super().__init__()
        self.conv1=nn.Conv2d(in_channels=in_channles,out_channels=out_channels,kernel_size=kernel_size,padding=1,stride=1)
        self.conv2 = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x=self.conv1(x)
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv2(result)
        output = self.sigmoid(output)
        return output

class MixingAttention(nn.Module):
    r""" Mixing Attention Module.
    Modified from Window based multi-head self attention (W-MSA) module with
    relative position bias.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        dwconv_kernel_size (int): The kernel size for dw-conv
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self,
                 dim,
                 window_size,
                 dwconv_kernel_size,
                 num_heads,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.dim = dim
        attn_dim = dim // 2
        self.window_size = window_size  # Wh, Ww
        self.dwconv_kernel_size = dwconv_kernel_size
        self.num_heads = num_heads
        head_dim = attn_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        # self.relative_position_bias_table = self.create_parameter(
        #     shape=((2 * window_size[0] - 1) * (2 * window_size[1] - 1),
        #            num_heads),
        #     default_initializer=zeros_)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1),
                        num_heads))
        # self.add_parameter("relative_position_bias_table",
        #                    self.relative_position_bias_table)

        # get pair-wise relative position index for each token inside the window
        # relative_coords = self._get_rel_pos()
        # self.relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        # self.register_buffer("relative_position_index",
        #                      self.relative_position_index)
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[0])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w))  # 3, Wd, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wd*Wh*Ww, Wd*Wh*Ww, 3
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        # relative_coords[:, :, 2] += self.window_size[2] - 1

        # relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1)
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        # prev proj layer
        self.proj_attn = nn.Linear(dim, dim // 2)
        self.proj_attn_norm = nn.LayerNorm(dim // 2)
        self.proj_cnn = nn.Linear(dim, dim)
        self.proj_cnn_norm = nn.LayerNorm(dim)
        self.channel_proj1 = nn.Linear(dim, dim // 2)
        self.act1 = nn.ReLU(inplace=True)

        # conv branch
        # self.dwconv1x1 = nn.Sequential(
        #     nn.Conv2d(
        #         dim, dim,
        #         kernel_size=1,
        #         padding=self.dwconv_kernel_size // 2,
        #         groups=dim
        #     ),
        #     nn.BatchNorm2d(dim),
        #     nn.ReLU()
        # )
        self.dwconv3x3 = nn.Sequential(
            nn.Conv2d(
                dim, dim,
                kernel_size=self.dwconv_kernel_size,
                padding=self.dwconv_kernel_size // 2,
                groups=dim
            ),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )

        # self.dwconv3x3_1 = nn.Sequential(
        #     nn.Conv2d(
        #         dim//2, dim//2,
        #         kernel_size=self.dwconv_kernel_size,
        #         padding=self.dwconv_kernel_size // 2,
        #
        #     ),
        #     nn.BatchNorm2d(dim//2),
        #     nn.ReLU()
        # )
        #
        # self.dwconv3x3_2 = nn.Sequential(
        #  nn.Conv2d(
        #         dim//2, dim//2,
        #         kernel_size=self.dwconv_kernel_size,
        #         padding=self.dwconv_kernel_size // 2,
        #
        #     ),
        #     nn.BatchNorm2d(dim//2),
        #     nn.ReLU()
        # )
        #
        # self.dwconv3x3_3 = nn.Sequential(
        #  nn.Conv2d(
        #         dim//2, dim//2,
        #         kernel_size=self.dwconv_kernel_size,
        #         padding=self.dwconv_kernel_size // 2,
        #
        #     ),
        #     nn.BatchNorm2d(dim//2),
        #     nn.ReLU()
        # )
        #
        # self.dwconv3x3_4 = nn.Sequential(
        #  nn.Conv2d(
        #         dim//2, dim//2,
        #         kernel_size=self.dwconv_kernel_size,
        #         padding=self.dwconv_kernel_size // 2,
        #
        #     ),
        #     nn.BatchNorm2d(dim//2),
        #     nn.ReLU()
        # )
        #
        # self.dwconv3x3_5 = nn.Sequential(
        #  nn.Conv2d(
        #         dim, dim,
        #         kernel_size=self.dwconv_kernel_size,
        #         padding=self.dwconv_kernel_size // 2,
        #
        #     ),
        #     nn.BatchNorm2d(dim),
        #     nn.ReLU()
        # )
        #
        # self.dwconv3x3_6 = nn.Sequential(
        #  nn.Conv2d(
        #         dim, dim,
        #         kernel_size=self.dwconv_kernel_size,
        #         padding=self.dwconv_kernel_size // 2,
        #
        #     ),
        #     nn.BatchNorm2d(dim),
        #     nn.ReLU()
        # )


        # self.GeLU = nn.ReLU(inplace=True)
        # self.dwconv5x5 = nn.Sequential(
        #     nn.Conv2d(
        #         dim, dim,
        #         kernel_size=5,
        #         padding=self.dwconv_kernel_size // 2,
        #         groups=dim
        #     ),
        #     nn.BatchNorm2d(dim),
        #     nn.ReLU()
        # )
        #


        # self.channel_interaction = nn.Sequential(
        #     nn.Conv2d(dim, dim // 8, kernel_size=1),
        #     nn.BatchNorm2d(dim // 8),
        #     nn.ReLU(),
        #     nn.Conv2d(dim // 8, dim // 2, kernel_size=1),
        # )

        # self.channel_interaction=CALayer(channel=dim)
        self.channel_interaction = nn.Sequential(
            nn.Conv2d(dim, dim // 8, kernel_size=1),
            nn.BatchNorm2d(dim // 8),
            nn.GELU(),
            nn.Conv2d(dim // 8, dim // 2, kernel_size=1),
        )


        self.projection = nn.Conv2d(dim, dim // 2, kernel_size=1)
        self.conv_norm = nn.BatchNorm2d(dim // 2)

        # window-attention branch
        self.qkv = nn.Linear(dim // 2, dim // 2 * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        # self.spatial_interaction = nn.Sequential(
        #     nn.Conv2d(dim // 2, dim // 8, kernel_size=1),
        #     nn.BatchNorm2d(dim // 8),
        #     nn.ReLU(),
        #     nn.Conv2d(dim // 8, 1, kernel_size=1)
        # )
        # self.spatial_interaction=SpatialAttention(in_channles=dim//2,out_channels=dim//8,kernel_size=3)
        self.spatial_interaction = nn.Sequential(
            nn.Conv2d(dim // 2, dim // 8, kernel_size=1),
            nn.BatchNorm2d(dim // 8),
            nn.GELU(),
            nn.Conv2d(dim // 8, 1, kernel_size=1)
        )
        self.attn_norm = nn.LayerNorm(dim // 2)

        # final projection
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table)
        self.softmax = nn.Softmax(dim=-1)

    # def _get_rel_pos(self):
    #     """ Get pair-wise relative position index for each token inside the window.
    #
    #     Args:
    #         window_size (tuple[int]): window size
    #     """
    #     coords_h = paddle.arange(self.window_size[0])
    #     coords_w = paddle.arange(self.window_size[1])
    #     coords = paddle.stack(paddle.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
    #     coords_flatten = paddle.flatten(coords, 1)  # 2, Wh*Ww
    #     coords_flatten_1 = coords_flatten.unsqueeze(axis=2)
    #     coords_flatten_2 = coords_flatten.unsqueeze(axis=1)
    #     relative_coords = coords_flatten_1 - coords_flatten_2
    #     relative_coords = relative_coords.transpose(
    #         [1, 2, 0])  # Wh*Ww, Wh*Ww, 2
    #     relative_coords[:, :, 0] += self.window_size[
    #                                     0] - 1  # shift to start from 0
    #     relative_coords[:, :, 1] += self.window_size[1] - 1
    #     relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
    #     return relative_coords

    def forward(self, x, H, W, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            H: the height of the feature map
            W: the width of the feature map
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        # B * H // win * W // win x win*win x C
        x_atten = self.proj_attn_norm(self.proj_attn(x))
        x_cnn = self.proj_cnn_norm(self.proj_cnn(x))
        # B * H // win * W // win x win*win x C --> B, C, H, W
        if H<=self.window_size[0]:
            self.window_size=(H,W)
            x_cnn=window_reverse2(x_cnn, self.window_size, H, W, x_cnn.shape[-1])
        else:
            x_cnn = window_reverse2(x_cnn, self.window_size, H, W, x_cnn.shape[-1])
        x_cnn = self.dwconv3x3(x_cnn)
        channel_interaction = self.channel_interaction(
            F.adaptive_avg_pool2d(x_cnn, output_size=1))
        x_cnn = self.projection(x_cnn)

        # attention branch
        B_, N, C = x_atten.shape
        qkv = self.qkv(x_atten).reshape([B_, N, 3, self.num_heads, C // self.num_heads]).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        # channel interaction
        x_cnn2v = F.sigmoid(channel_interaction).reshape(
            [-1, 1, self.num_heads, 1, C // self.num_heads])
        v = v.reshape(
            [x_cnn2v.shape[0], -1, self.num_heads, N, C // self.num_heads])
        v = v * x_cnn2v
        v = v.reshape([-1, self.num_heads, N, C // self.num_heads])

        q = q * self.scale
        # attn = paddle.mm(q, k.transpose([0, 1, 3, 2]))
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index[:N, :N].reshape(-1)].reshape(
            N, N, -1)  # Wd*Wh*Ww,Wd*Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wd*Wh*Ww, Wd*Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)  # B_, nH, N, N

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.reshape([B_ // nW, nW, self.num_heads, N, N]) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.reshape([-1, self.num_heads, N, N])
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        # x_atten = paddle.mm(attn, v).transpose([0, 2, 1, 3]).reshape([B_, N, C])
        x_atten = (attn @ v).transpose(1, 2).reshape(B_, N, C)

        # spatial interaction
        x_spatial = window_reverse2(x_atten, self.window_size, H, W, C)
        spatial_interaction = self.spatial_interaction(x_spatial)

        # x_cnn = self.projection(x_cnn)
        x_cnn = F.sigmoid(spatial_interaction) * x_cnn
        x_cnn = self.conv_norm(x_cnn)
        # B, C, H, W --> B * H // win * W // win x win*win x C
        x_cnn = window_partition2(x_cnn, self.window_size)

        # concat
        x_atten = self.attn_norm(x_atten)
        x = torch.cat([x_atten, x_cnn], dim=-1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # projection layers
        flops += N * self.dim * self.dim * 3 // 2
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops

class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def get_window_size(x_size, window_size, shift_size=None):
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)

class MixingBlock(nn.Module):
    r""" Mixing Block in MixFormer.
    Modified from Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        dwconv_kernel_size (int): kernel size for depth-wise convolution.
        shift_size (int): Shift size for SW-MSA. We do not use shift in MixFormer. Default: 0
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.ReLU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(7,7),
                 dwconv_kernel_size=3,
                 shift_size=(0,0),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.ReLU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        # assert self.shift_size == 0, "No shift in MixFormer"

        self.norm1 = norm_layer(dim)
        self.attn = MixingAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            dwconv_kernel_size=dwconv_kernel_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)
        self.H = None
        self.W = None
        self.sobel=Sobelxy(channels=dim)

    def forward(self, x, mask_matrix):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.
        """
        B, L, C = x.shape
        H, W = self.H, self.W
        # assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.reshape(B, H, W, C)
        x1=x.reshape(B,C,H,W)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        window_size, shift_size = get_window_size((H, W), self.window_size, self.shift_size)
        # pad_r = (self.window_size - W % self.window_size) % self.window_size
        # pad_b = (self.window_size - H % self.window_size) % self.window_size
        pad_b = (window_size[0] - H % window_size[0]) % window_size[0]
        pad_r = (window_size[1] - W % window_size[1]) % window_size[1]
        x = F.pad(x, [0, pad_l, 0, pad_r, 0, pad_b, 0, pad_t])
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(
            shifted_x, window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.reshape(
            [-1, window_size[0] * window_size[1],
             C])  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(
            x_windows, Hp, Wp, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.reshape(
            [-1, window_size[0], window_size[1], C])
        shifted_x = window_reverse(attn_windows, window_size, Hp,
                                   Wp, C)  # B H' W' C

        # reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :]

        x = x.reshape([B, H * W, C])

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x_sobel=self.sobel(x1)
        x_sobel = x_sobel.reshape([B, H * W, C])

        return x+x_sobel

    # def extra_repr(self):
    #     return "dim={}, input_resolution={}, num_heads={}, window_size={}, shift_size={}, mlp_ratio={}".format(
    #         self.dim, self.input_resolution, self.num_heads, self.window_size,
    #         self.shift_size, self.mlp_ratio)

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W

        # Mixing Attention
        flops += self.dim * H * W  # proj_cnn_norm
        flops += self.dim // 2 * H * W  # proj_attn_norm
        flops += self.dim * 1 * (self.conv_kernel_size ** 2) * H * W  # dwconv
        flops += self.dim * H * W  # BatchNorm2d
        flops += self.dim * self.dim // 2 * H * W  # conv1x1
        # channel_interaction
        flops += self.dim * self.dim // 8 * 1 * 1
        flops += self.dim // 8 * 1 * 1
        flops += self.dim // 8 * self.dim // 2 * 1 * 1
        # spatial_interaction
        flops += self.dim // 2 * self.dim //8 * H * W
        flops += self.dim // 8 * H * W
        flops += self.dim // 8 * 1 * H * W
        # branch norms
        flops += self.dim // 2 * H * W
        flops += self.dim // 2 * H * W
        # inside Mixing Attention
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)

        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class ConvMerging(nn.Module):
    r""" Conv Merging Layer.

    Args:
        dim (int): Number of input channels.
        out_dim (int): Output channels after the merging layer.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, out_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        # self.reduction = nn.Conv2d(dim, out_dim, kernel_size=2, stride=2,padding=0)  # h/2  w/2
        self.reduction = nn.Conv2d(dim, out_dim, kernel_size=3, stride=2, padding=1)
        # self.reduction = nn.Conv2d(dim, out_dim, kernel_size=3, stride=1, padding=1)  # h/2  w/2
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x, H, W):
        """
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        # assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view([B, H, W, C]).permute([0, 3, 1, 2])

        x = self.norm(x)
        x = self.reduction(x).flatten(2).permute(0, 2, 1)  # B, C, H, W -> B, H*W, C
        return x

    # def extra_repr(self) -> str:
    #     return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    """ A basic layer for one stage in MixFormer.
    Modified from Swin Transformer BasicLayer.

    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        dwconv_kernel_size (int): kernel size for depth-wise convolution.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        out_dim (int): Output channels for the downsample layer. Default: 0.
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=7,
                 dwconv_kernel_size=3,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 out_dim=0):
        super().__init__()
        self.window_size = window_size
        # self.shift_size = (0,0)
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList([
            MixingBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                dwconv_kernel_size=dwconv_kernel_size,
                shift_size=(0,0),
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i]
                if isinstance(drop_path, (np.ndarray, list)) else drop_path,
                norm_layer=norm_layer) for i in range(depth)
        ])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, out_dim=out_dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        b=x.shape[0]
        for blk in self.blocks:
            blk.H, blk.W = H, W
            x = blk(x, None)
        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2 # floor int
            return b,H, W, x_down, Wh, Ww
        else:
            return b, H, W, x, H, W

    # def extra_repr(self):
    #     return "dim={}, input_resolution={}, depth={}".format(
    #         self.dim, self.input_resolution, self.depth)

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class ConvEmbed(nn.Module):
    r""" Image to Conv Stem Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.stem = nn.Sequential(
            # nn.Conv2d(in_chans, embed_dim // 2, kernel_size=3, stride=patch_size[0] // 2, padding=1),
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(),
            nn.Conv2d(embed_dim // 2, embed_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(),
            nn.Conv2d(embed_dim // 2, embed_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(),
        )
        # self.proj = nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=patch_size[0] // 2, stride=patch_size[0] // 2)
        self.proj = nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=1,padding=1)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        if W % self.patch_size[1] != 0:
            x = F.pad(x, [0, self.patch_size[1] - W % self.patch_size[1], 0, 0])
        if H % self.patch_size[0] != 0:
            x = F.pad(x, [0, 0, 0, self.patch_size[0] - H % self.patch_size[0]])

        x = self.stem(x)
        x = self.proj(x)
        if self.norm is not None:
            _, _, Wh, Ww = x.shape
            x = x.flatten(2).transpose(1, 2)

            x = self.norm(x)

            x = x.reshape([-1, self.embed_dim, Wh, Ww])
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        # stem first 3x3 + BN
        flops = (Ho * 2) * (Wo * 2) * self.embed_dim // 2 * self.in_chans * 9
        flops += (Ho * 2) * (Wo * 2) * self.embed_dim // 2
        # stem second 3x3 + BN
        flops += (Ho * 2) * (Wo * 2) * self.embed_dim // 2 * self.embed_dim // 2 * 9
        flops += (Ho * 2) * (Wo * 2) * self.embed_dim // 2
        # stem third 3x3 + BN
        flops += (Ho * 2) * (Wo * 2) * self.embed_dim // 2 * self.embed_dim // 2 * 9
        flops += (Ho * 2) * (Wo * 2) * self.embed_dim // 2
        # proj
        flops += Ho * Wo * self.embed_dim * self.embed_dim // 2 * (
                self.patch_size[0] // 4 * self.patch_size[1] // 4)
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class MixFormer(nn.Module):
    """ A PaddlePaddle impl of MixFormer:
        `MixFormer: Mixing Features across Windows and Dimensions (CVPR 2022, Oral)`

    Modified from Swin Transformer.

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        dwconv_kernel_size (int): kernel size for depth-wise convolution.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self,
                 img_size=(128,128),
                 patch_size=4,
                 in_chans=1,
                 class_num=1000,
                 embed_dim=8,
                 depths=[2, 2, 8, 8],
                 num_heads=[4, 8, 16, 32],
                 window_size=(7,7),
                 dwconv_kernel_size=3,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 use_checkpoint=False,
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes = class_num
        self.num_layers = len(depths)
        if isinstance(embed_dim, int):
            embed_dim = [embed_dim * 2 ** i_layer for i_layer in range(self.num_layers)]
        # assert isinstance(embed_dim, list) and len(embed_dim) == self.num_layers
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(self.embed_dim[-1])
        self.mlp_ratio = mlp_ratio

        # split image into patches
        self.patch_embed = ConvEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            # embed_dim=embed_dim[0],
            embed_dim=8,
            norm_layer=norm_layer if self.patch_norm else None)
        # num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # # absolute position embedding
        # if self.ape:
        #     self.absolute_pos_embed = self.create_parameter(
        #         shape=(1, num_patches, self.embed_dim[0]), default_initializer=zeros_)
        #     self.add_parameter("absolute_pos_embed", self.absolute_pos_embed)
        #     trunc_normal_(self.absolute_pos_embed)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = np.linspace(0, drop_path_rate,
                          sum(depths)).tolist()  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(self.embed_dim[i_layer]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                dwconv_kernel_size=dwconv_kernel_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=ConvMerging,
                # if (i_layer < self.num_layers - 1) else None,
                out_dim=int(8 * 2 ** (i_layer + 1)))
                # if (i_layer < self.num_layers - 1) else 64*16)
            self.layers.append(layer)

        # self.norm = norm_layer(self.num_features)
        # self.last_proj = nn.Linear(self.num_features, 1280)
        # self.activate = nn.ReLU()
        # self.avgpool = nn.AdaptiveAvgPool1d(1)
        # self.head = nn.Linear(
        #     1280,
        #     num_classes) if self.num_classes > 0 else nn.Identity()

        # self.apply(self._init_weights)

    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         trunc_normal_(m.weight)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             zeros_(m.bias)
    #     elif isinstance(m, nn.LayerNorm):
    #         zeros_(m.bias)
    #         ones_(m.weight)

    def forward_features(self, x):
        x = self.patch_embed(x)
        _, _, Wh, Ww = x.shape
        x = x.flatten(2).transpose(1, 2)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x_out = []

        for layer in self.layers:
            b,H, W, x, Wh, Ww = layer(x, Wh, Ww)
            # x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
            x1=x.reshape(b,-1,Wh,Ww)

            x_out.append(x1)

        # x = self.norm(x)  # B L C
        # x = self.last_proj(x)
        # x = self.activate(x)
        # x = self.avgpool(x.permute(0, 2, 1))  # B C 1
        # x = torch.flatten(x, 1)
        return x_out


    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for _, layer in enumerate(self.layers):
            flops += layer.flops()
        # norm
        flops += self.num_features * self.patches_resolution[
            0] * self.patches_resolution[1] // (2 ** self.num_layers)
        # last proj
        flops += self.num_features * 1280 * self.patches_resolution[
            0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += 1280 * self.num_classes
        return flops

#
# def _load_pretrained(pretrained, model, model_url, use_ssld=False):
#     if pretrained is False:
#         pass
#     elif pretrained is True:
#         load_dygraph_pretrain_from_url(model, model_url, use_ssld=use_ssld)
#     elif isinstance(pretrained, str):
#         load_dygraph_pretrain(model, pretrained)
#     else:
#         raise RuntimeError(
#             "pretrained type is not available. Please use `string` or `boolean` type."
#         )


# def MixFormer_B0(pretrained=False, use_ssld=False, **kwargs):
#     model = MixFormer(
#         embed_dim=24,
#         depths=[1, 2, 6, 6],
#         num_heads=[3, 6, 12, 24],
#         drop_path_rate=0.,
#         **kwargs)
#     _load_pretrained(
#         pretrained,
#         model,
#         MODEL_URLS["mixformer-B0"],
#         use_ssld=use_ssld)
#     return model
#
#
# def MixFormer_B1(pretrained=False, use_ssld=False, **kwargs):
#     model = MixFormer(
#         embed_dim=32,
#         depths=[1, 2, 6, 6],
#         num_heads=[2, 4, 8, 16],
#         drop_path_rate=0.,
#         **kwargs)
#     _load_pretrained(
#         pretrained,
#         model,
#         MODEL_URLS["mixformer-B1"],
#         use_ssld=use_ssld)
#     return model
#
#
# def MixFormer_B2(pretrained=False, use_ssld=False, **kwargs):
#     model = MixFormer(
#         embed_dim=32,
#         depths=[2, 2, 8, 8],
#         num_heads=[2, 4, 8, 16],
#         drop_path_rate=0.05,
#         **kwargs)
#     _load_pretrained(
#         pretrained,
#         model,
#         MODEL_URLS["mixformer-B2"],
#         use_ssld=use_ssld)
#     return model
#
#
# def MixFormer_B3(pretrained=False, use_ssld=False, **kwargs):
#     model = MixFormer(
#         embed_dim=48,
#         depths=[2, 2, 8, 6],
#         num_heads=[3, 6, 12, 24],
#         drop_path_rate=0.1,
#         **kwargs)
#     _load_pretrained(
#         pretrained,
#         model,
#         MODEL_URLS["mixformer-B3"],
#         use_ssld=use_ssld)
#     return model
#
#
# def MixFormer_B4(pretrained=False, use_ssld=False, **kwargs):
#     model = MixFormer(
#         embed_dim=64,
#         depths=[2, 2, 8, 8],
#         num_heads=[4, 8, 16, 32],
#         drop_path_rate=0.2,
#         **kwargs)
#     _load_pretrained(
#         pretrained,
#         model,
#         MODEL_URLS["mixformer-B4"],
#         use_ssld=use_ssld)
#     return model
#
#
# def MixFormer_B5(pretrained=False, use_ssld=False, **kwargs):
#     model = MixFormer(
#         embed_dim=96,
#         depths=[1, 2, 8, 6],
#         num_heads=[6, 12, 24, 48],
#         drop_path_rate=0.3,
#         **kwargs)
#     _load_pretrained(
#         pretrained,
#         model,
#         MODEL_URLS["mixformer-B5"],
#         use_ssld=use_ssld)
#     return model
#
#
# def MixFormer_B6(pretrained=False, use_ssld=False, **kwargs):
#     model = MixFormer(
#         embed_dim=96,
#         depths=[2, 4, 16, 12],
#         num_heads=[6, 12, 24, 48],
#         drop_path_rate=0.5,
#         **kwargs)
#     _load_pretrained(
#         pretrained,
#         model,
#         MODEL_URLS["mixformer-B6"],
#         use_ssld=use_ssld)
#     return model

if __name__ == '__main__':
    model = MixFormer(embed_dim=8,depths=[2,2,4,8,8],num_heads=[4,8,16,16,32],drop_rate=0.2)
    # model.eval()9
    print('# generator parameters:', sum(param.numel() for param in model.parameters()))
    x1 = torch.rand((2, 1, 128 , 128))
    a=model(x1)
    print(a[0].shape)
    print(a[1].shape)
    print(a[2].shape)
    print(a[3].shape)
    print(a[4].shape)

    # x2 = torch.rand((1, 1, 128, 128))


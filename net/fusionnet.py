import torch
import torch.nn as nn
import torch.nn.functional as F
from net.dualmixformer import dual_MixFormer
from attention import DynamicConv

def maxpool():
    pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    return pool


def upsample(src, tar):
    src = F.interpolate(src, size=tar.shape[2:], mode='bilinear', align_corners=True)  # upsample
    return src


def dowmsample(src, tar):
    src = F.interpolate(src, size=tar.shape[2:], mode='bilinear', align_corners=True)  # upsample
    return src


class Conv_ex(nn.Module):
    def __init__(self, in_dim=1, out_dim=16):
        super(Conv_ex, self).__init__()
        act_fn = nn.LeakyReLU(0.2, inplace=False)
        # act_fn2 = nn.ReLU(inplace=False)  # nn.ReLU()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.down_1_0 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_dim, out_channels=self.out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_dim), act_fn,
            nn.Conv2d(in_channels=self.out_dim, out_channels=self.out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_dim), act_fn,
        )
        self.pool_1_0 = maxpool()

        self.down_1_1 = nn.Sequential(
            nn.Conv2d(in_channels=self.out_dim, out_channels=self.out_dim * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_dim * 2), act_fn,
            nn.Conv2d(in_channels=self.out_dim * 2, out_channels=self.out_dim * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_dim * 2), act_fn,
        )
        self.pool_1_1 = maxpool()

        self.down_2_0 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_dim, out_channels=self.out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_dim), act_fn,
            nn.Conv2d(in_channels=self.out_dim, out_channels=self.out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_dim), act_fn,
        )
        self.pool_2_0 = maxpool()

        self.down_2_1 = nn.Sequential(
            nn.Conv2d(in_channels=self.out_dim, out_channels=self.out_dim * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_dim * 2), act_fn,
            nn.Conv2d(in_channels=self.out_dim * 2, out_channels=self.out_dim * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_dim * 2), act_fn,
        )
        self.pool_2_1 = maxpool()

    def forward(self, x1, x2):
        conv_1_0 = self.down_1_0(x1)
        conv_1_0m = self.pool_1_0(conv_1_0)
        conv_1_1 = self.down_1_1(conv_1_0m)
        conv_1_1m = self.pool_1_1(conv_1_1)

        conv_2_0 = self.down_2_0(x1)
        conv_2_0m = self.pool_2_0(conv_2_0)
        conv_2_1 = self.down_2_1(conv_2_0m)
        conv_2_1m = self.pool_2_1(conv_2_1)

        return conv_1_1m, conv_2_1m


class ResidualBlock(nn.Module):
    def __init__(self,
                 in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
        )


    def forward(self, x):
        out = self.body(x)
        out = F.relu(out + x)
        return out


class BasicBlock(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 ksize=3, stride=1, pad=1, dilation=1):
        super(BasicBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, pad, dilation),
            nn.ReLU(inplace=True)
        )


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
            nn.Sigmoid()
        )


    def forward(self, x):
        out = self.body(x)
        return out


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.c1 = BasicBlock(channel, channel // reduction, 3, 1, 3, 3)
        self.c2 = BasicBlock(channel, channel // reduction, 3, 1, 5, 5)
        self.c3 = BasicBlock(channel, channel // reduction, 3, 1, 7, 7)
        self.c4 = BasicBlockSig((channel // reduction) * 3, channel, 3, 1, 1)

    def forward(self, x):
        y = self.avg_pool(x)
        c1 = self.c1(y)
        c2 = self.c2(y)
        c3 = self.c3(y)
        c_out = torch.cat([c1, c2, c3], dim=1)
        y = self.c4(c_out)
        return x * y


class DRLM(nn.Module):
    def __init__(self, in_channels=32, out_channels=32):
        super(DRLM, self).__init__()

        self.r1 = ResidualBlock(in_channels, out_channels)
        self.r2 = ResidualBlock(in_channels * 2, out_channels * 2)
        self.r3 = ResidualBlock(in_channels * 4, out_channels * 4)
        self.g = BasicBlock(in_channels * 8, out_channels, 1, 1, 0)
        self.ca = CALayer(in_channels)

    def forward(self, x):
        c0 = x

        r1 = self.r1(c0)
        c1 = torch.cat([c0, r1], dim=1)

        r2 = self.r2(c1)
        c2 = torch.cat([c1, r2], dim=1)

        r3 = self.r3(c2)
        c3 = torch.cat([c2, r3], dim=1)

        g = self.g(c3)
        out = self.ca(g)
        return out



class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.mean(x, 1).unsqueeze(1)


class MLP(nn.Module):
    """
    Linear Embedding:
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x



def conv_decod_block(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
        act_fn,
    )
    return model


def upx2(x):
    return nn.functional.interpolate(x, scale_factor=2)


def upx4(x):
    return nn.functional.interpolate(x, scale_factor=4)


def downx2(x):
    return nn.functional.interpolate(x, scale_factor=0.5)

class convblock(nn.Module):  # dynamic conv
    def __init__(self):
        super(convblock, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.dyconv1 = DynamicConv(in_planes=8, out_planes=8, kernel_size=3, stride=1, padding=3, dilation=3, bias=True)
        self.dyconv2 = DynamicConv(in_planes=8, out_planes=8, kernel_size=3, stride=1, padding=5, dilation=5, bias=True)
        self.dyconv3 = DynamicConv(in_planes=8, out_planes=8, kernel_size=3, stride=1, padding=7, dilation=7, bias=True)
        self.conv2 = nn.Conv2d(24, 32, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        x = self.conv1(x)
        x1 = self.dyconv1(x)
        x2 = self.dyconv2(x)
        x3 = self.dyconv3(x)
        x_out = torch.cat([x1, x2, x3], dim=1)
        x_out = self.conv2(x_out)
        return x_out


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None
        self.pool = maxpool()

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)

        x = self.pool(x)
        return x


class initConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1, relu=True, bn=True, bias=False):
        super(initConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class CrossPath(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.channel_proj1 = nn.Linear(dim, dim // reduction * 2)
        self.channel_proj2 = nn.Linear(dim, dim // reduction * 2)
        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU(inplace=True)
        self.cross_attn = CrossAttention(dim // reduction, num_heads=num_heads)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

    def forward(self, x1, x2):
        v1, v2 = self.cross_attn(x1, x2)
        out_x1 = self.norm1(x1 + v1)
        out_x2 = self.norm2(x2 + v2)
        return out_x1, out_x2


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, proj_drop=0., qkv_bias=True, qk_scale=None):
        super(CrossAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.k1v1 = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.k2v2 = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q1 = nn.Linear(dim, dim, bias=qkv_bias)
        self.q2 = nn.Linear(dim, dim, bias=qkv_bias)
        self.softmax1 = nn.Softmax(dim=-1)
        self.softmax2 = nn.Softmax(dim=-1)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x1, x2):
        B, N, C = x1.shape
        q1 = self.q1(x1).reshape(B, -1, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q2 = self.q2(x2).reshape(B, -1, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()

        k1, v1 = self.k1v1(x1).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1,
                                                                                              4).contiguous()
        k2, v2 = self.k2v2(x2).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1,
                                                                                              4).contiguous()

        q1 = q1 * self.scale
        attn1 = (q1 @ k2.transpose(-2, -1))
        attn1 = self.softmax1(attn1)
        x1 = (attn1 @ v2).transpose(1, 2).reshape(B, N, C)
        x1 = self.proj(x1)
        x1 = self.proj_drop(x1)

        q2 = q2 * self.scale
        attn2 = (q2 @ k1.transpose(-2, -1))
        attn2 = self.softmax2(attn2)
        x2 = (attn2 @ v1).transpose(1, 2).reshape(B, N, C)
        x2 = self.proj(x2)
        x2 = self.proj_drop(x2)

        return x1, x2


class CrossAttention_efficient(nn.Module):
    def __init__(self, dim, num_heads=8, proj_drop=0., qkv_bias=True, qk_scale=None):
        super(CrossAttention_efficient, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.k1v1 = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.k2v2 = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q1 = nn.Linear(dim, dim, bias=qkv_bias)
        self.q2 = nn.Linear(dim, dim, bias=qkv_bias)
        self.softmax1 = nn.Softmax(dim=-1)
        self.softmax2 = nn.Softmax(dim=-1)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax()

        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x1, x2):
        B, N, C = x1.shape
        q1 = self.q1(x1).reshape(B, -1, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q2 = self.q2(x2).reshape(B, -1, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        k1, v1 = self.k1v1(x1).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1,
                                                                                              4).contiguous()
        k2, v2 = self.k2v2(x2).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1,
                                                                                              4).contiguous()

        ctx1 = (k1.transpose(-2, -1) @ v1) * self.scale
        ctx1 = ctx1.softmax(dim=-2)
        ctx2 = (k2.transpose(-2, -1) @ v2) * self.scale
        ctx2 = ctx2.softmax(dim=-2)

        x1 = (q1 @ ctx2).transpose(1, 2).reshape(B, N, C)
        x2 = (q2 @ ctx1).transpose(1, 2).reshape(B, N, C)

        x1 = self.proj(x1)
        x1 = self.proj_drop(x1)

        x2 = self.proj(x2)
        x2 = self.proj_drop(x2)

        return x1, x2


class FeatureFusionModule(nn.Module):  # attn2
    def __init__(self, dim, reduction=1, num_heads=None, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.cross = CrossPath(dim=dim, reduction=reduction, num_heads=num_heads)
        self.conv = nn.Conv2d(in_channels=2 * dim, out_channels=dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        x1 = x1.flatten(2).transpose(1, 2)
        x2 = x2.flatten(2).transpose(1, 2)
        x1, x2 = self.cross(x1, x2)
        x1 = x1.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        x2 = x2.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        x = torch.cat((x1, x2), dim=1)
        out = self.conv(x)

        return out


class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()
        act_fn2 = nn.ReLU(inplace=True)
        self.in_dim = 16
        self.decoder_multi1 = Decoder_multi(channel=8)
        self.decoder_multi2 = Decoder_multi(channel=16)
        self.decoder_multi3 = Decoder_multi(channel=32)
        self.upattn1 = cbam_up(self.in_dim * 8, self.in_dim * 4, kernel_size=7)
        self.upattn2 = cbam_up(self.in_dim * 4, self.in_dim * 2, kernel_size=7)
        self.deconv_4_1 = conv_decod_block(self.in_dim * 2, self.in_dim * 1, act_fn2)
        self.deconv_5_1 = conv_decod_block(self.in_dim, 8, act_fn2)
        self.deconv_6_1 = conv_decod_block(8, 1, act_fn2)

    def forward(self, x, x1, x2, x3, x4, x5):
        yl = self.decoder_multi1(x1, x2, x3)
        ym = self.decoder_multi2(x2, x3, x4)
        yh = self.decoder_multi3(x3, x4, x5)
        y1 = self.upattn1(yh, ym)
        y2 = self.upattn2(y1, yl)
        y4 = self.deconv_4_1(upsample(y2, x1))
        y5 = self.deconv_5_1(upsample(y4, x))
        y6 = self.deconv_6_1(y5)
        output2 = torch.sigmoid(y6)

        return output2


class Attention_od(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, kernel_num=4, min_channel=16):
        super(Attention_od, self).__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = 1.0

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        self.bn = nn.BatchNorm2d(attention_channel)
        self.relu = nn.ReLU(inplace=True)

        self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)
        self.func_channel = self.get_channel_attention

        if in_planes == groups and in_planes == out_planes:  # depth-wise convolution
            self.func_filter = self.skip
        else:
            self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, bias=True)
            self.func_filter = self.get_filter_attention

        if kernel_size == 1:  # point-wise convolution
            self.func_spatial = self.skip
        else:
            self.spatial_fc = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1, bias=True)
            self.func_spatial = self.get_spatial_attention

        if kernel_num == 1:
            self.func_kernel = self.skip
        else:
            self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)
            self.func_kernel = self.get_kernel_attention

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def update_temperature(self, temperature):
        self.temperature = temperature

    @staticmethod
    def skip(_):
        return 1.0

    def get_channel_attention(self, x):
        channel_attention = torch.sigmoid(self.channel_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return channel_attention

    def get_filter_attention(self, x):
        filter_attention = torch.sigmoid(self.filter_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return filter_attention

    def get_spatial_attention(self, x):
        spatial_attention = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size)
        spatial_attention = torch.sigmoid(spatial_attention / self.temperature)
        return spatial_attention

    def get_kernel_attention(self, x):
        kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
        kernel_attention = F.softmax(kernel_attention / self.temperature, dim=1)
        return kernel_attention

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        return self.func_channel(x), self.func_filter(x), self.func_spatial(x), self.func_kernel(x)

class ODConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=1, dilation=1, groups=1,
                 reduction=0.0625, kernel_num=4):
        super(ODConv2d, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_num = kernel_num
        self.attention = Attention_od(in_planes, out_planes, kernel_size, groups=self.groups,
                                   reduction=reduction, kernel_num=kernel_num)
        self.weight = nn.Parameter(torch.randn(kernel_num, out_planes, in_planes//groups, kernel_size, kernel_size),
                                   requires_grad=True)
        self._initialize_weights()

        if self.kernel_size == 1 and self.kernel_num == 1:
            self._forward_impl = self._forward_impl_pw1x
        else:
            self._forward_impl = self._forward_impl_common

    def _initialize_weights(self):
        for i in range(self.kernel_num):
            nn.init.kaiming_normal_(self.weight[i], mode='fan_out', nonlinearity='relu')

    def update_temperature(self, temperature):
        self.attention.update_temperature(temperature)

    def _forward_impl_common(self, x):
        # Multiplying channel attention (or filter attention) to weights and feature maps are equivalent,
        # while we observe that when using the latter method the models will run faster with less gpu memory cost.
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        batch_size, in_planes, height, width = x.size()
        x = x * channel_attention
        x = x.reshape(1, -1, height, width)
        aggregate_weight = spatial_attention * kernel_attention * self.weight.unsqueeze(dim=0)
        aggregate_weight = torch.sum(aggregate_weight, dim=1).view(
            [-1, self.in_planes // self.groups, self.kernel_size, self.kernel_size])
        output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups * batch_size)
        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        output = output * filter_attention
        return output

    def _forward_impl_pw1x(self, x):
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        x = x * channel_attention
        output = F.conv2d(x, weight=self.weight.squeeze(dim=0), bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups)
        output = output * filter_attention
        return output

    def forward(self, x):
        return self._forward_impl(x)

class ResidualBlock_od(nn.Module):
    def __init__(self,
                 in_channels, out_channels):
        super(ResidualBlock_od, self).__init__()

        self.body = nn.Sequential(
            # nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            ODConv2d(in_planes=in_channels, out_planes=out_channels, kernel_size=3),
            nn.ReLU(inplace=True),
            # nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            ODConv2d(in_planes=in_channels, out_planes=out_channels, kernel_size=3),
            nn.ReLU(inplace=True),
        )

        # init_weights(self.modules)

    def forward(self, x):
        out = self.body(x)
        out = F.relu(out + x)
        return out


class Decoder_multi(nn.Module):
    def __init__(self, channel=32):
        super(Decoder_multi, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.channel = channel
        self.conv1 = ResidualBlock_od(in_channels=self.channel * 2, out_channels=self.channel * 2)
        self.conv2 = ResidualBlock_od(in_channels=self.channel * 4, out_channels=self.channel * 4)
        self.conv3 = ResidualBlock_od(in_channels=self.channel * 8, out_channels=self.channel * 8)
        self.conv5 = nn.Conv2d(in_channels=self.channel * 14, out_channels=self.channel * 4, kernel_size=1)

    def forward(self, x1, x2, x3):
        x1_down = dowmsample(x1, x2)
        x1_conv = self.conv1(x1_down)
        x3_up = upsample(x3, x2)
        x3_conv = self.conv3(x3_up)
        x2_conv =self.conv2(x2)
        x_cat = torch.cat((x1_conv, x2_conv, x3_conv), dim=1)
        x_out = self.conv5(x_cat)
        return x_out



from torch.nn import init
class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        return output


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output

class CBAMBlock(nn.Module):

    def __init__(self, channel=512, reduction=16, kernel_size=49):
        super().__init__()
        self.ca = ChannelAttention(channel=channel, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        # residual = x
        out1 = x * self.ca(x)
        out = out1 * self.sa(out1)
        return out

class cbam_up(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size):
        super(cbam_up, self).__init__()
        self.relu=nn.ReLU(inplace=True)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.cbam=CBAMBlock(channel=self.out_channels,kernel_size=self.kernel_size)
        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3, stride=1,
                               padding=1)
        self.conv2 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x1, x2):
        x1_conv = self.relu(self.conv1(x1))
        x1_up = upsample(x1_conv, x2)
        x_cat = torch.cat((x1_up, x2), dim=1)
        x_refine=self.conv2(x_cat)
        cbam_out=self.cbam(x_refine)
        out = cbam_out+x2

        return out

class basicBlock_dual(nn.Module):
    def __init__(self,
                 in_channels, out_channels):
        super(basicBlock_dual, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )


    def forward(self, x):
        out = self.body(x)
        out = F.relu(out)
        return out


class ECAAttention(nn.Module):

    def __init__(self, kernel_size=3, pool=None):
        super().__init__()
        self.gap = pool
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.gap(x)  # bs,c,1,1
        y = y.squeeze(-1).permute(0, 2, 1)  # bs,1,c
        y = self.conv(y)  # bs,1,c
        y = self.sigmoid(y)  # bs,1,c
        y = y.permute(0, 2, 1).unsqueeze(-1)  # bs,c,1,1
        return y


class split_dual_attn(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(split_dual_attn, self).__init__()

        self.conv1 = basicBlock_dual(in_channels=out_channels, out_channels=1)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.channel_attn = ECAAttention(pool=nn.AdaptiveAvgPool2d(1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.conv2(x)
        x_spacial = self.conv1(x1)
        attention_spacial = self.sigmoid(x_spacial)
        x2 = self.conv3(x)
        attention_channel = self.channel_attn(x2)


        return attention_spacial, attention_channel


class dualatt_up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(dualatt_up, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3, stride=1,
                               padding=1)
        self.attn = split_dual_attn(in_channels=self.in_channels, out_channels=self.out_channels)
    def forward(self, x1, x2):
        x1_conv = self.conv1(x1)
        x1_up = upsample(x1_conv, x2)
        x_cat = torch.cat((x1_up, x2), dim=1)
        attn1, attn2 = self.attn(x_cat)
        x2_attn1 = attn1 * x2
        x2_attn2 = attn2 * x2_attn1
        out = x2_attn2 + x1_up

        return out


class cross_modal(nn.Module):   # existing error
    def __init__(self, in_channels, out_channels):
        super(cross_modal, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.basic1 = basicBlock_dual(in_channels=self.in_channels, out_channels=1)
        self.basic2 = basicBlock_dual(in_channels=self.in_channels, out_channels=1)
        self.basic3 = basicBlock_dual(in_channels=self.in_channels, out_channels=1)
        self.conv1x1=nn.Conv2d(in_channels=2*self.in_channels,out_channels=self.out_channels,kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x11 = self.basic1(x1)
        x1_attn1 = self.sigmoid(x11)
        x1_intra = torch.mul(x1_attn1, x1)
        x21 = self.basic2(x2)
        x2_attn1 = self.sigmoid(x21)
        x2_intra = torch.mul(x2_attn1, x2)

        x3=x1+x2
        x31=self.basic3(x3)
        x3_attn1 = self.sigmoid(x31)

        x12 = torch.mul(x3_attn1, x1_intra)
        x1_out=x1+x12
        x22 = torch.mul(x3_attn1, x2_intra)
        x2_out = x2 + x22

        x_c = torch.cat((x1_out, x2_out), dim=1)
        out=self.conv1x1(x_c)
        return out



class FusionNet(nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()
        self.deep_net = dual_MixFormer(embed_dim=8, depths=[2, 2, 4, 8, 8], num_heads=[4, 8, 16, 16, 32], drop_rate=0.2)
        self.decoder = decoder()
        self.FFM1 = cross_modal(16, 16)
        self.FFM2 = cross_modal(32, 32)
        self.FFM3 = cross_modal(64, 64)
        self.FFM4 = cross_modal(128, 128)
        self.FFM5 = cross_modal(256, 256)
    def forward(self, x1, x2):
        x1_deep, x2_deep = self.deep_net(x1, x2)

        x_ffm1 = self.FFM1(x1_deep[0], x2_deep[0])

        x_ffm2 = self.FFM2(x1_deep[1], x2_deep[1])
        x_ffm3 = self.FFM3(x1_deep[2], x2_deep[2])
        x_ffm4 = self.FFM4(x1_deep[3], x2_deep[3])
        x_ffm5 = self.FFM5(x1_deep[4], x2_deep[4])
        out = self.decoder(x1, x_ffm1, x_ffm2, x_ffm3, x_ffm4, x_ffm5)

        return out
#

if __name__ == '__main__':

    c = FusionNet()
    c.eval()

    print('# generator parameters:', sum(param.numel() for param in c.parameters()))
    a = torch.rand((1, 1, 256, 256)).cuda()
    b = torch.rand((1, 1, 256, 256)).cuda()
    c=c.cuda()
    result=c(a,b)
    print(result.shape)


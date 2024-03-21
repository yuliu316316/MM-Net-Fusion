import numpy as np
import torch
from torch import nn
from torch.nn import init


class EMSA(nn.Module):

    def __init__(self, d_model, d_k, d_v, h, dropout=.1, H=7, W=7, ratio=3, apply_transform=True):

        super(EMSA, self).__init__()
        self.H = H
        self.W = W
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

        self.ratio = ratio
        if (self.ratio > 1):
            self.sr = nn.Sequential()
            self.sr_conv = nn.Conv2d(d_model, d_model, kernel_size=ratio + 1, stride=ratio, padding=ratio // 2,
                                     groups=d_model)
            self.sr_ln = nn.LayerNorm(d_model)

        self.apply_transform = apply_transform and h > 1
        if (self.apply_transform):
            self.transform = nn.Sequential()
            self.transform.add_module('conv', nn.Conv2d(h, h, kernel_size=1, stride=1))
            self.transform.add_module('softmax', nn.Softmax(-1))
            self.transform.add_module('in', nn.InstanceNorm2d(h))

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

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

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):

        b_s, nq, c = queries.shape
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)

        if (self.ratio > 1):
            x = queries.permute(0, 2, 1).view(b_s, c, self.H, self.W)  # bs,c,H,W
            x = self.sr_conv(x)  # bs,c,h,w
            x = x.contiguous().view(b_s, c, -1).permute(0, 2, 1)  # bs,n',c
            x = self.sr_ln(x)
            k = self.fc_k(x).view(b_s, -1, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, n')
            v = self.fc_v(x).view(b_s, -1, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, n', d_v)
        else:
            k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
            v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        if (self.apply_transform):
            att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, n')
            att = self.transform(att)  # (b_s, h, nq, n')
        else:
            att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, n')
            att = torch.softmax(att, -1)  # (b_s, h, nq, n')

        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)

        att = self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out


# if __name__ == '__main__':
#     input = torch.randn(50, 64, 512)
#     emsa = EMSA(d_model=512, d_k=512, d_v=512, h=8, H=8, W=8, ratio=2, apply_transform=True)
#     output = emsa(input, input, input)
#     print(output.shape)

import torch
from torch import nn
from torch.nn import functional as F


class Attention(nn.Module):
    def __init__(self, in_planes, ratio, K, temprature=30, init_weight=True):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.temprature = temprature
        assert in_planes > ratio
        hidden_planes = in_planes // ratio
        self.net = nn.Sequential(
            nn.Conv2d(in_planes, hidden_planes, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden_planes, K, kernel_size=1, bias=False)
        )

        if (init_weight):
            self._initialize_weights()

    def update_temprature(self):
        if (self.temprature > 1):
            self.temprature -= 1

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        att = self.avgpool(x)  # bs,dim,1,1
        att = self.net(att).view(x.shape[0], -1)  # bs,K
        return F.softmax(att / self.temprature, -1)


class DynamicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0, dilation=1, grounps=1, bias=True, K=4,
                 temprature=30, ratio=4, init_weight=True):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = grounps
        self.bias = bias
        self.K = K
        self.init_weight = init_weight
        self.attention = Attention(in_planes=in_planes, ratio=ratio, K=K, temprature=temprature,
                                   init_weight=init_weight)

        self.weight = nn.Parameter(torch.randn(K, out_planes, in_planes // grounps, kernel_size, kernel_size),
                                   requires_grad=True)
        if (bias):
            self.bias = nn.Parameter(torch.randn(K, out_planes), requires_grad=True)
        else:
            self.bias = None

        if (self.init_weight):
            self._initialize_weights()

        # TODO 初始化

    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])

    def forward(self, x):
        bs, in_planels, h, w = x.shape
        softmax_att = self.attention(x)  # bs,K
        x = x.view(1, -1, h, w)
        weight = self.weight.view(self.K, -1)  # K,-1
        aggregate_weight = torch.mm(softmax_att, weight).view(bs * self.out_planes, self.in_planes // self.groups,
                                                              self.kernel_size, self.kernel_size)  # bs*out_p,in_p,k,k

        if (self.bias is not None):
            bias = self.bias.view(self.K, -1)  # K,out_p
            aggregate_bias = torch.mm(softmax_att, bias).view(-1)  # bs,out_p
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              groups=self.groups * bs, dilation=self.dilation)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              groups=self.groups * bs, dilation=self.dilation)

        output = output.view(bs, self.out_planes, h, w)

        return output


# if __name__ == '__main__':
#     input = torch.randn(2, 32, 64, 64)
#     m = DynamicConv(in_planes=32, out_planes=64, kernel_size=3, stride=1, padding=1, bias=False)
#     out = m(input)
#     print(out)


# if __name__ == '__main__':
#     input = torch.randn(1,32, 64, 64, 64)
#     m = DynamicConv(in_planes=32, out_planes=32, kernel_size=3, stride=1, padding=1, bias=True)
#     print('# generator parameters:', sum(param.numel() for param in m.parameters()))
#     out = m(input)
#     print(out.shape)

import numpy as np
import torch
from torch import nn
from torch.nn import init
from collections import OrderedDict


class ECAAttention(nn.Module):

    def __init__(self, kernel_size=3, pool=None):
        super().__init__()
        # self.gap = nn.AdaptiveAvgPool2d(1)
        self.gap = pool
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.gap(x)  # bs,c,1,1
        y = y.squeeze(-1).squeeze(-1).permute(0, 2, 1)  # bs,1,c
        y = self.conv(y)  # bs,1,c
        y = self.sigmoid(y)  # bs,1,c
        y = y.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)  # bs,c,1,1
        return x * y.expand_as(x)


class ECAAttention_gai(nn.Module):

    def __init__(self, channels, kernel_size=3, pool=None):
        super().__init__()
        # self.gap = nn.AdaptiveAvgPool2d(1)
        self.gap = pool
        self.conv0 = nn.Conv2d(in_channels= channels, out_channels=channels, kernel_size=3, stride=1, padding=1,
                               bias=True)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, stride=1, padding=0,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, stride=1, padding=0,
                               bias=True)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        y = self.conv0(x)
        y = self.gap(y)  # bs,c,1,1
        y = y.squeeze(-1).squeeze(-1).permute(0, 2, 1)  # bs,1,c
        y = self.conv(y)  # bs,1,c

        # y = self.sigmoid(y)  # bs,1,c
        y = y.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)  # bs,c,1,1
        y1 = self.conv1(y)
        y2 = self.conv2(y)

        attn = self.softmax(torch.cat((y1.unsqueeze(1), y2.unsqueeze(1)), dim=1))
        # out1=x[:,0,:,:,:,:]*attn[:,0,:,:,:,:]
        # out2=x[:,1,:,:,:]*attn[:,1,:,:,:]

        # return x * y.expand_as(x)
        return attn


class ECAAttention_space_channel(nn.Module):

    def __init__(self, channels, kernel_size=3, pool=None):
        super().__init__()
        # self.gap = nn.AdaptiveAvgPool2d(1)
        self.gap = pool
        self.conv0 = nn.Conv2d(in_channels= 2*channels, out_channels=channels, kernel_size=3, stride=1, padding=1,
                               bias=True)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, stride=1, padding=0,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, stride=1, padding=0,
                               bias=True)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        y = self.conv0(x)
        y = self.gap(y)  # bs,c,1,1
        y = y.squeeze(-1).permute(0, 2, 1)  # bs,1,c
        y = self.conv(y)  # bs,1,c

        # y = self.sigmoid(y)  # bs,1,c
        y = y.permute(0, 2, 1).unsqueeze(-1)  # bs,c,1,1
        y1 = self.conv1(y)
        y2 = self.conv2(y)

        attn = self.softmax(torch.cat((y1.unsqueeze(1), y2.unsqueeze(1)), dim=1))
        # out1=x[:,0,:,:,:,:]*attn[:,0,:,:,:,:]
        # out2=x[:,1,:,:,:]*attn[:,1,:,:,:]

        # return x * y.expand_as(x)
        return attn
# if __name__ == '__main__':
#     input = torch.randn(1,32, 64, 64, 64)
#     eca = ECAAttention(kernel_size=3)
#     print('# generator parameters:', sum(param.numel() for param in eca.parameters()))
#     output = eca(input)
#     print(output.shape)


class dual_ECA(nn.Module):
    def __init__(self):
        super(dual_ECA, self).__init__()
        self.pool1 = ECAAttention(pool=nn.AdaptiveAvgPool2d(1))
        self.pool2 = ECAAttention(pool=nn.AdaptiveMaxPool2d(1))

    def forward(self, x):
        x1 = self.pool1(x)
        x2 = self.pool2(x)
        out = x1 + x2
        return out


# if __name__ == '__main__':
#     input = torch.randn(1,32, 64, 64, 64)
#     # input2 = torch.randn(1, 32, 64, 64, 64)
#
#     # eca = ECAAttention(kernel_size=3)
#     net=dual_ECA()
#     print('# generator parameters:', sum(param.numel() for param in net.parameters()))
#     output = net(input)
#     print(output.shape)

# %%
import numpy as np
import torch
from torch import flatten, nn
from torch.nn import init
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn import functional as F


class CoTAttention(nn.Module):

    def __init__(self, dim=32, kernel_size=3):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size

        self.key_embed = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=4, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )
        self.value_embed = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim)
        )

        factor = 4
        self.attention_embed = nn.Sequential(
            nn.Conv2d(2 * dim, 2 * dim // factor, 1, bias=False),
            nn.BatchNorm2d(2 * dim // factor),
            nn.ReLU(),
            nn.Conv2d(2 * dim // factor, kernel_size * kernel_size * dim, 1)
        )

    def forward(self, x):
        bs, c, h, w = x.shape
        k1 = self.key_embed(x)  # bs,c,h,w
        v = self.value_embed(x).view(bs, c, -1)  # bs,c,h,w

        y = torch.cat([k1, x], dim=1)  # bs,2c,h,w
        att = self.attention_embed(y)  # bs,c*k*k,h,w
        att = att.reshape(bs, c, self.kernel_size * self.kernel_size, h, w)
        att = att.mean(2, keepdim=False).view(bs, c, -1)  # bs,c,h*w
        k2 = F.softmax(att, dim=-1) * v
        k2 = k2.view(bs, c, h, w)

        return k1 + k2


class ResidualBlock(nn.Module):
    def __init__(self,
                 in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )

        # init_weights(self.modules)

    def forward(self, x):
        out = self.body(x)
        out = F.relu(out + x)
        return out


class cnn_attn1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(cnn_attn1, self).__init__()
        self.conv1 = ResidualBlock(in_channels=in_channels, out_channels=out_channels)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=3, stride=1, padding=1)
        # self.conv4 = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)
        # self.conv5 = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)
        self.eca = ECAAttention_space_channel(channels=in_channels, pool=nn.AdaptiveAvgPool2d(1))
        self.softmax = nn.Softmax(dim=1)
        self.conv = nn.Conv2d(in_channels=2 * in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x1, x2):
        # feas = torch.cat([x1, x2], dim=1)
        x_sum = x1 + x2
        x_conv = self.conv1(x_sum)
        x_split1 = self.conv2(x_conv)
        x_split2 = self.conv3(x_conv)
        attention_spacial = self.softmax(torch.cat((x_split1, x_split2), dim=1))
        # print(attention_spacial[:,0,:,:,:].shape)
        # print(x1.shape)
        x_spacial1 = x1 * attention_spacial[:, 0, :, :].unsqueeze(1)
        x_spacial2 = x2 * attention_spacial[:, 1, :, :].unsqueeze(1)

        x_cat = torch.cat((x_spacial1, x_spacial2), dim=1)

        attention_channel = self.eca(x_cat)

        # x_split3=self.conv4(x_eca)
        # x_split4=self.conv5(x_eca)
        #
        # attention_channel=self.softmax(torch.cat((x_split3,x_split4)))

        x_out1 = x_spacial1 * attention_channel[:, 0, :, :, :]
        x_out2 = x_spacial2 * attention_channel[:, 1, :, :, :]
        x = torch.cat((x_out1, x_out2), dim=1)
        out = self.conv(x)
        # x_out=x_out1+x_out2

        # y_in=self.conv4(x_spacial)

        # y_in1=x_spacial[:,]
        # y_in2=x_spacial[:]

        return out

class cnn_attn2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(cnn_attn2, self).__init__()
        self.conv1 = ResidualBlock(in_channels=in_channels, out_channels=out_channels)
        self.conv2 = nn.Conv2d(in_channels=2*in_channels, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=2*in_channels, out_channels=1, kernel_size=3, stride=1, padding=1)
        # self.conv4 = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)
        # self.conv5 = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)
        self.eca = ECAAttention_gai(channels=in_channels, pool=nn.AdaptiveAvgPool2d(1))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2):
        # feas = torch.cat([x1, x2], dim=1)
        x_sum = x1 + x2
        x_conv = self.conv1(x_sum)
        # x_split1 = self.conv2(x_conv)
        # x_split2 = self.conv3(x_conv)


        attention_channel = self.eca(x_conv)

        # x_split3=self.conv4(x_eca)
        # x_split4=self.conv5(x_eca)
        #
        # attention_channel=self.softmax(torch.cat((x_split3,x_split4)))

        x_out1 = x1 * attention_channel[:, 0, :, :, :, :]
        x_out2 = x2 * attention_channel[:, 1, :, :, :, :]
        x_cat = torch.cat((x_out1, x_out2), dim=1)
        x_split1 = self.conv2(x_cat)
        x_split2 = self.conv3(x_cat)
        # x_out=x_out1+x_out2

        # y_in=self.conv4(x_spacial)

        # y_in1=x_spacial[:,]
        # y_in2=x_spacial[:]
        attention_spacial = self.softmax(torch.cat((x_split1, x_split2), dim=1))
        # print(attention_spacial[:,0,:,:,:].shape)
        # print(x1.shape)
        x_spacial1 = x1 * attention_spacial[:, 0, :, :, :].unsqueeze(1)
        x_spacial2 = x2 * attention_spacial[:, 1, :, :, :].unsqueeze(1)
        x_out1=x1+x_spacial1
        x_out2=x2+x_spacial2


        return x_out1,x_out2




class PSA(nn.Module):

    def __init__(self, channel=512, reduction=4, S=4):
        super().__init__()
        self.S = S

        self.convs = []
        for i in range(S):
            # self.convs.append(nn.Conv2d(channel // S, channel // S, kernel_size=2 * (i + 1) + 1, padding=i + 1))
            self.convs.append(nn.Conv2d(channel // S, channel // S, kernel_size=2 * i + 1, padding=i ))

        self.se_blocks = []
        for i in range(S):
            self.se_blocks.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channel // S, channel // (S * reduction), kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // (S * reduction), channel // S, kernel_size=1, bias=False),
                nn.Sigmoid()
            ))

        self.softmax = nn.Softmax(dim=1)

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
        b, c, h, w = x.size()

        # Step1:SPC module
        SPC_out = x.view(b, self.S, c // self.S, h, w)  # bs,s,ci,h,w
        for idx, conv in enumerate(self.convs):
            SPC_out[:, idx, :, :, :] = conv(SPC_out[:, idx, :, :, :])

        # Step2:SE weight
        se_out = []
        for idx, se in enumerate(self.se_blocks):
            se_out.append(se(SPC_out[:, idx, :, :, :]))
        SE_out = torch.stack(se_out, dim=1)
        SE_out = SE_out.expand_as(SPC_out)

        # Step3:Softmax
        softmax_out = self.softmax(SE_out)

        # Step4:SPA
        PSA_out = SPC_out * softmax_out
        PSA_out = PSA_out.view(b, -1, h, w)

        return PSA_out

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd


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
        self.attention = Attention_od(in_planes, out_planes, kernel_size, groups=groups,
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

if __name__ == '__main__':
    a=ODConv2d(in_planes=2,out_planes=2,kernel_size=3)
    b = torch.rand((2, 2, 128, 128))
    result=a(b)
    print(result.shape)


if __name__ == '__main__':
    input = torch.randn(2, 8, 128, 128)
    psa = PSA(channel=8, reduction=1)
    output = psa(input)
    a = output.view(-1).sum()
    # a.backward()
    print(output.shape)

# if __name__ == '__main__':
#     # input1 = torch.randn(1,  32, 32, 32)
#     # input2 = torch.randn(1,32,  32, 32)
#     # cot = cnn_attn1(32, 32)
#     # output = cot(input1, input2)
#     # print(output[1].shape)
#     input = torch.randn(2, 32, 64, 64)
#     m = DynamicConv(in_planes=32, out_planes=64, kernel_size=3, stride=1, padding=1, bias=False)
#     out = m(input)
#     print(out.shape)

    # print(output)

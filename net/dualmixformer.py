import numpy as np
import torch
import torch.nn as nn
from net.mixformer import ConvMerging, BasicLayer, ConvEmbed


class dual_MixFormer(nn.Module):
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
                 img_size=(128, 128),
                 patch_size=4,
                 in_chans=1,
                 class_num=1000,
                 embed_dim=8,
                 depths=[2, 2, 8, 8],
                 num_heads=[4, 8, 16, 32],
                 window_size=(7, 7),
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
        # self.num_classes = num_classes = class_num
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

        self.patch_embed2 = ConvEmbed(
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
        self.pos_drop2 = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = np.linspace(0, drop_path_rate,
                          sum(depths)).tolist()  # stochastic depth decay rule

        # build layers
        self.layers1 = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer1 = BasicLayer(
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
            # if (i_layer < self.num_layers - 1) else 0)
            self.layers1.append(layer1)

        self.layers2 = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer2 = BasicLayer(
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
            # if (i_layer < self.num_layers - 1) else 0)
            self.layers2.append(layer2)

        # self.norm = norm_layer(self.num_features)
        # self.last_proj = nn.Linear(self.num_features, 1280)
        # self.activate = nn.GELU()
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

    def forward_features(self, x1, x2):
        x1 = self.patch_embed(x1)
        _, _, Wh1, Ww1 = x1.shape
        x1 = x1.flatten(2).transpose(1, 2)
        if self.ape:
            x1 = x1 + self.absolute_pos_embed
        x1 = self.pos_drop(x1)
        x1_out = []

        x2 = self.patch_embed2(x2)
        _, _, Wh2, Ww2 = x2.shape
        x2 = x2.flatten(2).transpose(1, 2)
        if self.ape:
            x2 = x2 + self.absolute_pos_embed
        x2 = self.pos_drop2(x2)
        x2_out = []

        for i in range(self.num_layers):
            b1, H1, W1, x1, Wh1, Ww1 = self.layers1[i](x1, Wh1, Ww1)
            # x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
            x1_o = x1.reshape(b1, -1, Wh1, Ww1)

            x1_out.append(x1_o)

            b2, H2, W2, x2, Wh2, Ww2 = self.layers2[i](x2, Wh2, Ww2)
            # x.view(B, H // window_size[0], window_size[0], W // window_size[2], window_size[2], C)
            x2_o = x2.reshape(b2, -1, Wh2, Ww2)

            x2_out.append(x2_o)

        return x1_out, x2_out


    def forward(self, x1, x2):
        x1_out, x2_out = self.forward_features(x1, x2)
        return x1_out, x2_out

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


if __name__ == '__main__':

    model = dual_MixFormer(embed_dim=8, depths=[2, 2, 4, 8, 8], num_heads=[4, 8, 16, 16, 32], drop_rate=0.2)
    model.eval()

    print('# generator parameters:', sum(param.numel() for param in model.parameters()))
    x1 = torch.rand((1, 1, 256, 256))
    x2 = torch.rand((1, 1, 256, 256))

    result1, result2 = model(x1, x2)

    print(result1[0].shape)
    print(result1[1].shape)
    print(result1[2].shape)
    print(result1[3].shape)
    print(result1[4].shape)

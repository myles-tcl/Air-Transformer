import torch
import torch.nn as nn


def conv_1xnxn(inp, oup, kernel_size=3, stride=3, groups=1):
    return nn.Conv3d(inp, oup, (1, kernel_size, kernel_size), (1, stride, stride), (0, 0, 0), groups=groups)


def conv_2xnxn_std(inp, oup, kernel_size=3, stride=3, groups=1):
    return nn.Conv3d(inp, oup, (2, kernel_size, kernel_size), (2, stride, stride), (0, 0, 0), groups=groups)


def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding from H,W to H/2, W/2
    if std == True： time from T to T-2
    """

    def __init__(self, img_size=(8, 8), patch_size=(2, 2), in_chans=100, embed_dim=400, std=False):
        super().__init__()
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size[0]
        self.patch_size = patch_size[0]
        self.num_patches = num_patches
        self.norm = nn.LayerNorm(embed_dim)
        if std:
            self.proj = conv_2xnxn_std(in_chans, embed_dim, kernel_size=patch_size[0], stride=1)
        else:
            self.proj = conv_1xnxn(in_chans, embed_dim, kernel_size=patch_size[0], stride=1)

    def forward(self, x):
        B, C, T, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        B, C, T, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.reshape(B, T, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        return x


class Attention(nn.Module):

    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., drop=0.):
        """multi-head attention

        Args:
            dim (_type_): 输入数据通道数
            num_heads (_type_): 头数
            qkv_bias (bool, optional): qkv偏执. Defaults to False.
            qk_scale (float, optional): qkv缩放因子. Defaults to None.
            attn_drop (float, optional): attention丢弃率. Defaults to 0..
            drop (float_, optional): mlp丢弃率. Defaults to 0..
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)
        self.softmax = nn.Softmax(dim=-1)

    # @get_local('attn_map')           # 装饰器，简洁高效获取attention map
    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.softmax(attn)
        attn_map = self.attn_drop(attn)

        x = (attn_map @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class STVBlock(nn.Module):

    def __init__(self, dim=100, num_heads=2, num_heads_chans=1, frame=4, mlp_ratio=4., drop=0., attn_drop=0., drop_path=0.,
                 qkv_bias=False, qk_scale=None, act_layer=nn.GELU, attention_type='devided_space_time', channel_attention=True, num_patches=64):
        """Spatial-temporal and variable self-attention

        Args:
            dim (int, optional): 输入数据通道数. Defaults to 100.
            num_heads (int, optional): self attention 头数. Defaults to 2.
            num_heads_chans: int, optional): channels attention 头数
            frame (int, optional): 输入数据的时间帧数. Defaults to 4.
            mlp_ratio (_type_, optional): mlp扩增倍数. Defaults to 4..
            drop (_type_, optional): mlp drop率. Defaults to 0..
            attn_drop (_type_, optional): self attention drop率. Defaults to 0..
            drop_path (_type_, optional): Droppath drop率. Defaults to 0..
            qkv_bias (bool, optional): qkv偏偏置. Defaults to False.
            qk_scale (_type_, optional): qkv缩放因子. Defaults to None.
            act_layer (_type_, optional): 激活层函数. Defaults to nn.GELU.
            attention_type (str, optional): 时空自注意力类型. Defaults to 'devided_space_time'.
            num_patches (int, optional): 单一时间frame中空间上像素点的个数, H*W. Defaults to 64.
        """
        super().__init__()
        self.attention_type = attention_type
        self.channel_attention = channel_attention
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, drop=drop)

        # temporal attention parameters
        if self.attention_type == 'devided_space_time':
            self.temporal_norm = nn.LayerNorm(dim)
            self.temporal_attn = Attention(dim, num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, drop=drop)
            self.temporal_fc = nn.Linear(dim, dim)

        # drop_path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp1 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if channel_attention:
            # channels attention parameters
            chans_dim = int(num_patches * frame)
            self.norm3 = nn.LayerNorm(chans_dim)
            self.chans_attn = Attention(chans_dim, num_heads_chans, qkv_bias=False, qk_scale=None, attn_drop=attn_drop, drop=drop)
            chans_mlp_hidden_dim = int(chans_dim * mlp_ratio)
            self.mlp2 = Mlp(in_features=chans_dim, hidden_features=chans_mlp_hidden_dim, act_layer=act_layer, drop=drop)
            self.norm4 = nn.LayerNorm(chans_dim)
        self.frame = frame

    def forward(self, x, x_pos_embed):
        B, C, T, H, W = x.shape

        x_pos_embed = x_pos_embed.reshape(B, 1, 1, H, W).repeat(1, C, T, 1, 1)
        x = x + x_pos_embed[:, :, -self.frame:, ::]

        if self.attention_type != 'devided_space_time':
            x = x.flatten(2).transpose(1, 2)  # embedding
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp1(self.norm2(x)))
            x = x.transpose(1, 2)

        if self.attention_type == 'devided_space_time':
            # temporal attention
            attn = x.view(B, C, T, H * W).permute(0, 3, 2, 1).contiguous()
            attn = attn.view(B * H * W, T, C)
            attn = attn + self.drop_path(self.temporal_attn(self.temporal_norm(attn)))
            # spatial attention
            attn = attn.view(B, H * W, T, C).permute(0, 2, 1, 3).contiguous()
            attn = attn.view(B * T, H * W, C)
            residual = x.view(B, C, T, H * W).permute(0, 2, 3, 1).contiguous()
            residual = residual.view(B * T, H * W, C)
            attn = residual + self.drop_path(self.attn(self.norm1(attn)))
            attn = attn.view(B, T * H * W, C)
            x = attn + self.drop_path(self.mlp1(self.norm2(attn)))
            x = x.transpose(1, 2)

        if self.channel_attention:
            # channels attention
            x = x + self.drop_path(self.chans_attn(self.norm3(x)))
            x = x + self.drop_path(self.mlp2(self.norm4(x)))
        x = x.reshape(B, C, T, H, W)
        return x


class STVBlocklayer(nn.Module):

    def __init__(self,
                 image_size=(8, 8),
                 in_chans=100,
                 attn_depth=2,
                 num_heads=2,
                 num_heads_chans=1,
                 frame=4,
                 mlp_ratio=4.,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 attention_type='devided_space_time',
                 channel_attention=True,
                 num_patches=64,
                 patchembed=False,
                 embed_dim=400,
                 std=False,
                 patch_size=(2, 2),
                 norm_layer=nn.LayerNorm([8, 5, 5])):
        super().__init__()

        # attn_block: 全局特征进行self-attention计算
        self.attn_block = nn.ModuleList([
            STVBlock(dim=embed_dim,
                     num_heads=num_heads,
                     num_heads_chans=num_heads_chans,
                     frame=frame,
                     mlp_ratio=mlp_ratio,
                     drop=drop,
                     attn_drop=attn_drop,
                     drop_path=drop_path,
                     attention_type=attention_type,
                     channel_attention=channel_attention,
                     num_patches=num_patches) for i in range(attn_depth)
        ])

        self.patchembed = patchembed
        if patchembed:
            # patch embedding: 用于第二阶段缩小图像尺寸
            self.patch_embed = PatchEmbed(img_size=image_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, std=std)

    def forward(self, x, x_pos_embed):
        if self.patchembed:
            x = self.patch_embed(x)

        for blk in self.attn_block:
            x = blk(x, x_pos_embed)

        return x


class Attention_decoder(nn.Module):

    def __init__(self, dim, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0., drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

    def forward(self, xq, xkv):
        B_q, N_q, C_q = xq.shape
        B_kv, N_kv, C_kv = xkv.shape
        q = self.q(xq).reshape(B_q, N_q, self.num_heads, C_q // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(xkv).reshape(B_kv, N_kv, 2, self.num_heads, C_kv // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn_map = self.attn_drop(attn)

        x = (attn_map @ v).transpose(1, 2).reshape(B_q, N_q, C_q)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Decoder(nn.Module):

    def __init__(self, inchans=88, dim=100, num_heads=2, mlp_ratio=4., drop=0., attn_drop=0., drop_path=0., qkv_bias=False, qk_scale=None, act_layer=nn.GELU, embedding=True):
        super().__init__()
        self.embedding = embedding
        if embedding:
            self.dim = dim
            self.embedding = nn.Linear(inchans, dim)
        # spatial attention parameters
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.attn = Attention_decoder(dim, num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, drop=drop)

        # drop_path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm4 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x_enc, x_dec):

        if self.embedding:
            # embedding
            B, N, C = x_dec.shape
            x_dec = self.norm1(self.embedding(x_dec))

        # attention
        x_enc.shape, x_dec.shape
        x = x_dec + self.drop_path(self.attn(self.norm3(x_dec), self.norm2(x_enc)))
        x = x + self.drop_path(self.mlp(self.norm4(x)))

        return x


class Air_Transformer(nn.Module):

    def __init__(self,
                 image_size=(8, 8),
                 patch_size=(2, 2),
                 in_chans=100,
                 std=[False, False],
                 c_depth=[1, 1],
                 attn_depth=[4, 2],
                 num_heads=[2, 4],
                 num_heads_chans=[2, 2],
                 frame=(4, 4),
                 mlp_ratio=4.,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 num_regress=1,
                 patchembed=[True, False],
                 embed_dim=[256, 512],
                 channel_attention=[True, False],
                 decoder_depth=3):
        """
        :param image_size:  输入数据图像大小
        :param patch_size:  若STVBlocklayer层数大于一时，控制patch—embed下采样大小
        :param in_chans:    输入数据的特征维度
        :param std:         若STVBlocklayer层数大于一时，控制patch—embed时是否对时间维度进行下采样
        :param c_depth:     cblock的层数, 列表格式, 列表中的数值大小表示STVBlocklayer层数，列表中元素个数n表示通过n-1次patch_embed层
        :param attn_depth:  attn_block的层数
        :param num_heads:   attn_block中时空自注意力的头数，通道自注意力的头数默认为时间帧数
        :param frame:       输入数据的时间帧数
        :param mlp_ratio:   mlp层中隐藏层的扩增倍数
        :param drop:        mlp中的drop_rate
        :param attn_drop:   自注意力的drop_rate
        :param drop_path:   DropPath的drop_rate
        :param num_regress: 回归头数
        """
        super().__init__()
        self.wid_s = image_size[0]
        self.pos_drop = nn.Dropout(drop)
        self.depth = len(attn_depth)
        self.std = std
        self.patchembed = patchembed
        self.patch_size = patch_size

        # STVBlocklayer
        self.encoder = nn.ModuleDict([
            [
                'block' + str(i),
                STVBlocklayer(
                    image_size=[i / (patch_size[0] * i) for i in image_size],
                    in_chans=in_chans if i == 0 else embed_dim[i - 1],
                    embed_dim=embed_dim[i],
                    attn_depth=attn_depth[i],
                    num_heads=num_heads[i],
                    num_heads_chans=num_heads_chans[i],
                    frame=frame[i],
                    mlp_ratio=mlp_ratio,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path,
                    attention_type='devided_space_time' if image_size[0] >= 16 else None,
                    num_patches=int((image_size[0] - i * (patch_size[0] - 1))**2),
                    std=std[i],
                    patch_size=patch_size,
                    patchembed=patchembed[i],
                    channel_attention=channel_attention[i],
                    # norm_layer = nn.LayerNorm([frame, int(np.ceil(image_size[0]/(2**i))), int(np.ceil(image_size[0]/(2**i)))])
                    # if std == False else nn.LayerNorm([frame - 2*i, int(np.ceil(image_size[0]/(2**i))), int(np.ceil(image_size[0]/(2**i)))])
                )
            ] for i in range(len(attn_depth))
        ])

        self.decoder = nn.ModuleDict([[
            'block' + str(i),
            Decoder(inchans=in_chans if i == 0 else embed_dim[i - 1],
                    dim=embed_dim[i],
                    num_heads=num_heads[i],
                    mlp_ratio=mlp_ratio,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path,
                    qkv_bias=False,
                    qk_scale=None,
                    act_layer=nn.GELU,
                    embedding=False if i == 0 else True)
        ] for i in range(len(attn_depth))])

        self.norm = nn.LayerNorm(embed_dim[-1])
        self.num_regress = num_regress
        # self.embedding = nn.Linear(embed_dim[-1], self.num_regress * embed_dim[-1])
        # self.avgpool = nn.AdaptiveAvgPool1d(1)
        # for i in range(self.num_regress):
        #     setattr(self, 'task_{}'.format(i), nn.Linear(embed_dim[-1], 1))
        self.head = nn.Linear(embed_dim[-1], self.num_regress)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, inputs):
        B, C, T, H, W = inputs.shape
        # postion embedding
        x_pos_embed = inputs[:, 0, -1, ::] + inputs[:, 1, -1, ::]
        inputs = inputs[:, 2:, ::]

        # encoder-decoder struction
        x = self.pos_drop(inputs)
        x_dec = inputs[:, :, -1, self.wid_s // 2, self.wid_s // 2].unsqueeze(1)

        for i in range(self.depth):
            # 卷积后空间维度收缩，缩放位置编码
            if self.patchembed[i]:
                B_p, H_p, W_p = x_pos_embed.shape
                x_pos_embed = torch.nn.functional.unfold(x_pos_embed.reshape(B_p, 1, H_p, W_p), self.patch_size)
                x_pos_embed = x_pos_embed.mean(dim=1).reshape(B_p, H_p + 1 - self.patch_size[0], W_p + 1 - self.patch_size[0])

            x = self.encoder['block' + str(i)](x, x_pos_embed)
            x_enc = x.flatten(2).transpose(1, 2)
            x_dec = self.decoder['block' + str(i)](x_enc, x_dec)
        
        # x_dec = x.flatten(2).sum(2).unsqueeze(1)
        x_dec = self.norm(x_dec)
        # x = self.embedding(x_dec).reshape(B, self.num_regress, -1)
        x = self.head(x_dec).flatten(1)


        return x


def AiT(frame=(8, 4, 2, 1), **kwargs):
    model = Air_Transformer(image_size=(5, 5),
                            in_chans=51,
                            attn_depth=[1, 3, 2, 1],
                            patch_size=(2, 2),
                            embed_dim=[51, 64, 96, 128],
                            patchembed=[False, True, True, True],
                            std=[False, True, True, True],
                            num_heads=[1, 1, 1, 1],
                            num_heads_chans=frame,
                            frame=frame,
                            channel_attention=[True, True, True, True],
                            drop=0.05,
                            attn_drop=0.05,
                            drop_path=0.05,
                            num_regress=3)
    return model

if __name__ == '__main__':
    # data = torch.load('/data/taochenliang/research-3/data/Training_M/5*0_0.01/Feature.pt')
    # x = torch.FloatTensor(data[:512])
    x = torch.randn((32, 53, 8, 5, 5))
    # print(x.shape)
    model = AiT()
    output = model(x)
    print(output.shape, output)

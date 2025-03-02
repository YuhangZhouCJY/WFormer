import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange
import torch
import numpy as np

from transformer_watermarking.blocks import ConvBNRelu

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))  # 将不可训练的的类型转换为可训练的类型
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)  # 两个1×1卷积看成了一个
        x1, x2 = self.dwconv(x).chunk(2, dim=1)  # 第 1 个维度方向切分成 2 块
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class Attention1(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention1, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, message):
        b, c, h, w = x.shape
        bm, cm, hm, wm = message.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        qkv_m = self.qkv_dwconv(self.qkv(message))
        q_m, k_m, v_m = qkv_m.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q_m = rearrange(q_m, 'bm (head cm) hm wm -> bm head cm (hm wm)', head=self.num_heads)

        # q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        q_m = torch.nn.functional.normalize(q_m, dim=-1)

        attn = (q_m @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)  # b c h w->b hw c->b c h w
        self.attn = Attention(dim, num_heads, bias)  # b c h  w->b head c//head hw ->b c h w
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


class TransformerBlock1(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock1, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)  # b c h w->b hw c->b c h w
        self.attn1 = Attention1(dim, num_heads, bias)  # b c h  w->b head c//head hw ->b c h w
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x, message):
        x = x + self.attn1(self.norm1(x), self.norm1(message))
        x = x + self.ffn(self.norm2(x))

        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=64, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


##########################################################################
## Resizing modules



class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 4, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

class Message_pre(nn.Module):
    def __init__(self, H, W, message_length, blocks=4, channels=64,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):
        super(Message_pre, self).__init__()
        self.H = H
        self.W = W
        message_convT_blocks = int(np.log2(H // int(np.sqrt(message_length))))
        message_se_blocks = max(blocks - message_convT_blocks, 1)

        self.message_pre_layer = nn.Sequential(
            ConvBNRelu(1, channels),
            # ExpandNet(channels, channels, blocks=message_convT_blocks),
            Upsample(channels),
            TransformerBlock(dim=channels, num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type),
            Upsample(channels),
            TransformerBlock(dim=channels, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type),
            Upsample(channels),
            TransformerBlock(dim=channels, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type),
            Upsample(channels),
            TransformerBlock(dim=channels, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type),
            TransformerBlock(dim=channels, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        )
        self.message_first_layer = TransformerBlock(dim=channels, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)

    def forward(self, message):
        size = int(np.sqrt(message.shape[1]))
        message_image = message.view(-1, 1, size, size)  # 第一个维度变为一维向量，其余的不变
        message_pre = self.message_pre_layer(message_image)
        intermediate2 = self.message_first_layer(message_pre)

        return intermediate2

class Embedding(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Embedding, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.liner = nn.Linear(dim * 2, dim)

    def forward(self, x, message):
        b, c, h, w = x.shape
        bm, cm, hm, wm = message.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        qkv_m = self.qkv_dwconv(self.qkv(message))
        q_m, k_m, v_m = qkv_m.chunk(3, dim=1)



        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q_m = rearrange(q_m, 'bm (head cm) hm wm -> bm head cm (hm wm)', head=self.num_heads)
        k_m = rearrange(k_m, 'bm (head cm) hm wm -> bm head cm (hm wm)', head=self.num_heads)
        v_m = rearrange(v_m, 'bm (head cm) hm wm -> bm head cm (hm wm)', head=self.num_heads)

        k_c = torch.cat([k, k_m], dim=2)
        v_c = torch.cat([v, v_m], dim=2)

        q = torch.nn.functional.normalize(q, dim=-1)
        k_c = torch.nn.functional.normalize(k_c, dim=-1)

        q_m = torch.nn.functional.normalize(q_m, dim=-1)
        # k_m = torch.nn.functional.normalize(k_m, dim=-1)

        attn1 = (q @ k_c.transpose(-2, -1)) * self.temperature
        attn1 = attn1.softmax(dim=-1)

        attn2 = (q_m @ k_c.transpose(-2, -1)) * self.temperature
        attn2 = attn2.softmax(dim=-1)

        out1 = (attn1 @ v_c)
        out2 = (attn2 @ v_c)

        out1 = rearrange(out1, 'b head c (h w) -> b (h w) (head c) ', head=self.num_heads, h=h, w=w)
        out2 = rearrange(out2, 'b head c (h w) -> b (h w) (head c)', head=self.num_heads, h=h, w=w)
        # print(out1.shape)
        # print(out2.shape)


        cat = torch.cat([out1, out2], dim=2)
        out = self.liner(cat)


        out = rearrange(out, 'b (h w) (head c) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        # out2 = rearrange(out2, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        # print(out.shape)

        out = self.project_out(out)
        # out2 = self.project_out(out2)
        return out

class Restormer(nn.Module):
    def __init__(self,
                 H = 128,
                 W = 128,
                 message_length=64,
                 inp_channels=3,
                 out_channels=3,
                 dim=64,
                 num_blocks=[2, 2, 2, 2],
                 num_refinement_blocks=4,
                 heads=[4, 4, 4, 4],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 dual_pixel_task=False  ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
                 ):

        super(Restormer, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.message_pre = Message_pre(H, W, message_length)

        self.encoder_level1 = TransformerBlock1(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type)
        self.encoder_level12 = TransformerBlock1(dim=dim, num_heads=heads[0],
                                                             ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                                             LayerNorm_type=LayerNorm_type)


        # self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 =TransformerBlock1(dim=dim, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type)
        self.encoder_level22 = TransformerBlock1(dim=dim, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                                                bias=bias, LayerNorm_type=LayerNorm_type)

        # self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.embedding = Embedding(dim, heads[1], bias)
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        # self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.encoder_level4 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

        self.output = nn.Conv2d(dim + 3, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img, message):

        inp_enc_level1 = self.patch_embed(inp_img)  # b 48 h w
        message_pre = self.message_pre(message)
        out_enc_level1 = self.encoder_level1(inp_enc_level1, message_pre)
        out_enc_level12 = self.encoder_level12(out_enc_level1, message_pre)

        # inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(out_enc_level12,message_pre)
        out_enc_level22 = self.encoder_level22(out_enc_level2,message_pre)
        # cat = torch.cat([message_pre, out_enc_level22], dim=1)
        # concat = self.concat(cat)
        embedding = self.embedding(out_enc_level22, message_pre)

        # inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(embedding)

        # inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.encoder_level4(out_enc_level3)

        # inp_dec_level3 = self.up4_3(latent)
        # inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        # inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        # out_dec_level3 = self.decoder_level3(inp_dec_level3)
        #
        # inp_dec_level2 = self.up3_2(out_dec_level3)
        # inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        # inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        # out_dec_level2 = self.decoder_level2(inp_dec_level2)
        #
        # inp_dec_level1 = self.up2_1(out_dec_level2)
        # inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        # out_dec_level1 = self.decoder_level1(inp_dec_level1)

        # out_dec_level1 = self.refinement(out_dec_level1)

        #### For Dual-Pixel Defocus Deblurring Task ####
        # if self.dual_pixel_task:
        #     out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
        #     out_dec_level1 = self.output(out_dec_level1)
        # ###########################
        # else:
        cat = torch.cat([latent, inp_img], dim=1)
        out_dec_level1 = self.output(cat)

        return out_dec_level1


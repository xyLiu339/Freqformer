"""
Implementation of ESDNet for image demoireing
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.nn.parameter import Parameter

class my_model(nn.Module):
    def __init__(self,
                 en_feature_num,
                 en_inter_num,
                 de_feature_num,
                 de_inter_num,
                 res_number=[1,1,1],
                 heads=[2,2,2],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):
        super(my_model, self).__init__()
        self.encoder = Encoder(feature_num=en_feature_num, inter_num=en_inter_num, res_number=res_number, heads=heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.decoder = Decoder(en_num=en_feature_num, feature_num=de_feature_num, inter_num=de_inter_num, res_number=[4,4,2], heads=heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)

    def forward(self, x):
        y_1, y_2, y_3 = self.encoder(x)
        out_1, out_2, out_3, fff = self.decoder(y_1, y_2, y_3)

        return out_1, out_2, out_3, fff

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)


class Decoder(nn.Module):
    def __init__(self, en_num, feature_num, inter_num, res_number,
                 heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(Decoder, self).__init__()
        self.preconv_3 = conv_relu(4 * en_num, feature_num, 3, padding=1)
        self.decoder_3 = Decoder_Level_Res(3, feature_num, inter_num, res_number=res_number[2], heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)

        self.preconv_2 = conv_relu(2 * en_num + feature_num, feature_num, 3, padding=1)
        self.decoder_2 = Decoder_Level_Res(2, feature_num, inter_num, res_number=res_number[1], heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)

        self.preconv_1 = conv_relu(en_num + feature_num, feature_num, 3, padding=1)
        self.decoder_1 = Decoder_Level_Res(1, feature_num, inter_num, res_number=res_number[0], heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)

    def forward(self, y_1, y_2, y_3):
        x_3 = y_3
        x_3 = self.preconv_3(x_3)
        out_3, feat_3 = self.decoder_3(x_3)

        x_2 = torch.cat([y_2, feat_3], dim=1)
        x_2 = self.preconv_2(x_2)
        out_2, feat_2 = self.decoder_2(x_2, other_input=feat_3)

        x_1 = torch.cat([y_1, feat_2], dim=1)
        x_1 = self.preconv_1(x_1)
        out_1, outf = self.decoder_1(x_1, feat=False, other_input=torch.cat([feat_2, F.interpolate(feat_3, scale_factor=2, mode='bilinear')], dim=1))

        return out_1, out_2, out_3, outf


class Encoder(nn.Module):
    def __init__(self, feature_num, inter_num, res_number,
                 heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(Encoder, self).__init__()
        self.conv_first = nn.Sequential(
            nn.Conv2d(12, feature_num, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True)
        )
        self.encoder_1 = Encoder_Level_Res(feature_num, inter_num, level=1, res_number=res_number[0], heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.encoder_2 = Encoder_Level_Res(2 * feature_num, inter_num, level=2, res_number=res_number[1], heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.encoder_3 = Encoder_Level_Res(4 * feature_num, inter_num, level=3, res_number=res_number[2], heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)

    def forward(self, x):
        x = F.pixel_unshuffle(x, 2)
        x = self.conv_first(x)

        out_feature_1, down_feature_1 = self.encoder_1(x)
        out_feature_2, down_feature_2 = self.encoder_2(down_feature_1)
        out_feature_3 = self.encoder_3(down_feature_2)

        return out_feature_1, out_feature_2, out_feature_3


class Encoder_Level(nn.Module):
    def __init__(self, feature_num, inter_num, level, sam_number):
        super(Encoder_Level, self).__init__()
        self.rdb = RDB(in_channel=feature_num, d_list=(1, 2, 1), inter_num=inter_num)
        self.sam_blocks = nn.ModuleList()
        for _ in range(sam_number):
            sam_block = SAM(in_channel=feature_num, d_list=(1, 2, 3, 2, 1), inter_num=inter_num)
            self.sam_blocks.append(sam_block)

        if level < 3:
            self.down = nn.Sequential(
                nn.Conv2d(feature_num, 2 * feature_num, kernel_size=3, stride=2, padding=1, bias=True),
                nn.ReLU(inplace=True)
            )
        self.level = level

    def forward(self, x):
        out_feature = self.rdb(x)
        for sam_block in self.sam_blocks:
            out_feature = sam_block(out_feature)
        if self.level < 3:
            down_feature = self.down(out_feature)
            return out_feature, down_feature
        return out_feature
    
class Encoder_Level_Res(nn.Module):
    def __init__(self, feature_num, inter_num, level, res_number,
                heads, ffn_expansion_factor = 2.66, bias = False, LayerNorm_type = 'WithBias'):
        super(Encoder_Level_Res, self).__init__()
        self.rdb = RDB(in_channel=feature_num, d_list=(1, 2, 1), inter_num=inter_num)
        self.res_blocks =nn.Sequential(*[TransformerBlock(dim=int(feature_num), num_heads=heads, ffn_expansion_factor=ffn_expansion_factor, \
            bias=bias, LayerNorm_type=LayerNorm_type) for i in range(res_number)])

        if level < 3:
            self.down = nn.Sequential(
                nn.Conv2d(feature_num, 2 * feature_num, kernel_size=3, stride=2, padding=1, bias=True),
                nn.ReLU(inplace=True)
            )
        self.level = level

    def forward(self, x):
        out_feature = self.rdb(x)
        out_feature = self.res_blocks(out_feature)
        if self.level < 3:
            down_feature = self.down(out_feature)
            return out_feature, down_feature
        return out_feature


class Decoder_Level_Res(nn.Module):
    def __init__(self, level, feature_num, inter_num, res_number,
                heads, ffn_expansion_factor = 2.66, bias = False, LayerNorm_type = 'WithBias'):
        super(Decoder_Level_Res, self).__init__()
        self.level = level
        self.rdb = RDB(feature_num, (1, 2, 1), inter_num)
        self.res_blocks =nn.Sequential(*[TransformerBlock(dim=int(feature_num), num_heads=heads, ffn_expansion_factor=ffn_expansion_factor, \
            bias=bias, LayerNorm_type=LayerNorm_type) for i in range(res_number)])
        self.conv = conv(in_channel=feature_num, out_channel=12, kernel_size=3, padding=1)
        if self.level == 2:
            self.combine_conv = conv(in_channel=feature_num * 2, out_channel=feature_num, kernel_size=3, padding=1)
        elif self.level == 1:
            self.combine_conv = conv(in_channel=feature_num * 3, out_channel=feature_num, kernel_size=3, padding=1)

    def forward(self, x, feat=True, other_input=None):
        # import ipdb; ipdb.set_trace()
        x = self.rdb(x) # (bs, de_feature_num(64), h // f, w // f)
        if len(self.res_blocks) == 2:
            x = self.res_blocks(x)
        else:
            assert len(self.res_blocks) == 4 and self.level <= 2
            for idx, blk in enumerate(self.res_blocks):
                if idx == 2:
                    x = blk(self.combine_conv(torch.cat([x, other_input], dim=1)))
                else:
                    x = blk(x)
                
        # for sam_block in self.sam_blocks:
        #     x = sam_block(x)  # (bs, de_feature_num(64), h // f, w // f)
        out = self.conv(x) # (bs, 12, h // f, w // f)
        out = F.pixel_shuffle(out, 2) # (bs, 12 // 4, h * 2 // f, w * 2 // f)

        if feat:
            feature = F.interpolate(x, scale_factor=2, mode='bilinear')
            return out, feature
        else:
            return out, x


class DB(nn.Module):
    def __init__(self, in_channel, d_list, inter_num):
        super(DB, self).__init__()
        self.d_list = d_list
        self.conv_layers = nn.ModuleList()
        c = in_channel
        for i in range(len(d_list)):
            dense_conv = conv_relu(in_channel=c, out_channel=inter_num, kernel_size=3, dilation_rate=d_list[i],
                                   padding=d_list[i])
            self.conv_layers.append(dense_conv)
            c = c + inter_num
        self.conv_post = conv(in_channel=c, out_channel=in_channel, kernel_size=1)

    def forward(self, x):
        t = x
        for conv_layer in self.conv_layers:
            _t = conv_layer(t)
            t = torch.cat([_t, t], dim=1)
        t = self.conv_post(t)
        return t


class SAM(nn.Module):
    def __init__(self, in_channel, d_list, inter_num):
        super(SAM, self).__init__()
        self.basic_block = DB(in_channel=in_channel, d_list=d_list, inter_num=inter_num)
        self.basic_block_2 = DB(in_channel=in_channel, d_list=d_list, inter_num=inter_num)
        self.basic_block_4 = DB(in_channel=in_channel, d_list=d_list, inter_num=inter_num)
        self.fusion = CSAF(3 * in_channel)

    def forward(self, x):
        x_0 = x
        x_2 = F.interpolate(x, scale_factor=0.5, mode='bilinear')
        x_4 = F.interpolate(x, scale_factor=0.25, mode='bilinear')

        y_0 = self.basic_block(x_0)
        y_2 = self.basic_block_2(x_2)
        y_4 = self.basic_block_4(x_4)

        y_2 = F.interpolate(y_2, scale_factor=2, mode='bilinear')
        y_4 = F.interpolate(y_4, scale_factor=4, mode='bilinear')

        y = self.fusion(y_0, y_2, y_4)
        y = x + y

        return y


class CSAF(nn.Module):
    def __init__(self, in_chnls, ratio=4):
        super(CSAF, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.compress1 = nn.Conv2d(in_chnls, in_chnls // ratio, 1, 1, 0)
        self.compress2 = nn.Conv2d(in_chnls // ratio, in_chnls // ratio, 1, 1, 0)
        self.excitation = nn.Conv2d(in_chnls // ratio, in_chnls, 1, 1, 0)

    def forward(self, x0, x2, x4):
        out0 = self.squeeze(x0)
        out2 = self.squeeze(x2)
        out4 = self.squeeze(x4)
        out = torch.cat([out0, out2, out4], dim=1)
        out = self.compress1(out)
        out = F.relu(out)
        out = self.compress2(out)
        out = F.relu(out)
        out = self.excitation(out)
        out = F.sigmoid(out)
        w0, w2, w4 = torch.chunk(out, 3, dim=1)
        x = x0 * w0 + x2 * w2 + x4 * w4

        return x


class RDB(nn.Module):
    def __init__(self, in_channel, d_list, inter_num):
        super(RDB, self).__init__()
        self.d_list = d_list
        self.conv_layers = nn.ModuleList()
        c = in_channel
        for i in range(len(d_list)):
            dense_conv = conv_relu(in_channel=c, out_channel=inter_num, kernel_size=3, dilation_rate=d_list[i],
                                   padding=d_list[i])
            self.conv_layers.append(dense_conv)
            c = c + inter_num
        self.conv_post = conv(in_channel=c, out_channel=in_channel, kernel_size=1)

    def forward(self, x):
        t = x
        for conv_layer in self.conv_layers:
            _t = conv_layer(t)
            t = torch.cat([_t, t], dim=1)

        t = self.conv_post(t)
        return t + x


class conv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, dilation_rate=1, padding=0, stride=1):
        super(conv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=True, dilation=dilation_rate)

    def forward(self, x_input):
        out = self.conv(x_input)
        return out


class conv_relu(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, dilation_rate=1, padding=0, stride=1):
        super(conv_relu, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=True, dilation=dilation_rate),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_input):
        out = self.conv(x_input)
        return out



from pdb import set_trace as stx
import numbers
from einops import rearrange



##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

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
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
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

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
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

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
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



##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x



##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x



##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)
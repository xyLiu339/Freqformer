import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.nn.parameter import Parameter

from model.nets_dat_res_conv_res import my_model as ResHIGH
from model.nets_res_res_conv_res import my_model as ResLOW

from pdb import set_trace as stx
import numbers
from einops import rearrange

class LXY_JointModel(nn.Module):
    def __init__(self, en_feature_num,
                 en_inter_num,
                 de_feature_num,
                 de_inter_num,
                 sam_number=1,
                 res_number=[1,1,1],
                 heads=[2,2,2],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):
        super(LXY_JointModel, self).__init__()
        self.model_high = ResHIGH(en_feature_num=en_feature_num,
                                      en_inter_num=en_inter_num,
                                      de_feature_num=de_feature_num,
                                      de_inter_num=de_inter_num,
                                      res_number=res_number,
                                      heads=heads,
                                      ffn_expansion_factor=ffn_expansion_factor,
                                      bias=bias,
                                      LayerNorm_type=LayerNorm_type)
        
        self.model_low = ResLOW(en_feature_num=en_feature_num,
                                      en_inter_num=en_inter_num,
                                      de_feature_num=de_feature_num,
                                      de_inter_num=de_inter_num,
                                      res_number=[1,2,2],
                                      heads=heads,
                                      ffn_expansion_factor=ffn_expansion_factor,
                                      bias=bias,
                                      LayerNorm_type=LayerNorm_type)
        
        self.fusion_pre_high = conv(in_channel=de_feature_num, out_channel=de_feature_num, kernel_size=3, padding=1)
        self.fusion_pre_low = conv(in_channel=de_feature_num, out_channel=de_feature_num, kernel_size=3, padding=1)
        
        self.fusion_module = nn.Sequential(*[TransformerBlock(dim=int(de_feature_num), num_heads=2, ffn_expansion_factor=ffn_expansion_factor, \
            bias=bias, LayerNorm_type=LayerNorm_type) for i in range(2)])
        
        self.out_conv = conv(in_channel=de_feature_num, out_channel=12, kernel_size=3, padding=1)
        
    def forward(self, high, low, h_ori, w_ori, x, y, mode="train"):
        h1, h2, h3, h_f = self.model_high(high)
        l1, l2, l3, l_f = self.model_low(low)
        
        if mode == "test":
            # assert h_f.shape[3] == 3840 // 2 or h_f.shape[3] == 1024 // 2 or h_f.shape[3] == 1920 // 2 or h_f.shape[3] == 256 // 2
            # h_f = h_f[:, :, 2:-2, :]
            l_f = F.interpolate(l_f, size=(h_f.shape[2], h_f.shape[3]), mode='bicubic',)
            # assert 0, f"{h_f.shape}, {l_f.shape}"
            fusion = self.fusion_pre_high(h_f) + self.fusion_pre_low(l_f)
            res = self.fusion_module(fusion)
            res = self.out_conv(res)
            res = F.pixel_shuffle(res, 2)
            
            return res
 
        l_f_cropped = torch.zeros_like(h_f)
        
        crop_size = h1.shape[2]

        # crop the low separately
        for i in range(l1.shape[0]):
            assert y[i]+crop_size <= h_ori[i] and x[i]+crop_size <= w_ori[i], f"{x[i], y[i], crop_size, h_ori[i], w_ori[i]}"
            l_f_temp = F.interpolate(l_f[i].unsqueeze(0), size=(h_ori[i] // 2, w_ori[i] // 2), mode='bicubic',)[0]
            l_f_cropped[i] = l_f_temp[:, y[i] // 2:(y[i]+crop_size) // 2, x[i] // 2:(x[i]+crop_size) // 2]
            
        fusion = self.fusion_pre_high(h_f) + self.fusion_pre_low(l_f_cropped)
        
        res = self.fusion_module(fusion)
        
        res = self.out_conv(res)
        res = F.pixel_shuffle(res, 2)
        
        return res, res, res

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)
        
        
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
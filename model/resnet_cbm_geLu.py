import torch
import torch.nn as nn
from math import ceil
class ECA(nn.Module):
    def __init__(self, in_channels, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y

class BAM(nn.Module):
    def __init__(self, in_channels, reduction=16, dilation=4):
        super(BAM, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, padding=0, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, padding=0, bias=True)
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=True)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        ca = self.channel_attention(x)
        sa = self.spatial_attention(x)
        y = self.sigmoid(ca + sa)
        return x * y

# Memory-efficient Siwsh using torch.jit.script borrowed from the code in (https://twitter.com/jeremyphoward/status/1188251041835315200)
# Currently use memory-efficient SiLU as default:
USE_MEMORY_EFFICIENT_SiLU = True

if USE_MEMORY_EFFICIENT_SiLU:
    @torch.jit.script
    def silu_fwd(x):
        return x.mul(torch.sigmoid(x))


    @torch.jit.script
    def silu_bwd(x, grad_output):
        x_sigmoid = torch.sigmoid(x)
        return grad_output * (x_sigmoid * (1. + x * (1. - x_sigmoid)))


    class SiLUJitImplementation(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return silu_fwd(x)

        @staticmethod
        def backward(ctx, grad_output):
            x = ctx.saved_tensors[0]
            return silu_bwd(x, grad_output)


    def silu(x, inplace=False):
        return SiLUJitImplementation.apply(x)

else:
    def silu(x, inplace=False):
        return x.mul_(x.sigmoid()) if inplace else x.mul(x.sigmoid())


class SiLU(nn.Module):
    def __init__(self, inplace=True):
        super(SiLU, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return silu(x, self.inplace)


import torch.nn.functional as F

class GELU(nn.Module):
    def __init__(self, inplace=True):
        super(GELU, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.gelu(x)

def ConvBNAct(out, in_channels, channels, kernel=1, stride=1, pad=0,
              num_group=1, active=True):
    out.append(nn.Conv2d(in_channels, channels, kernel,
                         stride, pad, groups=num_group, bias=False))
    out.append(nn.BatchNorm2d(channels))
    if active:
        out.append(GELU(inplace=True))


def ConvBNSiLU(out, in_channels, channels, kernel=1, stride=1, pad=0, num_group=1):
    out.append(nn.Conv2d(in_channels, channels, kernel,
                         stride, pad, groups=num_group, bias=False))
    out.append(nn.BatchNorm2d(channels))
    out.append(GELU(inplace=True))


class SE(nn.Module):
    def __init__(self, in_channels, channels, se_ratio=12):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, channels // se_ratio, kernel_size=1, padding=0),
            nn.BatchNorm2d(channels // se_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // se_ratio, channels, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y

class LinearBottleneck(nn.Module):
    def __init__(self, in_channels, channels, t, stride, use_eca=True, use_bam=False, eca_k_size=3, bam_reduction=16, bam_dilation=4, dropout=0.0, **kwargs):
        super(LinearBottleneck, self).__init__(**kwargs)
        self.use_shortcut = stride == 1 and in_channels <= channels
        self.in_channels = in_channels
        self.out_channels = channels
        self.drop = nn.Dropout2d(dropout)
        self.use_eca = use_eca
        self.use_bam = use_bam
        out = []
        if t != 1:
            dw_channels = in_channels * t
            ConvBNSiLU(out, in_channels=in_channels, channels=dw_channels)
        else:
            dw_channels = in_channels

        ConvBNAct(out, in_channels=dw_channels, channels=dw_channels, kernel=3, stride=stride, pad=1,
                  num_group=dw_channels, active=False)

        if self.use_eca:
            out.append(ECA(dw_channels, k_size=eca_k_size))

        if self.use_bam:
            out.append(BAM(dw_channels, reduction=bam_reduction, dilation=bam_dilation))

        out.append(GELU())
        ConvBNAct(out, in_channels=dw_channels, channels=channels, active=False)
        self.out = nn.Sequential(*out)

    def forward(self, x):
        out = self.out(x)
        if self.use_shortcut:
            out[:, 0:self.in_channels] += x
        out = self.drop(out)
        return out

class ReXNetAttentionGelu(nn.Module):
    def __init__(self, input_ch=16, final_ch=180, width_mult=1.0, depth_mult=1.0, classes=1000,
                 use_eca=True, use_bam=False, eca_k_size=3, bam_reduction=16, bam_dilation=4, dropout_ratio=0.2, drop_path=0.25, bn_momentum=0.9):
        super(ReXNetAttentionGelu, self).__init__()

        layers = [1, 2, 2, 3, 3, 5]
        strides = [1, 2, 2, 2, 1, 2]
        use_attention = [False, False, True, True, True, True]

        layers = [ceil(element * depth_mult) for element in layers]
        strides = sum([[element] + [1] * (layers[idx] - 1)
                       for idx, element in enumerate(strides)], [])
        if use_eca or use_bam:
            use_attention = sum([[element] * layers[idx] for idx, element in enumerate(use_attention)], [])
        else:
            use_attention = [False] * sum(layers[:])
        ts = [1] * layers[0] + [6] * sum(layers[1:])

        self.depth = sum(layers[:]) * 3
        stem_channel = 32 / width_mult if width_mult < 1.0 else 32
        inplanes = input_ch / width_mult if width_mult < 1.0 else input_ch

        features = []
        in_channels_group = []
        channels_group = []

        for i in range(self.depth // 3):
            if i == 0:
                in_channels_group.append(int(round(stem_channel * width_mult)))
                channels_group.append(int(round(inplanes * width_mult)))
            else:
                in_channels_group.append(int(round(inplanes * width_mult)))
                inplanes += final_ch / (self.depth // 3 * 1.0)
                channels_group.append(int(round(inplanes * width_mult)))

        ConvBNSiLU(features, 3, int(round(stem_channel * width_mult)), kernel=3, stride=2, pad=1)

        for block_idx, (in_c, c, t, s, att) in enumerate(zip(in_channels_group, channels_group, ts, strides, use_attention)):
            features.append(LinearBottleneck(in_channels=in_c,
                                             channels=c,
                                             t=t,
                                             stride=s,
                                             use_eca=att and use_eca, use_bam=att and use_bam, eca_k_size=eca_k_size, bam_reduction=bam_reduction, bam_dilation=bam_dilation, dropout=drop_path))

        pen_channels = int(1280 * width_mult)
        ConvBNSiLU(features, c, pen_channels)

        features.append(nn.AdaptiveAvgPool2d(1))
        self.features = nn.Sequential(*features)
        self.out = nn.Sequential(
            nn.Dropout(dropout_ratio),
            nn.Conv2d(pen_channels, classes, 1, bias=True))

    def extract_features(self, x):
        return self.features[:-1](x)

    def forward(self, x):
        x = self.features(x)
        x = self.out(x).flatten(1)
        return x

if __name__ == '__main__':
    model = ReXNetAttentionGelu(width_mult=1.0, classes=5, use_eca=True)
    out = model(torch.randn(32, 3, 224, 224))
    print(out.size())
    loss = out.sum()
    loss.backward()
    print('Checked a single forward/backward iteration')
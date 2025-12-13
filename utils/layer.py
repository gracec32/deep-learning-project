import torch
from torch import nn as nn
from torch.nn import init as init

# from DarkIR
# arxiv.org/abs/2412.13443

class CustomSequential(nn.Module):
    '''
    Similar to nn.Sequential, but it lets us introduce a second argument in the forward method
    so adaptors can be considered in the inference.
    '''
    def __init__(self, *args):
        super(CustomSequential, self).__init__()
        self.modules_list = nn.ModuleList(args)

    def forward(self, x, use_adapter=False):
        for module in self.modules_list:
            if hasattr(module, 'set_use_adapters'):
                module.set_use_adapters(use_adapter)
            x = module(x)
        return x


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class Adapter(nn.Module):
    def __init__(self, c, ffn_channel = None):
        super().__init__()
        if ffn_channel:
            ffn_channel = 2
        else:
            ffn_channel = c

        self.conv1 = nn.Conv2d(in_channels=c, out_channels=ffn_channel,
                                kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=ffn_channel, out_channels=c,
                                kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.depthwise = nn.Conv2d(in_channels=c, out_channels=ffn_channel,
                                    kernel_size=3, padding=1, stride=1, groups=c, bias=True, dilation=1)

    def forward(self, input):
        x = self.conv1(input) + self.depthwise(input)
        x = self.conv2(x)
        return x


class FreMLP(nn.Module):

    def __init__(self, nc, expand = 2):
        super(FreMLP, self).__init__()

        self.process1 = nn.Sequential(
            nn.Conv2d(nc, expand * nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(expand * nc, nc, 1, 1, 0)
            )

    def forward(self, x):
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(x, norm='backward')
        mag = torch.abs(x_freq)
        pha = torch.angle(x_freq)
        mag = self.process1(mag)
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)
        x_out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')
        return x_out


class Branch(nn.Module):
    '''
    Branch that lasts lonly the dilated convolutions
    '''
    def __init__(self, c, DW_Expand, dilation = 1):
        super().__init__()
        self.dw_channel = DW_Expand * c

        self.branch = nn.Sequential(
                       nn.Conv2d(in_channels=self.dw_channel, out_channels=self.dw_channel,
                                kernel_size=3, padding=dilation, stride=1, groups=self.dw_channel,
                                bias=True, dilation = dilation) # the dconv
        )

    def forward(self, input):
        return self.branch(input)

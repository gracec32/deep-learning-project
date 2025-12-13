import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init as init
import torch.distributed as dist

from collections import OrderedDict
from ptflops import get_model_complexity_info

from arch.color import HVIT, PHVIT
from arch.norms import LayerNormFunction, LayerNorm2d
from arch.layer import CustomSequential, SimpleGate, Adapter, FreMLP, Branch
from arch.attn import CAB


class EBlock(nn.Module):
    '''
    Encoder Block
    '''
    def __init__(self, c, DW_Expand=2, dilations = [1], extra_depth_wise = False):
        super().__init__()

        # we define the 2 branches
        self.dw_channel = DW_Expand * c
        self.extra_conv = nn.Conv2d(c, c, kernel_size=3, padding=1, stride=1,
                                    groups=c, bias=True, dilation=1) if extra_depth_wise else nn.Identity() #optional extra dw
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=self.dw_channel, kernel_size=1, padding=0, stride=1,
                                groups=1, bias=True, dilation = 1)

        self.branches = nn.ModuleList()
        for dilation in dilations:
            self.branches.append(Branch(c, DW_Expand, dilation = dilation))

        assert len(dilations) == len(self.branches)
        self.dw_channel = DW_Expand * c
        self.sca = nn.Sequential(
                       nn.AdaptiveAvgPool2d(1),
                       nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=self.dw_channel // 2,
                                            kernel_size=1, padding=0, stride=1,
                       groups=1, bias=True, dilation = 1),
        )
        self.sg1 = SimpleGate()
        self.conv3 = nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=c,
                                kernel_size=1, padding=0, stride=1, groups=1, bias=True, dilation = 1)

        # second step
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.freq = FreMLP(nc = c, expand=2)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        # for color transform + cross attn
        self.img_conv = nn.Conv2d(in_channels=c, out_channels=3, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.cab = CAB(dim=c, num_heads=1)
        self.PHVIT = PHVIT()

    def forward(self, inp):
        B, C, H, W = inp.shape
        y = inp

        dtypes = inp.dtype
        hvi = self.PHVIT(self.img_conv(y))

        x = self.norm1(y)
        x = self.conv1(self.extra_conv(x))
        z = 0
        for branch in self.branches:
            z += branch(x)

        z = self.sg1(z)
        x = self.sca(z) * z

        # grace's cross attension step
        attn = self.cab(x, hvi)
        x = self.conv3(attn)

        y = inp + self.beta * x

        # second step - frequency modulation
        x_step2 = self.norm2(y) # size [B, 2*C, H, W]
        x_freq = self.freq(x_step2) # size [B, C, H, W]
        x = y * x_freq
        x = y + x * self.gamma

        return x


class DBlock(nn.Module):
    '''
    Decoder Block
    '''
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, dilations = [1], extra_depth_wise = False):
        super().__init__()
        #we define the 2 branches
        self.dw_channel = DW_Expand * c

        self.conv1 = nn.Conv2d(in_channels=c, out_channels=self.dw_channel, kernel_size=1, padding=0, stride=1,
                                groups=1, bias=True, dilation = 1)
        self.extra_conv = nn.Conv2d(self.dw_channel, self.dw_channel, kernel_size=3, padding=1, stride=1,
                                    groups=c, bias=True, dilation=1) if extra_depth_wise else nn.Identity() #optional extra dw

        self.branches = nn.ModuleList()
        for dilation in dilations:
            self.branches.append(Branch(self.dw_channel, DW_Expand = 1, dilation = dilation))

        assert len(dilations) == len(self.branches)
        self.dw_channel = DW_Expand * c
        self.sca = nn.Sequential(
                       nn.AdaptiveAvgPool2d(1),
                       nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=self.dw_channel // 2,
                                            kernel_size=1, padding=0, stride=1,
                       groups=1, bias=True, dilation = 1),
        )
        self.sg1 = SimpleGate()
        self.sg2 = SimpleGate()
        self.conv3 = nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=c,
                                kernel_size=1, padding=0, stride=1, groups=1, bias=True, dilation = 1)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel,
                                kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c,
                                kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        # for color transform + cross attn
        self.img_conv = nn.Conv2d(in_channels=c, out_channels=3, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.cab = CAB(dim=c, num_heads=1)
        self.PHVIT = PHVIT()

    def forward(self, inp, adapter = None):
        B, C, H, W = inp.shape
        y = inp

        dtypes = inp.dtype
        hvi = self.PHVIT(self.img_conv(y))

        x = self.norm1(inp)
        x = self.extra_conv(self.conv1(x))
        z = 0
        for branch in self.branches:
            z += branch(x)

        z = self.sg1(z)
        x = self.sca(z) * z

        # cross attension step
        attn = self.cab(x, hvi)
        x = self.conv3(attn)

        x = self.conv3(x)
        y = inp + self.beta * x

        #second step
        x = self.conv4(self.norm2(y)) # size [B, 2*C, H, W]
        x = self.sg2(x)  # size [B, C, H, W]
        x = self.conv5(x) # size [B, C, H, W]
        x = y + x * self.gamma

        return x


class Model(nn.Module):

    def __init__(self, img_channel=3,
                 width=32,
                 middle_blk_num_enc=2,
                 middle_blk_num_dec=2,
                 enc_blk_nums=[1, 2, 3],
                 dec_blk_nums=[3, 1, 1],
                 dilations = [1, 4, 9],
                 extra_depth_wise = True):
        super(Model, self).__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                                bias=True)

        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                             bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                CustomSequential(
                    *[EBlock(chan, extra_depth_wise=extra_depth_wise) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks_enc = \
            CustomSequential(
                *[EBlock(chan, extra_depth_wise=extra_depth_wise) for _ in range(middle_blk_num_enc)]
            )
        self.middle_blks_dec = \
            CustomSequential(
                *[DBlock(chan, dilations=dilations, extra_depth_wise=extra_depth_wise) for _ in range(middle_blk_num_dec)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                CustomSequential(
                    *[DBlock(chan, dilations=dilations, extra_depth_wise=extra_depth_wise) for _ in range(num)]
                )
            )
        self.padder_size = 2 ** len(self.encoders)

        # this layer is needed for the computing of the middle loss - it isn't necessary for anything else
        self.side_out = nn.Conv2d(in_channels = width * 2**len(self.encoders), out_channels = img_channel,
                                kernel_size = 3, stride=1, padding=1)

        # color transformations
        self.HVIT = HVIT()
        self.PHVIT = PHVIT()

    def forward(self, input, side_loss = False, use_adapter = None):
        _, _, H, W = input.shape

        input = self.check_image_size(input)
        input = self.HVIT(input)

        x = self.intro(input)

        skips = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            skips.append(x)
            x = down(x)

        # we apply the encoder transforms
        x_light = self.middle_blks_enc(x)

        if side_loss:
            out_side = self.side_out(x_light)
        # apply the decoder transforms
        x = self.middle_blks_dec(x_light)
        x = x + x_light

        for decoder, up, skip in zip(self.decoders, self.ups, skips[::-1]):
            x = up(x)
            x = x + skip
            x = decoder(x)

        x = self.ending(x)
        x = self.PHVIT(x)
        x = x + input
        out = x[:, :, :H, :W] # we recover the original size of the image
        if side_loss:
            return out_side, out
        else:
            return out

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), value = 0)
        return x


def create():
    img_channel = 3
    width = 32

    enc_blks = [1, 2, 3]
    middle_blk_num_enc = 2
    middle_blk_num_dec = 2
    dec_blks = [3, 1, 1]
    residual_layers = None
    dilations = [1, 4, 9]
    extra_depth_wise = True

    net = Model(img_channel=img_channel,
                  width=width,
                  middle_blk_num_enc=middle_blk_num_enc,
                  middle_blk_num_dec= middle_blk_num_dec,
                  enc_blk_nums=enc_blks,
                  dec_blk_nums=dec_blks,
                  dilations = dilations,
                  extra_depth_wise = extra_depth_wise)

    new_state_dict = net.state_dict()
    inp_shape = (3, 256, 256)
    net.load_state_dict(new_state_dict)

    # macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)
    # print(macs, params)

    weights = net.state_dict()
    adapter_weights = {k: v for k, v in weights.items() if 'adapter' not in k}

    # print(net)
    return net


if __name__ == "__main__":
    create()

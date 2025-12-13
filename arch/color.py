import torch
import torch.nn as nn

# from HVI CidNET
# arxiv.org/abs/2502.20272

class HVIT(nn.Module):

    def __init__(self, density_k_init=0.2):
        super(HVIT, self).__init__()

        # density_k is a trainable parameter
        self.density_k = nn.Parameter(torch.tensor([density_k_init]))
        self.pi = torch.tensor(3.141592653589793)

    def forward(self, img):
        eps = 1e-8

        device = img.device
        dtypes = img.dtype

        hue = torch.zeros(img.shape[0], img.shape[2], img.shape[3], device=device, dtype=dtypes)

        value = img.max(1)[0].to(dtypes)
        img_min = img.min(1)[0].to(dtypes)

        hue[img[:, 2] == value] = 4.0 + ((img[:, 0] - img[:, 1]) / (value - img_min + eps))[img[:, 2] == value]
        hue[img[:, 1] == value] = 2.0 + ((img[:, 2] - img[:, 0]) / (value - img_min + eps))[img[:, 1] == value]
        hue[img[:, 0] == value] = (0.0 + ((img[:, 1] - img[:, 2]) / (value - img_min + eps)))[img[:, 0] == value] % 6

        hue[img_min == value] = 0.0
        hue = hue / 6.0

        saturation = (value - img_min) / (value + eps)
        saturation[value == 0] = 0

        hue = hue.unsqueeze(1)
        saturation = saturation.unsqueeze(1)
        value = value.unsqueeze(1)

        k = self.density_k
        this_k = k.item()
        color_sensitive = ((value * 0.5 * self.pi).sin() + eps).pow(k)

        ch = (2.0 * self.pi * hue).cos()
        cv = (2.0 * self.pi * hue).sin()

        H = color_sensitive * saturation * ch
        V = color_sensitive * saturation * cv
        I = value

        # concatenate the components to form the final HVI image [B, 3, H, W]
        xyz = torch.cat([H, V, I], dim=1)
        return xyz


class PHVIT(nn.Module):

    def __init__(self, density_k_init=0.2, alpha_s=1.3, alpha=1.0, gated=False, gated2=False):
        super(PHVIT, self).__init__()

        self.density_k = nn.Parameter(torch.tensor([density_k_init]))
        self.alpha_s = alpha_s
        self.alpha = alpha
        self.gated = gated
        self.gated2 = gated2
        self.pi = torch.tensor(3.141592653589793)

    def forward(self, img):
        eps = 1e-8

        device = img.device
        dtypes = img.dtype

        H, V, I = img[:, 0, :, :], img[:, 1, :, :], img[:, 2, :, :]

        H = torch.clamp(H, -1, 1)
        V = torch.clamp(V, -1, 1)
        I = torch.clamp(I, 0, 1)

        v = I
        k = self.density_k.item()

        color_sensitive = ((v * 0.5 * self.pi).sin() + eps).pow(k)

        H = H / (color_sensitive + eps)
        V = V / (color_sensitive + eps)

        H = torch.clamp(H, -1, 1)
        V = torch.clamp(V, -1, 1)

        h = torch.atan2(V + eps, H + eps) / (2 * self.pi)
        h = h % 1
        s = torch.sqrt(H ** 2 + V ** 2 + eps)

        if self.gated:
            s = s * self.alpha_s

        s = torch.clamp(s, 0, 1)
        v = torch.clamp(v, 0, 1)

        r = torch.zeros_like(h)
        g = torch.zeros_like(h)
        b = torch.zeros_like(h)

        hi = torch.floor(h * 6.0)
        f = h * 6.0 - hi
        p = v * (1. - s)
        q = v * (1. - (f * s))
        t = v * (1. - ((1. - f) * s))

        hi0 = hi == 0
        hi1 = hi == 1
        hi2 = hi == 2
        hi3 = hi == 3
        hi4 = hi == 4
        hi5 = hi == 5

        r[hi0] = v[hi0]
        g[hi0] = t[hi0]
        b[hi0] = p[hi0]

        r[hi1] = q[hi1]
        g[hi1] = v[hi1]
        b[hi1] = p[hi1]

        r[hi2] = p[hi2]
        g[hi2] = v[hi2]
        b[hi2] = t[hi2]

        r[hi3] = p[hi3]
        g[hi3] = q[hi3]
        b[hi3] = v[hi3]

        r[hi4] = t[hi4]
        g[hi4] = p[hi4]
        b[hi4] = v[hi4]

        r[hi5] = v[hi5]
        g[hi5] = p[hi5]
        b[hi5] = q[hi5]

        r = r.unsqueeze(1)
        g = g.unsqueeze(1)
        b = b.unsqueeze(1)

        # concatenate the components to form the final RGB image [B, 3, H, W]
        rgb = torch.cat([r, g, b], dim=1)

        if self.gated2:
            rgb = rgb * self.alpha

        return rgb

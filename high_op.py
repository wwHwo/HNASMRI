from abc import ABC
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count_table
import torch
import torch.nn as nn
from einops import rearrange

OPS = {
    'Local': lambda C, stride, affine: GlobalBlock(3),
    'Global': lambda C, stride, affine: LocalBlock(3),
}


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )


def conv_nxn_bn(inp, oup, kernal_size, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernal_size, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )


# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=32, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b p n (h d) -> b p h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    """Transformer block described in ViT.
    Paper: https://arxiv.org/abs/2010.11929
    Based on: https://github.com/lucidrains/vit-pytorch
    """

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class GlobalBlock(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size=16, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size
        mlp_dim = dim * 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.SiLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(channel, dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU())

        self.transformer = Transformer(dim, depth, 4, 32, mlp_dim, dropout)

        self.conv3 = nn.Sequential(
            nn.Conv2d(dim, channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU())
        self.conv4 = nn.Sequential(
            nn.Conv2d(2 * channel, channel, kernel_size, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.SiLU())

    def forward(self, x):
        y = x.clone()

        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)

        # Global representations
        _, _, h, w = x.shape
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d',
                      ph=self.ph, pw=self.pw)
        x = self.transformer(x)
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)',
                      h=h // self.ph, w=w // self.pw, ph=self.ph, pw=self.pw)

        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x


class LocalBlock(nn.Module):

    def __init__(self, channel=1):
        super().__init__()
        self.node1 = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, groups=channel, bias=False),
            nn.Conv2d(channel, 2 * channel, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(2 * channel, affine=True),
            nn.ReLU(inplace=False), )

        self.node2 = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(3 * channel, 3 * channel, kernel_size=3, stride=1, padding=1, groups=channel, bias=False),
            nn.Conv2d(3 * channel, 6 * channel, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(6 * channel, affine=True),
            nn.ReLU(inplace=False), )

        self.node3 = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(8 * channel, 8 * channel, kernel_size=3, stride=1, padding=1, groups=channel, bias=False),
            nn.Conv2d(8 * channel, 1 * channel, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(1 * channel, affine=True),
            nn.ReLU(inplace=False), )
        self.out = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(1 * channel, 1 * channel, kernel_size=3, stride=1, padding=1, groups=channel, bias=False),
            nn.Conv2d(1 * channel, 1 * channel, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(1 * channel, affine=True),
            nn.ReLU(inplace=False), )

    def forward(self, x):
        node1 = self.node1(x)
        node2 = self.node2(torch.cat((node1, x), 1))
        node3 = self.node3(torch.cat((node2, node1), 1))
        out = self.out(node3 + x)
        return out

import torch
import torch.nn as nn
from einops import rearrange
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count_table

OPS = {
    'Global': lambda C,  affine: GlobalBlock(32, ),
    'Lockal': lambda C, affine: LocalBlock(2),
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
    def __init__(self, dim, depth=1, channel=2, kernel_size=3, patch_size=16, dropout=0.):
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

    def __init__(self, channel=2):
        super().__init__()
        self.node1 = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, groups=channel, bias=False),
            nn.Conv2d(channel, channel, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel, affine=True),
            nn.ReLU(inplace=False), )

        self.node2 = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, groups=channel, bias=False),
            nn.Conv2d(channel, channel, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel, affine=True),
            nn.ReLU(inplace=False), )

        self.node3 = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, groups=channel, bias=False),
            nn.Conv2d(channel, channel, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(1 * channel, affine=True),
            nn.ReLU(inplace=False), )

    def forward(self, x):
        node1_1 = self.node1(x)
        node1_2 = self.node1(x)
        node1 = torch.add(node1_1, node1_2)
        node2_1 = self.node2(node1)
        node2_2 = self.node2(node1)
        node2 = torch.add(node2_1, node2_2)
        node3_1 = self.node3(node2)
        node3_2 = self.node3(node2)
        node3 = torch.add(node3_1, node3_2)
        return node3 + node2 + node1


class MixedOp(nn.Module):
    """ Mixed operation """

    def __init__(self, C, ):
        super().__init__()
        self._ops = nn.ModuleList()
        for primitive in OPS:
            op = OPS[primitive](C,affine=False)
            self._ops.append(op)

    def forward(self, x, weights):
        """
        Args:
            x: input
            weights: weight for each operation
        """
        return sum(w * op(x) for w, op in zip(weights, self._ops))


if __name__ == '__main__':
    trans = GlobalBlock(dim=32, depth=3, channel=1, kernel_size=3)
    input = torch.randn((1, 1, 256, 256))
    flop = FlopCountAnalysis(trans, input)
    print(flop_count_table(flop, max_depth=4, show_param_shapes=True))
    print(flop_count_str(flop))
    print("Total", flop.total() / 1e9)

import torch
import torch.nn as nn


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
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
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class BaseAttention(nn.Module):
    def __init__(self, dim, num_heads=8, num_tokens=120, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., regular_s=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads 
        self.regular_s = regular_s
        self.scale = qk_scale or head_dim ** -0.5
        
        if self.regular_s:
            self.reg = nn.Parameter(torch.ones(1, num_heads, num_tokens, num_tokens)/num_tokens, requires_grad=True)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = attn.softmax(dim=-1)
        if self.regular_s:
            attn = attn + self.reg.repeat(B, 1, 1, 1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class BaseBlock(nn.Module):
    def __init__(self, dim, num_heads, num_tokens=120, mlp_ratio=2, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, regular_s=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = BaseAttention(
            dim, num_heads=num_heads, num_tokens=num_tokens, qkv_bias=qkv_bias, qk_scale=qk_scale, 
            attn_drop=attn_drop, proj_drop=drop, regular_s=regular_s)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class BottAttention(nn.Module):
    def __init__(self, dim, qkv_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., num_tokens=120, regular_s=False):
        super().__init__()
        self.num_heads = num_heads
        att_dim = qkv_dim * num_heads
        self.qkv_dim = qkv_dim
        self.regular_s = regular_s
        self.scale = qk_scale or qkv_dim ** -0.5

        if self.regular_s:
            self.reg = nn.Parameter(torch.ones(1, num_heads, num_tokens, num_tokens)/num_tokens, requires_grad=True)

        self.qkv = nn.Linear(dim, qkv_dim * 3 * num_heads, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(att_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.qkv_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        if self.regular_s:
            attn = attn + self.reg.repeat(B, 1, 1, 1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.qkv_dim*self.num_heads)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



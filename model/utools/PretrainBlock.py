import torch
import torch.nn as nn
from einops import rearrange
from model.utools.attn import DropPath, Mlp
from model.utools.attn import BaseAttention, BaseBlock


class STParaBlock(nn.Module):
    """
        Spatiotemporal parallelism
    """
    def __init__(self, dim, num_heads=8, mlp_ratio=2, qkv_bias=False, qk_scale=None, 
                 drop=0., attn_drop=0., num_joints=25, num_frames=64, regular_s=False,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, spatial_norm=None, temporal_norm=None):
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm1 = norm_layer(dim)
        self.attn_sp = BaseAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, 
            attn_drop=attn_drop, proj_drop=drop, num_tokens=num_joints+1, regular_s=regular_s)
        self.norm3 = norm_layer(dim)
        self.mlp1 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.spatial_norm = spatial_norm or norm_layer(dim)
       
        self.norm2 = norm_layer(dim)
        self.attn_tp = BaseAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, 
            attn_drop=attn_drop, proj_drop=drop, num_tokens=num_frames+1, regular_s=regular_s)
        self.norm4 = norm_layer(dim)
        self.mlp2 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.temporal_norm = temporal_norm or norm_layer(dim)

    def forward(self, x, spatial_pos=None, temporal_pos=None, spatial_mask=None, temporal_mask=None):
        n, t, v, c = x.shape

        sp_x = rearrange(x, 'n t v c -> (n t) v c',)
        if spatial_pos is not None:
            sp_x = sp_x + spatial_pos
        sp_x = sp_x + self.drop_path(self.attn_sp(self.norm1(sp_x), spatial_mask))
        sp_x = sp_x + self.drop_path(self.mlp1(self.norm3(sp_x)))
        sp_x = self.spatial_norm(sp_x)
        sp_x = rearrange(sp_x, '(n t) v c -> n t v c', t=t)

        # tp_x = rearrange(sp_x, '(n t) v c -> n t (v c)', n=n, t=t)
        tp_x = rearrange(x, 'n t v c -> (n v) t c')
        if temporal_pos is not None:
            tp_x = tp_x + temporal_pos
        tp_x = tp_x + self.drop_path(self.attn_tp(self.norm2(tp_x), temporal_mask))
        tp_x = tp_x + self.drop_path(self.mlp2(self.norm4(tp_x)))
        tp_x = self.temporal_norm(tp_x)
        tp_x = rearrange(tp_x, '(n v) t c -> n t v c', v=v)

        return tp_x + sp_x
    

class STSerialBlock(nn.Module):
    """
        Spatiotemporal series
    """
    def __init__(self, dim, num_heads=8, mlp_ratio=2, qkv_bias=False, qk_scale=None, 
                 drop=0., attn_drop=0., num_joints=25, num_frames=64, regular_s=False,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, spatial_norm=None, temporal_norm=None):
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm1 = norm_layer(dim)
        self.attn_sp = BaseAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, 
            attn_drop=attn_drop, proj_drop=drop, num_tokens=num_joints+1, regular_s=regular_s)
        self.norm3 = norm_layer(dim)
        self.mlp1 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.spatial_norm = spatial_norm or norm_layer(dim)
       
        self.norm2 = norm_layer(dim)
        self.attn_tp = BaseAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, 
            attn_drop=attn_drop, proj_drop=drop, num_tokens=num_frames+1, regular_s=regular_s)
        self.norm4 = norm_layer(dim)
        self.mlp2 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.temporal_norm = temporal_norm or norm_layer(dim)

    def forward(self, x, spatial_pos=None, temporal_pos=None, spatial_mask=None, temporal_mask=None):
        n, t, v, c = x.shape
        sp_x = rearrange(x, 'n t v c -> (n t) v c',)
        if spatial_pos is not None:
            sp_x = sp_x + spatial_pos
        sp_x = sp_x + self.drop_path(self.attn_sp(self.norm1(sp_x), spatial_mask))
        sp_x = sp_x + self.drop_path(self.mlp1(self.norm3(sp_x)))
        sp_x = self.spatial_norm(sp_x)

        tp_x = rearrange(sp_x, '(n t) v c -> (n v) t c', n=n, t=t, v=v)
        if temporal_pos is not None:
            tp_x = tp_x + temporal_pos
        tp_x = tp_x + self.drop_path(self.attn_tp(self.norm2(tp_x), temporal_mask))
        tp_x = tp_x + self.drop_path(self.mlp2(self.norm4(tp_x)))
        tp_x = self.temporal_norm(tp_x)
        tp_x = rearrange(tp_x, '(n v) t c -> n t v c', n=n, t=t, v=v)

        return tp_x


class MotifBlock(nn.Module):
    def __init__(self, dim, num_heads, out_dim, mlp_ratio=2, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, num_tokens=120, regular_s=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = BaseAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, 
            num_tokens=num_tokens, regular_s=regular_s)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=out_dim, act_layer=act_layer, drop=drop)
        if dim != out_dim:
            self.res = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, out_dim)
            )
        else:
            self.res = lambda x: x

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = self.res(x) + self.drop_path(self.mlp(self.norm2(x)))
        return x


# 时间重复  空间重复得到
class STReBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=2, num_joints=25, num_frames=64, 
                 drop=0., attn_drop=0., qkv_bias=True, qk_scale=None, drop_path=0., 
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, re_depth=1, regular_s=False):
        super().__init__()
        self.re_depth = re_depth
        
        self.STRe = MotifBlock(2*dim, num_heads=num_heads, out_dim=dim, attn_drop=attn_drop, drop=drop, mlp_ratio=mlp_ratio, 
                               qkv_bias=qkv_bias, num_tokens=num_joints*num_frames, regular_s=regular_s,
                               act_layer=act_layer, norm_layer=norm_layer, qk_scale=qk_scale)
        if re_depth > 1:
            dpr = [x.item() for x in torch.linspace(0, drop_path, re_depth-1)]
            self.re = nn.ModuleList([
                BaseBlock(dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                          drop=drop, attn_drop=attn_drop, num_tokens=num_joints*num_frames, regular_s=regular_s,
                          drop_path=dpr[i], act_layer=act_layer, norm_layer=norm_layer) for i in range(re_depth-1)
            ])

    def forward(self, x):
        x = rearrange(x, 'n t v c -> n (t v) c')
        x = self.STRe(x)
        if self.re_depth > 1:
            for blk in self.re:
                x = blk(x)
                
        return x
    

class DecoderBlock(nn.Module):
    """
        Spatiotemporal series
    """
    def __init__(self, dim, num_heads=8, mlp_ratio=2, qkv_bias=False, qk_scale=None, 
                 drop=0., attn_drop=0., num_tokens=25, regular_s=False,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm1 = norm_layer(dim)
        self.attn_base = BaseAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, 
            attn_drop=attn_drop, proj_drop=drop, num_tokens=num_tokens+1, regular_s=regular_s)
        self.norm2 = norm_layer(dim)
        self.mlp1 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, input_x, attn_mask=None):
        hidden_x = input_x + self.drop_path(self.attn_base(self.norm1(input_x), attn_mask))
        output_x = hidden_x + self.drop_path(self.mlp1(self.norm2(hidden_x)))
        return output_x

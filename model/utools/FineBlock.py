import torch
import torch.nn as nn
from model.utools.attn import DropPath, Mlp
from model.utools.attn import BaseBlock


"""
Parameters:
    dim: int, The dimensionality of input feature
    num_heads: int, (default 8). If classifer with self-attention, the number of heads
    mlp_ratio: int, (default 2). The ratio of hidden dimensionality of MLP
    num_joints: int, (default, 25). The number of joint
    num_frames: int, (default, 64). The length of sequence
    num_labels: int, (default, 60). The number of label
    drop: float, (defalut, 0). The inactivation rate of MLP
    attn_drop: float, (default, 0). The inactivation rate of Attention
    qkv_bias: bool, (default, True). The bias of qkv
    qk_scale: tensor, (default, None). The scle of attention, default is dim**-0.5
    drop_path: float, (default, 0). The inactivation rate of Block
    act_layer: nn.functional, (default, nn.GELU). Activation function
    norm_layer: Module, (default, nn.LayerNorm).  Normalization layer
    pre_depth: int, (default, 2). The number of block of classifier
Input:
    x: tensor, the feature of sequence
    num_person: int, (default, 2). The number of person
Output:
    pre: tensor, The class probability 
"""


class ClassifierSTToken(nn.Module):
    """
        Time and space fuse information as input to classification
    """
    def __init__(self, dim, num_heads=8, mlp_ratio=2, num_joints=25, num_frames=64, num_labels=60, 
                 drop=0., attn_drop=0., qkv_bias=True, qk_scale=None, regular_s=False,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, pre_depth=1):
        super().__init__()
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, dim))
        self.temproal_pos_embed = nn.Parameter(torch.zeros(1, num_frames, dim))
        num_tokens = num_joints + num_frames
        dpr = [x.item() for x in torch.linspace(0, drop_path, pre_depth)]
        self.STRe = nn.ModuleList([
            BaseBlock(dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, 
                      num_tokens=num_tokens, regular_s=regular_s,
                      attn_drop=attn_drop, drop_path=dpr[i], act_layer=act_layer, norm_layer=norm_layer) for i in range(pre_depth)
        ])

        self.label_fc = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_labels)
        )

    def forward(self, x, num_person=2):
        n, t, v, c = x.shape
        spatial_embed = x[:, :1, 1:, :].squeeze()
        spatial_embed = spatial_embed + self.spatial_pos_embed

        temporal_embed = x[:, 1:, :1, :].squeeze()
        temporal_embed = temporal_embed + self.temproal_pos_embed

        y = torch.cat([spatial_embed, temporal_embed], dim=1)
        for blk in self.STRe:
            y = blk(y)

        y = y.mean(1)
        y = y.view(n//num_person, num_person, y.size(-1)).mean(1)
        pre = self.label_fc(y) 
        return pre
    

class ClassifierSTAdd(nn.Module):
    """
        Time and space fuse information as input to classification
    """
    def __init__(self, dim, num_heads=8, mlp_ratio=2, num_joints=25, num_frames=64, num_labels=60, 
                 drop=0., attn_drop=0., qkv_bias=True, qk_scale=None, regular_s=False, linear_type='seqence-fc',
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, pre_depth=1):
        super().__init__()
        self.feature_type, self.linear_layer = linear_type.split('-')
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, dim))
        self.temproal_pos_embed = nn.Parameter(torch.zeros(1, num_frames, dim))
        num_tokens = num_joints + num_frames
        dpr = [x.item() for x in torch.linspace(0, drop_path, pre_depth)]
        self.STRe = nn.ModuleList([
            BaseBlock(dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, 
                      num_tokens=num_tokens, regular_s=regular_s,
                      attn_drop=attn_drop, drop_path=dpr[i], act_layer=act_layer, norm_layer=norm_layer) for i in range(pre_depth)
        ])

        if self.linear_layer == 'fc':
            self.seq_fc = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim)
            )
        else:
            self.seq_fc = Mlp(dim, out_features=dim, hidden_features=dim*mlp_ratio, act_layer=act_layer, drop=drop)

        self.label_fc = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_labels)
        )

    def forward(self, x, num_person=2):
        n, t, v, c = x.shape
        spatial_embed = x[:, :1, 1:, :].squeeze()
        spatial_embed = spatial_embed + self.spatial_pos_embed

        temporal_embed = x[:, 1:, :1, :].squeeze()
        temporal_embed = temporal_embed + self.temproal_pos_embed

        seq_x = x[:, :1, :1, :].squeeze() if self.feature_type == 'seqence' else x[:, 1:, 1:, :].mean(2).mean(1)

        y = torch.cat([spatial_embed, temporal_embed], dim=1)
        for blk in self.STRe:
            y = blk(y)
        y = y.mean(1)

        # add seqence information
        seq_y = self.seq_fc(seq_x)
        y = seq_y + y

        y = y.view(n//num_person, num_person, y.size(-1)).mean(1)
        pre = self.label_fc(y) 
        return pre
    

class ClassifierSTAddScore(nn.Module):
    """
        Time and space fuse information as input to classification
    """
    def __init__(self, dim, num_heads=8, mlp_ratio=2, num_joints=25, num_frames=64, num_labels=60, 
                 drop=0., attn_drop=0., qkv_bias=True, qk_scale=None, regular_s=False, linear_type='sequence',
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, pre_depth=1):
        super().__init__()
        self.feature_type, _ = linear_type.split('-')
        assert self.feature_type in ['sequence', 'raw']
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, dim))
        self.temproal_pos_embed = nn.Parameter(torch.zeros(1, num_frames, dim))
        num_tokens = num_joints + num_frames
        dpr = [x.item() for x in torch.linspace(0, drop_path, pre_depth)]
        self.STRe = nn.ModuleList([
            BaseBlock(dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, 
                      num_tokens=num_tokens, regular_s=regular_s,
                      attn_drop=attn_drop, drop_path=dpr[i], act_layer=act_layer, norm_layer=norm_layer) for i in range(pre_depth)
        ])

        self.seq_fc = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_labels)
        )


        self.label_fc = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_labels)
        )

    def forward(self, x, num_person=2):
        n, t, v, c = x.shape
        spatial_embed = x[:, :1, 1:, :].squeeze()
        spatial_embed = spatial_embed + self.spatial_pos_embed

        temporal_embed = x[:, 1:, :1, :].squeeze()
        temporal_embed = temporal_embed + self.temproal_pos_embed

        seq_x = x[:, :1, :1, :].squeeze() if self.feature_type == 'seqence' else x[:, 1:, 1:, :].mean(2).mean(1)

        y = torch.cat([spatial_embed, temporal_embed], dim=1)
        for blk in self.STRe:
            y = blk(y)
        y = y.mean(1)

        y = y.view(n//num_person, num_person, y.size(-1)).mean(1)

        fussion_pre = self.label_fc(y)

        # add information
        seq_x = seq_x.view(n//num_person, num_person, seq_x.size(-1)).mean(1)
        seqence_pre = self.seq_fc(seq_x)

        return fussion_pre + seqence_pre
    

class ClassifierSTSeqRaw(nn.Module):
    """
        Time and space fuse information as input to classification
    """
    def __init__(self, dim, num_heads=8, mlp_ratio=2, num_joints=25, num_frames=64, num_labels=60, 
                 drop=0., attn_drop=0., qkv_bias=True, qk_scale=None, regular_s=False,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, pre_depth=1):
        super().__init__()
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, dim))
        self.temproal_pos_embed = nn.Parameter(torch.zeros(1, num_frames, dim))
        num_tokens = num_joints + num_frames
        dpr = [x.item() for x in torch.linspace(0, drop_path, pre_depth)]
        self.STRe = nn.ModuleList([
            BaseBlock(dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, 
                      num_tokens=num_tokens, regular_s=regular_s,
                      attn_drop=attn_drop, drop_path=dpr[i], act_layer=act_layer, norm_layer=norm_layer) for i in range(pre_depth)
        ])

        self.seq_fc = Mlp(dim, out_features=dim, hidden_features=dim*mlp_ratio, act_layer=act_layer, drop=drop)

        self.label_fc = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_labels)
        )

    def forward(self, x, num_person=2):
        n, t, v, c = x.shape
        spatial_embed = x[:, :1, 1:, :].squeeze()
        spatial_embed = spatial_embed + self.spatial_pos_embed

        temporal_embed = x[:, 1:, :1, :].squeeze()
        temporal_embed = temporal_embed + self.temproal_pos_embed

        seq_x = x[:, :1, :1, :].squeeze()

        y = torch.cat([spatial_embed, temporal_embed], dim=1)
        for blk in self.STRe:
            y = blk(y)
        y = y.mean(1)

        # add seqence information
        seq_y = self.seq_fc(seq_x)
        y = seq_y + y

        y = y.view(n//num_person, num_person, y.size(-1)).mean(1)
        pre = self.label_fc(y) 
        return pre
    


class ClassifierGlobal(nn.Module):
    """
        The global average of sequence features is used as input for classification
    """
    def __init__(self, dim, num_labels=60):
        super().__init__()

        self.label_fc = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_labels)
        )

    def forward(self, x, num_person=2):
        n, t, v, c = x.shape
        x = x[:, 1:, 1:, :]
        x = x.mean(1).mean(1)
        x = x.view(n//num_person, num_person, x.size(-1)).mean(1)
        pre = self.label_fc(x) 
        return pre


class ClassifierSSTGlobal(nn.Module):
    """
        The global average of sequence features is used as input for classification
    """
    def __init__(self, dim, num_labels=60):
        super().__init__()

        self.label_fc = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_labels)
        )

    def forward(self, x, num_person=2):
        n, t, v, c = x.shape
        spatial_embed = x[:, :1, 1:, :].squeeze()
        temporal_embed = x[:, 1:, :1, :].squeeze()
        sequence_embed = x[:, :1, :1, :].squeeze().unsqueeze(1)
        y = torch.cat([sequence_embed, spatial_embed, temporal_embed], dim=1).mean(1)
        y = y.view(n//num_person, num_person, c).mean(1)
        pre = self.label_fc(y) 
        return pre


class ClassifierThrid(nn.Module):
    """
        The global average of sequence features is used as input for classification
    """
    def __init__(self, dim, num_labels=60):
        super().__init__()

        self.seq_fc = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_labels)
        )
        self.spatial_fc = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_labels)
        )
        self.tmp_fc = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_labels)
        )

    def forward(self, x, num_person=2):
        n, t, v, c = x.shape
        spatial_embed = x[:, :1, 1:, :].squeeze().mean(1)
        temporal_embed = x[:, 1:, :1, :].squeeze().mean(1)
        sequence_embed = x[:, :1, :1, :].squeeze()
        
        sequence_embed = sequence_embed.view(n//num_person, num_person, c).mean(1)
        pre_seq = self.seq_fc(sequence_embed) 

        spatial_embed = spatial_embed.view(n//num_person, num_person, c).mean(1)
        pre_spa = self.spatial_fc(spatial_embed) 

        temporal_embed = temporal_embed.view(n//num_person, num_person, c).mean(1)
        pre_tmp = self.tmp_fc(temporal_embed) 

        pre = pre_seq + pre_spa + pre_tmp
        
        return pre


class ClassifierThridV2(nn.Module):
    """
        The global average of sequence features is used as input for classification
    """
    def __init__(self, dim, num_labels=60):
        super().__init__()

        self.label_fc = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_labels)
        )

    def forward(self, x, num_person=2):
        n, t, v, c = x.shape
        spatial_embed = x[:, :1, 1:, :].squeeze().mean(1)
        temporal_embed = x[:, 1:, :1, :].squeeze().mean(1)
        sequence_embed = x[:, :1, :1, :].squeeze()

        y = sequence_embed + temporal_embed + spatial_embed
        
        y = y.view(n//num_person, num_person, c).mean(1)
        pre = self.label_fc(y) 
        return pre

    

class ClassifierSSTToken(nn.Module):
    """
        Sequence, time and space fuse information as input to classification
    """
    def __init__(self, dim, num_heads=8, mlp_ratio=2, num_joints=25, num_frames=64, num_labels=60, 
                 drop=0., attn_drop=0., qkv_bias=True, qk_scale=None, regular_s=False,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, pre_depth=1):
        super().__init__()
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, dim))
        self.temproal_pos_embed = nn.Parameter(torch.zeros(1, num_frames, dim))
        num_tokens = 1 + num_joints + num_frames
        dpr = [x.item() for x in torch.linspace(0, drop_path, pre_depth)]
        self.STRe = nn.ModuleList([
            BaseBlock(dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, 
                      num_tokens=num_tokens, regular_s=regular_s,
                      attn_drop=attn_drop, drop_path=dpr[i], act_layer=act_layer, norm_layer=norm_layer) for i in range(pre_depth)
        ])

        self.label_fc = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_labels)
        )

    def forward(self, x, num_person=2):
        n, t, v, c = x.shape
        spatial_embed = x[:, :1, 1:, :].squeeze()
        spatial_embed = spatial_embed + self.spatial_pos_embed

        temporal_embed = x[:, 1:, :1, :].squeeze()
        temporal_embed = temporal_embed + self.temproal_pos_embed

        sequence_embed = x[:, :1, :1, :].squeeze().unsqueeze(1)

        y = torch.cat([sequence_embed, spatial_embed, temporal_embed], dim=1)
        for blk in self.STRe:
            y = blk(y)

        y = y.mean(1)
        y = y.view(n//num_person, num_person, y.size(-1)).mean(1)
        pre = self.label_fc(y) 
        return pre

    
class ClassifierSTSeparate(nn.Module):
    """
    Spatial and Temporal fussion feature to Label 
    """
    def __init__(self, dim, num_heads=8, mlp_ratio=2, num_joints=25, num_frames=64, num_labels=60, 
                 drop=0., attn_drop=0., qkv_bias=True, qk_scale=None, regular_s=False,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, pre_depth=1):
        super().__init__()
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, dim))
        self.temproal_pos_embed = nn.Parameter(torch.zeros(1, num_frames, dim))
        dpr = [x.item() for x in torch.linspace(0, drop_path, pre_depth)]
        self.SpatialRe = nn.ModuleList([
            BaseBlock(dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, 
                      num_tokens=num_joints, regular_s=regular_s,
                      attn_drop=attn_drop, drop_path=dpr[i], act_layer=act_layer, norm_layer=norm_layer) for i in range(pre_depth)
        ])

        self.TemporalRe = nn.ModuleList([
            BaseBlock(dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, 
                      num_tokens=num_frames, regular_s=regular_s,
                      attn_drop=attn_drop, drop_path=dpr[i], act_layer=act_layer, norm_layer=norm_layer) for i in range(pre_depth)
        ])

        self.label_fc = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_labels)
        )
    
    def forward(self, x, num_person=2):
        n = x.size(0)
        spatial_embed = x[:, :1, 1:, :].squeeze()
        spatial_embed = spatial_embed + self.spatial_pos_embed
        for sblk in self.SpatialRe:
            spatial_embed = sblk(spatial_embed)
        spatial_embed = spatial_embed.mean(1)
        spatial_embed = spatial_embed.view(n//num_person, num_person, spatial_embed.size(-1)).mean(1)

        temporal_embed = x[:, 1:, :1, :].squeeze()
        temporal_embed = temporal_embed + self.temproal_pos_embed
        for tblk in self.TemporalRe:
            temporal_embed = tblk(temporal_embed)
        temporal_embed = temporal_embed.mean(1)
        temporal_embed = temporal_embed.view(n//num_person, num_person, temporal_embed.size(-1)).mean(1)

        st_embed = spatial_embed + temporal_embed
        pre = self.label_fc(st_embed)
        return pre
    

class ClassifierSTSeparateV2(nn.Module):
    """
    Spatial and Temporal fussion feature to Label 
    """
    def __init__(self, dim, num_heads=8, mlp_ratio=2, num_joints=25, num_frames=64, num_labels=60, 
                 drop=0., attn_drop=0., qkv_bias=True, qk_scale=None, regular_s=False,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, pre_depth=1):
        super().__init__()
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, dim))
        self.temproal_pos_embed = nn.Parameter(torch.zeros(1, num_frames, dim))
        dpr = [x.item() for x in torch.linspace(0, drop_path, pre_depth)]
        self.SpatialRe = nn.ModuleList([
            BaseBlock(dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, 
                      num_tokens=num_joints, regular_s=regular_s,
                      attn_drop=attn_drop, drop_path=dpr[i], act_layer=act_layer, norm_layer=norm_layer) for i in range(pre_depth)
        ])

        self.TemporalRe = nn.ModuleList([
            BaseBlock(dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, 
                      num_tokens=num_frames, regular_s=regular_s,
                      attn_drop=attn_drop, drop_path=dpr[i], act_layer=act_layer, norm_layer=norm_layer) for i in range(pre_depth)
        ])

        self.spatial_fc = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_labels)
        )
        self.temporal_fc = nn.Sequential(
            nn.LayerNorm(dim), 
            nn.Linear(dim, num_labels)
        )
    
    def forward(self, x, num_person=2):
        n = x.size(0)
        spatial_embed = x[:, :1, 1:, :].squeeze()
        spatial_embed = spatial_embed + self.spatial_pos_embed
        for sblk in self.SpatialRe:
            spatial_embed = sblk(spatial_embed)
        spatial_embed = spatial_embed.mean(1)
        spatial_embed = spatial_embed.view(n//num_person, num_person, spatial_embed.size(-1)).mean(1)
        spatial_pre = self.spatial_fc(spatial_embed)

        temporal_embed = x[:, 1:, :1, :].squeeze()
        temporal_embed = temporal_embed + self.temproal_pos_embed
        for tblk in self.TemporalRe:
            temporal_embed = tblk(temporal_embed)
        temporal_embed = temporal_embed.mean(1)
        temporal_embed = temporal_embed.view(n//num_person, num_person, temporal_embed.size(-1)).mean(1)
        temporal_pre = self.temporal_fc(temporal_embed)

        pre = spatial_pre + temporal_pre
        return pre
    

class ClassifierSTSeparateV3(nn.Module):
    """
    Spatial and Temporal fussion feature to Label 
    """
    def __init__(self, dim, num_heads=8, mlp_ratio=2, num_joints=25, num_frames=64, num_labels=60, 
                 drop=0., attn_drop=0., qkv_bias=True, qk_scale=None, regular_s=False,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, pre_depth=1):
        super().__init__()
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, dim))
        self.temproal_pos_embed = nn.Parameter(torch.zeros(1, num_frames, dim))
        dpr = [x.item() for x in torch.linspace(0, drop_path, pre_depth)]
        self.SpatialRe = nn.ModuleList([
            BaseBlock(dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, 
                      num_tokens=num_joints, regular_s=regular_s,
                      attn_drop=attn_drop, drop_path=dpr[i], act_layer=act_layer, norm_layer=norm_layer) for i in range(pre_depth)
        ])

        self.TemporalRe = nn.ModuleList([
            BaseBlock(dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, 
                      num_tokens=num_frames, regular_s=regular_s,
                      attn_drop=attn_drop, drop_path=dpr[i], act_layer=act_layer, norm_layer=norm_layer) for i in range(pre_depth)
        ])

        self.spatial_fc = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_labels)
        )
        self.temporal_fc = nn.Sequential(
            nn.LayerNorm(dim), 
            nn.Linear(dim, num_labels)
        )

        self.sequence_fc = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_labels)
        )
    
    def forward(self, x, num_person=2):
        n = x.size(0)
        spatial_embed = x[:, :1, 1:, :].squeeze()
        spatial_embed = spatial_embed + self.spatial_pos_embed
        for sblk in self.SpatialRe:
            spatial_embed = sblk(spatial_embed)
        spatial_embed = spatial_embed.mean(1)
        spatial_embed = spatial_embed.view(n//num_person, num_person, spatial_embed.size(-1)).mean(1)
        spatial_pre = self.spatial_fc(spatial_embed)

        temporal_embed = x[:, 1:, :1, :].squeeze()
        temporal_embed = temporal_embed + self.temproal_pos_embed
        for tblk in self.TemporalRe:
            temporal_embed = tblk(temporal_embed)
        temporal_embed = temporal_embed.mean(1)
        temporal_embed = temporal_embed.view(n//num_person, num_person, temporal_embed.size(-1)).mean(1)
        temporal_pre = self.temporal_fc(temporal_embed)

        sequence_embed = x[:, :1, :1, :].squeeze()
        sequence_embed = sequence_embed.view(n//num_person, num_person, sequence_embed.size(-1)).mean(1)
        sequence_pre = self.sequence_fc(sequence_embed)

        pre = spatial_pre + temporal_pre + sequence_pre
        return pre


class ClassifierSTSeparateV4(nn.Module):
    """
    Spatial and Temporal fussion feature to Label 
    """
    def __init__(self, dim, num_heads=8, mlp_ratio=2, num_joints=25, num_frames=64, num_labels=60, 
                 drop=0., attn_drop=0., qkv_bias=True, qk_scale=None, regular_s=False,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, pre_depth=1):
        super().__init__()
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, dim))
        self.temproal_pos_embed = nn.Parameter(torch.zeros(1, num_frames, dim))
        dpr = [x.item() for x in torch.linspace(0, drop_path, pre_depth)]
        self.SpatialRe = nn.ModuleList([
            BaseBlock(dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, 
                      num_tokens=num_joints, regular_s=True,
                      attn_drop=attn_drop, drop_path=dpr[i], act_layer=act_layer, norm_layer=norm_layer) for i in range(pre_depth)
        ])

        self.TemporalRe = nn.ModuleList([
            BaseBlock(dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, 
                      num_tokens=num_frames, regular_s=regular_s,
                      attn_drop=attn_drop, drop_path=dpr[i], act_layer=act_layer, norm_layer=norm_layer) for i in range(pre_depth)
        ])

        self.spatial_fc = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_labels)
        )
        self.temporal_fc = nn.Sequential(
            nn.LayerNorm(dim), 
            nn.Linear(dim, num_labels)
        )

        self.sequence_fc = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_labels)
        )
    
    def forward(self, x, num_person=2):
        n = x.size(0)
        spatial_embed = x[:, :1, 1:, :].squeeze()
        spatial_embed = spatial_embed + self.spatial_pos_embed
        for sblk in self.SpatialRe:
            spatial_embed = sblk(spatial_embed)
        spatial_embed = spatial_embed.mean(1)
        spatial_embed = spatial_embed.view(n//num_person, num_person, spatial_embed.size(-1)).mean(1)
        spatial_pre = self.spatial_fc(spatial_embed)

        temporal_embed = x[:, 1:, :1, :].squeeze()
        temporal_embed = temporal_embed + self.temproal_pos_embed
        for tblk in self.TemporalRe:
            temporal_embed = tblk(temporal_embed)
        temporal_embed = temporal_embed.mean(1)
        temporal_embed = temporal_embed.view(n//num_person, num_person, temporal_embed.size(-1)).mean(1)
        temporal_pre = self.temporal_fc(temporal_embed)

        sequence_embed = x[:, :1, :1, :].squeeze()
        sequence_embed = sequence_embed.view(n//num_person, num_person, sequence_embed.size(-1)).mean(1)
        sequence_pre = self.sequence_fc(sequence_embed)

        pre = spatial_pre + temporal_pre + sequence_pre
        return pre


class ClassifierSTSeparateV5(nn.Module):
    """
    Spatial and Temporal fussion feature to Label 
    """
    def __init__(self, dim, num_heads=8, mlp_ratio=2, num_joints=25, num_frames=64, num_labels=60, 
                 drop=0., attn_drop=0., qkv_bias=True, qk_scale=None, regular_s=False,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, pre_depth=1):
        super().__init__()
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, dim))
        self.temproal_pos_embed = nn.Parameter(torch.zeros(1, num_frames, dim))
        dpr = [x.item() for x in torch.linspace(0, drop_path, pre_depth)]
        self.SpatialRe = nn.ModuleList([
            BaseBlock(dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, 
                      num_tokens=num_joints, regular_s=True,
                      attn_drop=attn_drop, drop_path=dpr[i], act_layer=act_layer, norm_layer=norm_layer) for i in range(pre_depth)
        ])

        self.TemporalRe = nn.ModuleList([
            BaseBlock(dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, 
                      num_tokens=num_frames, regular_s=True,
                      attn_drop=attn_drop, drop_path=dpr[i], act_layer=act_layer, norm_layer=norm_layer) for i in range(pre_depth)
        ])

        self.spatial_fc = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_labels)
        )
        self.temporal_fc = nn.Sequential(
            nn.LayerNorm(dim), 
            nn.Linear(dim, num_labels)
        )

        self.sequence_fc = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_labels)
        )
    
    def forward(self, x, num_person=2):
        n = x.size(0)
        spatial_embed = x[:, :1, 1:, :].squeeze()
        spatial_embed = spatial_embed + self.spatial_pos_embed
        for sblk in self.SpatialRe:
            spatial_embed = sblk(spatial_embed)
        spatial_embed = spatial_embed.mean(1)
        spatial_embed = spatial_embed.view(n//num_person, num_person, spatial_embed.size(-1)).mean(1)
        spatial_pre = self.spatial_fc(spatial_embed)

        temporal_embed = x[:, 1:, :1, :].squeeze()
        temporal_embed = temporal_embed + self.temproal_pos_embed
        for tblk in self.TemporalRe:
            temporal_embed = tblk(temporal_embed)
        temporal_embed = temporal_embed.mean(1)
        temporal_embed = temporal_embed.view(n//num_person, num_person, temporal_embed.size(-1)).mean(1)
        temporal_pre = self.temporal_fc(temporal_embed)

        sequence_embed = x[:, :1, :1, :].squeeze()
        sequence_embed = sequence_embed.view(n//num_person, num_person, sequence_embed.size(-1)).mean(1)
        sequence_pre = self.sequence_fc(sequence_embed)

        pre = spatial_pre + temporal_pre + sequence_pre
        return pre


class ClassifierSTSeparateV6(nn.Module):
    """
    Spatial and Temporal fussion feature to Label 
    """
    def __init__(self, dim, num_heads=8, mlp_ratio=2, num_joints=25, num_frames=64, num_labels=60, 
                 drop=0., attn_drop=0., qkv_bias=True, qk_scale=None, regular_s=False,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, pre_depth=1):
        super().__init__()
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, dim))
        self.temproal_pos_embed = nn.Parameter(torch.zeros(1, num_frames, dim))
        dpr = [x.item() for x in torch.linspace(0, drop_path, pre_depth)]
        self.SpatialRe = nn.ModuleList([
            BaseBlock(dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, 
                      num_tokens=num_joints, regular_s=False,
                      attn_drop=attn_drop, drop_path=dpr[i], act_layer=act_layer, norm_layer=norm_layer) for i in range(pre_depth)
        ])

        self.TemporalRe = nn.ModuleList([
            BaseBlock(dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, 
                      num_tokens=num_frames, regular_s=False,
                      attn_drop=attn_drop, drop_path=dpr[i], act_layer=act_layer, norm_layer=norm_layer) for i in range(pre_depth)
        ])

        self.spatial_fc = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_labels)
        )
        self.temporal_fc = nn.Sequential(
            nn.LayerNorm(dim), 
            nn.Linear(dim, num_labels)
        )

        self.sequence_fc = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_labels)
        )
    
    def forward(self, x, num_person=2):
        n = x.size(0)
        spatial_embed = x[:, :1, 1:, :].squeeze()
        spatial_embed = spatial_embed + self.spatial_pos_embed
        for sblk in self.SpatialRe:
            spatial_embed = sblk(spatial_embed)
        spatial_embed = spatial_embed.mean(1)
        spatial_embed = spatial_embed.view(n//num_person, num_person, spatial_embed.size(-1)).mean(1)
        spatial_pre = self.spatial_fc(spatial_embed)

        temporal_embed = x[:, 1:, :1, :].squeeze()
        temporal_embed = temporal_embed + self.temproal_pos_embed
        for tblk in self.TemporalRe:
            temporal_embed = tblk(temporal_embed)
        temporal_embed = temporal_embed.mean(1)
        temporal_embed = temporal_embed.view(n//num_person, num_person, temporal_embed.size(-1)).mean(1)
        temporal_pre = self.temporal_fc(temporal_embed)

        sequence_embed = x[:, :1, :1, :].squeeze()
        sequence_embed = sequence_embed.view(n//num_person, num_person, sequence_embed.size(-1)).mean(1)
        sequence_pre = self.sequence_fc(sequence_embed)

        pre = spatial_pre + temporal_pre + sequence_pre
        return pre


class ClassifierSTSeparateV7(nn.Module):
    """
    Spatial and Temporal fussion feature to Label 
    """
    def __init__(self, dim, num_heads=8, mlp_ratio=2, num_joints=25, num_frames=64, num_labels=60, 
                 drop=0., attn_drop=0., qkv_bias=True, qk_scale=None, regular_s=False,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, pre_depth=1):
        super().__init__()
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, dim))
        self.temproal_pos_embed = nn.Parameter(torch.zeros(1, num_frames, dim))
        dpr = [x.item() for x in torch.linspace(0, drop_path, pre_depth)]
        self.SpatialRe = nn.ModuleList([
            BaseBlock(dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, 
                      num_tokens=num_joints, regular_s=False,
                      attn_drop=attn_drop, drop_path=dpr[i], act_layer=act_layer, norm_layer=norm_layer) for i in range(pre_depth)
        ])

        self.TemporalRe = nn.ModuleList([
            BaseBlock(dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, 
                      num_tokens=num_frames, regular_s=True,
                      attn_drop=attn_drop, drop_path=dpr[i], act_layer=act_layer, norm_layer=norm_layer) for i in range(pre_depth)
        ])

        self.spatial_fc = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_labels)
        )
        self.temporal_fc = nn.Sequential(
            nn.LayerNorm(dim), 
            nn.Linear(dim, num_labels)
        )

        self.sequence_fc = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_labels)
        )
    
    def forward(self, x, num_person=2):
        n = x.size(0)
        spatial_embed = x[:, :1, 1:, :].squeeze()
        spatial_embed = spatial_embed + self.spatial_pos_embed
        for sblk in self.SpatialRe:
            spatial_embed = sblk(spatial_embed)
        spatial_embed = spatial_embed.mean(1)
        spatial_embed = spatial_embed.view(n//num_person, num_person, spatial_embed.size(-1)).mean(1)
        spatial_pre = self.spatial_fc(spatial_embed)

        temporal_embed = x[:, 1:, :1, :].squeeze()
        temporal_embed = temporal_embed + self.temproal_pos_embed
        for tblk in self.TemporalRe:
            temporal_embed = tblk(temporal_embed)
        temporal_embed = temporal_embed.mean(1)
        temporal_embed = temporal_embed.view(n//num_person, num_person, temporal_embed.size(-1)).mean(1)
        temporal_pre = self.temporal_fc(temporal_embed)

        sequence_embed = x[:, :1, :1, :].squeeze()
        sequence_embed = sequence_embed.view(n//num_person, num_person, sequence_embed.size(-1)).mean(1)
        sequence_pre = self.sequence_fc(sequence_embed)

        pre = spatial_pre + temporal_pre + sequence_pre
        return pre

    
class ClassifierSpatial(nn.Module):
    """
    Spatial and Temporal fussion feature to Label 
    """
    def __init__(self, dim, num_heads=8, mlp_ratio=2, num_joints=25, num_frames=64, num_labels=60, 
                 drop=0., attn_drop=0., qkv_bias=True, qk_scale=None, regular_s=False,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, pre_depth=1):
        super().__init__()
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, dim))
        dpr = [x.item() for x in torch.linspace(0, drop_path, pre_depth)]
        self.SpatialRe = nn.ModuleList([
            BaseBlock(dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, 
                      num_tokens=num_joints, regular_s=regular_s,
                      attn_drop=attn_drop, drop_path=dpr[i], act_layer=act_layer, norm_layer=norm_layer) for i in range(pre_depth)
        ])

        self.spatial_fc = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_labels)
        )

    
    def forward(self, x, num_person=2):
        n = x.size(0)
        # spatial_embed = x[:, :1, 1:, :].squeeze()   # 有序列融合时，v要减一
        spatial_embed = x[:, :1, :, :].squeeze()
        spatial_embed = spatial_embed + self.spatial_pos_embed
        for sblk in self.SpatialRe:
            spatial_embed = sblk(spatial_embed)
        spatial_embed = spatial_embed.mean(1)
        spatial_embed = spatial_embed.view(n//num_person, num_person, spatial_embed.size(-1)).mean(1)
        spatial_pre = self.spatial_fc(spatial_embed)

        return spatial_pre
    
class ClassifierTemporal(nn.Module):
    """
    Spatial and Temporal fussion feature to Label 
    """
    def __init__(self, dim, num_heads=8, mlp_ratio=2, num_joints=25, num_frames=64, num_labels=60, 
                 drop=0., attn_drop=0., qkv_bias=True, qk_scale=None, regular_s=False,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, pre_depth=1):
        super().__init__()
        self.temproal_pos_embed = nn.Parameter(torch.zeros(1, num_frames, dim))
        dpr = [x.item() for x in torch.linspace(0, drop_path, pre_depth)]
        self.TemporalRe = nn.ModuleList([
            BaseBlock(dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, 
                      num_tokens=num_frames, regular_s=regular_s,
                      attn_drop=attn_drop, drop_path=dpr[i], act_layer=act_layer, norm_layer=norm_layer) for i in range(pre_depth)
        ])

        self.temporal_fc = nn.Sequential(
            nn.LayerNorm(dim), 
            nn.Linear(dim, num_labels)
        )
    
    def forward(self, x, num_person=2):
        n = x.size(0)
        # temporal_embed = x[:, 1:, :1, :].squeeze()
        temporal_embed = x[:, :, :1, :].squeeze()
        temporal_embed = temporal_embed + self.temproal_pos_embed
        for tblk in self.TemporalRe:
            temporal_embed = tblk(temporal_embed)
        temporal_embed = temporal_embed.mean(1)
        temporal_embed = temporal_embed.view(n//num_person, num_person, temporal_embed.size(-1)).mean(1)
        temporal_pre = self.temporal_fc(temporal_embed)

        return temporal_pre


class ClassifierSeqence(nn.Module):
    """
        The global average of sequence features is used as input for classification
    """
    def __init__(self, dim, num_labels=60):
        super().__init__()

        self.label_fc = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_labels)
        )

    def forward(self, x, num_person=2):
        n, t, v, c = x.shape
        x = x[:, :1, :1, :].squeeze()
        x = x.view(n//num_person, num_person, x.size(-1)).mean(1)
        pre = self.label_fc(x) 
        return pre
    

class ClassifierHead(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=2, num_joints=25, num_frames=64, num_labels=60, 
                 drop=0., attn_drop=0., qkv_bias=True, qk_scale=None, regular_s=False, head_type='st', add_type='seqence-fc',
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, pre_depth=1):
        super().__init__()
        if head_type == 'st':
            self.head = ClassifierSTToken(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, num_joints=num_joints, num_frames=num_frames,
                                          num_labels=num_labels, drop=drop, attn_drop=attn_drop, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                                          regular_s=regular_s, drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer,
                                          pre_depth=pre_depth)
        elif head_type == 'stadd':
            self.head = ClassifierSTAdd(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, num_joints=num_joints, num_frames=num_frames, 
                                        num_labels=num_labels, drop=drop, attn_drop=attn_drop, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                                        regular_s=regular_s, drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer, linear_type=add_type,
                                        pre_depth=pre_depth)
        elif head_type == 'staddsc':
            self.head = ClassifierSTAddScore(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, num_joints=num_joints, num_frames=num_frames, 
                                             num_labels=num_labels, drop=drop, attn_drop=attn_drop, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                                             regular_s=regular_s, drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer, linear_type=add_type, 
                                             pre_depth=pre_depth)
        elif head_type == 'sts':
            self.head = ClassifierSTSeparate(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, num_joints=num_joints, num_frames=num_frames, 
                                             num_labels=num_labels, drop=drop, attn_drop=attn_drop, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                                             regular_s=regular_s, drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer, 
                                             pre_depth=pre_depth)
        elif head_type == 'stsv2':
            self.head = ClassifierSTSeparateV2(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, num_joints=num_joints, num_frames=num_frames, 
                                               num_labels=num_labels, drop=drop, attn_drop=attn_drop, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                                               regular_s=regular_s, drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer, 
                                               pre_depth=pre_depth)
        elif head_type == 'stsv3':
            self.head = ClassifierSTSeparateV3(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, num_joints=num_joints, num_frames=num_frames, 
                                               num_labels=num_labels, drop=drop, attn_drop=attn_drop, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                                               regular_s=regular_s, drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer, 
                                               pre_depth=pre_depth)
        elif head_type == 'stsv4':
            self.head = ClassifierSTSeparateV4(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, num_joints=num_joints, num_frames=num_frames, 
                                               num_labels=num_labels, drop=drop, attn_drop=attn_drop, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                                               regular_s=regular_s, drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer, 
                                               pre_depth=pre_depth)
        elif head_type == 'stsv5':
            self.head = ClassifierSTSeparateV5(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, num_joints=num_joints, num_frames=num_frames, 
                                               num_labels=num_labels, drop=drop, attn_drop=attn_drop, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                                               regular_s=regular_s, drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer, 
                                               pre_depth=pre_depth)
        elif head_type == 'stsv6':
            self.head = ClassifierSTSeparateV6(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, num_joints=num_joints, num_frames=num_frames, 
                                               num_labels=num_labels, drop=drop, attn_drop=attn_drop, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                                               regular_s=regular_s, drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer, 
                                               pre_depth=pre_depth)
        elif head_type == 'stsv7':
            self.head = ClassifierSTSeparateV7(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, num_joints=num_joints, num_frames=num_frames, 
                                               num_labels=num_labels, drop=drop, attn_drop=attn_drop, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                                               regular_s=regular_s, drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer, 
                                               pre_depth=pre_depth)
        elif head_type == 'spa':
            self.head = ClassifierSpatial(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, num_joints=num_joints, num_frames=num_frames, 
                                               num_labels=num_labels, drop=drop, attn_drop=attn_drop, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                                               regular_s=False, drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer, 
                                               pre_depth=pre_depth)
        elif head_type == 'spareg':
            self.head = ClassifierSpatial(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, num_joints=num_joints, num_frames=num_frames, 
                                               num_labels=num_labels, drop=drop, attn_drop=attn_drop, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                                               regular_s=True, drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer, 
                                               pre_depth=pre_depth)
        elif head_type == 'tmp':
            self.head = ClassifierTemporal(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, num_joints=num_joints, num_frames=num_frames, 
                                               num_labels=num_labels, drop=drop, attn_drop=attn_drop, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                                               regular_s=False, drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer, 
                                               pre_depth=pre_depth)
        elif head_type == 'tmpreg':
            self.head = ClassifierTemporal(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, num_joints=num_joints, num_frames=num_frames, 
                                               num_labels=num_labels, drop=drop, attn_drop=attn_drop, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                                               regular_s=True, drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer, 
                                               pre_depth=pre_depth)
        elif head_type == 'sst':
            self.head = ClassifierSSTToken(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, num_joints=num_joints, num_frames=num_frames, 
                                          num_labels=num_labels, drop=drop, attn_drop=attn_drop, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                                          regular_s=regular_s, drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer, 
                                          pre_depth=pre_depth)
        elif head_type == 'global':
            self.head = ClassifierGlobal(dim=dim, num_labels=num_labels)
        elif head_type == 's':
            self.head = ClassifierSeqence(dim=dim, num_labels=num_labels)
        elif head_type == 'sstglobal':
            self.head = ClassifierSSTGlobal(dim=dim, num_labels=num_labels)
        elif head_type == 'fusionFC':
            self.head = ClassifierThrid(dim=dim, num_labels=num_labels)
        elif head_type == 'fusionFCV2':
            self.head = ClassifierThridV2(dim=dim, num_labels=num_labels)
        else:
            raise ValueError("The Classifier Head Tye Error !!!")
        
    def forward(self, x, num_person=2):
        return self.head(x, num_person)
    
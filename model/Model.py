from functools import partial
import torch
import torch.nn as nn
from model.utools.PretrainBlock import STSerialBlock
from model.utools.FineBlock import ClassifierHead


class Model(nn.Module):
    def __init__(self, in_channels=3, num_frames=64, num_joints=25, embed_dim_ratio=32, temporal_patch=4, 
                 en_depth=4, 
                 num_labels=60, pre_depth=1, head_type='st',
                 mode='MAE',
                 input_frames_ratio=0.5, input_joints_ratio=0.5,
                 contrastive=True,
                 mlp_ratio=2., num_heads=4, qkv_bias=True, qk_scale=None, act_layer=nn.GELU, regular_s=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, norm_layer=None, drop_path_re_rate=0.1,
                 loss_type='mse-sim-con'):
        """ 
        Args:
            num_frame (int, tuple): input frame number
            num_joints (int, tuple): joints number
            in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
            embed_dim_ratio (int): embedding dimension ratio
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.temporal_patch = temporal_patch

        self.frames_ratio = input_frames_ratio
        self.joints_ratio = input_joints_ratio

        self.mode = mode

        self.contrastive = contrastive

        self.loss_type = loss_type
        
        dim = embed_dim_ratio
        self.embdedding = nn.Sequential(
            nn.Conv2d(in_channels, dim, 1),
            nn.BatchNorm2d(dim),
            act_layer()
        )
        
        # num_frames = int((num_frames - temporal_patch)/temporal_patch + 1)
        # num_joints = int((num_joints - spatial_patch)/spatial_patch + 1)
        num_frames = num_frames // temporal_patch
        num_joints = int(num_joints * temporal_patch)

        # fusion information
        self.Spatial_fusion = nn.Parameter(torch.zeros(1, 1, num_joints, dim))
        self.Temporal_fusion = nn.Parameter(torch.zeros(1, num_frames, 1, dim))
        self.Sequence_fusion = nn.Parameter(torch.zeros([1, 1, 1, dim]))

        self.spatial_pos = nn.Parameter(torch.zeros(1, num_joints, dim))
        self.temproal_pos = nn.Parameter(torch.zeros(1, num_frames, dim))
        self.fussion_pos = nn.Parameter(torch.zeros(1, 1, dim))


        self.spatail_norm = norm_layer(dim)
        self.temporal_norm = norm_layer(dim)


        # Module
        en_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, en_depth)]

        self.ST_encoder = nn.ModuleList([
            STSerialBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                            drop=drop_rate, attn_drop=attn_drop_rate, num_joints=num_joints, num_frames=num_frames, 
                            regular_s=regular_s, drop_path=en_dpr[i], act_layer=act_layer, norm_layer=norm_layer, 
                            spatial_norm=self.spatail_norm, temporal_norm=self.temporal_norm) for i in range(en_depth)
        ])
        
        self.fussion_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, embed_dim_ratio))
        self.spatial_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, embed_dim_ratio))
        self.temporal_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, embed_dim_ratio))

        self.fc = ClassifierHead(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, num_joints=num_joints, num_frames=num_frames, 
                                    num_labels=num_labels, drop=drop_rate, attn_drop=attn_drop_rate, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                                    regular_s=regular_s, head_type=head_type, drop_path=drop_path_re_rate, act_layer=act_layer, 
                                    norm_layer=norm_layer, pre_depth=pre_depth)
        self.initialize_weight()
    
    def initialize_weight(self):
        nn.init.normal_(self.temporal_mask, std=.02)
        nn.init.normal_(self.spatial_mask, std=.02)
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    
    def add_fusion(self, x, spatial_idx=None, temporal_idx=None):
        n = x.size(0)
        spatial_fusion = self.Spatial_fusion.expand(n, -1, -1, -1)  # n, 1, v, c
        temporal_fusion = self.Temporal_fusion.expand(n, -1, -1, -1)  # n, t, 1, c
        sequence_fusion = self.Sequence_fusion.expand(n, -1, -1, -1)  # n, 1, 1, c

        if spatial_idx is not None:
            spatial_fusion = spatial_fusion[:, :, spatial_idx, :]

        if temporal_idx is not None:
            temporal_fusion = temporal_fusion[:, temporal_idx, :, :]

        x = torch.cat([spatial_fusion, x], dim=1)  # n, t+1, v, c
        fusion = torch.cat([sequence_fusion, temporal_fusion], dim=1)  # n, t+1, 1, c
        x = torch.cat([fusion, x], dim=2)   # n, t+1, v+1, c
        return x 
    
    def forward_features(self, x, spatial_pos, temporal_pos):
        for idx, blk in enumerate(self.ST_encoder):
            if idx == 0:
                x = blk(x, spatial_pos, temporal_pos)
            else:
                x = blk(x)

        if self.contrastive:
            seq_fussion = x[:, :1, :1, :].squeeze()
            seq_fussion = self.fussion_head(seq_fussion)

            spatial_fussion = x[:, 1:, :1, :].squeeze()
            spatial_fussion = self.spatial_head(spatial_fussion)

            temporal_fussion = x[:, :1, 1:, :].squeeze()
            temporal_fussion = self.temporal_head(temporal_fussion)
            return x, seq_fussion, spatial_fussion, temporal_fussion 
            
        return x
    
    def pos_forward(self, spatial_idx=None, temporal_idx=None):
        if spatial_idx is not None:
            spatial_pos = torch.cat([self.fussion_pos, self.spatial_pos[:, spatial_idx]], dim=1)
        else:
            spatial_pos = torch.cat([self.fussion_pos, self.spatial_pos], dim=1)

        if temporal_idx is not None:
            temporal_pos = torch.cat([self.fussion_pos, self.temproal_pos[:, temporal_idx]], dim=1)
        else:
            temporal_pos = torch.cat([self.fussion_pos, self.temproal_pos], dim=1)

        return spatial_pos, temporal_pos


    def encoder_forward(self, x):
        n, c, t, v = x.size()
        x = x.view(n, c, t//self.temporal_patch, v*self.temporal_patch).permute(0, 2, 3, 1).contiguous()
        
        x = self.add_fusion(x)
        spatial_pos, temporal_pos = self.pos_forward()
        x = self.forward_features(x, spatial_pos=spatial_pos, temporal_pos=temporal_pos)
        return x

    def forward(self, data):
        n, c, t, v, m = data.size()
        data = data.permute(0, 4, 1, 2, 3).contiguous().view(n*m, c, t, v)

        x = self.embdedding(data)
        x = self.encoder_forward(x)
        pre = self.fc(x, m)

        return pre
    

if __name__ == '__main__':
    a_c = torch.rand([2, 3, 120, 25, 2])
    # embed_dim_rato: 128 --> 2.77     256 --> 
    m = Model(num_frames=120, num_joints=25, temporal_patch=6, en_depth=5, pre_depth=2, embed_dim_ratio=256)
    para = sum(pa.numel() for pa in m.parameters() if pa.requires_grad)
    print("Module Parameters: {:.2f}".format(para/1e6))
    la, (_, _, _) = m(a_c)
    print("Loss: {}".format(la))
    for name, param in m.named_parameters():
            if name.split('.')[0] not in ["ST_encoder"]:
                param.requires_grad = False
    para_e = sum(pa.numel() for pa in m.parameters() if pa.requires_grad)
    print("Encoder Module Parameters: {:.2f}".format(para_e/1e6))
    

# point_transformer_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PointTransformerLayer(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=2.0, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        # Transform input into Q, K, V
        self.qkv = nn.Linear(dim, dim*3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # MLP after attention
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(proj_drop)
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, pos):
        # x: [B, N, D], pos: [B, N, 2 or similar]
        B, N, D = x.shape

        # Compute Q,K,V
        qkv = self.qkv(self.norm1(x)) # [B, N, 3D]
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: [B, heads, N, head_dim]

        # Compute positional offsets
        # pos: [B, N, 2]
        # shape: [B, N, N, 2]
        rel_pos = pos.unsqueeze(2) - pos.unsqueeze(1)  # [B, N, N, 2]
        dist = torch.sum(rel_pos**2, dim=-1)  # [B, N, N]
        dist = torch.exp(-dist)  # a simple Gaussian kernel on distance
        # dist is now [B, N, N], we can incorporate into attention

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale  # [B, heads, N, N]
        # Incorporate a positional bias from dist:
        attn = attn + dist.unsqueeze(1) # broadcast over heads dimension

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        out = attn @ v  # [B, heads, N, head_dim]
        out = out.transpose(1,2).reshape(B, N, D)
        out = self.proj(out)
        out = self.proj_drop(out)

        # Residual + MLP
        x = x + out
        x = x + self.mlp(self.norm2(x))

        return x


class PointTransformerClassifier(nn.Module):
    def __init__(
        self,
        num_classes=10,
        seq_len=128,
        input_dim=8,
        embed_dim=64,
        depth=4,
        mlp_ratio=4.0,
        num_heads=4,
        drop=0.0,
        attn_drop=0.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        # data: [B, N, 8] -> xy: [B, N, 2], feats: [B, N, 6]
        self.coord_dim = 2
        self.feat_dim = input_dim - self.coord_dim

        self.feat_proj = nn.Linear(self.feat_dim, embed_dim)

        self.pos_mlp = nn.Sequential(
            nn.Linear(self.coord_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

        self.blocks = nn.ModuleList([
            PointTransformerLayer(
                dim=embed_dim, 
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio,
                attn_drop=attn_drop,
                proj_drop=drop
                )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x: [B, N, 8]
        B, N, D = x.shape
        coords = x[:, :, :2]  # [B, N, 2]
        feats = x[:, :, 2:]   # [B, N, 6]

        # Project feats
        feats = self.feat_proj(feats)  # [B, N, embed_dim]
        
        pos_emb = self.pos_mlp(coords) # [B, N, embed_dim]
        x = feats + pos_emb

        # Pass through point transformer blocks
        for blk in self.blocks:
            x = blk(x, coords)

        x = self.norm(x)
        # Global pooling: average pooling over points
        x = x.mean(dim=1)  # [B, embed_dim]

        logits = self.head(x) # [B, num_classes]
        return logits

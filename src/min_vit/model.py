import torch
from torch import nn
from min_vit.blocks import TransformerBlock


class MinViT(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, depth=4, n_classes=10):
        super().__init__()
        
        self.pos_emb = nn.Parameter(
            torch.randn(1, 65, embed_dim)
        )
        self.cls_token = nn.Parameter(
            torch.rand(1, 1, embed_dim)
        )
        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=embed_dim,
            kernel_size=4,
            stride=4
        )
        self.tblocks = nn.Sequential(
            *[
                TransformerBlock(
                    embed_dim, num_heads
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(
            embed_dim, n_classes
        )
        
    
    def forward(self, x):
        B = x.shape[0]
        emb = self.conv(x).flatten(2).transpose(1, 2)              # [B, 64, 128]
        cls_token = self.cls_token.expand(B, -1, -1)    
        emb = torch.cat((cls_token, emb), dim=1) + self.pos_emb     # [B, 64+1, 128]
        
        emb = self.tblocks(emb)
        emb = self.norm(emb)
        
        cls = emb[:, 0, :]
        logits = self.head(cls)
        
        return logits

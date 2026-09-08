from torch import nn


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(
            num_heads=num_heads,
            embed_dim=embed_dim,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

        
    def forward(self, x):
        norm_x = self.norm1(x)
        attn_out, _ = self.attention(
            norm_x, norm_x, norm_x
        )
        x = x + attn_out     # residual connection
        
        mlp_out = self.mlp(self.norm2(x)) # обогащаем признаки
        x = x + mlp_out
        
        return x
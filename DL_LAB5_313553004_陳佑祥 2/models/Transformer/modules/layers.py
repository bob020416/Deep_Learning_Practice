import torch.nn as nn
import torch
import math

#TODO1

class MultiHeadAttention(nn.Module):
    def __init__(self, dim=768, num_heads=16, attn_drop=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads  # d_k = d_v = 768 // 16 = 48
        self.scale = self.head_dim ** -0.5

        # Linear layers for projecting x to query, key, value
        self.qkv = nn.Linear(dim, dim * 3, bias=False)  # Output will be (batch_size, num_image_tokens, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)  # Final linear layer after concatenation of all heads

    def forward(self, x):
        batch_size, num_tokens, dim = x.shape

        # Project x to query, key, value
        qkv = self.qkv(x)  # Shape: (batch_size, num_tokens, dim * 3)
        qkv = qkv.reshape(batch_size, num_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # Shape: (3, batch_size, num_heads, num_tokens, head_dim)
        query, key, value = qkv[0], qkv[1], qkv[2]  # Each is now (batch_size, num_heads, num_tokens, head_dim)

        # Scaled dot-product attention
        attn_scores = (query @ key.transpose(-2, -1)) * self.scale  # Shape: (batch_size, num_heads, num_tokens, num_tokens)
        attn_probs = attn_scores.softmax(dim=-1)  # Normalize across the last dimension (num_tokens)
        attn_probs = self.attn_drop(attn_probs)  # Apply dropout

        # Apply attention to the value tensor
        attn_output = attn_probs @ value  # Shape: (batch_size, num_heads, num_tokens, head_dim)

        # Concatenate heads and pass through final linear layer
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, num_tokens, dim)  # (batch_size, num_tokens, dim)
        output = self.proj(attn_output)  # Final linear projection

        return output


class MLP(nn.Sequential):
    def __init__(self, dim=768, hidden_dim=3072, drop_rate=0.1):
        super(MLP, self).__init__(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=0.1)
        )
        
    def forward(self, input):
        return super().forward(input)
    
    
class TokenPredictor(nn.Sequential):
    def __init__(self, dim=768):
        super(TokenPredictor, self).__init__(
            nn.Linear(in_features=dim, out_features=dim),
            nn.GELU(),
            nn.LayerNorm(dim, eps=1e-12)
        )
        
    def forward(self, input):
        return super().forward(input)
    
    
class Encoder(nn.Module):
    def __init__(self, dim=768, hidden_dim=1536):
        super(Encoder, self).__init__()
        self.Attention = MultiHeadAttention(dim)
        self.LayerNorm1 = nn.LayerNorm(dim, eps=1e-12)
        self.LayerNorm2 = nn.LayerNorm(dim, eps=1e-12)
        self.MLP = MLP(dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        attn = self.Attention(x)
        attn = self.dropout(attn)
        
        x = x + attn
        x = self.LayerNorm1(x)
        
        mlp = self.MLP(x)
        x = x + mlp
        return self.LayerNorm2(x)
    
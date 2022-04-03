"""
0) Input: torch.Size([128, 11, 64])
1) K: torch.Size([128, 8, 11, 8]), Q: torch.Size([128, 8, 11, 8]), V: torch.Size([128, 8, 11, 8])
2) Attention: torch.Size([128, 8, 11, 11])
3) Attention: torch.Size([128, 8, 11, 11])
4) Softmax Attention: torch.Size([128, 8, 11, 11])
5) Attention: torch.Size([128, 8, 11, 11])
6) y: torch.Size([128, 8, 11, 8])
7) y: torch.Size([128, 11, 64])
8) Full Connected: torch.Size([128, 11, 64])
9) Dropout: torch.Size([128, 11, 64])


0) Input: torch.Size([128, 10, 64])
1) K: torch.Size([128, 8, 10, 8]), Q: torch.Size([128, 8, 10, 8]), V: torch.Size([128, 8, 10, 8])
2) Attention: torch.Size([128, 8, 10, 10])
3) Attention: torch.Size([128, 8, 10, 10])
4) Softmax Attention: torch.Size([128, 8, 10, 10])
5) Attention: torch.Size([128, 8, 10, 10])
6) y: torch.Size([128, 8, 10, 8])
7) y: torch.Size([128, 10, 64])
8) Full Connected: torch.Size([128, 10, 64])
9) Dropout: torch.Size([128, 10, 64])

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHead_Masked_SelfAttention(nn.Module):
    """
    A vanilla Multi-Head Masked Self-Attention layer with a
    projection at the end. Uncomment #print(f"...")s to observe
    output-flow for given input [batch_size, seq_len, embed_dim].
    """

    def __init__(self, embed_dim=512, num_heads=8, block_size=128, attention_dropout_rate=0.1, residual_dropout_rate=0.1):
        super().__init__()
        # key, query, value projections for all heads
        self.key   = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        # regularization
        self.attention_dropout = nn.Dropout(attention_dropout_rate)
        self.residual_dropout  = nn.Dropout(residual_dropout_rate)
        # output projection
        self.fc = nn.Linear(embed_dim, embed_dim)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))
        self.num_heads = num_heads

    def forward(self, x, layer_past=None):
        B, T, C = x.shape
        #print(f"8) Multi-Head Attention Input: {x.shape}")

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(  B, T, self.num_heads, C//self.num_heads).transpose(1, 2) # (B, num_heads, T, C//num_heads)
        q = self.query(x).view(B, T, self.num_heads, C//self.num_heads).transpose(1, 2) # (B, num_heads, T, C//num_heads)
        v = self.value(x).view(B, T, self.num_heads, C//self.num_heads).transpose(1, 2) # (B, num_heads, T, C//num_heads)
        #print(f"9) K: {k.shape}, Q: {q.shape}, V: {v.shape}")

        # self-attention: (B, num_heads, T, C//num_heads) x (B, num_heads, C//num_heads, T) ===> (B, num_heads, T, T)
        attention = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        #print(f"10) Attention: {attention.shape}")
        attention = attention.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        #print(attention[0][0])
        #print(f"11) Attention: {attention.shape}")
        normalized_attention = F.softmax(attention, dim=-1)
        #print(normalized_attention[0][0])
        #print(f"12) Softmax Attention: {normalized_attention.shape}")
        attention = self.attention_dropout(normalized_attention)
        #print(f"13) Attention: {attention.shape}")

        y = attention @ v # (B, num_heads, T, T) x (B, num_heads, T, C//num_heads) ===> (B, num_heads, T, C//num_heads)
        #print(f"14) Attention Output: {y.shape}")
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side (concat)
        #print(f"15) Multi-Head Attention Output: {y.shape}")

        # output projection
        y = self.fc(y)
        #print(f"16) Full Connected: {y.shape}")
        y = self.residual_dropout(y)
        #print(f"17) Dropout: {y.shape}")
        return y
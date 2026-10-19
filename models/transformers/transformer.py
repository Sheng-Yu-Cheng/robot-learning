import torch
import torch.nn as nn
import torch.nn.functional as f
from torch import Tensor
from jaxtyping import Float
import math


class SelfAttentionHead(nn.Module):
    # softmax(q @ k^T / sqrt(dk)) @ v
    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.Q = nn.Linear(latent_dim, latent_dim)
        self.K = nn.Linear(latent_dim, latent_dim)
        self.V = nn.Linear(latent_dim, latent_dim)
    
    def forward(self, x: Float[Tensor, "batch seq_len latent_dim"]) -> Float[Tensor, "batch seq_len latent_dim"]:
        q: Float[Tensor, "batch seq_len latent_dim"] = self.Q(x)
        k: Float[Tensor, "batch seq_len latent_dim"] = self.K(x)
        v: Float[Tensor, "batch seq_len latent_dim"] = self.V(x)
        weights: Float[Tensor, "batch seq_len seq_len"] = f.softmax((q @ k.transpose(-2, -1)) / math.sqrt(self.latent_dim), dim = -1)
        output: Float[Tensor, "batch seq_len latent_dim"] = weights @ v
        return output


class MultiHeadSelfAttention(nn.Module):
    # W @ softmax(q @ k^T / sqrt(dk)) @ v
    def __init__(self, latent_dim: int, head_amount: int):
        assert latent_dim % head_amount == 0
        super().__init__()
        self.latent_dim = latent_dim
        self.head_amount = head_amount
        self.head_dim = latent_dim // head_amount
        self.Q = nn.Linear(latent_dim, latent_dim)
        self.K = nn.Linear(latent_dim, latent_dim)
        self.V = nn.Linear(latent_dim, latent_dim)
        self.O = nn.Linear(latent_dim, latent_dim)
        
    def forward(self, x: Float[Tensor, "batch seq_len latent_dim"]) -> Float[Tensor, "batch seq_len latent_dim"]:
        batch, seq_len, latent_dim = x.shape
        q: Float[Tensor, "batch head_amount seq_len head_dim"] = self.Q(x).reshape(batch, seq_len, self.head_amount, self.head_dim).transpose(1, 2)
        k: Float[Tensor, "batch head_amount seq_len head_dim"] = self.K(x).reshape(batch, seq_len, self.head_amount, self.head_dim).transpose(1, 2)
        v: Float[Tensor, "batch head_amount seq_len head_dim"] = self.V(x).reshape(batch, seq_len, self.head_amount, self.head_dim).transpose(1, 2)
        weights: Float[Tensor, "batch head_amount seq_len seq_len"] = f.softmax((q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim), dim = -1)
        return self.O((weights @ v).transpose(1, 2).contiguous().reshape(batch, seq_len, self.latent_dim))


class LayerNorm(nn.Module):
    # y = gamma @ (x - mu) / sqrt(sigma^2 + eps) + beta
    def __init__(self, latent_dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(latent_dim))
        self.beta = nn.Parameter(torch.zeros(latent_dim))
    
    def forward(self, x: Float[Tensor, "batch seq_len latent_dim"]) -> Float[Tensor, "batch seq_len latent_dim"]:
        mean: Float[Tensor, "batch seq_len 1"] = x.mean(dim = -1, keepdim = True)
        variance: Float[Tensor, "batch seq_len 1"] = ((x - mean) ** 2).mean(dim = -1, keepdim = True)
        return self.gamma * (x - mean) / torch.sqrt(variance + self.eps) + self.beta
 

class TransformerBlock(nn.Module):
    # h = x + MHA(LN(x))
    # y = h + MLP(LN(h))
    def __init__(self, latent_dim: int, head_amount: int, activation = nn.ReLU): 
        super().__init__()
        self.latent_dim = latent_dim
        self.layer_norm1 = LayerNorm(latent_dim)
        self.multihead_attention = MultiHeadSelfAttention(latent_dim, head_amount)
        self.layer_norm2 = LayerNorm(latent_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 4), 
            activation(), 
            nn.Linear(latent_dim * 4,latent_dim)            
        )
        
    def forward(self, x: Float[Tensor, "batch seq_len latent_dim"]) -> Float[Tensor, "batch seq_len latent_dim"]:
        h: Float[Tensor, "batch seq_len latent_dim"] = x + self.multihead_attention(self.layer_norm1(x))
        y: Float[Tensor, "batch seq_len latent_dim"] = h + self.feed_forward(self.layer_norm2(h))
        return y
        

class PositionalEncoding(nn.Module):
    def __init__(self, latent_dim: int, max_len: int = 1024):
        super().__init__()
        assert latent_dim % 2 == 0
        
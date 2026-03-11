"""
neural/model.py — J.A.R.V.I.S Transformer Neural Network (from scratch)
Implements a GPT-style decoder-only transformer using only PyTorch primitives.

Architecture:
  - Token + Positional Embeddings
  - N × Transformer Decoder Blocks
      ↳ Multi-Head Causal Self-Attention (from scratch)
      ↳ Feed-Forward Network (with GELU activation)
      ↳ Layer Normalization (pre-norm style)
      ↳ Residual Connections
  - Final Linear projection → vocabulary logits
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple


# ─── Configuration ────────────────────────────────────────────────────────────

@dataclass
class JarvisConfig:
    vocab_size:   int   = 8000
    context_len:  int   = 256    # max sequence length
    embed_dim:    int   = 256    # embedding dimension
    num_heads:    int   = 8      # attention heads
    num_layers:   int   = 6      # transformer blocks
    ff_dim:       int   = 1024   # feed-forward inner dimension
    dropout:      float = 0.1
    bias:         bool  = True

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.num_heads


# ─── Building Blocks ──────────────────────────────────────────────────────────

class LayerNorm(nn.Module):
    """Layer normalization with optional bias — built from scratch."""

    def __init__(self, dim: int, bias: bool = True, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias   = nn.Parameter(torch.zeros(dim)) if bias else None
        self.eps    = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var  = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        out = self.weight * x_norm
        if self.bias is not None:
            out = out + self.bias
        return out


class MultiHeadCausalAttention(nn.Module):
    """
    Multi-Head Causal (masked) Self-Attention — implemented from scratch.
    
    Computes: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    with a causal mask so each token only attends to past tokens.
    """

    def __init__(self, cfg: JarvisConfig):
        super().__init__()
        assert cfg.embed_dim % cfg.num_heads == 0, \
            "embed_dim must be divisible by num_heads"

        self.num_heads = cfg.num_heads
        self.head_dim  = cfg.head_dim
        self.embed_dim = cfg.embed_dim
        self.dropout_p = cfg.dropout

        # Fused Q, K, V projection (3× embed → QKV)
        self.qkv_proj = nn.Linear(cfg.embed_dim, 3 * cfg.embed_dim, bias=cfg.bias)
        # Output projection
        self.out_proj = nn.Linear(cfg.embed_dim, cfg.embed_dim, bias=cfg.bias)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

        # Causal mask (lower-triangular) — registered as buffer (not a parameter)
        mask = torch.tril(torch.ones(cfg.context_len, cfg.context_len))
        self.register_buffer("causal_mask", mask.view(1, 1, cfg.context_len, cfg.context_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # batch, time, channels

        # QKV projection + split
        qkv = self.qkv_proj(x)                          # (B, T, 3C)
        q, k, v = qkv.split(self.embed_dim, dim=2)      # each: (B, T, C)

        # Reshape to (B, heads, T, head_dim)
        def reshape(t):
            return t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = reshape(q), reshape(k), reshape(v)

        # Scaled dot-product attention — from scratch
        scale  = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # (B, H, T, T)

        # Apply causal mask
        mask   = self.causal_mask[:, :, :T, :T]
        scores = scores.masked_fill(mask == 0, float("-inf"))

        attn   = F.softmax(scores, dim=-1)
        attn   = torch.nan_to_num(attn, nan=0.0)   # handle full -inf rows
        attn   = self.attn_drop(attn)

        # Weighted sum of values
        out = torch.matmul(attn, v)                             # (B, H, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)   # (B, T, C)
        out = self.resid_drop(self.out_proj(out))
        return out


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    FFN(x) = GELU(xW1 + b1)W2 + b2
    """

    def __init__(self, cfg: JarvisConfig):
        super().__init__()
        self.fc1   = nn.Linear(cfg.embed_dim, cfg.ff_dim, bias=cfg.bias)
        self.fc2   = nn.Linear(cfg.ff_dim, cfg.embed_dim, bias=cfg.bias)
        self.drop  = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)        # Gaussian Error Linear Unit
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """
    One Transformer Decoder Block.
    Pre-norm architecture:  x = x + Attn(LN(x))
                            x = x + FFN(LN(x))
    """

    def __init__(self, cfg: JarvisConfig):
        super().__init__()
        self.ln1  = LayerNorm(cfg.embed_dim, bias=cfg.bias)
        self.attn = MultiHeadCausalAttention(cfg)
        self.ln2  = LayerNorm(cfg.embed_dim, bias=cfg.bias)
        self.ffn  = FeedForward(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))   # residual + attention
        x = x + self.ffn(self.ln2(x))    # residual + feed-forward
        return x


class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding (from 'Attention Is All You Need').
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, embed_dim: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        return x + self.pe[:, :T, :]


# ─── Full Model ───────────────────────────────────────────────────────────────

class JarvisTransformer(nn.Module):
    """
    J.A.R.V.I.S. — GPT-style Decoder-Only Transformer, built from scratch.

    Pipeline:
        token_ids
        → Token Embedding + Sinusoidal Positional Encoding
        → N × TransformerBlock (Attention + FFN)
        → LayerNorm
        → Linear(embed → vocab) → logits
    """

    def __init__(self, cfg: JarvisConfig):
        super().__init__()
        self.cfg = cfg

        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.pos_enc     = SinusoidalPositionalEncoding(cfg.embed_dim, cfg.context_len)
        self.drop        = nn.Dropout(cfg.dropout)
        self.blocks      = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.num_layers)])
        self.ln_final    = LayerNorm(cfg.embed_dim, bias=cfg.bias)
        self.head        = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=False)

        # Weight tying: share token embedding and output projection weights
        self.head.weight = self.token_embed.weight

        self._init_weights()
        print(f"[Model] JarvisTransformer — {self.count_params():,} parameters")

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,                # (B, T)
        targets:   Optional[torch.Tensor] = None  # (B, T)
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        B, T = input_ids.shape
        assert T <= self.cfg.context_len, \
            f"Sequence length {T} exceeds context_len {self.cfg.context_len}"

        # Embedding + positional encoding
        x = self.token_embed(input_ids)   # (B, T, embed_dim)
        x = self.pos_enc(x)
        x = self.drop(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.ln_final(x)
        logits = self.head(x)              # (B, T, vocab_size)

        # Compute cross-entropy loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=0,   # PAD token = 0
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        prompt_ids:  torch.Tensor,
        max_new:     int   = 100,
        temperature: float = 0.8,
        top_k:       int   = 40,
    ) -> torch.Tensor:
        """
        Autoregressive generation via top-k sampling.
        Returns the full sequence including the prompt.
        """
        self.eval()
        ids = prompt_ids.clone()

        for _ in range(max_new):
            # Truncate to context window
            ctx = ids if ids.size(1) <= self.cfg.context_len \
                      else ids[:, -self.cfg.context_len:]

            logits, _ = self(ctx)
            logits = logits[:, -1, :] / temperature   # last token logits

            # Top-k filtering
            if top_k > 0:
                top_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                threshold   = top_vals[:, [-1]]
                logits      = logits.masked_fill(logits < threshold, float("-inf"))

            probs     = F.softmax(logits, dim=-1)
            next_id   = torch.multinomial(probs, num_samples=1)
            ids       = torch.cat([ids, next_id], dim=1)

            # Stop at EOS (token id 3)
            if next_id.item() == 3:
                break

        return ids

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, path: str):
        torch.save({"state_dict": self.state_dict(), "config": self.cfg}, path)
        print(f"[Model] Saved → {path}")

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "JarvisTransformer":
        data = torch.load(path, map_location=device)
        model = cls(data["config"])
        model.load_state_dict(data["state_dict"])
        model.to(device)
        print(f"[Model] Loaded from {path}")
        return model

"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           J.A.R.V.I.S — neural.py  ·  Stark Industries © 3000             ║
║                                                                              ║
║  Arquivo único consolidando todo o stack de IA do JARVIS:                   ║
║                                                                              ║
║   1. JarvisTokenizer   — BPE tokenizer do zero (sem libs externas)          ║
║   2. JarvisConfig      — Configuração do Transformer                        ║
║   3. LayerNorm         — Normalização de camada (from scratch)              ║
║   4. MultiHeadCausalAttention — Atenção causal multi-cabeça                 ║
║   5. FeedForward       — Rede feed-forward posicional (GELU)                ║
║   6. TransformerBlock  — Bloco decoder com pre-norm + residual              ║
║   7. SinusoidalPositionalEncoding — PE de 'Attention Is All You Need'       ║
║   8. JarvisTransformer — GPT-style decoder-only, geração top-k              ║
║   9. TextDataset       — Dataset sliding-window para treinamento            ║
║  10. TrainConfig       — Hiperparâmetros de treinamento                     ║
║  11. JarvisTrainer     — Loop AdamW + cosine LR decay + checkpointing       ║
║  12. EmbeddingEngine   — Mean-pool embeddings do Transformer                ║
║  13. VectorStore       — Busca por similaridade coseno (numpy)              ║
║  14. RAGEngine         — Retrieval Augmented Generation completo            ║
║  15. Insight           — Dataclass de insight de curiosidade                ║
║  16. CuriosityState    — Estado persistente do motor                        ║
║  17. CuriosityEngine   — Motor de curiosidade intrínseca autônomo           ║
║                                                                              ║
║  Fundamentos matemáticos presentes:                                          ║
║   • BPE:     merges iterativos sobre frequência de pares                     ║
║   • Atenção: Attn(Q,K,V) = softmax(QKᵀ / √dₖ) · V                         ║
║   • PE:      sin/cos positional encoding (Vaswani et al. 2017)              ║
║   • Loss:    cross-entropy com ignore_index=PAD                             ║
║   • LR:      warmup linear + cosine decay                                   ║
║   • Surpresa: S(w) = −log P(w)                                              ║
║   • Entropia: H = −Σ p·log₂(p)                                             ║
║   • Novidade: 1 − cos(vec, centróide_visto)                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ─── Standard Library ─────────────────────────────────────────────────────────
import os
import re
import sys
import json
import math
import time
import random
import threading
import collections
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Callable

# ─── Third-party ──────────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ══════════════════════════════════════════════════════════════════════════════
#  SEÇÃO 1 — BPE TOKENIZER
# ══════════════════════════════════════════════════════════════════════════════

class JarvisTokenizer:
    """
    Tokenizador BPE (Byte-Pair Encoding) implementado do zero.

    Algoritmo:
      1. Pré-tokenização no nível de caracteres
      2. Construção do vocabulário base com todos os chars únicos
      3. Iteração: encontrar o par mais frequente → mesclar → registrar merge
      4. Repetir até atingir vocab_size

    Tokens especiais:
        <PAD> = 0   padding (ignorado na loss)
        <UNK> = 1   token desconhecido
        <BOS> = 2   início de sequência
        <EOS> = 3   fim de sequência
        <SEP> = 4   separador entre passagens
    """

    SPECIAL_TOKENS = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3, "<SEP>": 4}
    PAD_ID = 0
    UNK_ID = 1
    BOS_ID = 2
    EOS_ID = 3
    SEP_ID = 4

    def __init__(self, vocab_size: int = 8000):
        self.vocab_size = vocab_size
        self.token2id: Dict[str, int] = dict(self.SPECIAL_TOKENS)
        self.id2token: Dict[int, str] = {v: k for k, v in self.SPECIAL_TOKENS.items()}
        self.merges:   List[Tuple[str, str]] = []
        self._trained  = False

    # ── Treinamento ───────────────────────────────────────────────────────────

    def train(self, corpus: str) -> None:
        """Treina o BPE no corpus fornecido."""
        print(f"[Tokenizer] Treinando BPE - vocab alvo: {self.vocab_size}")

        words      = self._pretokenize(corpus)
        word_freqs = collections.Counter(words)

        # Representação: cada palavra como tupla de chars + marcador </w>
        vocab: Dict[Tuple, int] = {
            tuple(list(w) + ["</w>"]): f
            for w, f in word_freqs.items()
        }

        # Vocabulário base: todos os caracteres únicos
        all_chars: set = set()
        for wt in vocab:
            all_chars.update(wt)

        next_id = len(self.token2id)
        for ch in sorted(all_chars):
            if ch not in self.token2id:
                self.token2id[ch] = next_id
                self.id2token[next_id] = ch
                next_id += 1

        # Merges iterativos
        num_merges = self.vocab_size - len(self.token2id)
        for i in range(max(0, num_merges)):
            pairs = self._get_stats(vocab)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            vocab = self._merge_vocab(best, vocab)
            self.merges.append(best)
            merged = best[0] + best[1]
            if merged not in self.token2id:
                self.token2id[merged] = next_id
                self.id2token[next_id] = merged
                next_id += 1
            if (i + 1) % 500 == 0:
                print(f"  [Tokenizer] Merge {i+1}/{num_merges} - vocab: {len(self.token2id)}")

        self._trained = True
        print(f"[Tokenizer] Concluido - vocab: {len(self.token2id)}")

    def _pretokenize(self, text: str) -> List[str]:
        text = text.lower()
        tokens = re.findall(r"[a-záéíóúàâêôãõüçñ]+|[0-9]+|[^\w\s]", text)
        return [t for t in tokens if t.strip()]

    def _get_stats(self, vocab: Dict[Tuple, int]) -> Dict[Tuple, int]:
        pairs: Dict[Tuple, int] = collections.defaultdict(int)
        for word, freq in vocab.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += freq
        return pairs

    def _merge_vocab(self, pair: Tuple[str, str],
                     vocab: Dict[Tuple, int]) -> Dict[Tuple, int]:
        new_vocab: Dict[Tuple, int] = {}
        bigram = pair[0] + pair[1]
        for word, freq in vocab.items():
            new_word, i = [], 0
            while i < len(word):
                if i < len(word)-1 and word[i] == pair[0] and word[i+1] == pair[1]:
                    new_word.append(bigram)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_vocab[tuple(new_word)] = freq
        return new_vocab

    # ── Codificação / Decodificação ───────────────────────────────────────────

    def encode(self, text: str, add_special: bool = True) -> List[int]:
        """Texto → lista de IDs."""
        ids = [self.token2id.get(t, self.UNK_ID) for t in self._tokenize(text)]
        return [self.BOS_ID] + ids + [self.EOS_ID] if add_special else ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """IDs → texto."""
        tokens = []
        for i in ids:
            tok = self.id2token.get(i, "<UNK>")
            if skip_special and tok in self.SPECIAL_TOKENS:
                continue
            tokens.append(tok)
        return "".join(tokens).replace("</w>", " ").strip()

    def _tokenize(self, text: str) -> List[str]:
        all_tokens = []
        for word in self._pretokenize(text):
            all_tokens.extend(self._apply_merges(list(word) + ["</w>"]))
        return all_tokens

    def _apply_merges(self, chars: List[str]) -> List[str]:
        word = list(chars)
        for pair in self.merges:
            new_word, i = [], 0
            while i < len(word):
                if i < len(word)-1 and word[i] == pair[0] and word[i+1] == pair[1]:
                    new_word.append(pair[0] + pair[1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
        return word

    # ── Utilitários ───────────────────────────────────────────────────────────

    @property
    def vocab_actual_size(self) -> int:
        return len(self.token2id)

    def pad_sequence(self, ids: List[int], max_len: int) -> List[int]:
        return ids[:max_len] if len(ids) >= max_len \
               else ids + [self.PAD_ID] * (max_len - len(ids))

    def save(self, path: str) -> None:
        Path(path).write_text(json.dumps({
            "vocab_size": self.vocab_size,
            "token2id":   self.token2id,
            "merges":     [list(m) for m in self.merges],
            "trained":    self._trained,
        }, ensure_ascii=False, indent=2))
        print(f"[Tokenizer] Salvo -> {path}")

    @classmethod
    def load(cls, path: str) -> "JarvisTokenizer":
        data = json.loads(Path(path).read_text())
        tok = cls(vocab_size=data["vocab_size"])
        tok.token2id  = data["token2id"]
        tok.id2token  = {int(v): k for k, v in data["token2id"].items()}
        tok.merges    = [tuple(m) for m in data["merges"]]
        tok._trained  = data["trained"]
        print(f"[Tokenizer] Carregado - vocab: {len(tok.token2id)}")
        return tok


# ══════════════════════════════════════════════════════════════════════════════
#  SEÇÃO 2 — TRANSFORMER (GPT-style, decoder-only)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class JarvisConfig:
    """
    Configuração do Transformer.

    Parâmetros dimensionais:
      embed_dim  → d_model (dimensão dos embeddings)
      num_heads  → h       (cabeças de atenção)
      ff_dim     → d_ff    (dimensão interna do FFN, geralmente 4× embed_dim)
      num_layers → N       (número de blocos Transformer)
      context_len→ T_max   (comprimento máximo de sequência)
    """
    vocab_size:  int   = 8000
    context_len: int   = 256
    embed_dim:   int   = 256
    num_heads:   int   = 8
    num_layers:  int   = 6
    ff_dim:      int   = 1024
    dropout:     float = 0.1
    bias:        bool  = True

    @property
    def head_dim(self) -> int:
        """Dimensão por cabeça: d_k = d_model / h"""
        return self.embed_dim // self.num_heads


# ── LayerNorm from scratch ────────────────────────────────────────────────────

class LayerNorm(nn.Module):
    """
    Normalização de camada implementada do zero.

    LN(x) = γ · (x − μ) / √(σ² + ε) + β

    onde μ e σ² são calculados por amostra (última dimensão),
    γ (weight) e β (bias) são parâmetros aprendíveis.
    """

    def __init__(self, dim: int, bias: bool = True, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias   = nn.Parameter(torch.zeros(dim)) if bias else None
        self.eps    = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean   = x.mean(dim=-1, keepdim=True)
        var    = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        out    = self.weight * x_norm
        if self.bias is not None:
            out = out + self.bias
        return out


# ── Multi-Head Causal Self-Attention ──────────────────────────────────────────

class MultiHeadCausalAttention(nn.Module):
    """
    Atenção multi-cabeça causal (masked) — implementada do zero.

    Fórmula:
        Attn(Q, K, V) = softmax( Q·Kᵀ / √d_k + M ) · V

    onde M é a máscara causal (−∞ nas posições futuras),
    garantindo que a posição i só atenda às posições ≤ i.

    Projeções fusionadas:
        [Q; K; V] = x · W_QKV    (W_QKV ∈ R^{d × 3d})
    Saída:
        out = concat(head_1, ..., head_h) · W_O
    """

    def __init__(self, cfg: JarvisConfig):
        super().__init__()
        assert cfg.embed_dim % cfg.num_heads == 0, \
            f"embed_dim ({cfg.embed_dim}) deve ser divisível por num_heads ({cfg.num_heads})"

        self.num_heads  = cfg.num_heads
        self.head_dim   = cfg.head_dim
        self.embed_dim  = cfg.embed_dim

        self.qkv_proj   = nn.Linear(cfg.embed_dim, 3 * cfg.embed_dim, bias=cfg.bias)
        self.out_proj   = nn.Linear(cfg.embed_dim, cfg.embed_dim,     bias=cfg.bias)
        self.attn_drop  = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

        # Máscara causal triangular inferior — buffer (não parâmetro)
        mask = torch.tril(torch.ones(cfg.context_len, cfg.context_len))
        self.register_buffer("causal_mask",
                             mask.view(1, 1, cfg.context_len, cfg.context_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # Projeção QKV fusionada
        q, k, v = self.qkv_proj(x).split(self.embed_dim, dim=2)

        def reshape(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = reshape(q), reshape(k), reshape(v)  # (B, H, T, head_dim)

        # Atenção escalada: scores = Q·Kᵀ / √d_k
        scale  = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # (B, H, T, T)

        # Máscara causal: posições futuras → −∞
        scores = scores.masked_fill(self.causal_mask[:, :, :T, :T] == 0,
                                    float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)  # rows completamente mascaradas
        attn = self.attn_drop(attn)

        # Combinação ponderada de valores
        out = torch.matmul(attn, v)                             # (B, H, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)   # (B, T, C)
        return self.resid_drop(self.out_proj(out))


# ── Feed-Forward Network ──────────────────────────────────────────────────────

class FeedForward(nn.Module):
    """
    FFN posicional com ativação GELU.

    FFN(x) = GELU(x · W₁ + b₁) · W₂ + b₂

    GELU(x) = x · Φ(x)   onde Φ é a CDF da Normal padrão.
    Comparado ao ReLU, GELU não tem gradiente zero para x < 0,
    resultando em treinamento mais estável em Transformers.
    """

    def __init__(self, cfg: JarvisConfig):
        super().__init__()
        self.fc1  = nn.Linear(cfg.embed_dim, cfg.ff_dim, bias=cfg.bias)
        self.fc2  = nn.Linear(cfg.ff_dim, cfg.embed_dim, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(F.gelu(self.fc1(x)))))


# ── Transformer Block ─────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """
    Bloco Transformer Decoder com arquitetura pre-norm.

    Fluxo:
        x  →  x + Attn( LN₁(x) )   ← conexão residual + atenção
           →  x + FFN(  LN₂(x) )   ← conexão residual + feed-forward

    Pre-norm (LN antes da sub-camada) é mais estável que post-norm
    para treinamento de modelos profundos sem warmup agressivo.
    """

    def __init__(self, cfg: JarvisConfig):
        super().__init__()
        self.ln1  = LayerNorm(cfg.embed_dim, bias=cfg.bias)
        self.attn = MultiHeadCausalAttention(cfg)
        self.ln2  = LayerNorm(cfg.embed_dim, bias=cfg.bias)
        self.ffn  = FeedForward(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# ── Positional Encoding ───────────────────────────────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    """
    Codificação posicional sinusoidal fixa.
    Vaswani et al. (2017) — 'Attention Is All You Need'.

    PE(pos, 2i)   = sin( pos / 10000^(2i / d_model) )
    PE(pos, 2i+1) = cos( pos / 10000^(2i / d_model) )

    Propriedades:
      • Determinístico (não aprendível) → sem parâmetros extras
      • Generaliza para sequências mais longas que as de treinamento
      • PE(pos+k) pode ser representado como transformação linear de PE(pos)
    """

    def __init__(self, embed_dim: int, max_len: int):
        super().__init__()
        pe  = torch.zeros(max_len, embed_dim)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


# ── Transformer Principal ─────────────────────────────────────────────────────

class JarvisTransformer(nn.Module):
    """
    J.A.R.V.I.S. — GPT-style Decoder-Only Transformer, do zero.

    Pipeline completo:
        token_ids  (B, T)
        → Token Embedding           → (B, T, d_model)
        → Sinusoidal Positional Enc → (B, T, d_model)
        → Dropout
        → N × TransformerBlock      → (B, T, d_model)
        → LayerNorm final
        → Linear (d_model → vocab)  → logits (B, T, V)

    Weight Tying:
        head.weight = token_embed.weight
        Reduz parâmetros e melhora generalização.
        (Press & Wolf, 2017 — "Using the Output Embedding to Improve LMs")

    Inicialização:
        Pesos lineares e embeddings: N(0, 0.02)
        Biases: zeros
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

        # Weight tying
        self.head.weight = self.token_embed.weight
        self._init_weights()
        print(f"[Transformer] {self.count_params():,} parâmetros treináveis")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets:   Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            input_ids: (B, T) — sequências de token IDs
            targets:   (B, T) — IDs alvo para calcular a loss (opcional)
        Returns:
            logits: (B, T, V)
            loss:   escalar ou None
        """
        B, T = input_ids.shape
        assert T <= self.cfg.context_len, \
            f"Comprimento {T} excede context_len {self.cfg.context_len}"

        x = self.token_embed(input_ids)   # (B, T, d_model)
        x = self.pos_enc(x)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)

        x      = self.ln_final(x)
        logits = self.head(x)              # (B, T, V)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=0,            # PAD_ID = 0
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
        Geração autoregressiva via amostragem top-k.

        Algoritmo:
          1. Forward no contexto atual
          2. Pegar logits do último token
          3. Dividir por temperature (temperatura > 1 → mais aleatório)
          4. Zerar todos exceto os top_k maiores
          5. Softmax → distribuição de probabilidade
          6. Amostrar próximo token
          7. Concatenar e repetir até EOS ou max_new
        """
        self.eval()
        ids = prompt_ids.clone()

        for _ in range(max_new):
            ctx    = ids[:, -self.cfg.context_len:]
            logits, _ = self(ctx)
            logits = logits[:, -1, :] / temperature

            if top_k > 0:
                top_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits      = logits.masked_fill(logits < top_vals[:, [-1]], float("-inf"))

            probs   = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            ids     = torch.cat([ids, next_id], dim=1)

            if next_id.item() == JarvisTokenizer.EOS_ID:
                break

        return ids

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, path: str) -> None:
        torch.save({"state_dict": self.state_dict(), "config": self.cfg}, path)
        print(f"[Transformer] Salvo -> {path}")

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "JarvisTransformer":
        data  = torch.load(path, map_location=device, weights_only=False)
        model = cls(data["config"])
        model.load_state_dict(data["state_dict"])
        model.to(device)
        print(f"[Transformer] Carregado -> {path}")
        return model


# ══════════════════════════════════════════════════════════════════════════════
#  SEÇÃO 3 — PIPELINE DE TREINAMENTO
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrainConfig:
    """
    Hiperparâmetros de treinamento.

    Otimizador:  AdamW com weight decay separado para parâmetros 2D+ e 1D
    Schedule:    warmup linear → cosine decay
      lr(t) =
        lr_base × t/warmup                  se t < warmup
        lr_base × 0.5 × (1 + cos(π × p))   senão (p = progresso em [0,1])
    """
    learning_rate:    float = 3e-4
    weight_decay:     float = 1e-2
    beta1:            float = 0.9
    beta2:            float = 0.95
    grad_clip:        float = 1.0
    warmup_steps:     int   = 100
    max_steps:        int   = 2000
    lr_decay:         bool  = True
    batch_size:       int   = 16
    context_len:      int   = 256
    log_interval:     int   = 50
    eval_interval:    int   = 200
    eval_iters:       int   = 20
    checkpoint_dir:   str   = "data/models"
    checkpoint_every: int   = 500
    device:           str   = "cpu"


class TextDataset(Dataset):
    """
    Dataset de janela deslizante sobre sequência tokenizada.

    Para cada índice i:
        x = token_ids[i : i + T]
        y = token_ids[i+1 : i + T + 1]

    Próximo-token prediction: o modelo aprende a prever cada
    token dado todos os tokens anteriores na janela.
    """

    def __init__(self, token_ids: List[int], context_len: int):
        self.ids = token_ids
        self.ctx = context_len
        self.n   = len(token_ids) - context_len - 1

    def __len__(self) -> int:
        return max(0, self.n)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk = self.ids[idx: idx + self.ctx + 1]
        return (torch.tensor(chunk[:-1], dtype=torch.long),
                torch.tensor(chunk[1:],  dtype=torch.long))


class JarvisTrainer:
    """
    Loop de treinamento completo para o JarvisTransformer.

    Fluxo por step:
      1. Ajustar LR pelo schedule cosine
      2. Obter batch (cicla infinitamente pelo DataLoader)
      3. Forward → loss (cross-entropy)
      4. Backward → acumular gradientes
      5. Gradient clipping (‖∇‖₂ ≤ grad_clip) → estabilidade
      6. AdamW step → atualizar pesos
      7. Log / eval / checkpoint periódico

    Separação decay/no-decay no AdamW:
      • Parâmetros 2D+ (matrizes de peso) → recebem weight decay
      • Parâmetros 1D (bias, LayerNorm) → sem weight decay
    """

    def __init__(
        self,
        model:     JarvisTransformer,
        tokenizer: JarvisTokenizer,
        corpus:    str,
        cfg:       TrainConfig,
    ):
        self.model     = model.to(cfg.device)
        self.tokenizer = tokenizer
        self.cfg       = cfg
        self.device    = cfg.device
        self.history:  List[Dict] = []
        self.step      = 0

        Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        print("[Trainer] Tokenizando corpus...")
        token_ids = tokenizer.encode(corpus, add_special=False)
        print(f"[Trainer] Tokens totais: {len(token_ids):,}")

        split     = int(len(token_ids) * 0.9)
        self.train_dataset = TextDataset(token_ids[:split], cfg.context_len)
        self.val_dataset   = TextDataset(token_ids[split:], cfg.context_len)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True, drop_last=True, num_workers=0,
        )

        # AdamW com weight decay seletivo
        decay   = [p for p in model.parameters() if p.dim() >= 2]
        nodecay = [p for p in model.parameters() if p.dim() < 2]
        self.optimizer = torch.optim.AdamW(
            [{"params": decay,   "weight_decay": cfg.weight_decay},
             {"params": nodecay, "weight_decay": 0.0}],
            lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2),
        )

    def _get_lr(self, step: int) -> float:
        """Schedule: warmup linear + cosine decay."""
        cfg = self.cfg
        if not cfg.lr_decay:
            return cfg.learning_rate
        if step < cfg.warmup_steps:
            return cfg.learning_rate * step / max(1, cfg.warmup_steps)
        progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
        return max(cfg.learning_rate * 0.1,
                   cfg.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress)))

    @torch.no_grad()
    def evaluate(self) -> float:
        """Calcula perda média no conjunto de validação."""
        self.model.eval()
        if len(self.val_dataset) == 0:
            return 0.0
        loader = DataLoader(self.val_dataset, batch_size=self.cfg.batch_size,
                            shuffle=True)
        losses = []
        for i, (x, y) in enumerate(loader):
            if i >= self.cfg.eval_iters:
                break
            x, y = x.to(self.device), y.to(self.device)
            _, loss = self.model(x, y)
            if loss is not None:
                losses.append(loss.item())
        self.model.train()
        return sum(losses) / len(losses) if losses else 0.0

    def train(
        self,
        callback: Optional[Callable[[Dict], None]] = None
    ) -> List[Dict]:
        """
        Executa o loop principal de treinamento.
        callback(info) é chamado a cada log_interval para streaming ao frontend.
        """
        self.model.train()
        cfg      = self.cfg
        iterator = iter(self.train_loader)
        t0       = time.time()
        loss     = None

        print(f"[Trainer] Iniciando - {cfg.max_steps} steps")

        for step in range(cfg.max_steps):
            self.step = step

            # Atualiza LR
            lr = self._get_lr(step)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

            # Obtém batch
            try:
                x, y = next(iterator)
            except StopIteration:
                iterator = iter(self.train_loader)
                x, y    = next(iterator)

            x, y = x.to(self.device), y.to(self.device)

            # Forward → backward → clip → step
            self.optimizer.zero_grad()
            _, loss = self.model(x, y)
            if loss is None:
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            self.optimizer.step()

            if step % cfg.log_interval == 0:
                info = {
                    "step":     step,
                    "loss":     round(loss.item(), 4),
                    "lr":       round(lr, 6),
                    "elapsed":  round(time.time() - t0, 1),
                    "progress": round(step / cfg.max_steps * 100, 1),
                }
                self.history.append(info)
                print(f"  step {step:5d} | loss {info['loss']:.4f}"
                      f" | lr {info['lr']:.2e} | {info['elapsed']:.0f}s")
                if callback:
                    callback(info)

            if step > 0 and step % cfg.eval_interval == 0:
                val_loss = self.evaluate()
                print(f"  [EVAL] step {step} | val_loss {val_loss:.4f}")
                if callback:
                    callback({"step": step, "val_loss": round(val_loss, 4)})

            if step > 0 and step % cfg.checkpoint_every == 0:
                self._checkpoint(step, loss.item())

        self._checkpoint(self.step, loss.item() if loss else 0.0, final=True)
        print("[Trainer] Treinamento concluído.")
        return self.history

    def _checkpoint(self, step: int, loss: float, final: bool = False) -> None:
        tag  = "final" if final else f"step_{step}"
        path = os.path.join(self.cfg.checkpoint_dir, f"jarvis_{tag}.pt")
        self.model.save(path)
        hist_path = os.path.join(self.cfg.checkpoint_dir, "train_history.json")
        Path(hist_path).write_text(json.dumps(self.history, indent=2))
        print(f"[Trainer] Checkpoint -> {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  SEÇÃO 4 — EMBEDDINGS E RAG
# ══════════════════════════════════════════════════════════════════════════════

class EmbeddingEngine:
    """
    Gera embeddings de texto de tamanho fixo a partir do Transformer.

    Estratégia: mean-pooling das hidden states finais sobre posições não-PAD.

        emb(x) = ( Σᵢ hᵢ · [xᵢ ≠ PAD] ) / ( Σᵢ [xᵢ ≠ PAD] )

    Normalização L2 final para busca por coseno eficiente.
    """

    def __init__(self, model: JarvisTransformer, tokenizer: JarvisTokenizer,
                 device: str = "cpu"):
        self.model     = model.to(device)
        self.tokenizer = tokenizer
        self.device    = device
        self.model.eval()

    @torch.no_grad()
    def embed(self, text: str, max_len: int = 128) -> np.ndarray:
        """Texto → vetor numpy (embed_dim,) normalizado."""
        ids   = self.tokenizer.pad_sequence(
            self.tokenizer.encode(text, add_special=True), max_len)
        ids_t = torch.tensor([ids], dtype=torch.long, device=self.device)

        # Extrai hidden states antes da cabeça de linguagem
        x = self.model.token_embed(ids_t)
        x = self.model.pos_enc(x)
        for block in self.model.blocks:
            x = block(x)
        x = self.model.ln_final(x)

        # Mean pooling sobre tokens não-PAD
        mask = (ids_t != JarvisTokenizer.PAD_ID).float().unsqueeze(-1)
        vec  = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        vec  = F.normalize(vec, dim=-1)
        return vec.squeeze(0).cpu().numpy()

    def embed_batch(self, texts: List[str], max_len: int = 128) -> np.ndarray:
        return np.stack([self.embed(t, max_len) for t in texts])


class VectorStore:
    """
    Store vetorial em memória com busca por similaridade coseno.

    Armazenamento: matriz numpy (N, D) de vetores normalizados.
    Busca:  scores = vectors @ query   →   argsort descendente
    Complexidade: O(N·D) por consulta (linear, adequado para milhares de docs).
    """

    def __init__(self):
        self.vectors:  Optional[np.ndarray] = None
        self.metadata: List[Dict]           = []

    def add(self, vectors: np.ndarray, meta: List[Dict]) -> None:
        assert len(vectors) == len(meta)
        self.vectors  = vectors if self.vectors is None \
                        else np.vstack([self.vectors, vectors])
        self.metadata.extend(meta)

    def search(self, query_vec: np.ndarray, top_k: int = 5) -> List[Dict]:
        if self.vectors is None or len(self.vectors) == 0:
            return []
        q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-9)
        scores = self.vectors @ q_norm
        top_i  = np.argsort(scores)[::-1][:top_k]
        results = []
        for i in top_i:
            entry = dict(self.metadata[i])
            entry["score"] = float(scores[i])
            results.append(entry)
        return results

    def __len__(self) -> int:
        return len(self.metadata)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.save(f"{path}_vecs.npy", self.vectors)
        Path(f"{path}_meta.json").write_text(
            json.dumps(self.metadata, ensure_ascii=False, indent=2))
        print(f"[VectorStore] Salvo {len(self)} entradas -> {path}")

    def load(self, path: str) -> None:
        vec_path = f"{path}_vecs.npy"
        if Path(vec_path).exists():
            self.vectors  = np.load(vec_path)
            self.metadata = json.loads(Path(f"{path}_meta.json").read_text())
            print(f"[VectorStore] Carregado {len(self)} entradas")


class RAGEngine:
    """
    Retrieval Augmented Generation (RAG) para o J.A.R.V.I.S.

    Pipeline:
      Indexação:
        documento → chunks → embed() → VectorStore

      Consulta:
        query → embed(query) → top-k retrieval → contexto textual
             → construir prompt → JarvisTransformer.generate()

    O RAG permite que o modelo responda com base em documentos externos
    sem precisar retrainer — o contexto relevante é injetado no prompt.
    """

    CHUNK_SIZE    = 200
    CHUNK_OVERLAP = 40

    def __init__(self, model: JarvisTransformer, tokenizer: JarvisTokenizer,
                 store: VectorStore, device: str = "cpu"):
        self.embedder  = EmbeddingEngine(model, tokenizer, device)
        self.model     = model
        self.tokenizer = tokenizer
        self.store     = store
        self.device    = device

    def index_document(self, text: str, source: str) -> int:
        chunks = self._chunk(text)
        if not chunks:
            return 0
        vecs = self.embedder.embed_batch(chunks, max_len=128)
        meta = [{"text": c, "source": source, "chunk": i}
                for i, c in enumerate(chunks)]
        self.store.add(vecs, meta)
        print(f"[RAG] Indexado '{source}' - {len(chunks)} chunks")
        return len(chunks)

    def _chunk(self, text: str) -> List[str]:
        words  = text.split()
        step   = self.CHUNK_SIZE - self.CHUNK_OVERLAP
        chunks = []
        for i in range(0, len(words), max(1, step)):
            c = " ".join(words[i: i + self.CHUNK_SIZE])
            if len(c.split()) >= 10:
                chunks.append(c)
        return chunks

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        if len(self.store) == 0:
            return []
        return self.store.search(self.embedder.embed(query), top_k=top_k)

    @torch.no_grad()
    def answer(self, query: str, top_k: int = 3, max_new: int = 80,
               temperature: float = 0.7) -> Dict:
        results = self.retrieve(query, top_k)
        if not results:
            return {"answer": "Nenhum documento indexado ainda, Senhor.",
                    "sources": [], "method": "no_context"}

        context = "\n\n".join(
            f"[{r['source']} | {r['score']:.2f}]\n{r['text']}"
            for r in results
        )
        prompt  = f"Contexto:\n{context[:800]}\n\nPergunta: {query}\nResposta:"
        ids     = self.tokenizer.encode(prompt[:500], add_special=True)
        ids     = ids[:self.model.cfg.context_len - max_new]
        ids_t   = torch.tensor([ids], dtype=torch.long, device=self.device)

        out_ids = self.model.generate(ids_t, max_new=max_new,
                                      temperature=temperature, top_k=40)
        answer  = self.tokenizer.decode(out_ids[0, len(ids):].tolist())

        return {
            "answer":  answer if answer.strip() else "Processando, Senhor…",
            "sources": [{"source": r["source"],
                         "score":  round(r["score"], 3)} for r in results],
            "method":  "rag",
        }


# ══════════════════════════════════════════════════════════════════════════════
#  SEÇÃO 5 — MOTOR DE CURIOSIDADE INTRÍNSECA
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Insight:
    """
    Unidade de conhecimento gerada pelo motor de curiosidade.
    Contém o trecho original, resumo, tags, métricas e conexões.
    """
    id:               str
    source:           str
    chunk_text:       str
    summary:          str
    tags:             List[str]
    curiosity_score:  float      # score composto [0, 1]
    novelty_score:    float      # distância do centróide visto
    entropy_score:    float      # entropia de Shannon local
    surprise_score:   float      # −log P(w) médio
    connections:      List[str]  # IDs de insights relacionados
    timestamp:        float
    times_surfaced:   int  = 0
    is_new:           bool = True

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class CuriosityState:
    """Estado persistente do motor de curiosidade entre reinicializações."""
    total_analyses:    int             = 0
    insights_found:    int             = 0
    last_analysis_ts:  float           = 0.0
    seen_chunk_hashes: List[str]       = field(default_factory=list)
    topic_index:       Dict[str, List[str]] = field(default_factory=dict)
    corpus_vocab:      Dict[str, int]  = field(default_factory=dict)


class CuriosityEngine:
    """
    Motor de Motivação Intrínseca do J.A.R.V.I.S.
    Inspirado em: Pathak et al. (2017) — 'Curiosity-Driven Exploration'.

    Ciclo autônomo (background thread, a cada CYCLE_INTERVAL segundos):
      1. Lê todos os corpora de data/embeddings/
      2. Divide em chunks sobrepostos
      3. Calcula 4 métricas de "interesse" por chunk:
           • Surpresa:  S(w) = −log P(w)          [teoria da informação]
           • Entropia:  H = −Σ p·log₂(p)          [Shannon]
           • Novidade:  1 − cos(vec, centróide)   [espaço vetorial]
           • Densidade: proporção de termos técnicos + números + siglas
      4. Score composto: 0.35·S + 0.25·H + 0.25·N + 0.15·D
      5. Seleciona top-K mais interessantes
      6. Extrai tags temáticas e gera resumo heurístico
      7. Detecta conexões entre insights por sobreposição de tags
      8. Persiste e notifica o Brain via callback
    """

    CYCLE_INTERVAL   = 45
    MIN_CHUNK_WORDS  = 20
    MAX_CHUNK_WORDS  = 150
    CHUNK_OVERLAP    = 30
    TOP_K_INSIGHTS   = 5
    MAX_INSIGHTS     = 200
    RESURFACE_AFTER  = 10

    W_SURPRISE = 0.35
    W_ENTROPY  = 0.25
    W_NOVELTY  = 0.25
    W_DENSITY  = 0.15

    TECHNICAL_TERMS = {
        "lidar","slam","esp32","vespa","sensor","mqtt","neural","tensor",
        "embedding","transformer","attention","gradient","entropy",
        "algorithm","autonomous","navigation","pid","pwm","i2c","spi",
        "uart","freertos","kalman","ekf","odometria","robô","protocolo",
        "frequência","tensão","corrente","potência","algoritmo","iot",
        "nuvem","firebase","automação","controle","atuador","microcontrolador",
        "lidar","rocket","tank","mq-02","aht10","brushless","sirene",
    }

    STOPWORDS = {
        "de","do","da","dos","das","em","no","na","nos","nas","um","uma",
        "uns","umas","o","a","os","as","e","é","que","se","por","para",
        "com","como","mais","mas","ou","ao","à","the","and","of","to",
        "in","is","it","that","for","on","with","as","this","are","from",
        "was","has","have","be","not","at","an","or","but","which","its",
    }

    THEME_DICT = {
        "LiDAR / Sensoriamento":   ["lidar","ld14p","varredura","360","distância"],
        "Navegação / SLAM":        ["slam","mapeamento","navegação","odometria","trajetória"],
        "Segurança Ativa":         ["alarme","sirene","detecção","anomalia","alerta"],
        "Gás / Ambiente":          ["mq-02","gás","inflamável","temperatura","umidade","aht10"],
        "Motores / Atuadores":     ["motor","brushless","fan","driver","pwm","esteira"],
        "ESP32 / Embarcado":       ["esp32","vespa","freertos","i2c","spi","uart","firmware"],
        "IoT / Nuvem":             ["mqtt","firebase","nuvem","wi-fi","http","api","banco"],
        "IA / Machine Learning":   ["neural","modelo","treinamento","embedding","transformer"],
        "Hardware / Eletrônica":   ["tensão","corrente","resistor","capacitor","circuito","bateria"],
        "Chassi / Estrutura":      ["rocket tank","chassi","esteira","suporte","3d"],
        "Controle / PID":          ["pid","controle","feedback","setpoint","erro","ganho"],
        "Comunicação":             ["websocket","serial","protocolo","pacote","byte"],
    }

    def __init__(self, corpus_dir: str = "data/embeddings",
                 output_dir: str  = "data/curiosity",
                 on_insight: Optional[Callable[[Insight], None]] = None):
        self.corpus_dir     = Path(corpus_dir)
        self.output_dir     = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.on_insight     = on_insight
        self.state          = CuriosityState()
        self.insights:      Dict[str, Insight] = {}
        self._running       = False
        self._thread:       Optional[threading.Thread] = None
        self._cycle         = 0
        self._seen_vectors: List[np.ndarray] = []
        self._load_state()
        print("[Curiosity] Motor inicializado.")

    # ── Controle ──────────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"[Curiosity] Ciclos a cada {self.CYCLE_INTERVAL}s")

    def stop(self) -> None:
        self._running = False

    # ── Loop Principal ────────────────────────────────────────────────────────

    def _loop(self) -> None:
        time.sleep(10)   # aguarda boot
        while self._running:
            try:
                self._run_cycle()
            except Exception as e:
                print(f"[Curiosity] Erro no ciclo: {e}")
            time.sleep(self.CYCLE_INTERVAL)

    def _run_cycle(self) -> None:
        self._cycle += 1
        corpora = list(self.corpus_dir.glob("*_corpus.txt"))
        if not corpora:
            return

        all_text = "\n".join(p.read_text(encoding="utf-8") for p in corpora)
        self._update_global_vocab(all_text)

        candidates = []
        for corpus_path in corpora:
            source = corpus_path.stem.replace("_corpus", "")
            text   = corpus_path.read_text(encoding="utf-8")
            for chunk in self._chunk_text(text):
                h = self._hash(chunk)
                if h in self.state.seen_chunk_hashes:
                    if self._cycle % self.RESURFACE_AFTER != 0:
                        continue
                s = self._surprise(chunk)
                e = self._entropy(chunk)
                n = self._novelty(chunk)
                d = self._density(chunk)
                score = (self.W_SURPRISE*s + self.W_ENTROPY*e +
                         self.W_NOVELTY*n  + self.W_DENSITY*d)
                candidates.append({"chunk": chunk, "source": source,
                                   "hash": h, "score": score,
                                   "surprise": s, "entropy": e,
                                   "novelty": n, "density": d})

        if not candidates:
            return

        top = self._dedup(sorted(candidates, key=lambda x: x["score"],
                                 reverse=True))[: self.TOP_K_INSIGHTS]
        new_insights = []
        for item in top:
            ins = self._build_insight(item)
            self.insights[ins.id] = ins
            new_insights.append(ins)
            if item["hash"] not in self.state.seen_chunk_hashes:
                self.state.seen_chunk_hashes.append(item["hash"])
                if len(self.state.seen_chunk_hashes) > 2000:
                    self.state.seen_chunk_hashes = self.state.seen_chunk_hashes[-1000:]
            for tag in ins.tags:
                self.state.topic_index.setdefault(tag, [])
                if ins.id not in self.state.topic_index[tag]:
                    self.state.topic_index[tag].append(ins.id)

        self._detect_connections(new_insights)
        self.state.total_analyses += 1
        self.state.insights_found += len(new_insights)
        self.state.last_analysis_ts = time.time()
        self._save_state()
        self._save_insights()

        print(f"[Curiosity] Ciclo #{self._cycle} - {len(new_insights)} insights"
              f" | total: {len(self.insights)}")

        if self.on_insight:
            for ins in new_insights:
                self.on_insight(ins)

    # ── Métricas de Curiosidade ───────────────────────────────────────────────

    def _surprise(self, text: str) -> float:
        """S(w) = −log P(w), média sobre tokens do chunk."""
        if not self.state.corpus_vocab:
            return 0.5
        words  = self._tok(text)
        if not words:
            return 0.0
        total  = sum(self.state.corpus_vocab.values()) or 1
        scores = []
        for w in words:
            f = self.state.corpus_vocab.get(w, 0)
            scores.append(1.0 if f == 0 else min(1.0, -math.log(f/total) / 15.0))
        return min(1.0, sum(scores) / len(scores))

    def _entropy(self, text: str) -> float:
        """H = −Σ p(w)·log₂p(w)   normalizado pelo máximo teórico."""
        words = self._tok(text)
        if len(words) < 5:
            return 0.0
        freq  = collections.Counter(words)
        total = len(words)
        H     = -sum((c/total)*math.log2(c/total) for c in freq.values())
        H_max = math.log2(total)
        return min(1.0, H / H_max) if H_max > 0 else 0.0

    def _novelty(self, text: str) -> float:
        """1 − cosine_similarity(vec, centróide_visto)."""
        vec = self._bow(text)
        if vec is None:
            return 1.0
        if not self._seen_vectors:
            self._seen_vectors.append(vec)
            return 1.0
        centroid = np.mean(self._seen_vectors[-50:], axis=0)
        nv, nc   = np.linalg.norm(vec), np.linalg.norm(centroid)
        if nv == 0 or nc == 0:
            return 0.5
        cos = float(np.dot(vec, centroid) / (nv * nc))
        self._seen_vectors.append(vec)
        return min(1.0, max(0.0, (1.0 - cos) / 2.0))

    def _density(self, text: str) -> float:
        """Proporção de termos técnicos + números + siglas + unidades."""
        words = text.lower().split()
        if not words:
            return 0.0
        tech     = sum(1 for w in words if w.strip(".,;:") in self.TECHNICAL_TERMS)
        nums     = sum(1 for w in words if re.search(r"\d", w))
        acronyms = sum(1 for w in words if w.isupper() and len(w) > 1)
        units    = sum(1 for w in words if w in {
            "hz","mhz","ghz","v","mv","a","ma","w","kb","mb","gb",
            "ms","us","ns","m","cm","mm","kg","g","°c","rpm","baud"})
        score = (tech*2 + nums + acronyms*1.5 + units*2) / (len(words)*2)
        return min(1.0, score)

    # ── Análise Temática e Resumo ─────────────────────────────────────────────

    def _extract_tags(self, text: str) -> List[str]:
        lower = text.lower()
        tags  = [theme for theme, kws in self.THEME_DICT.items()
                 if any(kw in lower for kw in kws)]
        freq  = collections.Counter(self._tok(text)).most_common(5)
        for w, c in freq:
            if c >= 2 and len(w) > 4:
                tags.append(w.capitalize())
        return list(dict.fromkeys(tags))[:8]

    def _generate_summary(self, text: str, tags: List[str]) -> str:
        sents = re.split(r"[.!?]\s+", text.strip())
        sents = [s.strip() for s in sents if len(s.split()) > 6]
        if not sents:
            return text[:200] + "…"
        def score(s):
            ws = s.lower().split()
            return (sum(1 for w in ws if w in self.TECHNICAL_TERMS)*2 +
                    sum(1 for w in ws if re.search(r"\d", w)))
        best    = sorted(sents, key=score, reverse=True)[:2]
        summary = " ".join(best)
        if tags:
            summary = f"[{', '.join(tags[:3])}] {summary}"
        return (summary[:350] + "…") if len(summary) > 350 else summary

    def _build_insight(self, item: Dict) -> Insight:
        tags    = self._extract_tags(item["chunk"])
        summary = self._generate_summary(item["chunk"], tags)
        ins_id  = f"ins_{int(time.time()*1000)}_{random.randint(100,999)}"
        return Insight(
            id             = ins_id,
            source         = item["source"],
            chunk_text     = item["chunk"][:500],
            summary        = summary,
            tags           = tags,
            curiosity_score= round(item["score"],    4),
            novelty_score  = round(item["novelty"],  4),
            entropy_score  = round(item["entropy"],  4),
            surprise_score = round(item["surprise"], 4),
            connections    = [],
            timestamp      = time.time(),
        )

    def _detect_connections(self, insights: List[Insight]) -> None:
        all_ins = list(self.insights.values())
        for ins in insights:
            for other in all_ins:
                if other.id == ins.id:
                    continue
                shared = set(ins.tags) & set(other.tags)
                if len(shared) >= 2:
                    if other.id not in ins.connections:
                        ins.connections.append(other.id)
                    if ins.id not in other.connections:
                        other.connections.append(ins.id)

    # ── Utilitários ───────────────────────────────────────────────────────────

    def _chunk_text(self, text: str) -> List[str]:
        words  = text.split()
        step   = self.MAX_CHUNK_WORDS - self.CHUNK_OVERLAP
        return [
            " ".join(words[i: i + self.MAX_CHUNK_WORDS])
            for i in range(0, len(words), max(1, step))
            if len(words[i: i + self.MAX_CHUNK_WORDS]) >= self.MIN_CHUNK_WORDS
        ]

    def _tok(self, text: str) -> List[str]:
        return [w.lower().strip(".,;:!?\"'()[]")
                for w in text.split()
                if len(w) > 2 and
                w.lower().strip(".,;:!?\"'()[]") not in self.STOPWORDS]

    def _bow(self, text: str, dim: int = 256) -> Optional[np.ndarray]:
        words = self._tok(text)
        if not words:
            return None
        vec = np.zeros(dim, dtype=np.float32)
        for w in words:
            vec[hash(w) % dim] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def _hash(self, text: str) -> str:
        return str(hash(" ".join(text.split()[:15])) & 0xFFFFFFFF)

    def _update_global_vocab(self, text: str) -> None:
        self.state.corpus_vocab.clear()
        for w in self._tok(text):
            self.state.corpus_vocab[w] = self.state.corpus_vocab.get(w, 0) + 1

    def _dedup(self, items: List[Dict]) -> List[Dict]:
        seen, result = set(), []
        for item in items:
            prefix = " ".join(item["chunk"].split()[:8])
            if prefix not in seen:
                seen.add(prefix)
                result.append(item)
        return result

    # ── API Pública ───────────────────────────────────────────────────────────

    def get_top_insights(self, n: int = 10,
                         tag: str = None) -> List[Insight]:
        pool = list(self.insights.values())
        if tag:
            pool = [i for i in pool if tag in i.tags]
        return sorted(pool, key=lambda x: x.curiosity_score, reverse=True)[:n]

    def get_topics(self) -> Dict[str, int]:
        return {t: len(ids) for t, ids in self.state.topic_index.items()}

    def get_random_insight(self) -> Optional[Insight]:
        if not self.insights:
            return None
        pool   = sorted(self.insights.values(), key=lambda x: x.times_surfaced)
        chosen = random.choice(pool[: max(1, len(pool)//2)])
        chosen.times_surfaced += 1
        chosen.is_new          = False
        return chosen

    def search_insights(self, query: str) -> List[Insight]:
        q = query.lower()
        scored = []
        for ins in self.insights.values():
            s = (3 if q in ins.chunk_text.lower() else 0) + \
                (2 if q in ins.summary.lower()    else 0) + \
                (1 if any(q in t.lower() for t in ins.tags) else 0)
            if s > 0:
                scored.append((s, ins))
        return [ins for _, ins in sorted(scored, reverse=True)[:10]]

    def get_stats(self) -> Dict:
        return {
            "total_analyses":  self.state.total_analyses,
            "insights_found":  len(self.insights),
            "topics":          len(self.state.topic_index),
            "cycle":           self._cycle,
            "last_analysis":   self.state.last_analysis_ts,
            "is_running":      self._running,
            "top_topics":      sorted(self.get_topics().items(),
                                      key=lambda x: x[1], reverse=True)[:8],
        }

    # ── Persistência ──────────────────────────────────────────────────────────

    def _save_state(self) -> None:
        (self.output_dir / "curiosity_state.json").write_text(
            json.dumps({
                "total_analyses":    self.state.total_analyses,
                "insights_found":    self.state.insights_found,
                "last_analysis_ts":  self.state.last_analysis_ts,
                "seen_chunk_hashes": self.state.seen_chunk_hashes[-500:],
                "topic_index":       self.state.topic_index,
            }, ensure_ascii=False, indent=2), encoding="utf-8")

    def _save_insights(self) -> None:
        data = sorted([ins.to_dict() for ins in self.insights.values()],
                      key=lambda x: x["curiosity_score"], reverse=True)
        (self.output_dir / "insights.json").write_text(
            json.dumps(data[:self.MAX_INSIGHTS], ensure_ascii=False, indent=2),
            encoding="utf-8")

    def _load_state(self) -> None:
        sp = self.output_dir / "curiosity_state.json"
        ip = self.output_dir / "insights.json"
        if sp.exists():
            d = json.loads(sp.read_text(encoding="utf-8"))
            self.state.total_analyses    = d.get("total_analyses", 0)
            self.state.insights_found    = d.get("insights_found", 0)
            self.state.last_analysis_ts  = d.get("last_analysis_ts", 0.0)
            self.state.seen_chunk_hashes = d.get("seen_chunk_hashes", [])
            self.state.topic_index       = d.get("topic_index", {})
            print(f"[Curiosity] Estado restaurado - "
                  f"{self.state.total_analyses} análises anteriores")
        if ip.exists():
            for d in json.loads(ip.read_text(encoding="utf-8")):
                try:
                    ins = Insight(**d)
                    ins.is_new = False
                    self.insights[ins.id] = ins
                except Exception:
                    pass
            print(f"[Curiosity] {len(self.insights)} insights restaurados")


# ══════════════════════════════════════════════════════════════════════════════
#  EXPORTS — compatibilidade com brain.py
# ══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Tokenizer
    "JarvisTokenizer",
    # Model
    "JarvisConfig", "LayerNorm", "MultiHeadCausalAttention",
    "FeedForward", "TransformerBlock", "SinusoidalPositionalEncoding",
    "JarvisTransformer",
    # Trainer
    "TrainConfig", "TextDataset", "JarvisTrainer",
    # Embeddings & RAG
    "EmbeddingEngine", "VectorStore", "RAGEngine",
    # Curiosity
    "Insight", "CuriosityState", "CuriosityEngine",
]

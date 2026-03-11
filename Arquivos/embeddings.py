"""
neural/embeddings.py — J.A.R.V.I.S Vector Store & RAG (Pure Local)
Geração de respostas utilizando estritamente o modelo Transformer local.
"""

import json
import math
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn.functional as F

# Tentativa de importação do FAISS para aceleração de busca vetorial
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from model import JarvisTransformer
from tokenizer import BasicTokenizer as JarvisTokenizer

# ─── Embedding Engine ─────────────────────────────────────────────────────────

class EmbeddingEngine:
    """Gera embeddings densos a partir do modelo treinado."""
    def __init__(self, model: JarvisTransformer, tokenizer: JarvisTokenizer, device: str = "cpu"):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def embed(self, text: str, max_len: int = 128) -> np.ndarray:
        ids = self.tokenizer.encode(text, add_special=True)
        ids = self.tokenizer.pad_sequence(ids, max_len)
        ids_t = torch.tensor([ids], dtype=torch.long, device=self.device)

        x = self.model.token_embed(ids_t)
        x = self.model.pos_enc(x)
        for block in self.model.blocks:
            x = block(x)
        x = self.model.ln_final(x)

        mask = (ids_t != self.tokenizer.PAD_ID).float().unsqueeze(-1)
        vec = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        vec = F.normalize(vec, dim=-1)
        return vec.squeeze(0).cpu().numpy().astype('float32')

    def embed_batch(self, texts: List[str], max_len: int = 128) -> np.ndarray:
        return np.stack([self.embed(t, max_len) for t in texts])

# ─── Vector Store (FAISS) ─────────────────────────────────────────────────────

class VectorStore:
    def __init__(self, dimension: int = 256):
        self.dimension = dimension
        self.metadata: List[Dict] = []
        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatIP(dimension)
        else:
            self.index = None
            self.vectors = None

    def add(self, vectors: np.ndarray, meta: List[Dict]) -> None:
        vectors = vectors.astype('float32')
        if FAISS_AVAILABLE:
            self.index.add(vectors)
        else:
            self.vectors = vectors if self.vectors is None else np.vstack([self.vectors, vectors])
        self.metadata.extend(meta)

    def search(self, query_vec: np.ndarray, top_k: int = 5) -> List[Dict]:
        if not self.metadata: return []
        query_vec = query_vec.reshape(1, -1).astype('float32')
        if FAISS_AVAILABLE:
            scores, indices = self.index.search(query_vec, top_k)
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1 or idx >= len(self.metadata): continue
                entry = dict(self.metadata[idx])
                entry["score"] = float(score)
                results.append(entry)
            return results
        return []

    def __len__(self) -> int:
        return len(self.metadata)

# ─── RAG Engine (Pure Local) ──────────────────────────────────────────────────

class RAGEngine:
    CHUNK_SIZE = 250
    CHUNK_OVERLAP = 40

    def __init__(self, model, tokenizer, store, device="cpu"):
        self.embedder = EmbeddingEngine(model, tokenizer, device)
        self.model = model
        self.tokenizer = tokenizer
        self.store = store
        self.device = device

    def index_document(self, text: str, source: str) -> int:
        chunks = self._chunk_text(text)
        if not chunks: return 0
        vecs = self.embedder.embed_batch(chunks, max_len=128)
        meta = [{"text": c, "source": source, "chunk": i} for i, c in enumerate(chunks)]
        self.store.add(vecs, meta)
        return len(chunks)

    def _chunk_text(self, text: str) -> List[str]:
        words = text.split()
        chunks = [ " ".join(words[i : i + self.CHUNK_SIZE]) 
                  for i in range(0, len(words), self.CHUNK_SIZE - self.CHUNK_OVERLAP) ]
        return [c for c in chunks if len(c.split()) > 10]

    def build_context(self, results: List[Dict]) -> str:
        return "\n".join([f"[{r['source']}]: {r['text']}" for r in results])

    @torch.no_grad()
    def answer(self, query: str, top_k: int = 3, max_new: int = 100) -> Dict:
        """Pipeline RAG Local: Recuperação + Geração Autoregressiva."""
        q_vec = self.embedder.embed(query)
        results = self.store.search(q_vec, top_k=top_k)

        if not results:
            return {
                "answer": "Senhor, não detetei dados relevantes no Datalake para esta consulta.",
                "sources": [],
                "method": "local_no_context"
            }

        context = self.build_context(results)
        prompt = f"Contexto:\n{context}\n\nPergunta: {query}\nResposta:"

        # Codificação para tensores (limitada pelo context_len do modelo)
        ids = self.tokenizer.encode(prompt, add_special=True)
        ids = ids[-(self.model.cfg.context_len - max_new):]
        ids_t = torch.tensor([ids], dtype=torch.long, device=self.device)

        # Inferência local no modelo Transformer
        out_ids = self.model.generate(ids_t, max_new=max_new, temperature=0.7)
        
        # Extração apenas da parte gerada (pós-prompt)
        new_tokens = out_ids[0, len(ids):].tolist()
        answer = self.tokenizer.decode(new_tokens)

        return {
            "answer": answer if answer.strip() else "Processando tensores, Senhor...",
            "sources": [{"source": r["source"], "score": round(r["score"], 3)} for r in results],
            "method": "local_rag_faiss"
        }
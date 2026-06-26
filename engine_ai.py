"""
engine_ai.py - local direct inference layer for JarvisV3.

This replaces the older V2 DirectML-only experiment with a safer local adapter:
it reuses the active JarvisBrain model/RAG when available and never calls
external APIs.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict


class DirectInferenceEngine:
    """Runs local inference against the in-process Jarvis brain."""

    def __init__(self, brain: Any):
        self.brain = brain

    async def answer(self, prompt: str) -> Dict[str, Any]:
        prompt = (prompt or "").strip()
        if not prompt:
            return {
                "text": "Entrada vazia.",
                "intent": "direct_inference",
                "action": None,
                "data": {},
            }

        if getattr(self.brain, "is_trained", False) and getattr(self.brain, "rag", None):
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, self._rag_answer, prompt)
            return {
                "text": result.get("answer", ""),
                "intent": "direct_inference",
                "action": "show_sources" if result.get("sources") else None,
                "data": {"sources": result.get("sources", [])},
            }

        if getattr(self.brain, "tokenizer", None) and getattr(self.brain, "model", None):
            loop = asyncio.get_running_loop()
            text = await loop.run_in_executor(None, self._generate_raw, prompt)
            return {
                "text": text,
                "intent": "direct_inference",
                "action": None,
                "data": {"mode": "local_transformer"},
            }

        return {
            "text": (
                "Motor de inferência local pronto, mas o modelo ainda não foi "
                "treinado. Carregue PDFs e execute o treinamento local primeiro."
            ),
            "intent": "direct_inference",
            "action": "show_train_panel",
            "data": {"is_trained": False},
        }

    def _rag_answer(self, prompt: str) -> Dict[str, Any]:
        return self.brain.rag.answer(prompt, top_k=3, max_new=80)

    def _generate_raw(self, prompt: str) -> str:
        import torch

        tokenizer = self.brain.tokenizer
        model = self.brain.model
        context_len = getattr(getattr(model, "cfg", None), "context_len", 128)
        ids = tokenizer.encode(prompt, add_special=True)[-context_len:]
        device = next(model.parameters()).device
        prompt_ids = torch.tensor([ids], dtype=torch.long, device=device)
        out = model.generate(prompt_ids, max_new=80, temperature=0.8, top_k=40)
        decoded = tokenizer.decode(out[0].detach().cpu().tolist())
        return decoded.strip() or "Geração local concluída sem texto decodificável."

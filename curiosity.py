"""
neural/curiosity.py — J.A.R.V.I.S Intrinsic Curiosity Engine

Implementa motivação intrínseca: o sistema analisa e reanalisará
documentos autonomamente, buscando trechos "interessantes" com base
em métricas de surpresa informacional, novidade, densidade semântica
e conexões entre documentos.

Fundamentos matemáticos:
  - Surpresa: S(x) = -log P(x)          (teoria da informação)
  - Entropia local: H = -Σ p·log(p)     (Shannon)
  - Novidade: dist(x, centróide_visto)  (espaço vetorial)
  - TF-IDF: relevância local vs. global
  - Score de curiosidade: combinação ponderada dos anteriores
"""

import re
import math
import time
import json
import random
import threading
import collections
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass, field, asdict
import numpy as np


# ─── Estruturas de Dados ──────────────────────────────────────────────────────

@dataclass
class Insight:
    """Um insight gerado pelo motor de curiosidade."""
    id:           str
    source:       str          # documento de origem
    chunk_text:   str          # trecho original
    summary:      str          # resumo gerado pelo JARVIS
    tags:         List[str]    # temas identificados
    curiosity_score: float     # 0.0 → 1.0
    novelty_score:   float
    entropy_score:   float
    surprise_score:  float
    connections:  List[str]    # IDs de insights relacionados
    timestamp:    float
    times_surfaced: int = 0    # quantas vezes foi apresentado
    is_new:       bool  = True

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class CuriosityState:
    """Estado persistente do motor de curiosidade."""
    total_analyses:    int   = 0
    insights_found:    int   = 0
    last_analysis_ts:  float = 0.0
    seen_chunk_hashes: List[str] = field(default_factory=list)
    topic_index:       Dict[str, List[str]] = field(default_factory=dict)  # tema → [insight_ids]
    corpus_vocab:      Dict[str, int] = field(default_factory=dict)         # palavra → freq global


# ─── Motor de Curiosidade ─────────────────────────────────────────────────────

class CuriosityEngine:
    """
    Motor de curiosidade intrínseca do J.A.R.V.I.S.

    Ciclo autônomo (roda em background thread):
    1.  Lê todos os corpora indexados
    2.  Divide em chunks e calcula métricas de interesse
    3.  Seleciona os mais "curiosos"
    4.  Gera tags temáticas automáticas
    5.  Detecta conexões entre documentos diferentes
    6.  Armazena insights e notifica o Brain via callback

    Métricas de "interessante":
      • Surpresa informacional  → palavras raras no corpus global
      • Entropia local          → densidade informacional do trecho
      • Novidade vetorial       → distância dos chunks já vistos
      • Densidade de entidades  → nomes, números, termos técnicos
      • Score composto          → média ponderada
    """

    # Hiperparâmetros
    CYCLE_INTERVAL   = 45     # segundos entre ciclos
    MIN_CHUNK_WORDS  = 20
    MAX_CHUNK_WORDS  = 150
    TOP_K_INSIGHTS   = 5      # insights por ciclo
    OVERLAP          = 30     # palavras de overlap entre chunks
    MAX_INSIGHTS     = 200    # limite do índice
    RESURFACE_AFTER  = 10     # reanalisar após N ciclos

    # Pesos do score composto
    W_SURPRISE = 0.35
    W_ENTROPY  = 0.25
    W_NOVELTY  = 0.25
    W_DENSITY  = 0.15

    # Vocabulário técnico que eleva o score
    TECHNICAL_TERMS = {
        "lidar", "slam", "esp32", "sensor", "mqtt", "neural", "tensor",
        "embedding", "transformer", "attention", "gradient", "entropy",
        "algorithm", "autonomous", "navigation", "pid", "pwm", "i2c",
        "spi", "uart", "freertos", "kalman", "ekf", "odometria", "robô",
        "rede", "protocolo", "frequência", "tensão", "corrente", "potência",
        "algoritmo", "banco", "dados", "iot", "nuvem", "firebase",
        "automação", "controle", "atuador", "microcontrolador",
    }

    # Stopwords pt-BR + en
    STOPWORDS = {
        "de","do","da","dos","das","em","no","na","nos","nas","um","uma",
        "uns","umas","o","a","os","as","e","é","que","se","por","para",
        "com","como","mais","mas","ou","ao","à","the","and","of","to",
        "in","is","it","that","for","on","with","as","this","are","from",
        "was","has","have","be","not","at","an","or","but","which","its",
    }

    def __init__(
        self,
        corpus_dir:  str = "data/embeddings",
        output_dir:  str = "data/curiosity",
        on_insight:  Optional[Callable[[Insight], None]] = None,
    ):
        self.corpus_dir  = Path(corpus_dir)
        self.output_dir  = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.on_insight  = on_insight   # callback quando novo insight é gerado
        self.state       = CuriosityState()
        self.insights:   Dict[str, Insight] = {}
        self._running    = False
        self._thread:    Optional[threading.Thread] = None
        self._cycle      = 0

        # Vetor de "memória" — centróides dos chunks já vistos (simplificado)
        self._seen_vectors: List[np.ndarray] = []

        self._load_state()
        print("[Curiosity] Motor de curiosidade inicializado.")

    # ─── Controle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Inicia o ciclo autônomo em background."""
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print("[Curiosity] Motor iniciado - ciclos a cada "
              f"{self.CYCLE_INTERVAL}s")

    def stop(self) -> None:
        self._running = False
        print("[Curiosity] Motor parado.")

    # ─── Loop Principal ───────────────────────────────────────────────────────

    def _loop(self) -> None:
        # Primeiro ciclo após 10s do boot
        time.sleep(10)
        while self._running:
            try:
                self._run_cycle()
            except Exception as e:
                print(f"[Curiosity] Erro no ciclo: {e}")
            time.sleep(self.CYCLE_INTERVAL)

    def _run_cycle(self) -> None:
        self._cycle += 1
        print(f"[Curiosity] Ciclo #{self._cycle} - analisando documentos...")

        corpora = list(self.corpus_dir.glob("*_corpus.txt"))
        if not corpora:
            print("[Curiosity] Nenhum corpus encontrado. Aguardando...")
            return

        # Atualizar vocabulário global (para cálculo de surpresa)
        all_text = "\n".join(p.read_text(encoding="utf-8") for p in corpora)
        self._update_global_vocab(all_text)

        cycle_insights = []

        for corpus_path in corpora:
            source = corpus_path.stem.replace("_corpus", "")
            text   = corpus_path.read_text(encoding="utf-8")
            chunks = self._chunk_text(text)

            for chunk in chunks:
                chunk_hash = self._hash_text(chunk)

                # Pula se já foi visto recentemente (exceto resurface)
                if chunk_hash in self.state.seen_chunk_hashes:
                    if self._cycle % self.RESURFACE_AFTER != 0:
                        continue

                # Calcula métricas
                surprise = self._surprise_score(chunk)
                entropy  = self._entropy_score(chunk)
                novelty  = self._novelty_score(chunk)
                density  = self._density_score(chunk)

                # Score composto
                score = (
                    self.W_SURPRISE * surprise +
                    self.W_ENTROPY  * entropy  +
                    self.W_NOVELTY  * novelty  +
                    self.W_DENSITY  * density
                )

                cycle_insights.append({
                    "chunk":    chunk,
                    "source":   source,
                    "hash":     chunk_hash,
                    "score":    score,
                    "surprise": surprise,
                    "entropy":  entropy,
                    "novelty":  novelty,
                    "density":  density,
                })

        if not cycle_insights:
            return

        # Selecionar os TOP_K mais interessantes
        top = sorted(cycle_insights, key=lambda x: x["score"], reverse=True)
        top = self._deduplicate(top)[: self.TOP_K_INSIGHTS]

        new_insights = []
        for item in top:
            insight = self._build_insight(item)
            self.insights[insight.id] = insight
            new_insights.append(insight)

            # Marcar como visto
            if item["hash"] not in self.state.seen_chunk_hashes:
                self.state.seen_chunk_hashes.append(item["hash"])
                if len(self.state.seen_chunk_hashes) > 2000:
                    self.state.seen_chunk_hashes = self.state.seen_chunk_hashes[-1000:]

            # Indexar temas
            for tag in insight.tags:
                self.state.topic_index.setdefault(tag, [])
                if insight.id not in self.state.topic_index[tag]:
                    self.state.topic_index[tag].append(insight.id)

        # Detectar conexões entre insights deste ciclo
        self._detect_connections(new_insights)

        # Atualizar estado
        self.state.total_analyses += 1
        self.state.insights_found += len(new_insights)
        self.state.last_analysis_ts = time.time()

        # Persistir
        self._save_state()
        self._save_insights()

        print(f"[Curiosity] Ciclo #{self._cycle} - "
              f"{len(new_insights)} novos insights | "
              f"total: {len(self.insights)}")

        # Notificar Brain
        if self.on_insight:
            for ins in new_insights:
                self.on_insight(ins)

    # ─── Métricas de Interesse ────────────────────────────────────────────────

    def _surprise_score(self, text: str) -> float:
        """
        Surpresa informacional: palavras raras no corpus global
        têm alta surpresa S(w) = -log P(w).
        Score = média da surpresa das palavras do chunk.
        """
        if not self.state.corpus_vocab:
            return 0.5

        words = self._tokenize(text)
        if not words:
            return 0.0

        total_freq = sum(self.state.corpus_vocab.values()) or 1
        surprises  = []

        for w in words:
            freq = self.state.corpus_vocab.get(w, 0)
            if freq == 0:
                surprises.append(1.0)
            else:
                prob = freq / total_freq
                surprises.append(min(1.0, -math.log(prob) / 15.0))

        return min(1.0, sum(surprises) / len(surprises))

    def _entropy_score(self, text: str) -> float:
        """
        Entropia de Shannon local:
        H = -Σ p(w)·log₂ p(w)
        Alta entropia → conteúdo denso e variado → mais interessante.
        """
        words  = self._tokenize(text)
        if len(words) < 5:
            return 0.0

        freq   = collections.Counter(words)
        total  = len(words)
        H      = -sum((c / total) * math.log2(c / total) for c in freq.values())
        H_max  = math.log2(total)
        return min(1.0, H / H_max) if H_max > 0 else 0.0

    def _novelty_score(self, text: str) -> float:
        """
        Novidade: quão diferente é este chunk dos já vistos.
        Usa bag-of-words normalizado como vetor.
        dist = 1 - cosine_similarity(vec, centróide_visto)
        """
        vec = self._bow_vector(text)
        if vec is None:
            return 1.0   # nunca visto → máxima novidade

        if not self._seen_vectors:
            self._seen_vectors.append(vec)
            return 1.0

        centroid = np.mean(self._seen_vectors[-50:], axis=0)
        norm_v   = np.linalg.norm(vec)
        norm_c   = np.linalg.norm(centroid)

        if norm_v == 0 or norm_c == 0:
            return 0.5

        cos_sim  = float(np.dot(vec, centroid) / (norm_v * norm_c))
        novelty  = (1.0 - cos_sim) / 2.0   # normaliza para [0, 1]
        self._seen_vectors.append(vec)
        return min(1.0, max(0.0, novelty))

    def _density_score(self, text: str) -> float:
        """
        Densidade de entidades técnicas:
        números, termos técnicos, siglas, unidades de medida.
        """
        words = text.lower().split()
        if not words:
            return 0.0

        tech_hits   = sum(1 for w in words if w.strip(".,;:") in self.TECHNICAL_TERMS)
        num_hits    = sum(1 for w in words if re.search(r'\d', w))
        acronym_hits = sum(1 for w in words if w.isupper() and len(w) > 1)
        unit_hits   = sum(1 for w in words if w in {"hz","mhz","ghz","v","mv","a","ma","w","kb","mb","gb","ms","us","ns","m","cm","mm","kg","g","°c"})

        score = (tech_hits * 2 + num_hits + acronym_hits * 1.5 + unit_hits * 2) / (len(words) * 2)
        return min(1.0, score)

    # ─── Análise Temática ─────────────────────────────────────────────────────

    def _extract_tags(self, text: str) -> List[str]:
        """
        Extrai tags temáticas automaticamente por frequência + dicionário.
        """
        text_lower = text.lower()
        tags = []

        THEME_DICT = {
            "LiDAR / Sensoriamento":    ["lidar", "ld14p", "varredura", "360", "distância"],
            "Navegação / SLAM":         ["slam", "mapeamento", "navegação", "odometria", "trajetória"],
            "Segurança Ativa":          ["alarme", "sirene", "detecção", "anomalia", "alerta"],
            "Gás / Ambiente":           ["mq-02", "gás", "inflamável", "temperatura", "umidade", "aht10"],
            "Motores / Atuadores":      ["motor", "brushless", "fan", "driver", "pwm", "esteira"],
            "ESP32 / Embarcado":        ["esp32", "vespa", "freertos", "i2c", "spi", "uart", "firmware"],
            "IoT / Nuvem":              ["mqtt", "firebase", "nuvem", "wi-fi", "http", "api", "banco"],
            "IA / Machine Learning":    ["neural", "modelo", "treinamento", "embedding", "transformer", "ia"],
            "Hardware / Eletrônica":    ["tensão", "corrente", "resistor", "capacitor", "circuito", "bateria"],
            "Chassi / Estrutura":       ["rocket tank", "chassi", "esteira", "suporte", "impresso", "3d"],
            "Controle / PID":           ["pid", "controle", "feedback", "setpoint", "erro", "ganho"],
            "Comunicação":              ["websocket", "serial", "protocolo", "pacote", "byte"],
        }

        for theme, keywords in THEME_DICT.items():
            if any(kw in text_lower for kw in keywords):
                tags.append(theme)

        # Tags de palavras-chave frequentes (TF local)
        words = self._tokenize(text)
        freq  = collections.Counter(words).most_common(5)
        for w, c in freq:
            if c >= 2 and len(w) > 4 and w not in self.STOPWORDS:
                tags.append(w.capitalize())

        return list(dict.fromkeys(tags))[:8]  # dedup + limite

    def _generate_summary(self, text: str, tags: List[str]) -> str:
        """
        Gera um resumo estilo JARVIS do trecho.
        (Heurística baseada em sentenças-chave — sem modelo externo.)
        """
        sentences = re.split(r'[.!?]\s+', text.strip())
        sentences = [s.strip() for s in sentences if len(s.split()) > 6]

        if not sentences:
            return text[:200] + "…"

        # Pontua sentenças pela presença de termos técnicos e números
        def sent_score(s):
            words = s.lower().split()
            return (
                sum(1 for w in words if w in self.TECHNICAL_TERMS) * 2 +
                sum(1 for w in words if re.search(r'\d', w))
            )

        best = sorted(sentences, key=sent_score, reverse=True)[:2]
        summary = " ".join(best)

        if tags:
            summary = f"[{', '.join(tags[:3])}] {summary}"

        return summary[:350] + ("…" if len(summary) > 350 else "")

    # ─── Construção do Insight ────────────────────────────────────────────────

    def _build_insight(self, item: Dict) -> Insight:
        tags    = self._extract_tags(item["chunk"])
        summary = self._generate_summary(item["chunk"], tags)
        ins_id  = f"ins_{int(time.time()*1000)}_{random.randint(100,999)}"

        return Insight(
            id              = ins_id,
            source          = item["source"],
            chunk_text      = item["chunk"][:500],
            summary         = summary,
            tags            = tags,
            curiosity_score = round(item["score"],    4),
            novelty_score   = round(item["novelty"],  4),
            entropy_score   = round(item["entropy"],  4),
            surprise_score  = round(item["surprise"], 4),
            connections     = [],
            timestamp       = time.time(),
            is_new          = True,
        )

    def _detect_connections(self, insights: List[Insight]) -> None:
        """
        Detecta conexões semânticas entre insights via sobreposição de tags.
        Também conecta com insights existentes.
        """
        all_ins = list(self.insights.values())
        for ins in insights:
            ins_tags = set(ins.tags)
            for other in all_ins:
                if other.id == ins.id:
                    continue
                shared = ins_tags & set(other.tags)
                if len(shared) >= 2:
                    if other.id not in ins.connections:
                        ins.connections.append(other.id)
                    if ins.id not in other.connections:
                        other.connections.append(ins.id)

    # ─── Utilitários ──────────────────────────────────────────────────────────

    def _chunk_text(self, text: str) -> List[str]:
        words  = text.split()
        chunks = []
        step   = self.MAX_CHUNK_WORDS - self.OVERLAP
        for i in range(0, len(words), max(1, step)):
            chunk = " ".join(words[i: i + self.MAX_CHUNK_WORDS])
            if len(chunk.split()) >= self.MIN_CHUNK_WORDS:
                chunks.append(chunk)
        return chunks

    def _tokenize(self, text: str) -> List[str]:
        return [
            w.lower().strip(".,;:!?\"'()[]")
            for w in text.split()
            if len(w) > 2 and w.lower().strip(".,;:!?\"'()[]") not in self.STOPWORDS
        ]

    def _bow_vector(self, text: str, dim: int = 256) -> Optional[np.ndarray]:
        """Bag-of-words comprimido por hashing (feature hashing)."""
        words = self._tokenize(text)
        if not words:
            return None
        vec = np.zeros(dim, dtype=np.float32)
        for w in words:
            idx = hash(w) % dim
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def _hash_text(self, text: str) -> str:
        """Hash simples para identificar chunks únicos."""
        words = text.split()[:15]
        return str(hash(" ".join(words)) & 0xFFFFFFFF)

    def _update_global_vocab(self, text: str) -> None:
        """Atualiza o vocabulário global para cálculo de surpresa."""
        self.state.corpus_vocab.clear()
        for w in self._tokenize(text):
            self.state.corpus_vocab[w] = self.state.corpus_vocab.get(w, 0) + 1

    def _deduplicate(self, items: List[Dict]) -> List[Dict]:
        """Remove chunks muito similares (pelos primeiros tokens)."""
        seen_prefixes = set()
        result = []
        for item in items:
            prefix = " ".join(item["chunk"].split()[:8])
            if prefix not in seen_prefixes:
                seen_prefixes.add(prefix)
                result.append(item)
        return result

    # ─── API Pública ──────────────────────────────────────────────────────────

    def get_top_insights(self, n: int = 10, tag: str = None) -> List[Insight]:
        """Retorna os N insights com maior score de curiosidade."""
        pool = list(self.insights.values())
        if tag:
            pool = [i for i in pool if tag in i.tags]
        return sorted(pool, key=lambda x: x.curiosity_score, reverse=True)[:n]

    def get_topics(self) -> Dict[str, int]:
        """Retorna todos os temas e quantidade de insights por tema."""
        return {t: len(ids) for t, ids in self.state.topic_index.items()}

    def get_random_insight(self) -> Optional[Insight]:
        """Retorna um insight aleatório (para surfacing proativo)."""
        if not self.insights:
            return None
        pool = sorted(self.insights.values(), key=lambda x: x.times_surfaced)
        # Preferir os menos vistos com score alto
        pool = pool[:max(1, len(pool) // 2)]
        chosen = random.choice(pool)
        chosen.times_surfaced += 1
        chosen.is_new = False
        return chosen

    def get_stats(self) -> Dict:
        return {
            "total_analyses":  self.state.total_analyses,
            "insights_found":  len(self.insights),
            "topics":          len(self.state.topic_index),
            "cycle":           self._cycle,
            "last_analysis":   self.state.last_analysis_ts,
            "is_running":      self._running,
            "top_topics":      sorted(
                self.get_topics().items(),
                key=lambda x: x[1], reverse=True
            )[:8],
        }

    def search_insights(self, query: str) -> List[Insight]:
        """Busca insights por palavra-chave."""
        q = query.lower()
        results = []
        for ins in self.insights.values():
            score = 0
            if q in ins.chunk_text.lower(): score += 3
            if q in ins.summary.lower():    score += 2
            if any(q in t.lower() for t in ins.tags): score += 1
            if score > 0:
                results.append((score, ins))
        return [ins for _, ins in sorted(results, reverse=True)[:10]]

    # ─── Persistência ─────────────────────────────────────────────────────────

    def _save_state(self) -> None:
        path = self.output_dir / "curiosity_state.json"
        data = {
            "total_analyses":    self.state.total_analyses,
            "insights_found":    self.state.insights_found,
            "last_analysis_ts":  self.state.last_analysis_ts,
            "seen_chunk_hashes": self.state.seen_chunk_hashes[-500:],
            "topic_index":       self.state.topic_index,
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _save_insights(self) -> None:
        path = self.output_dir / "insights.json"
        data = [ins.to_dict() for ins in self.insights.values()]
        # Limitar tamanho
        data = sorted(data, key=lambda x: x["curiosity_score"], reverse=True)
        data = data[: self.MAX_INSIGHTS]
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_state(self) -> None:
        state_path   = self.output_dir / "curiosity_state.json"
        insight_path = self.output_dir / "insights.json"

        if state_path.exists():
            data = json.loads(state_path.read_text(encoding="utf-8"))
            self.state.total_analyses    = data.get("total_analyses", 0)
            self.state.insights_found    = data.get("insights_found", 0)
            self.state.last_analysis_ts  = data.get("last_analysis_ts", 0.0)
            self.state.seen_chunk_hashes = data.get("seen_chunk_hashes", [])
            self.state.topic_index       = data.get("topic_index", {})
            print(f"[Curiosity] Estado restaurado - "
                  f"{self.state.total_analyses} análises anteriores")

        if insight_path.exists():
            data = json.loads(insight_path.read_text(encoding="utf-8"))
            for d in data:
                try:
                    ins = Insight(**d)
                    ins.is_new = False
                    self.insights[ins.id] = ins
                except Exception:
                    pass
            print(f"[Curiosity] {len(self.insights)} insights restaurados")

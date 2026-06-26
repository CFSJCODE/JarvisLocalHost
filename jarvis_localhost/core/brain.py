"""
brain.py — J.A.R.V.I.S Central Intelligence
Orchestrates the neural model, RAG, PDF processor and system monitor.
"""

import re
import json
import asyncio
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List

from jarvis_localhost.ai.neural import (
    JarvisTransformer, JarvisConfig,
    JarvisTokenizer, JarvisTrainer, TrainConfig,
    VectorStore, RAGEngine,
    CuriosityEngine, Insight,
)
from jarvis_localhost.processing.pdf_processor import PDFProcessor
from jarvis_localhost.monitoring.system_monitor import SystemMonitor
from jarvis_localhost.storage.database import JarvisDB
from jarvis_localhost.projects.project_manager import ProjectManager
from jarvis_localhost.integrations.cluster_client import ClusterClient, ClusterError
from jarvis_localhost.integrations.local_voice import LocalVoice


# ─── Conversation State ───────────────────────────────────────────────────────

JARVIS_GREETINGS = [
    "Sistemas online, Senhor. Como posso auxiliá-lo?",
    "À disposição, Senhor. Todos os sistemas respondem normalmente.",
    "Prontamente, Senhor. Aguardando suas ordens.",
]

PROJECT_QUESTIONS = {
    "nome":      r"(nome|título|chama|projeto)",
    "tipo":      r"(tipo|categoria|classe|área)",
    "descricao": r"(descrição|descreva|trata-se|sobre)",
}


class ConversationMemory:
    """Stores short-term dialogue context."""
    MAX_HISTORY = 20

    def __init__(self):
        self.turns: List[Dict] = []

    def add(self, role: str, text: str):
        self.turns.append({"role": role, "text": text})
        if len(self.turns) > self.MAX_HISTORY:
            self.turns.pop(0)

    def recent(self, n: int = 6) -> List[Dict]:
        return self.turns[-n:]

    def last_user(self) -> str:
        for t in reversed(self.turns):
            if t["role"] == "user":
                return t["text"]
        return ""


# ─── JARVIS Brain ─────────────────────────────────────────────────────────────

class JarvisBrain:
    """
    Central intelligence for J.A.R.V.I.S.
    Manages: model lifecycle, PDF ingestion, RAG, system monitoring, chat.
    """

    MODEL_PATH     = "data/models/jarvis_final.pt"
    TOKENIZER_PATH = "data/models/jarvis_tokenizer.json"
    STORE_PATH     = "data/embeddings/vector_store"

    def __init__(self):
        self.monitor   = SystemMonitor()
        self.processor = PDFProcessor()
        self.memory    = ConversationMemory()
        self.db        = JarvisDB()
        self.pm        = ProjectManager()
        self.cluster_boot_error: Optional[str] = None
        try:
            self.cluster = ClusterClient.from_env()
        except Exception as e:
            self.cluster = None
            self.cluster_boot_error = str(e)
        self.voice     = LocalVoice.from_env()
        self.session_id = __import__('uuid').uuid4().hex[:8]
        self.projects: List[Dict] = self.db.list_projects()

        # Model state
        self.tokenizer: Optional[JarvisTokenizer]  = None
        self.model:     Optional[JarvisTransformer] = None
        self.rag:       Optional[RAGEngine]         = None
        self.store:     VectorStore                 = VectorStore()

        self.is_trained    = False
        self.is_training   = False
        self.train_progress: Dict = {}

        # Curiosity engine — autonomous document analysis
        self.curiosity_callback: Optional[Callable] = None
        self.curiosity = CuriosityEngine(
            corpus_dir = "data/embeddings",
            output_dir = "data/curiosity",
            on_insight = self._on_new_insight,
        )

        # Deterministic responses for special intents
        # (used before/after model is trained)
        self._intent_rules = self._build_intent_rules()

        self._try_load_existing()
        # Start curiosity engine after boot. It only reads local corpora and
        # surfaces insights; it does not execute commands or change code.
        self.curiosity.start()
        print("[Brain] J.A.R.V.I.S. online.")

    def shutdown(self) -> None:
        """Stop background workers before the FastAPI process exits."""
        if getattr(self, "curiosity", None):
            self.curiosity.stop()

    # ─── Boot ─────────────────────────────────────────────────────────────────

    def _try_load_existing(self):
        """Try to load previously trained model and token store."""
        try:
            if Path(self.TOKENIZER_PATH).exists():
                self.tokenizer = JarvisTokenizer.load(self.TOKENIZER_PATH)
                print("[Brain] Tokenizer loaded.")

            if Path(self.MODEL_PATH).exists() and self.tokenizer:
                self.model = JarvisTransformer.load(self.MODEL_PATH)
                self.store.load(self.STORE_PATH)
                self.rag = RAGEngine(self.model, self.tokenizer, self.store)
                self.is_trained = True
                print("[Brain] Modelo neural carregado. RAG ativo.")
        except Exception as e:
            print(f"[Brain] Could not load model: {e}")

    # ─── Intent Rules ─────────────────────────────────────────────────────────

    def _build_intent_rules(self):
        return [
            # Wake / greeting
            {
                "pattern": r"jarvis.*(acord|acor|awake|online|ai\b)",
                "response": "Para o senhor sempre.",
                "intent":  "wake",
            },
            # Status
            {
                "pattern": r"\b(status|relat[oó]rio|diagnóstico|como (está|vai))\b",
                "response": None,   # dynamic
                "intent":  "status",
            },
            # New project
            {
                "pattern": r"(quero criar|criar|novo).*(projeto|arquivo|file)",
                "response": None,
                "intent":  "new_project",
            },
            # Train model
            {
                "pattern": r"(treinar|trein|train|aprender|learn).*(model|ia|neural|jarvis)",
                "response": None,
                "intent":  "train",
            },
            # Curiosity / autonomous learning
            {
                "pattern": r"(curiosidade|curioso|insight|descobriu|aprendeu|tau)",
                "response": None,
                "intent": "curiosity",
            },
            # Optional local/LAN cluster offload
            {
                "pattern": r"(cluster|aether|worker|workers|offload|processamento pesado|gpt local)",
                "response": None,
                "intent": "cluster",
            },
            # Offline voice/TTS status
            {
                "pattern": r"(voz|falar|fala|tts|wake word|microfone)",
                "response": None,
                "intent": "voice",
            },
            # System info
            {
                "pattern": r"(cpu|memória|mem[oó]ria|disco|rede|temperatura|sistema|hardware)",
                "response": None,
                "intent":  "system_info",
            },
            # Time
            {
                "pattern": r"\b(hora|que horas|time)\b",
                "response": None,
                "intent":  "time",
            },
            # Thank you
            {
                "pattern": r"\b(obrigad[ao]|valeu|thanks|thank you)\b",
                "response": "É sempre um prazer servir, Senhor.",
                "intent":  "thanks",
            },
        ]

    def _detect_intent(self, text: str) -> str:
        lower = text.lower()
        for rule in self._intent_rules:
            if re.search(rule["pattern"], lower):
                return rule["intent"]
        return "chat"

    # ─── Chat ─────────────────────────────────────────────────────────────────

    async def chat(self, user_text: str) -> Dict[str, Any]:
        """
        Main chat entry point. Returns response dict with:
        text, intent, action, data.
        """
        self.memory.add("user", user_text)
        self.db.save_message("user", user_text, self.session_id, intent=None)
        intent   = self._detect_intent(user_text)
        response = await self._handle_intent(intent, user_text)
        self.memory.add("jarvis", response["text"])
        self.db.save_message("jarvis", response["text"], self.session_id,
                             intent=intent,
                             sources=response.get("data", {}).get("sources"))
        return response

    async def _handle_intent(self, intent: str, text: str) -> Dict:
        # Fixed-response intents
        for rule in self._intent_rules:
            if rule["intent"] == intent and rule["response"]:
                return {"text": rule["response"], "intent": intent, "action": None, "data": {}}

        # Dynamic intents
        if intent == "status":
            return self._handle_status()

        elif intent == "new_project":
            return {
                "text":   "Claro, Senhor. Ativei o formulário de novo projeto no painel lateral. "
                          "Preencha os dados e registrarei imediatamente nos arquivos da Stark Industries.",
                "intent": intent,
                "action": "open_project_form",
                "data":   {},
            }

        elif intent == "train":
            return {
                "text":   "Entendido, Senhor. Para iniciar o treinamento do modelo neural, "
                          "faça upload de documentos PDF primeiro usando o painel de documentos. "
                          "Quando pronto, pressione 'Iniciar Treinamento'.",
                "intent": intent,
                "action": "show_train_panel",
                "data":   {"is_trained": self.is_trained},
            }

        elif intent == "curiosity":
            ins = self.get_random_insight()
            if ins:
                return {
                    "text": (
                        "Encontrei um ponto curioso nos documentos, Senhor: "
                        f"{ins.get('summary', '')}"
                    ),
                    "intent": intent,
                    "action": "show_curiosity",
                    "data": {"insight": ins},
                }
            return {
                "text": (
                    "Meu motor de curiosidade está ativo, mas ainda não há insights. "
                    "Carregue PDFs ou aguarde o próximo ciclo de análise local."
                ),
                "intent": intent,
                "action": "show_curiosity",
                "data": {},
            }

        elif intent == "cluster":
            return self._handle_cluster_status()

        elif intent == "voice":
            status = self.get_voice_status()
            enabled = "ativada" if status.get("enabled") else "desativada"
            active = "em escuta lógica" if status.get("active") else "em espera"
            return {
                "text": (
                    f"Voz local {enabled}, Senhor. Wake word: "
                    f"{status.get('wake_word')}. Estado: {active}. "
                    "Nenhum reconhecimento por API externa é usado."
                ),
                "intent": intent,
                "action": None,
                "data": status,
            }

        elif intent == "system_info":
            snap = self.monitor.snapshot()
            cpu  = snap["cpu"]
            mem  = snap["memory"]
            disk = snap["disk"]
            reply = (
                f"Relatório do sistema, Senhor: "
                f"CPU em {cpu['percent']}% a {cpu['freq_mhz']} MHz, "
                f"memória {mem['percent']}% utilizada "
                f"({mem['used_gb']}GB de {mem['total_gb']}GB), "
                f"disco com {disk['percent']}% de uso. "
            )
            if cpu.get("temperature"):
                reply += f"Temperatura da CPU: {cpu['temperature']}°C. "
            alerts = self.monitor.check_alerts()
            if alerts:
                reply += " ⚠ Alertas: " + "; ".join(alerts)
            else:
                reply += "Todos os parâmetros dentro do normal."
            return {"text": reply, "intent": intent, "action": "update_metrics",
                    "data": snap}

        elif intent == "time":
            import datetime
            now = datetime.datetime.now().strftime("%H:%M:%S")
            return {"text": f"São exatamente {now}, Senhor.", "intent": intent,
                    "action": None, "data": {}}

        elif intent == "chat":
            return await self._rag_or_fallback(text)

        return {"text": "Processando, Senhor…", "intent": intent, "action": None, "data": {}}

    def _handle_status(self) -> Dict:
        snap = self.monitor.snapshot()
        cpu  = snap["cpu"]
        mem  = snap["memory"]
        trained_str = "Modelo neural ativo e treinado." if self.is_trained \
                      else "Modelo neural aguardando treinamento."
        docs_str = f"{len(self.store)} chunks indexados." if len(self.store) > 0 \
                   else "Nenhum documento indexado ainda."
        text = (
            f"Status geral: sistemas operacionais. "
            f"CPU {cpu['percent']}%, RAM {mem['percent']}%. "
            f"{trained_str} {docs_str} "
            f"Plataforma: {snap['platform']}. "
            f"Uptime: {snap['uptime_hours']:.1f}h."
        )
        return {"text": text, "intent": "status", "action": "update_metrics",
                "data": snap}

    def _handle_cluster_status(self) -> Dict:
        snap = self.get_cluster_snapshot()
        status = snap.get("status", {})
        workers = snap.get("workers", [])
        if self.cluster_boot_error:
            text = f"Conector Aether bloqueado na inicialização: {self.cluster_boot_error}"
        elif not status.get("enabled"):
            text = (
                "Offload Aether está desativado. O Jarvis continua 100% local nesta "
                "máquina. Para liberar workers locais/LAN, configure "
                "JARVIS_CLUSTER_ENABLED=1 e JARVIS_CLUSTER_URL."
            )
        elif status.get("error"):
            text = f"Cluster Aether configurado, mas indisponível: {status['error']}"
        else:
            cluster = status.get("cluster", {})
            text = (
                "Cluster Aether conectado. "
                f"Workers online: {cluster.get('workers_online', len(workers))}; "
                f"tarefas em fila: {cluster.get('tasks_queued', 0)}. "
                "Offload permitido apenas por tarefas explícitas e allowlist local."
            )
        return {
            "text": text,
            "intent": "cluster",
            "action": "show_cluster",
            "data": snap,
        }

    async def _rag_or_fallback(self, text: str) -> Dict:
        """Try RAG if model is trained; otherwise use rule-based fallback."""
        if self.is_trained and self.rag and len(self.store) > 0:
            result = self.rag.answer(text, top_k=3, max_new=80)
            return {
                "text":   result["answer"],
                "intent": "chat",
                "action": "show_sources" if result["sources"] else None,
                "data":   {"sources": result["sources"]},
            }

        # Fallback responses before model is trained
        fallbacks = [
            "Entendido, Senhor. Processando sua solicitação.",
            "Informação registrada. Posso ajudar com mais alguma coisa?",
            "Analisando, Senhor. Quando o modelo neural estiver treinado, "
            "poderei responder com base nos documentos carregados.",
            "Compreendido. Faça upload de documentos PDF para que eu possa "
            "aprender e responder com mais precisão.",
        ]
        import random
        return {"text": random.choice(fallbacks), "intent": "chat",
                "action": None, "data": {}}

    # ─── PDF Processing & Training ────────────────────────────────────────────

    def process_pdf(self, pdf_path: str) -> Dict:
        """Process a PDF and index it for RAG. Returns stats."""
        result = self.processor.process(pdf_path)
        doc_id = self.db.save_document(
            filename=result.filename,
            path=pdf_path,
            stats=result.stats,
            corpus=result.training_corpus,
        )

        # If model is ready, index immediately
        if self.is_trained and self.rag:
            n = self.rag.index_document(result.training_corpus, result.filename)
            self.store.save(self.STORE_PATH)
            self.db.mark_indexed(doc_id)
            return {**result.stats, "indexed_chunks": n, "filename": result.filename, "doc_id": doc_id}

        # Store corpus for later training
        Path("data/embeddings").mkdir(parents=True, exist_ok=True)
        return {**result.stats, "indexed_chunks": 0, "filename": result.filename, "doc_id": doc_id}

    def start_training(
        self,
        progress_callback: Optional[Callable[[Dict], None]] = None
    ) -> None:
        """Start model training in a background thread."""
        if self.is_training:
            return
        self.is_training = True
        self.train_progress = {"percent": 0, "message": "Preparando treinamento."}

        def emit(info: Dict) -> None:
            self.train_progress = {**self.train_progress, **info, "updated_at": time.time()}
            if progress_callback:
                progress_callback(info)

        def _run():
            run_id = self.db.start_training_run()
            history: List[Dict] = []
            try:
                corpora = list(Path("data/embeddings").glob("*_corpus.txt"))
                if not corpora:
                    emit({"error": "Nenhum documento encontrado. Faça upload de PDFs primeiro."})
                    self.db.update_training_run(
                        run_id,
                        finished_at=time.time(),
                        status="error",
                        history=history,
                    )
                    return

                emit({"percent": 2, "message": "Lendo corpus local."})
                texts = [p.read_text(encoding="utf-8", errors="replace") for p in corpora]
                full_corpus = "\n\n".join(texts)

                # 1. Train tokenizer
                if not self.tokenizer:
                    emit({"percent": 5, "message": "Treinando tokenizador BPE."})
                    self.tokenizer = JarvisTokenizer(vocab_size=4000)
                    self.tokenizer.train(full_corpus)
                    Path("data/models").mkdir(parents=True, exist_ok=True)
                    self.tokenizer.save(self.TOKENIZER_PATH)

                # 2. Build model
                cfg = JarvisConfig(
                    vocab_size  = self.tokenizer.vocab_actual_size,
                    context_len = 128,
                    embed_dim   = 128,
                    num_heads   = 4,
                    num_layers  = 4,
                    ff_dim      = 512,
                    dropout     = 0.1,
                )
                self.model = JarvisTransformer(cfg)

                # 3. Train
                train_cfg = TrainConfig(
                    max_steps       = 1000,
                    batch_size      = 8,
                    context_len     = 128,
                    log_interval    = 25,
                    eval_interval   = 200,
                    checkpoint_every= 500,
                )

                def _train_callback(info: Dict) -> None:
                    history.append(info)
                    emit(info)

                trainer = JarvisTrainer(self.model, self.tokenizer, full_corpus, train_cfg)
                trainer.train(callback=_train_callback)

                # 4. Setup RAG and index docs
                self.store = VectorStore()
                self.rag   = RAGEngine(self.model, self.tokenizer, self.store)

                for p in corpora:
                    text = p.read_text(encoding="utf-8", errors="replace")
                    stem = p.stem.replace("_corpus", "")
                    self.rag.index_document(text, stem)

                self.store.save(self.STORE_PATH)
                self.is_trained = True
                self.model.save(self.MODEL_PATH)

                self.db.update_training_run(
                    run_id,
                    finished_at=time.time(),
                    status="done",
                    steps=len(history),
                    vocab_size=self.tokenizer.vocab_actual_size,
                    history=history[-200:],
                )
                emit({"done": True, "percent": 100, "message": "Treinamento concluído, Senhor!"})

            except Exception as e:
                print(f"[Brain] Training error: {e}")
                self.db.update_training_run(
                    run_id,
                    finished_at=time.time(),
                    status="error",
                    history=history[-200:],
                )
                emit({"error": str(e)})
            finally:
                self.is_training = False

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

    # ─── Projects ─────────────────────────────────────────────────────────────

    def save_project(self, project: Dict) -> Dict:
        """Save project to SQLite and generate project files."""
        saved = self.db.save_project(
            name        = project.get("name", "Projeto"),
            type_       = project.get("type", ""),
            priority    = project.get("priority", "BETA"),
            description = project.get("description", ""),
            tags        = project.get("tags", []),
        )
        # Generate project files
        try:
            zip_path = self.pm.generate(saved)
            self.db.update_project(saved["id"], files=[zip_path])
            saved["zip_path"] = zip_path
        except Exception as e:
            print(f"[Brain] Project file gen error: {e}")
        self.projects = self.db.list_projects()
        return saved

    def list_projects(self) -> List[Dict]:
        self.projects = self.db.list_projects()
        return self.projects

    def get_project(self, pid: str) -> Optional[Dict]:
        return self.db.get_project(pid)

    def get_chat_history(self, limit: int = 50) -> List[Dict]:
        return self.db.get_history(limit=limit)

    def get_db_stats(self) -> Dict:
        return self.db.get_stats()

    # ─── Optional Cluster / Voice Integrations ───────────────────────────────

    def get_cluster_snapshot(self) -> Dict:
        if self.cluster_boot_error or not self.cluster:
            return {
                "status": {
                    "enabled": False,
                    "error": self.cluster_boot_error or "Cluster não inicializado.",
                },
                "workers": [],
                "tasks": [],
            }
        return self.cluster.snapshot()

    def get_cluster_workers(self) -> Dict:
        if not self.cluster:
            return {"enabled": False, "workers": [], "error": self.cluster_boot_error}
        return self.cluster.workers()

    def get_cluster_tasks(self) -> Dict:
        if not self.cluster:
            return {"enabled": False, "tasks": [], "error": self.cluster_boot_error}
        return self.cluster.tasks()

    def submit_cluster_task(
        self,
        command: str,
        required_tags: Optional[List[str]] = None,
        timeout_seconds: int = 120,
        priority: int = 5,
    ) -> Dict:
        if not self.cluster:
            raise ClusterError(self.cluster_boot_error or "Cluster não inicializado.")
        return self.cluster.submit_task(
            command,
            required_tags=required_tags,
            timeout_seconds=timeout_seconds,
            priority=priority,
        )

    def get_voice_status(self) -> Dict:
        return self.voice.status()

    def speak(self, text: str) -> Dict:
        return self.voice.speak(text)

    def update_wake_state(self, text: str) -> Dict:
        return self.voice.update_wake_state(text)

    # ─── System Metrics ───────────────────────────────────────────────────────

    def get_metrics(self) -> Dict:
        return self.monitor.snapshot()
    # ─── Curiosity Engine Integration ─────────────────────────────────────────

    def _on_new_insight(self, insight) -> None:
        """Callback chamado pelo CuriosityEngine quando um novo insight é gerado."""
        # Persist to DB
        try:
            self.db.save_insight(insight.to_dict())
        except Exception as e:
            print(f"[Brain] Insight DB save error: {e}")
        if self.curiosity_callback:
            self.curiosity_callback({
                "type":    "curiosity_insight",
                "insight": insight.to_dict(),
            })

    def set_curiosity_callback(self, cb: Callable) -> None:
        """Registra o callback WebSocket para notificações de curiosidade."""
        self.curiosity_callback = cb

    def get_insights(self, n: int = 20, tag: str = None) -> List[Dict]:
        # Try DB first (persisted), fallback to in-memory
        db_ins = self.db.get_insights(limit=n, tag=tag)
        if db_ins:
            return db_ins
        return [i.to_dict() for i in self.curiosity.get_top_insights(n, tag)]

    def get_random_insight(self) -> Optional[Dict]:
        ins = self.db.get_random_insight()
        if ins:
            return ins
        mem_ins = self.curiosity.get_random_insight()
        return mem_ins.to_dict() if mem_ins else None

    def get_curiosity_stats(self) -> Dict:
        stats = self.curiosity.get_stats()
        stats["db_insights"] = self.db.get_stats().get("insights", 0)
        timeline = self.db.get_curiosity_timeline(limit=20)
        if not timeline and stats.get("cycle"):
            timeline = [{
                "cycle_num": stats.get("cycle", 0),
                "insights_found": stats.get("insights_found", 0),
                "docs_scanned": len(list(Path("data/embeddings").glob("*_corpus.txt"))),
                "top_score": 0,
                "ts": stats.get("last_analysis", time.time()),
            }]
        stats["curiosity_timeline"] = timeline
        return stats

    def search_insights(self, query: str) -> List[Dict]:
        db_res = self.db.search_insights(query)
        if db_res:
            return db_res
        return [i.to_dict() for i in self.curiosity.search_insights(query)]

    def get_topics(self) -> Dict:
        db_topics = self.db.get_topics()
        if db_topics:
            return db_topics
        return self.curiosity.get_topics()

    def get_metrics_history(self, minutes: int = 30) -> List[Dict]:
        return self.db.get_metrics_history(minutes)

    def save_metrics_snapshot(self, snap: Dict):
        self.db.save_metric(snap)

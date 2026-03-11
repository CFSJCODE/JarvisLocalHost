import asyncio
import threading
import glob
import os
import torch
import time
import uuid
from pathlib import Path
from typing import Dict, Any, Callable

try:
    import psutil
except ImportError:
    psutil = None

from pdf_processor import PDFProcessor
from system_monitor import SystemMonitor
from model import JarvisTransformer, JarvisConfig
from tokenizer import BasicTokenizer
from trainer import AITrainer
from embeddings import RAGEngine, VectorStore

class JarvisBrain:
    """
    Controlador Mestre. Rege a engenharia de dados (PDFs), treina o 
    modelo de linguagem autoregressivo, e ativa o pipeline de Inferência Local.
    """
    def __init__(self):
        # Saturação de threads para o Ryzen 5 4600G
        torch.set_num_threads(12)
        
        self.monitor = SystemMonitor()
        self.pdf_proc = PDFProcessor()
        
        self.tokenizer = BasicTokenizer()
        self.model = None
        self.store = VectorStore(dimension=256) 
        self.rag_engine = None
        
        self.is_training = False
        self.is_trained = False
        self.projects = []
        self.conversation_history = []

    def get_metrics(self) -> Dict[str, Any]:
        if psutil:
            return {
                "cpu": psutil.cpu_percent(),
                "ram": psutil.virtual_memory().percent,
                "disk": psutil.disk_usage('/').percent,
                "platform": "x86_64 (Ryzen 5 Local Mode)"
            }
        return {"cpu": 0, "ram": 0, "disk": 0, "status": "psutil_missing"}

    def process_pdf(self, filepath: str) -> Dict[str, Any]:
        res = self.pdf_proc.process(filepath)
        return {"words": res.word_count, "pages": res.total_pages, "file": filepath}

    def start_training(self, progress_callback: Callable[[Dict], None]):
        if self.is_training: return
        self.is_training = True

        def _train_thread():
            try:
                progress_callback({"percent": 5, "epoch": 0, "total": 10, "loss": "Sincronizando Datalake..."})
                
                corpus_files = glob.glob("data/embeddings/*_corpus.txt")
                if not corpus_files:
                    progress_callback({"error": "Datalake vazio. Carregue PDFs."})
                    return
                
                full_text = ""
                for f in corpus_files:
                    with open(f, 'r', encoding='utf-8') as doc:
                        full_text += doc.read() + " "
                
                self.tokenizer.build_vocab(full_text)
                
                cfg = JarvisConfig(
                    vocab_size=self.tokenizer.vocab_size, 
                    context_len=512, 
                    embed_dim=256, 
                    num_heads=8, 
                    num_layers=6
                )
                self.model = JarvisTransformer(cfg)
                
                trainer = AITrainer(self.model, self.tokenizer, full_text, batch_size=32)
                
                def _trainer_callback(info):
                    global_pct = 15 + int((info["epoch"] / info["total"]) * 70)
                    info["percent"] = global_pct
                    progress_callback(info)

                trainer.train_loop(epochs=10, callback=_trainer_callback)
                
                self.rag_engine = RAGEngine(self.model, self.tokenizer, self.store)
                for f in corpus_files:
                    with open(f, 'r', encoding='utf-8') as doc:
                        self.rag_engine.index_document(doc.read(), os.path.basename(f))
                
                self.is_trained = True
                progress_callback({"percent": 100, "epoch": 10, "total": 10, "loss": "Convergência Local OK"})
                
            except Exception as e:
                progress_callback({"error": f"Colapso de Hardware: {str(e)}"})
            finally:
                self.is_training = False

        threading.Thread(target=_train_thread, daemon=True).start()

    async def chat(self, message: str) -> Dict[str, Any]:
        """Inferência estritamente local."""
        if not self.is_trained or self.rag_engine is None:
            return {"answer": "Aviso: Os protocolos de inferência local requerem inicialização prévia, Senhor."}
        
        self.conversation_history.append(f"Sr. Stark: {message}")
        if len(self.conversation_history) > 6: self.conversation_history.pop(0)
            
        try:
            # Chamada síncrona: o processamento ocorre inteiramente na CPU local
            res = self.rag_engine.answer(message)
            
            # Indexação da resposta para memória de curto prazo
            learning_chunk = f"Conhecimento Adquirido: '{message}' -> '{res['answer']}'"
            self.rag_engine.index_document(learning_chunk, source="memoria_online_jarvis")
            
            return res
        except Exception as e:
            return {"answer": f"Ocorreu um erro na decodificação do tensor local: {str(e)}"}
        
    def save_project(self, project: dict) -> dict:
        if "id" not in project: project["id"] = uuid.uuid4().hex
        for i, p in enumerate(self.projects):
            if p.get("id") == project["id"]:
                self.projects[i] = project
                return project
        self.projects.append(project)
        return project
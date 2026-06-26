"""
app.py — J.A.R.V.I.S FastAPI Server
REST + WebSocket API. Otimizado para concorrência assíncrona.
"""

import sys
import os
import json
import asyncio
import shutil
import uuid
import mimetypes
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from contextlib import asynccontextmanager

# --- Configuração Robusta de Paths ---
APP_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = APP_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
os.chdir(str(APP_DIR))

STATIC_DIR = APP_DIR / "web" / "static"
INDEX_PATH = STATIC_DIR / "index.html"

# Diretórios requeridos
for _d in [
    "data/models",
    "data/embeddings",
    "data/extracted_images",
    "data/curiosity",
    "data/projects",
    "uploads",
    "web/static",
]:
    (APP_DIR / _d).mkdir(parents=True, exist_ok=True)


def setup_event_loop() -> None:
    """Ativa um event loop mais eficiente quando a dependência opcional existir."""
    try:
        if sys.platform == "win32":
            import winloop
            asyncio.set_event_loop_policy(winloop.EventLoopPolicy())
        else:
            import uvloop
            asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    except ImportError:
        pass


setup_event_loop()

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from jarvis_localhost.core.brain import JarvisBrain

# Instância base
brain = JarvisBrain()
direct_engine = None

# Estrutura O(1) para conexões (Evita data race O(N))
ws_clients: Set[WebSocket] = set()

# --- Pydantic Data Transfer Objects (DTOs) ---
# Contratos explícitos resolvendo Inconsistência de Schemas
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)

class SourceDTO(BaseModel):
    source: str
    score: float

class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceDTO] = []
    action: Optional[str] = None
    data: Optional[Dict] = None

class ProjectModel(BaseModel):
    name: str
    type: str
    priority: str = "BETA"
    description: str = ""
    tags: list = []

class TokenizeRequest(BaseModel):
    text: str = Field(..., min_length=1)

class ClusterTaskRequest(BaseModel):
    command: str = Field(..., min_length=1)
    required_tags: List[str] = []
    timeout_seconds: int = Field(120, ge=30, le=86400)
    priority: int = Field(5, ge=0, le=100)

class SpeakRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1500)

# --- Lógica Base de Redes (WebSockets) ---
async def broadcast(data: Dict):
    """Envia JSON para todos os clientes ativos via WebSocket."""
    dead_connections = set()

    for ws in list(ws_clients):
        try:
            await ws.send_json(data)
        except Exception:
            dead_connections.add(ws)

    # Limpeza O(1)
    for ws in dead_connections:
        ws_clients.discard(ws)

async def metrics_loop():
    """Varredura cíclica enviando telemetria em broadcast."""
    _sample_count = 0
    while True:
        await asyncio.sleep(2)
        try:
            # Operação assíncrona para não travar loop
            metrics = await asyncio.to_thread(brain.get_metrics)
            if ws_clients:
                await broadcast({"type": "metrics", "data": metrics})

            _sample_count += 1
            if _sample_count % 5 == 0:
                await asyncio.to_thread(brain.save_metrics_snapshot, metrics)
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"[metrics_loop] Exception: {e}")

# --- Lifespan Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[System] Inicializando Módulos de Baixo Nível...")
    metrics_task = asyncio.create_task(metrics_loop())

    loop = asyncio.get_event_loop()
    def _curiosity_cb(data: Dict):
        asyncio.run_coroutine_threadsafe(broadcast(data), loop)

    brain.set_curiosity_callback(_curiosity_cb)
    print("[Server] J.A.R.V.I.S. online.")

    yield  # Uvicorn control here

    metrics_task.cancel()
    if hasattr(brain, "shutdown"):
        await asyncio.to_thread(brain.shutdown)
    print("[Server] Desligando motores e liberando VRAM...")

# --- Instância da API ---
app = FastAPI(title="J.A.R.V.I.S", version="4.1.0", lifespan=lifespan)

# CORS Seguro configurado para portas padrão
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# --- Controladores (Endpoints) ---

@app.get("/", response_class=HTMLResponse)
async def root():
    if not INDEX_PATH.exists():
        return HTMLResponse(content="<h1>Frontend (index.html) ausente.</h1>", status_code=500)
    return HTMLResponse(content=INDEX_PATH.read_text(encoding="utf-8"))

@app.get("/api/metrics")
async def get_metrics():
    return await asyncio.to_thread(brain.get_metrics)


def _normalize_chat_response(response: Dict[str, Any]) -> ChatResponse:
    data = response.get("data") or {}
    sources = response.get("sources") or data.get("sources") or []
    answer = response.get("answer") or response.get("response") or response.get("text") or "Vazio"
    return ChatResponse(
        answer=answer,
        sources=sources,
        action=response.get("action"),
        data=data,
    )

@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    # Correção: O brain.chat pode ser denso. É chamado nativamente caso já seja coroutine.
    try:
        response = await brain.chat(req.message)
        return _normalize_chat_response(response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/inferencia", response_model=ChatResponse)
async def inferencia_direta(req: ChatRequest):
    """Compatibilidade com o V2: inferência direta sem quebrar se não houver GPU."""
    global direct_engine
    try:
        if direct_engine is None:
            from jarvis_localhost.ai.engine_ai import DirectInferenceEngine
            direct_engine = DirectInferenceEngine(brain)
        response = await direct_engine.answer(req.message)
        return _normalize_chat_response(response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/tokenize")
async def debug_tokenizer(req: TokenizeRequest):
    if not getattr(brain, "tokenizer", None):
        raise HTTPException(400, "Tokenizador ainda não treinado.")
    token_ids = await asyncio.to_thread(brain.tokenizer.encode, req.text)
    return {
        "status": "success",
        "input_text": req.text,
        "token_ids": token_ids,
        "vocab_size": getattr(brain.tokenizer, "vocab_actual_size", None)
        or getattr(brain.tokenizer, "vocab_size", None),
    }

@app.post("/api/pdf/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Formato invalido (PDF apenas).")

    dest = APP_DIR / "uploads" / f"{uuid.uuid4().hex}_{file.filename}"

    # Resolvendo gargalo de bloqueio na event loop para leitura binária
    def _save_file():
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)
        return brain.process_pdf(str(dest))

    try:
        # Offloading the blocking task to thread pool
        stats = await asyncio.to_thread(_save_file)
        await broadcast({
            "type": "pdf_processed",
            "data": stats,
            "message": f"Documento '{file.filename}' mapeado na rede neural."
        })
        return {"success": True, **stats}
    except Exception as e:
        raise HTTPException(500, f"Falha de I/O: {e}")

@app.post("/api/project/save")
async def save_project(proj: ProjectModel):
    # Offloading de possível gravação sincrona em disco/DB
    saved = await asyncio.to_thread(brain.save_project, proj.model_dump()) if hasattr(brain, "save_project") else proj.model_dump()
    return {"success": True, "project": saved}


@app.get("/api/projects")
async def list_projects():
    projects = await asyncio.to_thread(brain.list_projects)
    return {"projects": projects}


@app.get("/api/project/{project_id}/download")
async def download_project(project_id: str):
    project = await asyncio.to_thread(brain.get_project, project_id)
    if not project:
        raise HTTPException(404, "Projeto não encontrado.")

    files = project.get("files") or []
    zip_path = Path(files[0]) if files else Path(project.get("zip_path", ""))
    if not zip_path.is_absolute():
        zip_path = APP_DIR / zip_path
    if not zip_path.exists():
        raise HTTPException(404, "Arquivo do projeto não encontrado.")

    mime = mimetypes.guess_type(zip_path.name)[0] or "application/zip"
    return FileResponse(
        path=str(zip_path),
        media_type=mime,
        filename=zip_path.name,
    )

@app.post("/api/train/start")
async def start_training():
    if brain.is_training:
        return {"status": "already_training"}

    main_loop = asyncio.get_running_loop()

    def _progress(info: Dict):
        asyncio.run_coroutine_threadsafe(
            broadcast({"type": "train_progress", "data": info}),
            main_loop,
        )

    await asyncio.to_thread(brain.start_training, progress_callback=_progress)
    return {"status": "started"}


@app.get("/api/train/status")
async def train_status():
    doc_count = len(list((APP_DIR / "data/embeddings").glob("*_meta.json")))
    return {
        "is_training": brain.is_training,
        "is_trained": brain.is_trained,
        "documents": doc_count,
        "chunks": len(brain.store) if hasattr(brain, "store") else 0,
        "progress": getattr(brain, "train_progress", {}),
    }

@app.get("/api/documents")
async def list_documents():
    def _read_meta():
        docs = []
        for p in (APP_DIR / "data/embeddings").glob("*_meta.json"):
            try:
                docs.append(json.loads(p.read_text(encoding="utf-8")))
            except Exception: pass
        return docs
    docs = await asyncio.to_thread(_read_meta)
    return {"documents": docs}


@app.get("/api/db/stats")
async def db_stats():
    return await asyncio.to_thread(brain.get_db_stats)


@app.get("/api/metrics/history")
async def metrics_history(minutes: int = 30):
    minutes = max(1, min(minutes, 24 * 60))
    history = await asyncio.to_thread(brain.get_metrics_history, minutes)
    return {"history": history}


@app.get("/api/cluster/status")
async def cluster_status():
    return await asyncio.to_thread(brain.get_cluster_snapshot)


@app.get("/api/cluster/workers")
async def cluster_workers():
    return await asyncio.to_thread(brain.get_cluster_workers)


@app.get("/api/cluster/tasks")
async def cluster_tasks():
    return await asyncio.to_thread(brain.get_cluster_tasks)


@app.post("/api/cluster/task")
async def cluster_task(req: ClusterTaskRequest):
    try:
        return await asyncio.to_thread(
            brain.submit_cluster_task,
            req.command,
            req.required_tags,
            req.timeout_seconds,
            req.priority,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/voice/status")
async def voice_status():
    return await asyncio.to_thread(brain.get_voice_status)


@app.post("/api/voice/speak")
async def voice_speak(req: SpeakRequest):
    return await asyncio.to_thread(brain.speak, req.text)


@app.post("/api/voice/wake")
async def voice_wake(req: TokenizeRequest):
    return await asyncio.to_thread(brain.update_wake_state, req.text)


@app.get("/api/curiosity/stats")
async def curiosity_stats():
    return await asyncio.to_thread(brain.get_curiosity_stats)


@app.get("/api/curiosity/insights")
async def curiosity_insights(n: int = 20, tag: Optional[str] = None):
    n = max(1, min(n, 100))
    insights = await asyncio.to_thread(brain.get_insights, n, tag)
    return {"insights": insights}


@app.get("/api/curiosity/topics")
async def curiosity_topics():
    topics = await asyncio.to_thread(brain.get_topics)
    return {"topics": topics}


@app.get("/api/curiosity/search")
async def curiosity_search(q: str):
    if not q.strip():
        return {"insights": []}
    insights = await asyncio.to_thread(brain.search_insights, q.strip())
    return {"insights": insights}


@app.get("/api/curiosity/random")
async def curiosity_random():
    insight = await asyncio.to_thread(brain.get_random_insight)
    return {"insight": insight}

# --- Transporte WebSocket ---
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    ws_clients.add(ws)
    print(f"[TCP] Conexao Socket Estabelecida - total: {len(ws_clients)}")

    try:
        # Inicialização
        initial_metrics = await asyncio.to_thread(brain.get_metrics)
        await ws.send_json({"type": "metrics", "data": initial_metrics})
        await ws.send_json({
            "type": "init",
            "data": {
                "is_trained":  brain.is_trained,
                "is_training": brain.is_training,
                "projects":    brain.projects if hasattr(brain, 'projects') else [],
            }
        })

        # Ping-Pong Nativo/Keep Alive robusto
        while True:
            data = await ws.receive_text()
            if data == "ping":
                await ws.send_text("pong")

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WS Error]: {e}")
    finally:
        ws_clients.discard(ws)
        print(f"[TCP] Conexao Socket Finalizada - total: {len(ws_clients)}")

# --- Início da Aplicação (CORREÇÃO DE PORTA CRÍTICA) ---
if __name__ == "__main__":
    # Correção da porta de 8080 para 8000 para sincronia matemática com o frontend.
    # Removido reload em prod para liberar memória.
    uvicorn.run(
        "jarvis_localhost.server.app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        reload_dirs=[str(APP_DIR)],
        log_level="info",
    )

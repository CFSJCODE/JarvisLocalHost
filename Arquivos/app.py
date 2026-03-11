"""
app.py — J.A.R.V.I.S FastAPI Server
REST + WebSocket API serving the frontend.
"""

import sys
import os
import json
import asyncio
import shutil
import uuid
from pathlib import Path
from typing import Dict, Any
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ─── Fix module path ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Ensure required data directories exist
for _d in ["data/models", "data/embeddings", "data/extracted_images", "uploads", "static"]:
    Path(_d).mkdir(parents=True, exist_ok=True)

# Generate a dummy index.html if it doesn't exist to prevent 404
if not Path("static/index.html").exists():
    Path("static/index.html").write_text("<h1>J.A.R.V.I.S Online</h1>", encoding="utf-8")

# Importação da engine neural nativa
from brain import JarvisBrain

# Boot brain
brain = JarvisBrain()

# Active WebSocket connections
ws_clients: list[WebSocket] = []


# ─── WebSocket Broadcast ──────────────────────────────────────────────────────
async def broadcast(data: Dict):
    """Send JSON to all connected WebSocket clients."""
    dead = []
    for ws in ws_clients:
        try:
            await ws.send_json(data)
        except Exception:
            dead.append(ws)
    for ws in dead:
        ws_clients.remove(ws)


# ─── Background Metrics Task ──────────────────────────────────────────────────
async def metrics_loop():
    """Every 2 seconds, push live system metrics to connected clients."""
    while True:
        await asyncio.sleep(2)
        if ws_clients:
            try:
                metrics = brain.get_metrics()
                await broadcast({"type": "metrics", "data": metrics})
            except Exception as e:
                print(f"[metrics_loop] {e}")


# Gerenciamento de ciclo de vida moderno do FastAPI
@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(metrics_loop())
    print("[Server] J.A.R.V.I.S. server started.")
    yield
    task.cancel()
    print("[Server] J.A.R.V.I.S. server stopped.")


# ─── App Setup ────────────────────────────────────────────────────────────────
app = FastAPI(title="J.A.R.V.I.S", version="4.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")


# ─── Pydantic Models ──────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str

class ProjectModel(BaseModel):
    name: str
    type: str
    priority: str = "BETA"
    description: str = ""
    tags: list = []

class TokenizeRequest(BaseModel):
    text: str


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    html = Path("static/index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)

@app.get("/api/metrics")
async def get_metrics():
    return brain.get_metrics()

# Chat
@app.post("/api/chat")
async def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(400, "Empty message")
    response = await brain.chat(req.message)
    return response

# NLP Debugging / Tokenizer
@app.post("/api/tokenize")
async def debug_tokenizer(req: TokenizeRequest):
    """Rota direta para interagir com o Tokenizador da rede neural."""
    if not brain.is_trained or not hasattr(brain, 'tokenizer'):
        raise HTTPException(400, "A rede neural (e o tokenizador) ainda não foram inicializados. Faça o treino primeiro.")
    
    # Conecta o input da rede ao método encode() do tokenizer.py
    token_ids = brain.tokenizer.encode(req.text)
    
    return {
        "status": "success",
        "input_text": req.text,
        "tensor_ids": token_ids,
        "vocab_size": brain.tokenizer.vocab_size
    }

# PDF Upload & Processing
@app.post("/api/pdf/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted.")

    # Save to uploads dir
    uploads = Path("uploads")
    uploads.mkdir(exist_ok=True)
    dest = uploads / f"{uuid.uuid4().hex}_{file.filename}"

    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Process
    try:
        stats = brain.process_pdf(str(dest))
        await broadcast({
            "type": "pdf_processed",
            "data": stats,
            "message": f"Documento '{file.filename}' processado: {stats['words']} palavras, {stats['pages']} páginas.",
        })
        return {"success": True, **stats}
    except Exception as e:
        raise HTTPException(500, f"Processing error: {e}")

# Project Management
@app.post("/api/project/save")
async def save_project(proj: ProjectModel):
    # .model_dump() is valid for Pydantic v2
    saved = brain.save_project(proj.model_dump())
    return {"success": True, "project": saved}

@app.get("/api/projects")
async def list_projects():
    return {"projects": brain.projects}

# Model Training Pipeline
@app.post("/api/train/start")
async def start_training():
    if brain.is_training: 
        return {"status": "already_training"}
    
    # 1. Capturamos o Event Loop ativo da thread principal (FastAPI)
    main_loop = asyncio.get_running_loop()
    
    def _progress(info: Dict):
        try:
            # 2. Injetamos de forma segura a rotina usando o loop capturado
            asyncio.run_coroutine_threadsafe(
                broadcast({"type": "train_progress", "data": info}), 
                main_loop
            )
        except Exception as e: 
            print(f"Erro no broadcast de progresso: {e}")

    brain.start_training(progress_callback=_progress)
    return {"status": "started"}

@app.get("/api/train/status")
async def train_status():
    return {
        "is_training": brain.is_training,
        "is_trained": brain.is_trained,
        "documents": len(list(Path("data/embeddings").glob("*_corpus.txt"))),
        "chunks": len(brain.store) if hasattr(brain, 'store') else 0,
    }

@app.get("/api/documents")
async def list_documents():
    docs = []
    for p in Path("data/embeddings").glob("*_meta.json"):
        try:
            meta = json.loads(p.read_text(encoding="utf-8"))
            docs.append(meta)
        except Exception:
            pass
    return {"documents": docs}


# ─── WebSockets ───────────────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    ws_clients.append(ws)
    print(f"[WS] Client connected — total: {len(ws_clients)}")
    try:
        # Send initial metrics
        await ws.send_json({"type": "metrics", "data": brain.get_metrics()})
        await ws.send_json({
            "type": "init",
            "data": {
                "is_trained": brain.is_trained,
                "is_training": brain.is_training,
                "projects": brain.projects,
            }
        })
        # Keep alive
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        if ws in ws_clients:
            ws_clients.remove(ws)
        print(f"[WS] Client disconnected — total: {len(ws_clients)}")


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
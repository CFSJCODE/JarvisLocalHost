"""
app.py — J.A.R.V.I.S FastAPI Server
REST + WebSocket API. Otimizado para concorrência assíncrona.
"""

import asyncio
import hmac
import json
import mimetypes
import os
import re
import secrets
import sys
import time
import unicodedata
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlsplit

# --- Configuração Robusta de Paths ---
_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _PACKAGE_ROOT.parent
REPO_ROOT = _REPO_ROOT
sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv
    _env_file = _REPO_ROOT / ".env"
    if _env_file.is_file():
        load_dotenv(_env_file)
except ImportError:
    pass


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
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from jarvis_localhost.core.brain import JarvisBrain
from jarvis_localhost.logging_config import configure_logging
from jarvis_localhost.processing.pdf_processor import EmptyDocumentError
from jarvis_localhost.paths import (
    CORPUS_ROOT,
    PACKAGE_ROOT,
    STATIC_ROOT,
    UPLOADS_ROOT,
    assert_runtime_path,
    ensure_runtime_directories,
)
from jarvis_localhost.sovereign import POLICY, SovereignModeViolation

APP_DIR = PACKAGE_ROOT
STATIC_DIR = STATIC_ROOT
INDEX_PATH = STATIC_DIR / "index.html"
ensure_runtime_directories()


def _bounded_env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    """Read a bounded integer without exposing the environment value in errors."""

    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    return max(minimum, min(value, maximum))


LOCAL_HOSTNAMES = frozenset({"localhost", "127.0.0.1", "::1"})
MUTATING_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
SESSION_COOKIE = "jarvis_local_session"
SESSION_TTL_SECONDS = _bounded_env_int(
    "JARVIS_LOCAL_SESSION_TTL_SECONDS", 8 * 60 * 60, 5 * 60, 24 * 60 * 60
)
MAX_LOCAL_SESSIONS = _bounded_env_int("JARVIS_MAX_LOCAL_SESSIONS", 128, 8, 1024)
MAX_PDF_BYTES = _bounded_env_int(
    "JARVIS_MAX_PDF_BYTES", 64 * 1024 * 1024, 1024, 512 * 1024 * 1024
)
PDF_COPY_CHUNK_BYTES = 1024 * 1024
PDF_UPLOAD_CONCURRENCY = _bounded_env_int("JARVIS_PDF_CONCURRENCY", 1, 1, 4)
PDF_UPLOAD_SEMAPHORE = asyncio.Semaphore(PDF_UPLOAD_CONCURRENCY)
SERVER_PORT = _bounded_env_int("JARVIS_PORT", 8000, 1024, 65535)

# The cookie is an opaque random identifier.  CSRF values are kept server-side
# and returned only by the same-origin session endpoint; neither value is logged.
_local_sessions: "OrderedDict[str, Tuple[str, float]]" = OrderedDict()


class PDFUploadRejected(ValueError):
    """Expected PDF validation failure with a safe client-facing message."""

    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def _header_values(scope: Dict[str, Any], name: bytes) -> List[str]:
    return [
        value.decode("latin-1")
        for key, value in scope.get("headers", [])
        if key.lower() == name
    ]


def _parse_local_authority(authority: str) -> Optional[Tuple[str, Optional[int]]]:
    """Return a normalized loopback host/port, rejecting ambiguous authorities."""

    if not authority or any(char.isspace() for char in authority):
        return None
    try:
        parsed = urlsplit(f"//{authority}")
    except ValueError:
        return None
    if parsed.username or parsed.password or parsed.path or parsed.query or parsed.fragment:
        return None
    try:
        port = parsed.port
    except ValueError:
        return None
    host = (parsed.hostname or "").casefold()
    if host not in LOCAL_HOSTNAMES:
        return None
    return host, port


def _origin_matches_host(origin: str, host_header: str, request_scheme: str) -> bool:
    """Require an exact same-origin loopback origin (including effective port)."""

    host = _parse_local_authority(host_header)
    if host is None:
        return False
    try:
        parsed = urlsplit(origin)
    except ValueError:
        return False
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.username
        or parsed.password
        or parsed.path
        or parsed.query
        or parsed.fragment
    ):
        return False
    try:
        origin_port = parsed.port
    except ValueError:
        return False
    origin_host = (parsed.hostname or "").casefold()
    if origin_host not in LOCAL_HOSTNAMES or parsed.scheme != request_scheme:
        return False
    default_port = 443 if request_scheme == "https" else 80
    return (origin_host, origin_port or default_port) == (host[0], host[1] or default_port)


def _prune_sessions(now: Optional[float] = None) -> None:
    current = time.monotonic() if now is None else now
    expired = [token for token, (_, expiry) in _local_sessions.items() if expiry <= current]
    for token in expired:
        _local_sessions.pop(token, None)
    while len(_local_sessions) >= MAX_LOCAL_SESSIONS:
        _local_sessions.popitem(last=False)


def _lookup_session(session_id: Optional[str]) -> Optional[str]:
    if not session_id:
        return None
    now = time.monotonic()
    record = _local_sessions.get(session_id)
    if record is None or record[1] <= now:
        _local_sessions.pop(session_id, None)
        return None
    csrf_token, _ = record
    _local_sessions[session_id] = (csrf_token, now + SESSION_TTL_SECONDS)
    _local_sessions.move_to_end(session_id)
    return csrf_token


def _create_session() -> Tuple[str, str]:
    _prune_sessions()
    session_id = secrets.token_urlsafe(32)
    csrf_token = secrets.token_urlsafe(32)
    _local_sessions[session_id] = (
        csrf_token,
        time.monotonic() + SESSION_TTL_SECONDS,
    )
    return session_id, csrf_token


def _csrf_matches(expected: Optional[str], supplied: str) -> bool:
    if expected is None or not supplied or len(supplied) > 128 or not supplied.isascii():
        return False
    return hmac.compare_digest(expected.encode("ascii"), supplied.encode("ascii"))


def _session_for_request(request: Request) -> Tuple[str, str, bool]:
    session_id = request.cookies.get(SESSION_COOKIE)
    csrf_token = _lookup_session(session_id)
    if csrf_token is not None and session_id is not None:
        return session_id, csrf_token, False
    session_id, csrf_token = _create_session()
    return session_id, csrf_token, True


def _set_session_cookie(response: Any, request: Request, session_id: str) -> None:
    response.set_cookie(
        SESSION_COOKIE,
        session_id,
        max_age=SESSION_TTL_SECONDS,
        httponly=True,
        secure=request.url.scheme == "https",
        samesite="strict",
        path="/",
    )


# (audit fix, F5) structured logging replaces bare print() for the server
# lifecycle; see jarvis_localhost/logging_config.py.
logger = configure_logging()


def _safe_error_log(context: str, exc: BaseException) -> None:
    """Log only an exception class; paths, commands and secrets stay private."""

    logger.warning("[%s] %s", context, type(exc).__name__)


async def _run_blocking_safely(function: Any, *args: Any) -> Any:
    """Let a bounded worker finish before request-cancellation cleanup runs."""

    task = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        try:
            await task
        except Exception as exc:
            _safe_error_log("cancelled_worker", exc)
        raise


def _sanitize_pdf_filename(filename: Optional[str]) -> str:
    raw = unicodedata.normalize("NFKC", filename or "")
    basename = raw.replace("\\", "/").rsplit("/", 1)[-1].strip().strip(". ")
    if not basename.casefold().endswith(".pdf"):
        raise PDFUploadRejected(400, "Formato inválido (PDF apenas).")
    stem = basename[:-4]
    cleaned = "".join(
        char if (char.isalnum() or char in {" ", ".", "_", "-"}) else "_"
        for char in stem
        if not unicodedata.category(char).startswith("C")
    )
    cleaned = re.sub(r"[ ._]+$", "", re.sub(r"\s+", " ", cleaned)).strip(" ._")
    if not cleaned:
        cleaned = "documento"
    return f"{cleaned[:100]}.pdf"


def _copy_validated_pdf(source: Any, destination: Path, max_bytes: int) -> int:
    """Copy a PDF in bounded chunks and remove every partial destination."""

    total = 0
    try:
        first_chunk = source.read(PDF_COPY_CHUNK_BYTES)
        if not first_chunk.startswith(b"%PDF-"):
            raise PDFUploadRejected(400, "Arquivo rejeitado: assinatura PDF ausente.")
        with destination.open("xb") as target:
            chunk = first_chunk
            while chunk:
                total += len(chunk)
                if total > max_bytes:
                    raise PDFUploadRejected(413, "Arquivo PDF excede o limite permitido.")
                target.write(chunk)
                chunk = source.read(PDF_COPY_CHUNK_BYTES)
        return total
    except Exception:
        destination.unlink(missing_ok=True)
        raise

# Instância base
brain = JarvisBrain()
direct_engine = None

# Estrutura O(1) para conexões (Evita data race O(N))
ws_clients: Set[WebSocket] = set()

# --- Pydantic Data Transfer Objects (DTOs) ---
# Contratos explícitos resolvendo Inconsistência de Schemas
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=16_384)

class SourceDTO(BaseModel):
    source: str = "Documento"
    score: float = 0.0

class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceDTO] = Field(default_factory=list)
    action: Optional[str] = None
    data: Optional[Dict[str, Any]] = None

class ProjectModel(BaseModel):
    name: str = Field(..., min_length=1, max_length=160)
    type: str = Field(..., min_length=1, max_length=80)
    priority: str = Field("BETA", max_length=32)
    description: str = Field("", max_length=8_192)
    tags: List[str] = Field(default_factory=list)

class TokenizeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=16_384)

class ClusterTaskRequest(BaseModel):
    command: str = Field(..., min_length=1, max_length=4_096)
    required_tags: List[str] = Field(default_factory=list)
    timeout_seconds: int = Field(120, ge=30, le=86400)
    priority: int = Field(5, ge=0, le=100)

class SpeakRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1500)

class FeedbackRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=16_384)
    answer: str = Field(..., min_length=1, max_length=65_536)
    interaction_id: Optional[str] = None
    session_id: Optional[str] = None
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    accepted: bool = True
    explicit_rating: Optional[int] = Field(None, ge=1, le=5)
    regenerated: bool = False
    implicit_reward: float = 0.0
    intent: Optional[str] = None
    outline: List[str] = Field(default_factory=list)
    rejected_answer: Optional[str] = None

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
        except Exception as exc:
            _safe_error_log("metrics_loop", exc)

# --- Lifespan Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[System] Inicializando módulos de baixo nível...")
    metrics_task = asyncio.create_task(metrics_loop())

    loop = asyncio.get_event_loop()
    def _curiosity_cb(data: Dict):
        asyncio.run_coroutine_threadsafe(broadcast(data), loop)

    brain.set_curiosity_callback(_curiosity_cb)
    logger.info("[Server] J.A.R.V.I.S. online.")

    try:
        yield  # Uvicorn control here
    finally:
        metrics_task.cancel()
        with suppress(asyncio.CancelledError):
            await metrics_task
        if hasattr(brain, "shutdown"):
            await asyncio.to_thread(brain.shutdown)
        logger.info("[Server] Desligando motores e liberando VRAM...")

# --- Instância da API ---
app = FastAPI(title="J.A.R.V.I.S", version="4.1.0", lifespan=lifespan)

# CORS is retained for compatibility with the two explicit loopback names.  The
# request guard below still requires exact same-origin Host/Origin matching.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        f"http://localhost:{SERVER_PORT}",
        f"http://127.0.0.1:{SERVER_PORT}",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-Jarvis-CSRF"],
)


@app.middleware("http")
async def local_request_guard(request: Request, call_next: Any):
    """Reject DNS rebinding/cross-origin requests and protect every mutation."""

    host_values = _header_values(request.scope, b"host")
    if len(host_values) != 1 or _parse_local_authority(host_values[0]) is None:
        return JSONResponse({"detail": "Host local inválido."}, status_code=400)

    origin_values = _header_values(request.scope, b"origin")
    if len(origin_values) > 1 or (
        origin_values
        and not _origin_matches_host(
            origin_values[0], host_values[0], request.scope.get("scheme", "http")
        )
    ):
        return JSONResponse({"detail": "Origem local inválida."}, status_code=403)

    if request.method.upper() in MUTATING_METHODS:
        # Browser mutations must prove both same-origin context and possession of
        # the per-session CSRF value.  Generic errors avoid token-oracle details.
        if len(origin_values) != 1:
            return JSONResponse({"detail": "Sessão local inválida."}, status_code=403)
        expected = _lookup_session(request.cookies.get(SESSION_COOKIE))
        supplied = request.headers.get("X-Jarvis-CSRF", "")
        if not _csrf_matches(expected, supplied):
            return JSONResponse({"detail": "Sessão local inválida."}, status_code=403)

        if request.url.path == "/api/pdf/upload":
            content_lengths = _header_values(request.scope, b"content-length")
            if len(content_lengths) > 1:
                return JSONResponse({"detail": "Upload inválido."}, status_code=400)
            if content_lengths:
                try:
                    declared_size = int(content_lengths[0])
                except ValueError:
                    return JSONResponse({"detail": "Upload inválido."}, status_code=400)
                # Multipart framing is bounded separately from the PDF payload.
                if declared_size < 0 or declared_size > MAX_PDF_BYTES + 2 * 1024 * 1024:
                    return JSONResponse(
                        {"detail": "Arquivo PDF excede o limite permitido."},
                        status_code=413,
                    )

    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
    response.headers["Content-Security-Policy"] = "frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
    if request.url.path == "/api/session":
        response.headers["Cache-Control"] = "no-store"
    return response


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# --- Controladores (Endpoints) ---

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    if not INDEX_PATH.exists():
        response = HTMLResponse(content="Frontend (index.html) ausente.", status_code=500)
    else:
        response = HTMLResponse(content=INDEX_PATH.read_text(encoding="utf-8"))
    session_id, _, created = _session_for_request(request)
    if created:
        _set_session_cookie(response, request, session_id)
    return response


@app.get("/api/session")
async def local_session(request: Request):
    session_id, csrf_token, created = _session_for_request(request)
    response = JSONResponse({"csrf_token": csrf_token, "sovereign": POLICY.enabled})
    if created:
        _set_session_cookie(response, request, session_id)
    return response

@app.get("/api/metrics")
async def get_metrics():
    return await asyncio.to_thread(brain.get_metrics)


def _normalize_chat_response(response: Dict[str, Any]) -> ChatResponse:
    data = response.get("data") or {}
    raw_sources = response.get("sources") or data.get("sources") or []
    sources_list: List[SourceDTO] = []
    for item in raw_sources:
        if isinstance(item, dict):
            source_text = (
                item.get("source")
                or item.get("label")
                or item.get("filename")
                or item.get("marker")
                or "Documento"
            )
            score = float(item.get("score", 0.0))
            sources_list.append(SourceDTO(source=str(source_text), score=score))
        elif isinstance(item, str):
            sources_list.append(SourceDTO(source=item, score=0.0))
        elif isinstance(item, SourceDTO):
            sources_list.append(item)
    answer = response.get("answer") or response.get("response") or response.get("text") or "Vazio"
    return ChatResponse(
        answer=answer,
        sources=sources_list,
        action=response.get("action"),
        data=data,
    )

@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    # Correção: O brain.chat pode ser denso. É chamado nativamente caso já seja coroutine.
    try:
        response = await brain.chat(req.message)
        return _normalize_chat_response(response)
    except Exception as exc:
        _safe_error_log("chat", exc)
        raise HTTPException(status_code=500, detail="Falha interna ao gerar resposta.") from None


@app.post("/api/chat/feedback")
async def chat_feedback(req: FeedbackRequest):
    """Register explicit or implicit user feedback for preference learning (DPO)."""
    try:
        interaction_id = req.interaction_id or f"int_{uuid.uuid4().hex[:16]}"
        session_id = req.session_id or "default_session"
        await asyncio.to_thread(
            brain.db.save_interaction_feedback,
            interaction_id=interaction_id,
            session_id=session_id,
            question=req.question,
            answer=req.answer,
            sources=req.sources,
            accepted=req.accepted,
            explicit_rating=req.explicit_rating,
            regenerated=req.regenerated,
            implicit_reward=req.implicit_reward,
            intent=req.intent,
            outline=req.outline,
        )
        pref_id = None
        if req.rejected_answer and req.accepted:
            pref_id = await asyncio.to_thread(
                brain.db.save_preference_pair,
                prompt=req.question,
                chosen_answer=req.answer,
                rejected_answer=req.rejected_answer,
                chosen_sources=req.sources,
                rejected_sources=None,
                reward_delta=1.0,
            )
        return {
            "status": "success",
            "interaction_id": interaction_id,
            "preference_pair_id": pref_id,
        }
    except Exception as exc:
        _safe_error_log("feedback", exc)
        raise HTTPException(status_code=500, detail="Falha ao salvar feedback.") from None


@app.get("/api/linguistics/stats")
async def linguistics_stats():
    """Retrieve corpus-wide linguistic and rhetorical metrics."""
    try:
        return await asyncio.to_thread(brain.db.get_linguistics_summary)
    except Exception as exc:
        _safe_error_log("linguistics_stats", exc)
        raise HTTPException(status_code=500, detail="Falha ao obter sumário linguístico.") from None


@app.get("/api/feedback/stats")
async def feedback_stats():
    """Retrieve preference memory metrics and recently recorded pairs."""
    try:
        pairs = await asyncio.to_thread(brain.db.get_preference_pairs, 100)
        return {
            "total_preference_pairs": len(pairs),
            "recent_pairs": pairs[:10],
        }
    except Exception as exc:
        _safe_error_log("feedback_stats", exc)
        raise HTTPException(status_code=500, detail="Falha ao obter estatísticas de preferências.") from None



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
    except Exception as exc:
        _safe_error_log("inferencia", exc)
        raise HTTPException(status_code=500, detail="Inferência local indisponível.") from None


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
    destination: Optional[Path] = None
    try:
        safe_filename = _sanitize_pdf_filename(file.filename)
        destination = assert_runtime_path(
            UPLOADS_ROOT / f"{uuid.uuid4().hex}_{safe_filename}"
        )
        async with PDF_UPLOAD_SEMAPHORE:
            await _run_blocking_safely(
                _copy_validated_pdf,
                file.file,
                destination,
                MAX_PDF_BYTES,
            )
            stats = await _run_blocking_safely(brain.process_pdf, str(destination))
        if not isinstance(stats, dict):
            raise TypeError("PDF processor returned a non-object result")
        await broadcast(
            {
                "type": "pdf_processed",
                "data": stats,
                "message": f"Documento '{safe_filename}' mapeado na rede neural.",
            }
        )
        return {"success": True, **stats}
    except asyncio.CancelledError:
        if destination is not None:
            destination.unlink(missing_ok=True)
        raise
    except PDFUploadRejected as exc:
        if destination is not None:
            destination.unlink(missing_ok=True)
        raise HTTPException(exc.status_code, exc.detail) from None
    except SovereignModeViolation as exc:
        if destination is not None:
            destination.unlink(missing_ok=True)
        _safe_error_log("pdf_upload", exc)
        raise HTTPException(400, f"Rejeitado pela política soberana: {exc}") from None
    except EmptyDocumentError as exc:
        if destination is not None:
            destination.unlink(missing_ok=True)
        _safe_error_log("pdf_upload", exc)
        raise HTTPException(400, str(exc)) from None
    except Exception as exc:
        if destination is not None:
            destination.unlink(missing_ok=True)
        _safe_error_log("pdf_upload", exc)
        raise HTTPException(500, "Falha interna ao processar o PDF.") from None
    finally:
        await file.close()

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
    raw_path = files[0] if files else project.get("zip_path")
    if not raw_path:
        raise HTTPException(404, "Arquivo do projeto não encontrado.")
    zip_path = Path(raw_path)
    if not zip_path.is_absolute():
        zip_path = APP_DIR / zip_path
    try:
        zip_path = assert_runtime_path(zip_path)
    except ValueError:
        raise HTTPException(404, "Arquivo do projeto não encontrado.") from None
    if not zip_path.is_file():
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
    doc_count = len(list(CORPUS_ROOT.glob("*_meta.json")))
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
        for p in CORPUS_ROOT.glob("*_meta.json"):
            try:
                docs.append(json.loads(p.read_text(encoding="utf-8")))
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue
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
    if POLICY.enabled:
        raise HTTPException(
            status_code=403,
            detail="Execução de comandos de cluster desativada no modo soberano.",
        )
    try:
        return await asyncio.to_thread(
            brain.submit_cluster_task,
            req.command,
            req.required_tags,
            req.timeout_seconds,
            req.priority,
        )
    except Exception as exc:
        _safe_error_log("cluster_task", exc)
        raise HTTPException(
            status_code=400,
            detail="Tarefa rejeitada pela política local do cluster.",
        ) from None


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
    host_values = _header_values(ws.scope, b"host")
    origin_values = _header_values(ws.scope, b"origin")
    scheme = "https" if ws.scope.get("scheme") == "wss" else "http"
    session_valid = _lookup_session(ws.cookies.get(SESSION_COOKIE)) is not None
    if (
        len(host_values) != 1
        or _parse_local_authority(host_values[0]) is None
        or len(origin_values) != 1
        or not _origin_matches_host(origin_values[0], host_values[0], scheme)
        or not session_valid
    ):
        await ws.close(code=1008, reason="Sessão WebSocket local inválida.")
        return

    await ws.accept()
    ws_clients.add(ws)
    logger.info("[TCP] Conexao Socket Estabelecida - total: %d", len(ws_clients))

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
    except Exception as exc:
        _safe_error_log("websocket", exc)
    finally:
        ws_clients.discard(ws)
        logger.info("[TCP] Conexao Socket Finalizada - total: %d", len(ws_clients))

# --- Início da Aplicação (CORREÇÃO DE PORTA CRÍTICA) ---
if __name__ == "__main__":
    # Suporte a porta customizada via JARVIS_PORT mantendo 8000 como padrao.
    # reload=False e explicito (nao apenas o default do uvicorn) porque e uma
    # propriedade de seguranca/memoria: o auto-reload do uvicorn nunca deve
    # chegar a producao, mesmo que um default de biblioteca mude no futuro ou
    # que um valor de depuracao local seja esquecido.
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=SERVER_PORT,
        log_level="info",
        reload=False,
    )

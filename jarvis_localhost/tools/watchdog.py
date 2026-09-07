"""Watchdog conservador para a ingestao/treino noturno do JARVIS LocalHost.

Nunca mata processos nem exclui artefatos do corpus. Arquivos ausentes do
manifesto exigem recuperacao explicita. Um POST aceito nao prova conclusao:
o watchdog acompanha o treino ate o servidor confirmar o pipeline pronto.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
VENV_PY = REPO / "jarvis_localhost" / ".venv" / (
    "Scripts/python.exe" if os.name == "nt" else "bin/python"
)
if not VENV_PY.is_file():
    VENV_PY = Path(sys.executable)
CORPUS = REPO / "jarvis_localhost" / "data" / "embeddings"
MANIFEST = CORPUS / "corpus_manifest.json"
TOOLS = REPO / "jarvis_localhost" / "tools"
WATCHLOG = TOOLS / "watchdog.log"
WATCHLOCK = TOOLS / ".watchdog.lock"
INGEST_FOLDER = Path(r"E:\Acadêmico\Livros E Arquivos De Estudos\Robótica")
INGEST_LOG = TOOLS / "robotica_ingest_log3.jsonl"
INGEST_STDOUT = TOOLS / "robotica_ingest_stdout3.log"
PORT = 8008
BASE_URL = f"http://127.0.0.1:{PORT}"


def log(msg: str) -> None:
    line = f"{datetime.now(timezone.utc).isoformat()} {msg}"
    print(line, flush=True)
    with WATCHLOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def port_open() -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(2.0)
        return sock.connect_ex(("127.0.0.1", PORT)) == 0


def process_running(needle: str) -> bool:
    """Fail closed if a Python command line cannot be inspected."""
    import psutil

    for process in psutil.process_iter(["name", "cmdline"]):
        name = (process.info.get("name") or "").lower()
        if not name.startswith("python"):
            continue
        command = process.info.get("cmdline")
        if not command:
            raise RuntimeError("nao foi possivel verificar um processo Python")
        if needle in command:
            return True
    return False


def find_orphans() -> list[Path]:
    """List unregistered artifacts without modifying any corpus data."""
    if not CORPUS.is_dir():
        return []
    manifest = (
        json.loads(MANIFEST.read_text(encoding="utf-8"))
        if MANIFEST.is_file() else {}
    )
    authorized: set[str] = set()
    for doc in (manifest.get("documents") or {}).values():
        for key in ("chunks_file", "corpus_file", "metadata_file"):
            name = doc.get(key)
            if name:
                authorized.add(name)
    return sorted(
        path for path in CORPUS.iterdir()
        if path.is_file() and path.name.startswith("doc_")
        and path.name not in authorized
    )


def _launch(command: list[str], output: Path) -> None:
    options = (
        {"creationflags": subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS}
        if os.name == "nt" else {"start_new_session": True}
    )
    with output.open("a", encoding="utf-8") as stream:
        subprocess.Popen(
            command, cwd=str(REPO), stdout=stream, stderr=subprocess.STDOUT,
            **options,
        )


def start_server() -> None:
    log("iniciando servidor...")
    _launch(
        [str(VENV_PY), "-m", "jarvis_localhost.server.app"],
        TOOLS / "server_watchdog_stdout.log",
    )


def start_ingest() -> None:
    log("iniciando ingestao...")
    _launch(
        [str(VENV_PY), "-u", "-m", "jarvis_localhost.tools.ingest_folder",
         str(INGEST_FOLDER), "--log", str(INGEST_LOG)],
        INGEST_STDOUT,
    )


def missing_pdfs() -> list[Path]:
    """Compare PDF hashes with the manifest, never the upload client's log."""
    import hashlib

    if not INGEST_FOLDER.is_dir():
        raise FileNotFoundError("pasta de ingestao indisponivel")
    if not MANIFEST.is_file():
        return [p for p in INGEST_FOLDER.rglob("*.pdf") if p.is_file()]
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    have = {d.get("document_sha256")
            for d in (manifest.get("documents") or {}).values()}

    def digest(path: Path) -> str:
        hasher = hashlib.sha256()
        with path.open("rb") as fh:
            for block in iter(lambda: fh.read(1024 * 1024), b""):
                hasher.update(block)
        return hasher.hexdigest()

    return [p for p in sorted(INGEST_FOLDER.rglob("*.pdf"))
            if p.is_file() and digest(p) not in have]


# Existing ingestion policy is kept; this threshold does not certify why a
# document was skipped. Explicit rejection accounting belongs to ingest_folder.
MAX_MISSING_TOLERADO = 25


def ingest_done() -> bool:
    missing = missing_pdfs()
    log(f"faltando no manifesto: {len(missing)} PDF(s)")
    return len(missing) <= MAX_MISSING_TOLERADO


def train_status() -> dict:
    import httpx

    with httpx.Client(base_url=BASE_URL, timeout=30.0) as client:
        response = client.get("/api/train/status")
        response.raise_for_status()
        status = response.json()
        if not isinstance(status, dict) or any(
            not isinstance(status.get(key), bool)
            for key in ("is_training", "is_trained")
        ):
            raise ValueError("resposta de status de treino invalida")
        return status


def start_training() -> bool:
    """Dispara o treino. Idempotente: o servidor devolve already_training."""
    import httpx

    headers = {"Origin": BASE_URL}
    with httpx.Client(base_url=BASE_URL, timeout=120.0, headers=headers) as client:
        session = client.get("/api/session")
        session.raise_for_status()
        csrf = session.json()["csrf_token"]
        resp = client.post("/api/train/start", headers={"X-Jarvis-CSRF": csrf})
        log(f"POST /api/train/start -> {resp.status_code}")
        resp.raise_for_status()
        return resp.json().get("status") in {"started", "already_training"}


def monitor(*, training_only: bool = False) -> int:
    log("=== watchdog iniciado ===")
    training_observed = False
    while True:
        try:
            if not port_open():
                log(f"servidor NAO responde na porta {PORT}")
                if not process_running("jarvis_localhost.server.app"):
                    orphans = find_orphans()
                    if orphans:
                        log(
                            f"{len(orphans)} artefato(s) sem registro preservado(s); "
                            "recuperacao manual necessaria antes do reinicio"
                        )
                        return 1
                    start_server()
                    training_observed = False
                    time.sleep(180)
                else:
                    log("processo server.app existe mas porta fechada; ainda iniciando")
            else:
                status = train_status()
                if status.get("is_training"):
                    training_observed = True
                    log(f"TREINANDO: {status.get('progress')}")
                    time.sleep(600)
                    continue
                progress = status.get("progress") or {}
                if progress.get("error") or progress.get("cancelled"):
                    log(f"TREINO INTERROMPIDO: {progress}. Watchdog encerrado sem repetir o inicio.")
                    return 1
                if status.get("is_trained"):
                    log(f"TREINO CONCLUIDO ({status.get('documents')} docs). Watchdog encerrado.")
                    return 0
                if training_observed:
                    log("treino parou sem pipeline concluido; verificacao necessaria")
                    return 1
                if training_only:
                    log(f"retomando treino no corpus atual ({status.get('documents')} docs)")
                    if start_training():
                        training_observed = True
                        time.sleep(60)
                        continue
                    log("falha ao iniciar treino; tentara de novo no proximo ciclo")
                elif not process_running("jarvis_localhost.tools.ingest_folder"):
                    if ingest_done():
                        log(f"ingestao apta para treino ({status.get('documents')} docs); iniciando")
                        if start_training():
                            training_observed = True
                            time.sleep(60)
                            continue
                        log("falha ao iniciar treino; tentara de novo no proximo ciclo")
                    else:
                        log("ingestao parada e incompleta; relancando")
                        start_ingest()
                else:
                    log(f"ok: {status.get('documents')} docs, ingestao em andamento")
        except Exception as exc:  # noqa: BLE001 -- retry observation failures
            log(f"erro no watchdog: {exc!r}")
        time.sleep(300)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-only", action="store_true",
        help="acompanhar/retomar apenas o treino no corpus atual, sem iniciar ingestao",
    )
    args = parser.parse_args(argv)
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from jarvis_localhost.integrations.process_lock import InterProcessFileLock

    lock = InterProcessFileLock(WATCHLOCK)
    if not lock.acquire():
        log("outro watchdog ja esta ativo; esta instancia sera encerrada")
        return 2
    try:
        return monitor(training_only=args.training_only)
    finally:
        lock.release()


if __name__ == "__main__":
    raise SystemExit(main())

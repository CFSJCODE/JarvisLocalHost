"""Ingesta em lote de PDFs de uma pasta para o corpus do JARVIS LocalHost.

Uso:
    .venv\\Scripts\\python.exe -m jarvis_localhost.tools.ingest_folder <pasta> [--base-url URL]

Itera recursivamente todos os *.pdf da pasta dada, obtem sessao+CSRF em
/api/session e envia cada arquivo para /api/pdf/upload. Documentos sem
camada de texto (scans) sao esperados como rejeicao (nao erro) pelo modo
soberano -- sao registrados separadamente. Nao apaga nem move nada na
pasta de origem; so le os arquivos.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx


def find_pdfs(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.pdf") if p.is_file())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, help="Pasta a ingerir (recursivo)")
    parser.add_argument("--base-url", default="http://127.0.0.1:8008")
    parser.add_argument("--log", type=Path, default=None, help="Arquivo JSONL de log")
    args = parser.parse_args()

    folder: Path = args.folder
    if not folder.is_dir():
        print(f"ERRO: pasta nao encontrada: {folder}", file=sys.stderr)
        return 1

    pdfs = find_pdfs(folder)
    print(f"Encontrados {len(pdfs)} PDF(s) em {folder}")
    if not pdfs:
        return 0

    base_url = args.base_url.rstrip("/")
    log_path = args.log or (folder / "_ingest_log.jsonl")

    headers_common = {"Origin": base_url}

    # A ingestao reverifica (SHA-256 + reparse) todos os documentos ja
    # ingeridos a cada novo upload (corpus/manifest.py::validate_artifacts),
    # entao o tempo por PDF cresce com o tamanho do corpus. Timeout generoso
    # para nao abortar um upload que o servidor completaria com sucesso.
    client = httpx.Client(base_url=base_url, timeout=3600.0, headers=headers_common)

    def fetch_session() -> str:
        resp = client.get("/api/session")
        resp.raise_for_status()
        data = resp.json()
        print(f"Sessao (re)obtida. sovereign={data.get('sovereign')}")
        return data["csrf_token"]

    csrf = fetch_session()

    results = {"success": 0, "rejected_expected": 0, "error": 0}
    started = time.monotonic()

    with log_path.open("a", encoding="utf-8") as logf:
        for i, pdf in enumerate(pdfs, 1):
            rel = pdf.relative_to(folder)
            entry = {"file": str(rel), "ts": None}
            try:
                # A sessao local expira apos JARVIS_LOCAL_SESSION_TTL_SECONDS
                # (8h por padrao). Um lote longo pode ultrapassar isso, entao
                # uma unica tentativa de renovacao em caso de 403 "sessao
                # invalida" evita que o resto do lote falhe em cascata.
                for attempt in range(2):
                    with pdf.open("rb") as fh:
                        files = {"file": (pdf.name, fh, "application/pdf")}
                        r = client.post(
                            "/api/pdf/upload",
                            files=files,
                            headers={"X-Jarvis-CSRF": csrf},
                        )
                    if r.status_code == 403 and attempt == 0:
                        print(f"[{i}/{len(pdfs)}] sessao expirada, renovando...")
                        csrf = fetch_session()
                        continue
                    break
                if r.status_code == 200:
                    data = r.json()
                    results["success"] += 1
                    entry.update({"status": "success", "http": 200, "stats": data})
                    print(f"[{i}/{len(pdfs)}] OK  {rel}")
                elif r.status_code == 400 and (
                    "extractable chunks" in r.text
                    or "política soberana" in r.text
                    or "soberana" in r.text
                ):
                    results["rejected_expected"] += 1
                    entry.update({"status": "rejected_expected", "http": 400, "detail": r.json().get("detail")})
                    print(f"[{i}/{len(pdfs)}] SKIP (esperado: sem texto/soberania) {rel}")
                else:
                    results["error"] += 1
                    entry.update({"status": "error", "http": r.status_code, "detail": r.text[:500]})
                    print(f"[{i}/{len(pdfs)}] ERRO http={r.status_code} {rel} :: {r.text[:200]}")
            except Exception as exc:  # noqa: BLE001 -- registra qualquer falha de rede/IO e segue
                results["error"] += 1
                entry.update({"status": "error", "detail": str(exc)})
                print(f"[{i}/{len(pdfs)}] EXCEPTION {rel} :: {exc}")
            logf.write(json.dumps(entry, ensure_ascii=False) + "\n")
            logf.flush()

    elapsed = time.monotonic() - started
    print(
        f"\nConcluido em {elapsed:.1f}s. "
        f"sucesso={results['success']} rejeitado_esperado={results['rejected_expected']} "
        f"erro={results['error']} (log: {log_path})"
    )
    return 0 if results["error"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

import json, pathlib

root = pathlib.Path(r"E:\SoftwareProjects\JarvisLocalHost\jarvis_localhost\data\embeddings")
manifest = json.loads((root / "corpus_manifest.json").read_text(encoding="utf-8"))

docs = manifest.get("documents", manifest) if isinstance(manifest, dict) else manifest
if isinstance(docs, dict):
    items = list(docs.items())
else:
    items = [(d.get("document_id", "?"), d) for d in docs]

print("total documentos no manifesto:", len(items))
ghosts = []
for doc_id, d in items:
    stats = d.get("stats", {})
    chunks = stats.get("canonical_chunks", d.get("chunks_records"))
    words = stats.get("words", d.get("words"))
    if (chunks in (0, None)) or (words in (0, None)):
        ghosts.append((doc_id, d.get("filename"), chunks, words))

print("candidatos a documento fantasma (0 chunks/palavras):", len(ghosts))
for doc_id, fn, chunks, words in ghosts:
    print(" -", doc_id, "|", fn, "| chunks=", chunks, "| words=", words)

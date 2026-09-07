"""Thread-safe, checksum-verified local vector store."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Iterable

import numpy as np


def _normalized(vectors: np.ndarray) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2:
        raise ValueError("vectors must be a 2D matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("vectors must contain only finite values")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if matrix.shape[0] and np.any(norms <= 1e-12):
        raise ValueError("vectors must have non-zero norm")
    return matrix / np.maximum(norms, 1e-12)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class VectorStore:
    def __init__(self) -> None:
        self.vectors: np.ndarray | None = None
        self.style_vectors: np.ndarray | None = None
        self.rhetorical_vectors: np.ndarray | None = None
        self.metadata: list[dict] = []
        self._lock = threading.RLock()

    def __len__(self) -> int:
        with self._lock:
            return len(self.metadata)

    def clear(self) -> None:
        with self._lock:
            self.vectors = None
            self.style_vectors = None
            self.rhetorical_vectors = None
            self.metadata = []

    def add(
        self,
        vectors: np.ndarray,
        metadata: Iterable[dict],
        *,
        style_vectors: np.ndarray | None = None,
        rhetorical_vectors: np.ndarray | None = None,
    ) -> None:
        entries = [dict(item) for item in metadata]
        matrix = np.ascontiguousarray(_normalized(vectors), dtype=np.float32)
        if matrix.shape[0] != len(entries):
            raise ValueError("vector and metadata counts differ")
        if any(not entry.get("chunk_id") for entry in entries):
            raise ValueError("every vector entry must carry a stable chunk_id")
        incoming_ids = [str(entry["chunk_id"]) for entry in entries]
        if len(incoming_ids) != len(set(incoming_ids)):
            raise ValueError("incoming vector entries contain duplicate chunk_ids")

        style_mat = (
            np.ascontiguousarray(_normalized(style_vectors), dtype=np.float32)
            if style_vectors is not None
            else None
        )
        rhet_mat = (
            np.ascontiguousarray(_normalized(rhetorical_vectors), dtype=np.float32)
            if rhetorical_vectors is not None
            else None
        )

        with self._lock:
            if not self.metadata:
                self.vectors = matrix
                self.style_vectors = style_mat
                self.rhetorical_vectors = rhet_mat
                self.metadata = entries
                return
            existing = {entry["chunk_id"]: index for index, entry in enumerate(self.metadata)}
            new_vectors: list[np.ndarray] = []
            new_style: list[np.ndarray] = []
            new_rhet: list[np.ndarray] = []
            new_entries: list[dict] = []
            for i, (vector, entry) in enumerate(zip(matrix, entries)):
                index = existing.get(entry["chunk_id"])
                s_vec = style_mat[i] if style_mat is not None else None
                r_vec = rhet_mat[i] if rhet_mat is not None else None
                if index is None:
                    if self.vectors is not None and vector.shape[0] != self.vectors.shape[1]:
                        raise ValueError("vector dimension mismatch")
                    new_vectors.append(vector)
                    if s_vec is not None:
                        new_style.append(s_vec)
                    if r_vec is not None:
                        new_rhet.append(r_vec)
                    new_entries.append(entry)
                else:
                    if self.vectors is None or vector.shape[0] != self.vectors.shape[1]:
                        raise ValueError("vector dimension mismatch")
                    self.vectors[index] = vector
                    if self.style_vectors is not None and s_vec is not None:
                        self.style_vectors[index] = s_vec
                    if self.rhetorical_vectors is not None and r_vec is not None:
                        self.rhetorical_vectors[index] = r_vec
                    self.metadata[index] = entry
            if new_vectors:
                stacked_new = np.ascontiguousarray(np.vstack(new_vectors), dtype=np.float32)
                self.vectors = np.ascontiguousarray(
                    np.vstack((self.vectors, stacked_new)) if self.vectors is not None else stacked_new,
                    dtype=np.float32,
                )
                if new_style and (self.style_vectors is not None or len(new_style) == len(new_vectors)):
                    st_stacked = np.ascontiguousarray(np.vstack(new_style), dtype=np.float32)
                    self.style_vectors = np.ascontiguousarray(
                        np.vstack((self.style_vectors, st_stacked))
                        if self.style_vectors is not None
                        else st_stacked,
                        dtype=np.float32,
                    )
                if new_rhet and (self.rhetorical_vectors is not None or len(new_rhet) == len(new_vectors)):
                    rh_stacked = np.ascontiguousarray(np.vstack(new_rhet), dtype=np.float32)
                    self.rhetorical_vectors = np.ascontiguousarray(
                        np.vstack((self.rhetorical_vectors, rh_stacked))
                        if self.rhetorical_vectors is not None
                        else rh_stacked,
                        dtype=np.float32,
                    )
                self.metadata.extend(new_entries)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        *,
        min_score: float = -1.0,
    ) -> list[dict]:
        return self.search_multi(
            query_vector=query_vector,
            top_k=top_k,
            min_score=min_score,
        )

    def search_multi(
        self,
        query_vector: np.ndarray,
        *,
        target_style_vector: np.ndarray | None = None,
        target_rhetorical_vector: np.ndarray | None = None,
        top_k: int = 5,
        weights: tuple[float, float, float] = (0.7, 0.2, 0.1),
        min_score: float = -1.0,
        rhetorical_role_filter: str | None = None,
    ) -> list[dict]:
        if top_k <= 0:
            return []
        with self._lock:
            if self.vectors is None or not self.metadata:
                return []
            query = _normalized(np.asarray(query_vector, dtype=np.float32))[0]
            if query.shape[0] != self.vectors.shape[1]:
                raise ValueError("query vector dimension mismatch")
            
            w_c, w_s, w_r = weights
            scores = (self.vectors @ query) * w_c

            if target_style_vector is not None and self.style_vectors is not None:
                q_style = _normalized(np.asarray(target_style_vector, dtype=np.float32))[0]
                if q_style.shape[0] == self.style_vectors.shape[1]:
                    scores += (self.style_vectors @ q_style) * w_s

            if target_rhetorical_vector is not None and self.rhetorical_vectors is not None:
                q_rhet = _normalized(np.asarray(target_rhetorical_vector, dtype=np.float32))[0]
                if q_rhet.shape[0] == self.rhetorical_vectors.shape[1]:
                    scores += (self.rhetorical_vectors @ q_rhet) * w_r

            n_scores = len(scores)
            if n_scores <= top_k:
                indices = np.argsort(scores, kind="stable")[::-1]
            else:
                top_part = np.argpartition(scores, -top_k)[-top_k:]
                indices = top_part[np.argsort(scores[top_part], kind="stable")[::-1]]
            output: list[dict] = []
            for index in indices:
                score = float(scores[index])
                if score < min_score:
                    continue
                entry = dict(self.metadata[int(index)])
                if rhetorical_role_filter:
                    role = entry.get("rhetorical_role") or entry.get("linguistic", {}).get("rhetorical_role")
                    if role and role != rhetorical_role_filter:
                        continue
                entry["score"] = score
                output.append(entry)
                if len(output) >= top_k:
                    break
            return output

    @staticmethod
    def _paths(prefix: str | Path) -> tuple[Path, Path, Path]:
        base = Path(prefix)
        return (
            Path(str(base) + "_vecs.npy"),
            Path(str(base) + "_meta.json"),
            Path(str(base) + "_manifest.json"),
        )

    def save(
        self,
        prefix: str | Path,
        *,
        corpus_sha256: str | None = None,
        tokenizer_sha256: str | None = None,
        encoder_sha256: str | None = None,
    ) -> None:
        vector_path, metadata_path, manifest_path = self._paths(prefix)
        vector_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            matrix = (
                self.vectors.copy()
                if self.vectors is not None
                else np.empty((0, 0), dtype=np.float32)
            )
            metadata = [dict(entry) for entry in self.metadata]

        if matrix.size and not np.isfinite(matrix).all():
            raise ValueError("vector index contains non-finite values")
        chunk_ids = [str(entry.get("chunk_id", "")) for entry in metadata]
        if any(not chunk_id for chunk_id in chunk_ids):
            raise ValueError("vector metadata contains a missing chunk_id")
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("vector metadata contains duplicate chunk_ids")

        vector_temp = vector_path.with_suffix(vector_path.suffix + ".tmp")
        metadata_temp = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
        with vector_temp.open("wb") as handle:
            np.save(handle, matrix, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        metadata_temp.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(vector_temp, vector_path)
        os.replace(metadata_temp, metadata_path)
        manifest = {
            "format_version": 2,
            "entries": len(metadata),
            "dimension": int(matrix.shape[1]) if matrix.ndim == 2 else 0,
            "vectors_sha256": _sha256(vector_path),
            "metadata_sha256": _sha256(metadata_path),
            "corpus_sha256": corpus_sha256,
            "tokenizer_sha256": tokenizer_sha256,
            "encoder_sha256": encoder_sha256,
            "normalized": True,
        }
        manifest_temp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
        manifest_temp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.replace(manifest_temp, manifest_path)

    def load(
        self,
        prefix: str | Path,
        *,
        expected_corpus_sha256: str | None = None,
        expected_tokenizer_sha256: str | None = None,
        expected_encoder_sha256: str | None = None,
    ) -> None:
        vector_path, metadata_path, manifest_path = self._paths(prefix)
        if not (vector_path.is_file() and metadata_path.is_file() and manifest_path.is_file()):
            raise FileNotFoundError("vector index generation is incomplete")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format_version") != 2:
            raise ValueError("unsupported vector index manifest version")
        if manifest.get("vectors_sha256") != _sha256(vector_path):
            raise ValueError("vector index checksum mismatch")
        if manifest.get("metadata_sha256") != _sha256(metadata_path):
            raise ValueError("vector metadata checksum mismatch")
        with vector_path.open("rb") as handle:
            matrix = np.load(handle, allow_pickle=False)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, list) or any(
            not isinstance(entry, dict) for entry in metadata
        ):
            raise ValueError("vector metadata must be a list of objects")
        if matrix.ndim != 2 or matrix.shape[0] != len(metadata):
            raise ValueError("vector index shape is inconsistent with metadata")
        if int(manifest.get("entries", -1)) != len(metadata):
            raise ValueError("vector entry count does not match manifest")
        if int(manifest.get("dimension", -1)) != matrix.shape[1]:
            raise ValueError("vector index dimension does not match manifest")
        if manifest.get("normalized") is not True:
            raise ValueError("vector index does not declare normalized vectors")
        if matrix.size and not np.isfinite(matrix).all():
            raise ValueError("vector index contains non-finite values")
        if matrix.shape[0]:
            norms = np.linalg.norm(matrix.astype(np.float32, copy=False), axis=1)
            if not np.allclose(norms, 1.0, rtol=1e-4, atol=1e-5):
                raise ValueError("vector index contains non-normalized vectors")
        chunk_ids = [str(entry.get("chunk_id", "")) for entry in metadata]
        if any(not chunk_id for chunk_id in chunk_ids):
            raise ValueError("vector metadata contains a missing chunk_id")
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("vector metadata contains duplicate chunk_ids")
        expected_lineage = {
            "corpus_sha256": expected_corpus_sha256,
            "tokenizer_sha256": expected_tokenizer_sha256,
            "encoder_sha256": expected_encoder_sha256,
        }
        for field, expected in expected_lineage.items():
            if expected is not None and manifest.get(field) != expected:
                raise ValueError(f"vector index {field} mismatch")
        with self._lock:
            self.vectors = matrix.astype(np.float32, copy=False)
            self.metadata = [dict(entry) for entry in metadata]


__all__ = ["VectorStore"]

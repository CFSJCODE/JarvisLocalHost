"""Absolute runtime paths for Jarvis LocalHost.

Runtime state must never depend on the process current working directory.  In
particular this keeps private PDFs, corpora, indexes and checkpoints under the
package data directory already excluded by ``.gitignore``.
"""

from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
DATA_ROOT = PACKAGE_ROOT / "data"
MODELS_ROOT = DATA_ROOT / "models"
CORPUS_ROOT = DATA_ROOT / "embeddings"
IMAGES_ROOT = DATA_ROOT / "extracted_images"
CURIOSITY_ROOT = DATA_ROOT / "curiosity"
PROJECTS_ROOT = DATA_ROOT / "projects"
INTEGRATIONS_ROOT = DATA_ROOT / "integrations"
UPLOADS_ROOT = PACKAGE_ROOT / "uploads"
STATIC_ROOT = PACKAGE_ROOT / "web" / "static"

RUNTIME_DIRECTORIES = (
    DATA_ROOT,
    MODELS_ROOT,
    CORPUS_ROOT,
    IMAGES_ROOT,
    CURIOSITY_ROOT,
    PROJECTS_ROOT,
    INTEGRATIONS_ROOT,
    UPLOADS_ROOT,
)


def ensure_runtime_directories() -> None:
    """Create only the known runtime directories under the package root."""

    for directory in RUNTIME_DIRECTORIES:
        directory.mkdir(parents=True, exist_ok=True)


def assert_runtime_path(path: Path) -> Path:
    """Resolve *path* and ensure it remains inside a Jarvis runtime root."""

    resolved = path.resolve(strict=False)
    allowed_roots = (DATA_ROOT.resolve(), UPLOADS_ROOT.resolve())
    if not any(resolved == root or root in resolved.parents for root in allowed_roots):
        raise ValueError(f"Runtime path escapes Jarvis data roots: {resolved}")
    return resolved

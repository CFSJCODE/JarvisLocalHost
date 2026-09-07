"""Centralized structured logging for Jarvis LocalHost.

(audit fix, F5) The codebase had zero uses of the standard ``logging``
module before this file: every runtime message -- including error paths in
the central orchestrator (``core/brain.py``) and the FastAPI server
lifecycle (``server/app.py``) -- went through bare ``print()`` calls. That
loses level/timestamp/module information, cannot be redirected to a log
file for post-mortem debugging, and is silently lost entirely when the
process is launched without a visible console (a common case for a
long-running local desktop service). This module provides one place to
configure real logging; call ``configure_logging()`` once at process start
(``server/app.py`` does this) and use ``logging.getLogger(__name__)``
elsewhere.

Scope note (updated 2026-08-31): the server lifecycle (server/app.py), the
central orchestrator (core/brain.py), the PDF pipeline
(processing/pdf_processor.py) and the persistence layer
(storage/database.py) all use this logger now. Deliberately left as
print()/stderr, not migrated: tools/corpus_hygiene.py, tools/hardware_probe.py
and tools/directml_smoke.py are CLI scripts whose whole contract is clean
JSON/text on stdout for the operator (or a pipe) to read directly --
routing that through a leveled, timestamped logger would break that
contract, not improve it. integrations/team_bus_server.py is a separate,
deliberately dependency-free MCP stdio server (see its own module
docstring) that reserves stdout exclusively for newline-delimited
JSON-RPC and already writes its one diagnostic line to stderr; importing
this module's logging setup into that process would both violate its
stdlib-only constraint and risk that isolation. legacy/engine_AI_legacy.py
is unreferenced dead code (see docs/ARCHITECTURE.md, "Ferramentas De
Manutencao" is about maintained tools; this file is not one of them) and
was left untouched rather than edited without a reason to run it.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
from pathlib import Path

from jarvis_localhost.paths import DATA_ROOT

LOG_ROOT = DATA_ROOT / "logs"
DEFAULT_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"

_configured = False


def configure_logging(
    *,
    level: int | str | None = None,
    log_file: str | Path | None = "jarvis.log",
    to_stdout: bool = True,
) -> logging.Logger:
    """Configure the ``jarvis`` logger tree once per process.

    Idempotent: calling this more than once (e.g. once from a CLI entry
    point and once from a test) reconfigures cleanly instead of stacking
    duplicate handlers, which is the standard failure mode of ad-hoc
    ``logging.basicConfig()`` calls scattered across a codebase.

    ``level`` accepts a standard logging level or its name; it defaults to
    the ``JARVIS_LOG_LEVEL`` environment variable, then ``INFO``.
    ``log_file`` is resolved under ``data/logs/`` (created if needed) unless
    an absolute path is given; pass ``None`` to disable file logging (e.g.
    in unit tests that should not touch the filesystem).
    """

    global _configured
    resolved_level = level if level is not None else os.getenv("JARVIS_LOG_LEVEL", "INFO")
    if isinstance(resolved_level, str):
        resolved_level = logging.getLevelName(resolved_level.upper())
        if not isinstance(resolved_level, int):
            resolved_level = logging.INFO

    logger = logging.getLogger("jarvis")
    logger.setLevel(resolved_level)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    logger.propagate = False

    formatter = logging.Formatter(DEFAULT_FORMAT)
    if to_stdout:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if log_file is not None:
        file_path = Path(log_file)
        if not file_path.is_absolute():
            LOG_ROOT.mkdir(parents=True, exist_ok=True)
            file_path = LOG_ROOT / file_path
        # Rotate to bound disk use on a long-running local service.
        file_handler = logging.handlers.RotatingFileHandler(
            file_path, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _configured = True
    return logger


_PACKAGE_PREFIX = "jarvis_localhost."


def get_logger(name: str) -> logging.Logger:
    """Return a child of the ``jarvis`` logger tree, configuring defaults if needed.

    (audit fix, F5 follow-up) ``name`` is normally a module's ``__name__``
    (e.g. ``"jarvis_localhost.core.brain"``). The previous version of this
    function only special-cased names that already started with the literal
    substring ``"jarvis"`` -- which ``"jarvis_localhost.core.brain"`` also
    matches, since it happens to start with those six letters -- so it was
    returned unprefixed, as its own unrelated top-level logger with no
    handlers, instead of becoming a child of the ``"jarvis"`` logger that
    ``configure_logging()`` attaches handlers to. Python's logging hierarchy
    is dot-separated, not string-prefix matching: "jarvis_localhost" is not
    an ancestor of "jarvis" just because the characters overlap. The result
    was silent data loss -- every ``logger.info(...)`` call from a real
    module (core/brain.py, processing/pdf_processor.py, storage/database.py)
    was dropped with no output anywhere, confirmed by the audit's own real
    E2E run producing no matching lines in ``data/logs/jarvis.log``. Fixed
    by rewriting the package prefix to the ``jarvis.`` namespace so real
    module loggers become proper children of ``"jarvis"`` and inherit its
    handlers via normal propagation; see
    tests/test_logging_config.py::LoggerHierarchyTests, which asserts a
    message logged through a real module-shaped name actually reaches a
    handler attached to the ``"jarvis"`` logger, not just that no exception
    is raised.
    """

    if not _configured:
        configure_logging()
    if name == "jarvis" or name.startswith("jarvis."):
        resolved = name
    elif name.startswith(_PACKAGE_PREFIX):
        resolved = "jarvis." + name[len(_PACKAGE_PREFIX):]
    else:
        resolved = f"jarvis.{name}"
    return logging.getLogger(resolved)


__all__ = ["configure_logging", "get_logger", "LOG_ROOT"]

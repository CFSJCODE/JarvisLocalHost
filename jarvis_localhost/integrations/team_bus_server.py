"""Dependency-free MCP stdio server for local senior-agent collaboration.

The server deliberately uses only the Python 3.10 standard library.  Multiple
stdio server processes share a SQLite database, so ChatGPT Desktop/Codex and
Antigravity can exchange messages and coordinate tasks without a network
listener.  Standard output is reserved exclusively for newline-delimited MCP
JSON-RPC messages.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple


SERVER_NAME = "jarvis-local-team-bus"
SERVER_TITLE = "Jarvis Local Team Bus"
SERVER_VERSION = "1.0.0"
DEFAULT_PROTOCOL_VERSION = "2025-06-18"
SUPPORTED_PROTOCOL_VERSIONS = {
    "2025-11-25",
    "2025-06-18",
    "2025-03-26",
    "2024-11-05",
}

DB_PATH_ENV = "JARVIS_TEAM_BUS_DB"
DB_PATH_ENV_ALIAS = "JARVIS_TEAM_BUS_DB_PATH"
BUSY_TIMEOUT_MS = 10_000
MAX_JSON_INPUT_BYTES = 4 * 1024 * 1024
MAX_JSON_VALUE_BYTES = 1024 * 1024
MAX_COMMAND_CHARS = 32_768
MAX_COMMAND_OUTPUT_CHARS = 1_000_000
WRITE_RETRIES = 7

TASK_STATUSES = {
    "open",
    "claimed",
    "in_progress",
    "completed",
    "failed",
    "cancelled",
}
TERMINAL_TASK_STATUSES = {"completed", "failed", "cancelled"}
TASK_TRANSITIONS = {
    "open": {"cancelled"},
    "claimed": {"in_progress", "completed", "failed", "open", "cancelled"},
    "in_progress": {"completed", "failed", "open", "cancelled"},
    "completed": set(),
    "failed": {"open"},
    "cancelled": {"open"},
}


class TeamBusError(RuntimeError):
    """Expected domain or validation error suitable for an MCP tool result."""


class ValidationError(TeamBusError):
    """Raised when tool arguments are invalid."""


class NotFoundError(TeamBusError):
    """Raised when a requested agent, message, or task does not exist."""


class ConflictError(TeamBusError):
    """Raised for optimistic-concurrency or idempotency conflicts."""


class ProtocolError(RuntimeError):
    """JSON-RPC/MCP protocol-level failure."""

    def __init__(self, code: int, message: str, data: Any = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _json_dumps(value: Any, *, sort_keys: bool = False) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=sort_keys,
        )
    except (TypeError, ValueError) as exc:
        raise ValidationError("O valor deve ser JSON válido e finito.") from exc


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Constante JSON inválida: {value}")


def _json_loads(raw: str) -> Any:
    return json.loads(raw, parse_constant=_reject_json_constant)


def _json_text(value: Any, field: str, max_bytes: int = MAX_JSON_VALUE_BYTES) -> str:
    raw = _json_dumps(value)
    if len(raw.encode("utf-8")) > max_bytes:
        raise ValidationError(
            f"'{field}' excede o limite de {max_bytes} bytes em JSON."
        )
    return raw


def _canonical_hash(value: Any) -> str:
    raw = _json_dumps(value, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _validate_allowed_keys(args: Mapping[str, Any], allowed: Sequence[str]) -> None:
    unexpected = sorted(set(args) - set(allowed))
    if unexpected:
        raise ValidationError(
            "Argumentos desconhecidos: " + ", ".join(repr(item) for item in unexpected)
        )


def _string(
    args: Mapping[str, Any],
    name: str,
    *,
    required: bool = True,
    default: Optional[str] = None,
    max_length: int = 256,
    allow_empty: bool = False,
) -> Optional[str]:
    if name not in args:
        if required:
            raise ValidationError(f"O argumento '{name}' é obrigatório.")
        return default
    value = args[name]
    if value is None and not required:
        return default
    if not isinstance(value, str):
        raise ValidationError(f"'{name}' deve ser texto.")
    if value != value.strip():
        raise ValidationError(f"'{name}' não pode começar ou terminar com espaços.")
    if not allow_empty and not value:
        raise ValidationError(f"'{name}' não pode ser vazio.")
    if len(value) > max_length:
        raise ValidationError(f"'{name}' excede {max_length} caracteres.")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValidationError(f"'{name}' contém caracteres de controle.")
    return value


def _agent_name(args: Mapping[str, Any], name: str = "agent") -> str:
    value = _string(args, name, max_length=128)
    assert value is not None
    if value == "*":
        raise ValidationError("'*' é reservado para mensagens de broadcast.")
    return value


def _integer(
    args: Mapping[str, Any],
    name: str,
    *,
    required: bool = True,
    default: Optional[int] = None,
    minimum: Optional[int] = None,
    maximum: Optional[int] = None,
) -> Optional[int]:
    if name not in args:
        if required:
            raise ValidationError(f"O argumento '{name}' é obrigatório.")
        return default
    value = args[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"'{name}' deve ser um número inteiro.")
    if minimum is not None and value < minimum:
        raise ValidationError(f"'{name}' deve ser no mínimo {minimum}.")
    if maximum is not None and value > maximum:
        raise ValidationError(f"'{name}' deve ser no máximo {maximum}.")
    return value


def _number(
    args: Mapping[str, Any],
    name: str,
    *,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    if name not in args:
        return default
    value = args[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"'{name}' deve ser numérico.")
    result = float(value)
    if not minimum <= result <= maximum:
        raise ValidationError(f"'{name}' deve estar entre {minimum} e {maximum}.")
    return result


def _boolean(args: Mapping[str, Any], name: str, default: bool) -> bool:
    if name not in args:
        return default
    value = args[name]
    if not isinstance(value, bool):
        raise ValidationError(f"'{name}' deve ser booleano.")
    return value


def _idempotency_key(args: Mapping[str, Any]) -> Optional[str]:
    return _string(
        args,
        "idempotency_key",
        required=False,
        default=None,
        max_length=256,
    )


def _default_database_path() -> Path:
    configured = os.getenv(DB_PATH_ENV) or os.getenv(DB_PATH_ENV_ALIAS)
    if configured:
        return Path(configured).expanduser().resolve()
    return (
        Path(__file__).resolve().parents[1]
        / "data"
        / "integrations"
        / "team_bus.sqlite"
    )


class TeamBus:
    """SQLite-backed collaboration primitives shared by all MCP processes."""

    def __init__(self, database_path: Optional[Path] = None):
        self.database_path = (database_path or _default_database_path()).resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            str(self.database_path),
            timeout=BUSY_TIMEOUT_MS / 1000,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA synchronous=NORMAL")
        return connection

    def _initialize_schema(self) -> None:
        schema = """
        CREATE TABLE IF NOT EXISTS agents (
            name TEXT PRIMARY KEY,
            role TEXT NOT NULL,
            capabilities_json TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            status TEXT NOT NULL CHECK (status IN ('active', 'inactive')),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            last_seen_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sender TEXT NOT NULL REFERENCES agents(name),
            recipient TEXT REFERENCES agents(name),
            channel TEXT NOT NULL,
            kind TEXT NOT NULL,
            content_json TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            correlation_id TEXT,
            reply_to INTEGER REFERENCES messages(id),
            created_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_messages_recipient_id
            ON messages(recipient, id);
        CREATE INDEX IF NOT EXISTS idx_messages_channel_id
            ON messages(channel, id);

        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            description TEXT,
            payload_json TEXT NOT NULL,
            result_json TEXT,
            priority INTEGER NOT NULL,
            status TEXT NOT NULL CHECK (
                status IN ('open', 'claimed', 'in_progress', 'completed', 'failed', 'cancelled')
            ),
            created_by TEXT NOT NULL REFERENCES agents(name),
            assigned_to TEXT REFERENCES agents(name),
            version INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            claimed_at TEXT,
            completed_at TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_tasks_status_priority_id
            ON tasks(status, priority DESC, id);
        CREATE INDEX IF NOT EXISTS idx_tasks_assigned_id
            ON tasks(assigned_to, id);

        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            actor TEXT,
            action TEXT NOT NULL,
            target_type TEXT,
            target_id TEXT,
            success INTEGER NOT NULL CHECK (success IN (0, 1)),
            details_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_audit_actor_id ON audit_log(actor, id);
        CREATE INDEX IF NOT EXISTS idx_audit_action_id ON audit_log(action, id);

        CREATE TABLE IF NOT EXISTS idempotency (
            scope TEXT NOT NULL,
            key TEXT NOT NULL,
            request_hash TEXT NOT NULL,
            state TEXT NOT NULL CHECK (state IN ('in_progress', 'completed')),
            response_json TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (scope, key)
        );
        """
        last_error: Optional[Exception] = None
        for attempt in range(WRITE_RETRIES):
            connection: Optional[sqlite3.Connection] = None
            try:
                connection = self._connect()
                mode = connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]
                if str(mode).lower() != "wal":
                    raise RuntimeError("SQLite não ativou journal_mode=WAL.")
                connection.executescript(schema)
                return
            except sqlite3.OperationalError as exc:
                last_error = exc
                if not self._is_busy(exc) or attempt == WRITE_RETRIES - 1:
                    raise
                time.sleep(min(0.025 * (2**attempt), 0.5))
            finally:
                if connection is not None:
                    connection.close()
        if last_error is not None:
            raise last_error

    @staticmethod
    def _is_busy(error: sqlite3.OperationalError) -> bool:
        text = str(error).lower()
        return "locked" in text or "busy" in text

    def _write(self, operation: Callable[[sqlite3.Connection], Any]) -> Any:
        last_error: Optional[Exception] = None
        for attempt in range(WRITE_RETRIES):
            connection = self._connect()
            try:
                connection.execute("BEGIN IMMEDIATE")
                result = operation(connection)
                connection.commit()
                return result
            except sqlite3.OperationalError as exc:
                connection.rollback()
                last_error = exc
                if not self._is_busy(exc) or attempt == WRITE_RETRIES - 1:
                    raise
                time.sleep(min(0.025 * (2**attempt), 0.5))
            except Exception:
                connection.rollback()
                raise
            finally:
                connection.close()
        if last_error is not None:
            raise last_error
        raise RuntimeError("A transação SQLite não foi executada.")

    def _idempotent_write(
        self,
        scope: str,
        key: Optional[str],
        request_payload: Any,
        operation: Callable[[sqlite3.Connection], Dict[str, Any]],
    ) -> Dict[str, Any]:
        request_hash = _canonical_hash(request_payload)

        def transaction(connection: sqlite3.Connection) -> Dict[str, Any]:
            if key is not None:
                existing = connection.execute(
                    "SELECT request_hash, state, response_json "
                    "FROM idempotency WHERE scope=? AND key=?",
                    (scope, key),
                ).fetchone()
                if existing is not None:
                    if existing["request_hash"] != request_hash:
                        raise ConflictError(
                            "A chave de idempotência já foi usada com outros argumentos."
                        )
                    if existing["state"] == "in_progress":
                        raise ConflictError(
                            "A operação desta chave de idempotência ainda está em execução."
                        )
                    return _json_loads(existing["response_json"])

            result = operation(connection)
            if key is not None:
                now = _utc_now()
                connection.execute(
                    "INSERT INTO idempotency "
                    "(scope, key, request_hash, state, response_json, created_at, updated_at) "
                    "VALUES (?, ?, ?, 'completed', ?, ?, ?)",
                    (scope, key, request_hash, _json_dumps(result), now, now),
                )
            return result

        return self._write(transaction)

    def _reserve_external_idempotency(
        self, scope: str, key: Optional[str], request_payload: Any
    ) -> Tuple[str, str, Optional[Dict[str, Any]]]:
        request_hash = _canonical_hash(request_payload)
        if key is None:
            return "new", request_hash, None

        def transaction(connection: sqlite3.Connection) -> Tuple[str, str, Optional[Dict[str, Any]]]:
            existing = connection.execute(
                "SELECT request_hash, state, response_json "
                "FROM idempotency WHERE scope=? AND key=?",
                (scope, key),
            ).fetchone()
            if existing is not None:
                if existing["request_hash"] != request_hash:
                    raise ConflictError(
                        "A chave de idempotência já foi usada com outros argumentos."
                    )
                if existing["state"] == "in_progress":
                    raise ConflictError(
                        "A execução externa desta chave está em andamento ou terminou "
                        "sem confirmação; use uma nova chave somente após verificar o efeito."
                    )
                return "completed", request_hash, _json_loads(existing["response_json"])

            now = _utc_now()
            connection.execute(
                "INSERT INTO idempotency "
                "(scope, key, request_hash, state, response_json, created_at, updated_at) "
                "VALUES (?, ?, ?, 'in_progress', NULL, ?, ?)",
                (scope, key, request_hash, now, now),
            )
            return "new", request_hash, None

        return self._write(transaction)

    @staticmethod
    def _insert_audit(
        connection: sqlite3.Connection,
        *,
        actor: Optional[str],
        action: str,
        target_type: Optional[str],
        target_id: Optional[Any],
        success: bool,
        details: Mapping[str, Any],
    ) -> int:
        cursor = connection.execute(
            "INSERT INTO audit_log "
            "(actor, action, target_type, target_id, success, details_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                actor,
                action,
                target_type,
                None if target_id is None else str(target_id),
                1 if success else 0,
                _json_text(dict(details), "audit.details"),
                _utc_now(),
            ),
        )
        return int(cursor.lastrowid)

    def _complete_external(
        self,
        *,
        scope: str,
        key: Optional[str],
        request_hash: str,
        result: Dict[str, Any],
        actor: str,
        success: bool,
        details: Mapping[str, Any],
    ) -> Dict[str, Any]:
        def transaction(connection: sqlite3.Connection) -> Dict[str, Any]:
            final = dict(result)
            audit_id = self._insert_audit(
                connection,
                actor=actor,
                action="execute_command",
                target_type="process",
                target_id=None,
                success=success,
                details=details,
            )
            final["audit_id"] = audit_id
            if key is not None:
                cursor = connection.execute(
                    "UPDATE idempotency SET state='completed', response_json=?, updated_at=? "
                    "WHERE scope=? AND key=? AND request_hash=? AND state='in_progress'",
                    (_json_dumps(final), _utc_now(), scope, key, request_hash),
                )
                if cursor.rowcount != 1:
                    raise ConflictError(
                        "A reserva idempotente da execução não pôde ser concluída."
                    )
            return final

        return self._write(transaction)

    @staticmethod
    def _ensure_agent(connection: sqlite3.Connection, name: str) -> sqlite3.Row:
        row = connection.execute(
            "SELECT * FROM agents WHERE name=? AND status='active'", (name,)
        ).fetchone()
        if row is None:
            raise NotFoundError(f"Agente ativo não registrado: {name}")
        return row

    @staticmethod
    def _agent_result(row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "name": row["name"],
            "role": row["role"],
            "capabilities": _json_loads(row["capabilities_json"]),
            "metadata": _json_loads(row["metadata_json"]),
            "status": row["status"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "last_seen_at": row["last_seen_at"],
        }

    @staticmethod
    def _message_result(row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "id": int(row["id"]),
            "from_agent": row["sender"],
            "to_agent": row["recipient"] or "*",
            "channel": row["channel"],
            "kind": row["kind"],
            "content": _json_loads(row["content_json"]),
            "metadata": _json_loads(row["metadata_json"]),
            "correlation_id": row["correlation_id"],
            "reply_to": row["reply_to"],
            "created_at": row["created_at"],
        }

    @staticmethod
    def _task_result(row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "id": int(row["id"]),
            "title": row["title"],
            "description": row["description"],
            "payload": _json_loads(row["payload_json"]),
            "result": (
                None if row["result_json"] is None else _json_loads(row["result_json"])
            ),
            "priority": int(row["priority"]),
            "status": row["status"],
            "created_by": row["created_by"],
            "assigned_to": row["assigned_to"],
            "version": int(row["version"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "claimed_at": row["claimed_at"],
            "completed_at": row["completed_at"],
        }

    @staticmethod
    def _audit_result(row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "id": int(row["id"]),
            "actor": row["actor"],
            "action": row["action"],
            "target_type": row["target_type"],
            "target_id": row["target_id"],
            "success": bool(row["success"]),
            "details": _json_loads(row["details_json"]),
            "created_at": row["created_at"],
        }

    def ping(self, args: Mapping[str, Any]) -> Dict[str, Any]:
        _validate_allowed_keys(args, [])
        connection = self._connect()
        try:
            journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
            agents = connection.execute(
                "SELECT COUNT(*) FROM agents WHERE status='active'"
            ).fetchone()[0]
            open_tasks = connection.execute(
                "SELECT COUNT(*) FROM tasks WHERE status IN ('open', 'claimed', 'in_progress')"
            ).fetchone()[0]
        finally:
            connection.close()
        return {
            "ok": True,
            "server": SERVER_NAME,
            "version": SERVER_VERSION,
            "time": _utc_now(),
            "database": str(self.database_path),
            "journal_mode": str(journal_mode).lower(),
            "active_agents": int(agents),
            "active_tasks": int(open_tasks),
        }

    def register_agent(self, args: Mapping[str, Any]) -> Dict[str, Any]:
        _validate_allowed_keys(
            args, ["name", "role", "capabilities", "metadata", "idempotency_key"]
        )
        name = _agent_name(args, "name")
        role = _string(
            args, "role", required=False, default="agent", max_length=128
        )
        assert role is not None
        capabilities = args.get("capabilities", [])
        if not isinstance(capabilities, list) or len(capabilities) > 100:
            raise ValidationError("'capabilities' deve ser uma lista de até 100 textos.")
        normalized_capabilities: List[str] = []
        for capability in capabilities:
            if not isinstance(capability, str) or not capability.strip():
                raise ValidationError("Cada capability deve ser texto não vazio.")
            if capability != capability.strip() or len(capability) > 128:
                raise ValidationError("Capability inválida ou maior que 128 caracteres.")
            normalized_capabilities.append(capability)
        metadata = args.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValidationError("'metadata' deve ser um objeto JSON.")
        capabilities_json = _json_text(
            normalized_capabilities, "capabilities", max_bytes=64 * 1024
        )
        metadata_json = _json_text(metadata, "metadata", max_bytes=256 * 1024)
        key = _idempotency_key(args)
        payload = {
            "name": name,
            "role": role,
            "capabilities": normalized_capabilities,
            "metadata": metadata,
        }

        def operation(connection: sqlite3.Connection) -> Dict[str, Any]:
            now = _utc_now()
            connection.execute(
                "INSERT INTO agents "
                "(name, role, capabilities_json, metadata_json, status, created_at, "
                " updated_at, last_seen_at) "
                "VALUES (?, ?, ?, ?, 'active', ?, ?, ?) "
                "ON CONFLICT(name) DO UPDATE SET "
                "role=excluded.role, capabilities_json=excluded.capabilities_json, "
                "metadata_json=excluded.metadata_json, status='active', "
                "updated_at=excluded.updated_at, last_seen_at=excluded.last_seen_at",
                (name, role, capabilities_json, metadata_json, now, now, now),
            )
            row = connection.execute(
                "SELECT * FROM agents WHERE name=?", (name,)
            ).fetchone()
            audit_id = self._insert_audit(
                connection,
                actor=name,
                action="register_agent",
                target_type="agent",
                target_id=name,
                success=True,
                details={"role": role, "capability_count": len(normalized_capabilities)},
            )
            return {"agent": self._agent_result(row), "audit_id": audit_id}

        return self._idempotent_write(
            f"register_agent:{name}", key, payload, operation
        )

    def post_message(self, args: Mapping[str, Any]) -> Dict[str, Any]:
        _validate_allowed_keys(
            args,
            [
                "from_agent",
                "to_agent",
                "content",
                "channel",
                "kind",
                "metadata",
                "correlation_id",
                "reply_to",
                "idempotency_key",
            ],
        )
        sender = _agent_name(args, "from_agent")
        recipient_raw = _string(args, "to_agent", max_length=128)
        assert recipient_raw is not None
        recipient = None if recipient_raw == "*" else recipient_raw
        if recipient is not None and any(
            ord(char) < 32 or ord(char) == 127 for char in recipient
        ):
            raise ValidationError("'to_agent' contém caracteres de controle.")
        if "content" not in args:
            raise ValidationError("O argumento 'content' é obrigatório.")
        content = args["content"]
        content_json = _json_text(content, "content")
        channel = _string(
            args, "channel", required=False, default="default", max_length=128
        )
        kind = _string(
            args, "kind", required=False, default="message", max_length=64
        )
        assert channel is not None and kind is not None
        metadata = args.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValidationError("'metadata' deve ser um objeto JSON.")
        metadata_json = _json_text(metadata, "metadata", max_bytes=256 * 1024)
        correlation_id = _string(
            args,
            "correlation_id",
            required=False,
            default=None,
            max_length=256,
        )
        reply_to = _integer(
            args, "reply_to", required=False, default=None, minimum=1
        )
        key = _idempotency_key(args)
        payload = {
            "from_agent": sender,
            "to_agent": recipient_raw,
            "content": content,
            "channel": channel,
            "kind": kind,
            "metadata": metadata,
            "correlation_id": correlation_id,
            "reply_to": reply_to,
        }

        def operation(connection: sqlite3.Connection) -> Dict[str, Any]:
            self._ensure_agent(connection, sender)
            if recipient is not None:
                self._ensure_agent(connection, recipient)
            if reply_to is not None:
                exists = connection.execute(
                    "SELECT 1 FROM messages WHERE id=?", (reply_to,)
                ).fetchone()
                if exists is None:
                    raise NotFoundError(f"Mensagem reply_to não encontrada: {reply_to}")
            cursor = connection.execute(
                "INSERT INTO messages "
                "(sender, recipient, channel, kind, content_json, metadata_json, "
                " correlation_id, reply_to, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    sender,
                    recipient,
                    channel,
                    kind,
                    content_json,
                    metadata_json,
                    correlation_id,
                    reply_to,
                    _utc_now(),
                ),
            )
            message_id = int(cursor.lastrowid)
            row = connection.execute(
                "SELECT * FROM messages WHERE id=?", (message_id,)
            ).fetchone()
            audit_id = self._insert_audit(
                connection,
                actor=sender,
                action="post_message",
                target_type="message",
                target_id=message_id,
                success=True,
                details={
                    "to_agent": recipient_raw,
                    "channel": channel,
                    "kind": kind,
                    "correlation_id": correlation_id,
                },
            )
            return {"message": self._message_result(row), "audit_id": audit_id}

        return self._idempotent_write(
            f"post_message:{sender}", key, payload, operation
        )

    def fetch_messages(self, args: Mapping[str, Any]) -> Dict[str, Any]:
        _validate_allowed_keys(
            args, ["agent", "after_id", "limit", "channel", "include_broadcast"]
        )
        agent = _agent_name(args)
        after_id = _integer(
            args, "after_id", required=False, default=0, minimum=0
        )
        limit = _integer(
            args, "limit", required=False, default=50, minimum=1, maximum=200
        )
        assert after_id is not None and limit is not None
        channel = _string(
            args, "channel", required=False, default=None, max_length=128
        )
        include_broadcast = _boolean(args, "include_broadcast", True)

        connection = self._connect()
        try:
            self._ensure_agent(connection, agent)
            conditions = ["id > ?"]
            parameters: List[Any] = [after_id]
            if include_broadcast:
                conditions.append("(recipient=? OR recipient IS NULL)")
            else:
                conditions.append("recipient=?")
            parameters.append(agent)
            if channel is not None:
                conditions.append("channel=?")
                parameters.append(channel)
            parameters.append(limit)
            rows = connection.execute(
                "SELECT * FROM messages WHERE "
                + " AND ".join(conditions)
                + " ORDER BY id ASC LIMIT ?",
                parameters,
            ).fetchall()
        finally:
            connection.close()
        messages = [self._message_result(row) for row in rows]
        return {
            "messages": messages,
            "count": len(messages),
            "next_after_id": messages[-1]["id"] if messages else after_id,
        }

    def create_task(self, args: Mapping[str, Any]) -> Dict[str, Any]:
        _validate_allowed_keys(
            args,
            [
                "created_by",
                "title",
                "description",
                "payload",
                "priority",
                "idempotency_key",
            ],
        )
        creator = _agent_name(args, "created_by")
        title = _string(args, "title", max_length=512)
        assert title is not None
        description = _string(
            args,
            "description",
            required=False,
            default=None,
            max_length=20_000,
            allow_empty=True,
        )
        task_payload = args.get("payload", {})
        payload_json = _json_text(task_payload, "payload")
        priority = _integer(
            args, "priority", required=False, default=0, minimum=-100, maximum=100
        )
        assert priority is not None
        key = _idempotency_key(args)
        request_payload = {
            "created_by": creator,
            "title": title,
            "description": description,
            "payload": task_payload,
            "priority": priority,
        }

        def operation(connection: sqlite3.Connection) -> Dict[str, Any]:
            self._ensure_agent(connection, creator)
            now = _utc_now()
            cursor = connection.execute(
                "INSERT INTO tasks "
                "(title, description, payload_json, result_json, priority, status, "
                " created_by, assigned_to, version, created_at, updated_at, "
                " claimed_at, completed_at) "
                "VALUES (?, ?, ?, NULL, ?, 'open', ?, NULL, 1, ?, ?, NULL, NULL)",
                (title, description, payload_json, priority, creator, now, now),
            )
            task_id = int(cursor.lastrowid)
            row = connection.execute(
                "SELECT * FROM tasks WHERE id=?", (task_id,)
            ).fetchone()
            audit_id = self._insert_audit(
                connection,
                actor=creator,
                action="create_task",
                target_type="task",
                target_id=task_id,
                success=True,
                details={"priority": priority},
            )
            return {"task": self._task_result(row), "audit_id": audit_id}

        return self._idempotent_write(
            f"create_task:{creator}", key, request_payload, operation
        )

    def claim_task(self, args: Mapping[str, Any]) -> Dict[str, Any]:
        _validate_allowed_keys(args, ["task_id", "agent", "idempotency_key"])
        task_id = _integer(args, "task_id", minimum=1)
        agent = _agent_name(args)
        key = _idempotency_key(args)
        assert task_id is not None
        request_payload = {"task_id": task_id, "agent": agent}

        def operation(connection: sqlite3.Connection) -> Dict[str, Any]:
            self._ensure_agent(connection, agent)
            existing = connection.execute(
                "SELECT * FROM tasks WHERE id=?", (task_id,)
            ).fetchone()
            if existing is None:
                raise NotFoundError(f"Tarefa não encontrada: {task_id}")
            if existing["status"] != "open" or existing["assigned_to"] is not None:
                raise ConflictError(
                    f"Tarefa {task_id} não está disponível; status={existing['status']}."
                )
            now = _utc_now()
            cursor = connection.execute(
                "UPDATE tasks SET status='claimed', assigned_to=?, version=version+1, "
                "updated_at=?, claimed_at=?, completed_at=NULL "
                "WHERE id=? AND status='open' AND assigned_to IS NULL",
                (agent, now, now, task_id),
            )
            if cursor.rowcount != 1:
                raise ConflictError(f"Tarefa {task_id} foi reivindicada por outro agente.")
            row = connection.execute(
                "SELECT * FROM tasks WHERE id=?", (task_id,)
            ).fetchone()
            audit_id = self._insert_audit(
                connection,
                actor=agent,
                action="claim_task",
                target_type="task",
                target_id=task_id,
                success=True,
                details={"version": int(row["version"])},
            )
            return {"task": self._task_result(row), "audit_id": audit_id}

        return self._idempotent_write(
            f"claim_task:{agent}", key, request_payload, operation
        )

    def update_task(self, args: Mapping[str, Any]) -> Dict[str, Any]:
        _validate_allowed_keys(
            args,
            ["task_id", "actor", "status", "result", "expected_version", "idempotency_key"],
        )
        task_id = _integer(args, "task_id", minimum=1)
        actor = _agent_name(args, "actor")
        status = _string(
            args, "status", required=False, default=None, max_length=32
        )
        if status is not None and status not in TASK_STATUSES:
            raise ValidationError(
                "'status' deve ser um de: " + ", ".join(sorted(TASK_STATUSES))
            )
        has_result = "result" in args
        if status is None and not has_result:
            raise ValidationError("Informe ao menos 'status' ou 'result'.")
        result_value = args.get("result")
        result_json = _json_text(result_value, "result") if has_result else None
        expected_version = _integer(
            args, "expected_version", required=False, default=None, minimum=1
        )
        key = _idempotency_key(args)
        assert task_id is not None
        request_payload = {
            "task_id": task_id,
            "actor": actor,
            "status": status,
            "has_result": has_result,
            "result": result_value,
            "expected_version": expected_version,
        }

        def operation(connection: sqlite3.Connection) -> Dict[str, Any]:
            self._ensure_agent(connection, actor)
            current = connection.execute(
                "SELECT * FROM tasks WHERE id=?", (task_id,)
            ).fetchone()
            if current is None:
                raise NotFoundError(f"Tarefa não encontrada: {task_id}")
            if actor not in {current["created_by"], current["assigned_to"]}:
                raise ConflictError(
                    "Somente o criador ou o agente responsável pode atualizar a tarefa."
                )
            current_version = int(current["version"])
            if expected_version is not None and expected_version != current_version:
                raise ConflictError(
                    f"Versão divergente: esperada {expected_version}, atual {current_version}."
                )
            old_status = current["status"]
            new_status = status or old_status
            if new_status != old_status and new_status not in TASK_TRANSITIONS[old_status]:
                raise ConflictError(
                    f"Transição de status inválida: {old_status} -> {new_status}."
                )
            assigned_to = current["assigned_to"]
            claimed_at = current["claimed_at"]
            if new_status == "open":
                assigned_to = None
                claimed_at = None
            completed_at = _utc_now() if new_status in TERMINAL_TASK_STATUSES else None
            final_result_json = result_json if has_result else current["result_json"]
            now = _utc_now()
            cursor = connection.execute(
                "UPDATE tasks SET status=?, assigned_to=?, result_json=?, "
                "version=version+1, updated_at=?, claimed_at=?, completed_at=? "
                "WHERE id=? AND version=?",
                (
                    new_status,
                    assigned_to,
                    final_result_json,
                    now,
                    claimed_at,
                    completed_at,
                    task_id,
                    current_version,
                ),
            )
            if cursor.rowcount != 1:
                raise ConflictError(f"Tarefa {task_id} foi modificada simultaneamente.")
            row = connection.execute(
                "SELECT * FROM tasks WHERE id=?", (task_id,)
            ).fetchone()
            audit_id = self._insert_audit(
                connection,
                actor=actor,
                action="update_task",
                target_type="task",
                target_id=task_id,
                success=True,
                details={
                    "from_status": old_status,
                    "to_status": new_status,
                    "version": int(row["version"]),
                    "result_updated": has_result,
                },
            )
            return {"task": self._task_result(row), "audit_id": audit_id}

        return self._idempotent_write(
            f"update_task:{actor}", key, request_payload, operation
        )

    def list_tasks(self, args: Mapping[str, Any]) -> Dict[str, Any]:
        _validate_allowed_keys(
            args, ["status", "assigned_to", "created_by", "after_id", "limit"]
        )
        status = _string(
            args, "status", required=False, default=None, max_length=32
        )
        if status is not None and status not in TASK_STATUSES:
            raise ValidationError(
                "'status' deve ser um de: " + ", ".join(sorted(TASK_STATUSES))
            )
        assigned_to = _string(
            args, "assigned_to", required=False, default=None, max_length=128
        )
        created_by = _string(
            args, "created_by", required=False, default=None, max_length=128
        )
        after_id = _integer(
            args, "after_id", required=False, default=0, minimum=0
        )
        limit = _integer(
            args, "limit", required=False, default=100, minimum=1, maximum=200
        )
        assert after_id is not None and limit is not None
        conditions = ["id > ?"]
        parameters: List[Any] = [after_id]
        if status is not None:
            conditions.append("status=?")
            parameters.append(status)
        if assigned_to is not None:
            conditions.append("assigned_to=?")
            parameters.append(assigned_to)
        if created_by is not None:
            conditions.append("created_by=?")
            parameters.append(created_by)
        parameters.append(limit)
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT * FROM tasks WHERE "
                + " AND ".join(conditions)
                + " ORDER BY id ASC LIMIT ?",
                parameters,
            ).fetchall()
        finally:
            connection.close()
        tasks = [self._task_result(row) for row in rows]
        return {
            "tasks": tasks,
            "count": len(tasks),
            "next_after_id": max((task["id"] for task in tasks), default=after_id),
        }

    def get_audit_log(self, args: Mapping[str, Any]) -> Dict[str, Any]:
        _validate_allowed_keys(
            args, ["actor", "action", "success", "after_id", "limit"]
        )
        actor = _string(
            args, "actor", required=False, default=None, max_length=128
        )
        action = _string(
            args, "action", required=False, default=None, max_length=128
        )
        success = args.get("success")
        if success is not None and not isinstance(success, bool):
            raise ValidationError("'success' deve ser booleano.")
        after_id = _integer(
            args, "after_id", required=False, default=0, minimum=0
        )
        limit = _integer(
            args, "limit", required=False, default=100, minimum=1, maximum=500
        )
        assert after_id is not None and limit is not None
        conditions = ["id > ?"]
        parameters: List[Any] = [after_id]
        if actor is not None:
            conditions.append("actor=?")
            parameters.append(actor)
        if action is not None:
            conditions.append("action=?")
            parameters.append(action)
        if success is not None:
            conditions.append("success=?")
            parameters.append(1 if success else 0)
        parameters.append(limit)
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT * FROM audit_log WHERE "
                + " AND ".join(conditions)
                + " ORDER BY id ASC LIMIT ?",
                parameters,
            ).fetchall()
        finally:
            connection.close()
        entries = [self._audit_result(row) for row in rows]
        return {
            "entries": entries,
            "count": len(entries),
            "next_after_id": entries[-1]["id"] if entries else after_id,
        }

    def execute_command(self, args: Mapping[str, Any]) -> Dict[str, Any]:
        _validate_allowed_keys(
            args,
            [
                "actor",
                "command",
                "cwd",
                "timeout_seconds",
                "max_output_chars",
                "shell",
                "idempotency_key",
            ],
        )
        actor = _agent_name(args, "actor")
        if "command" not in args:
            raise ValidationError("O argumento 'command' é obrigatório.")
        command = args["command"]
        if isinstance(command, str):
            if not command.strip():
                raise ValidationError("'command' não pode ser vazio.")
            if len(command) > MAX_COMMAND_CHARS:
                raise ValidationError(
                    f"'command' excede {MAX_COMMAND_CHARS} caracteres."
                )
            default_shell = True
            command_for_process: Any = command
            command_kind = "string"
        elif isinstance(command, list):
            if not command or len(command) > 256:
                raise ValidationError(
                    "'command' como lista deve conter entre 1 e 256 argumentos."
                )
            normalized: List[str] = []
            total_chars = 0
            for item in command:
                if not isinstance(item, str) or "\x00" in item:
                    raise ValidationError(
                        "Cada item de 'command' deve ser texto sem byte NUL."
                    )
                total_chars += len(item)
                normalized.append(item)
            if not normalized[0] or total_chars > MAX_COMMAND_CHARS:
                raise ValidationError("Lista 'command' vazia ou grande demais.")
            default_shell = False
            command_for_process = normalized
            command_kind = "argv"
        else:
            raise ValidationError("'command' deve ser texto ou lista de textos.")

        shell = _boolean(args, "shell", default_shell)
        if isinstance(command_for_process, list) and shell:
            raise ValidationError(
                "Use 'command' como texto quando 'shell=true'; listas usam shell=false."
            )
        cwd_raw = _string(
            args,
            "cwd",
            required=False,
            default=None,
            max_length=32_768,
            allow_empty=False,
        )
        cwd = Path(cwd_raw).expanduser().resolve() if cwd_raw else Path.cwd().resolve()
        if not cwd.exists() or not cwd.is_dir():
            raise ValidationError(f"Diretório de trabalho inexistente: {cwd}")
        timeout_seconds = _number(
            args,
            "timeout_seconds",
            default=120.0,
            minimum=0.1,
            maximum=3600.0,
        )
        max_output_chars = _integer(
            args,
            "max_output_chars",
            required=False,
            default=100_000,
            minimum=1,
            maximum=MAX_COMMAND_OUTPUT_CHARS,
        )
        assert max_output_chars is not None
        key = _idempotency_key(args)

        connection = self._connect()
        try:
            self._ensure_agent(connection, actor)
        finally:
            connection.close()

        request_payload = {
            "actor": actor,
            "command": command_for_process,
            "cwd": str(cwd),
            "timeout_seconds": timeout_seconds,
            "max_output_chars": max_output_chars,
            "shell": shell,
        }
        scope = f"execute_command:{actor}"
        state, request_hash, cached = self._reserve_external_idempotency(
            scope, key, request_payload
        )
        if state == "completed":
            assert cached is not None
            return cached

        started_at = time.monotonic()
        command_hash = _canonical_hash(command_for_process)
        process: Optional[subprocess.Popen[bytes]] = None
        stdout_collector = _CappedCollector(max_output_chars)
        stderr_collector = _CappedCollector(max_output_chars)
        timed_out = False
        spawn_error: Optional[str] = None
        spawn_error_type: Optional[str] = None

        try:
            popen_kwargs: Dict[str, Any] = {
                "cwd": str(cwd),
                "shell": shell,
                "stdin": subprocess.DEVNULL,
                "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE,
            }
            if os.name == "nt":
                popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                popen_kwargs["start_new_session"] = True
            process = subprocess.Popen(command_for_process, **popen_kwargs)
            assert process.stdout is not None and process.stderr is not None
            stdout_thread = threading.Thread(
                target=stdout_collector.read, args=(process.stdout,), daemon=True
            )
            stderr_thread = threading.Thread(
                target=stderr_collector.read, args=(process.stderr,), daemon=True
            )
            stdout_thread.start()
            stderr_thread.start()
            try:
                process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                _terminate_process_tree(process)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            stdout_thread.join(timeout=5)
            stderr_thread.join(timeout=5)
        except (OSError, subprocess.SubprocessError) as exc:
            spawn_error = str(exc)[:2_000]
            spawn_error_type = type(exc).__name__
            if process is not None and process.poll() is None:
                _terminate_process_tree(process)

        duration_ms = int((time.monotonic() - started_at) * 1000)
        stdout, stdout_truncated = stdout_collector.result()
        stderr, stderr_truncated = stderr_collector.result()
        exit_code = process.returncode if process is not None else None
        ok = (
            process is not None
            and not timed_out
            and spawn_error is None
            and exit_code == 0
        )
        result: Dict[str, Any] = {
            "ok": ok,
            "started": process is not None,
            "exit_code": exit_code,
            "timed_out": timed_out,
            "duration_ms": duration_ms,
            "cwd": str(cwd),
            "stdout": stdout,
            "stderr": stderr,
            "stdout_truncated": stdout_truncated,
            "stderr_truncated": stderr_truncated,
            "stdout_bytes": stdout_collector.total_bytes,
            "stderr_bytes": stderr_collector.total_bytes,
        }
        if spawn_error is not None:
            result["error"] = spawn_error
            result["error_type"] = spawn_error_type

        audit_details = {
            "command_sha256": command_hash,
            "command_kind": command_kind,
            "shell": shell,
            "cwd": str(cwd),
            "timeout_seconds": timeout_seconds,
            "max_output_chars": max_output_chars,
            "started": process is not None,
            "exit_code": exit_code,
            "timed_out": timed_out,
            "duration_ms": duration_ms,
            "stdout_bytes": stdout_collector.total_bytes,
            "stderr_bytes": stderr_collector.total_bytes,
            "stdout_truncated": stdout_truncated,
            "stderr_truncated": stderr_truncated,
            "error_type": spawn_error_type,
        }
        return self._complete_external(
            scope=scope,
            key=key,
            request_hash=request_hash,
            result=result,
            actor=actor,
            success=ok,
            details=audit_details,
        )


class _CappedCollector:
    """Drain a subprocess pipe while retaining a bounded UTF-8-safe result."""

    def __init__(self, max_chars: int):
        self.max_chars = max_chars
        self.max_bytes = max_chars * 4
        self.buffer = bytearray()
        self.total_bytes = 0
        self.byte_truncated = False

    def read(self, stream: Any) -> None:
        try:
            while True:
                chunk = stream.read(8192)
                if not chunk:
                    break
                self.total_bytes += len(chunk)
                remaining = self.max_bytes - len(self.buffer)
                if remaining > 0:
                    self.buffer.extend(chunk[:remaining])
                if len(chunk) > remaining:
                    self.byte_truncated = True
        finally:
            try:
                stream.close()
            except OSError:
                pass

    def result(self) -> Tuple[str, bool]:
        text = bytes(self.buffer).decode("utf-8", errors="replace")
        char_truncated = len(text) > self.max_chars
        if char_truncated:
            text = text[: self.max_chars]
        return text, self.byte_truncated or char_truncated


def _terminate_process_tree(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill.exe", "/PID", str(process.pid), "/T", "/F"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
                check=False,
            )
            return
        except (OSError, subprocess.SubprocessError):
            pass
    else:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            return
        except (OSError, ProcessLookupError):
            pass
    try:
        process.kill()
    except OSError:
        pass


OBJECT_SCHEMA: Dict[str, Any] = {"type": "object", "additionalProperties": True}
JSON_VALUE_SCHEMA: Dict[str, Any] = {
    "description": "Any JSON value up to the documented size limit."
}
IDEMPOTENCY_PROPERTY = {
    "type": "string",
    "minLength": 1,
    "maxLength": 256,
    "description": "Caller-generated key that makes this mutation replay-safe.",
}


def _object_schema(
    properties: Mapping[str, Any], required: Sequence[str] = ()
) -> Dict[str, Any]:
    schema: Dict[str, Any] = {
        "type": "object",
        "properties": dict(properties),
        "additionalProperties": False,
    }
    if required:
        schema["required"] = list(required)
    return schema


TOOLS: List[Dict[str, Any]] = [
    {
        "name": "ping",
        "title": "Team bus health",
        "description": "Check the local team bus, shared database, and active counts.",
        "inputSchema": _object_schema({}),
        "annotations": {"readOnlyHint": True, "idempotentHint": True},
    },
    {
        "name": "register_agent",
        "title": "Register team agent",
        "description": "Register or refresh a stable agent identity in the shared bus.",
        "inputSchema": _object_schema(
            {
                "name": {"type": "string", "minLength": 1, "maxLength": 128},
                "role": {"type": "string", "maxLength": 128, "default": "agent"},
                "capabilities": {
                    "type": "array",
                    "items": {"type": "string", "maxLength": 128},
                    "maxItems": 100,
                    "default": [],
                },
                "metadata": OBJECT_SCHEMA,
                "idempotency_key": IDEMPOTENCY_PROPERTY,
            },
            ["name"],
        ),
        "annotations": {"idempotentHint": True},
    },
    {
        "name": "post_message",
        "title": "Post team message",
        "description": "Send JSON content to a registered agent or '*' for broadcast.",
        "inputSchema": _object_schema(
            {
                "from_agent": {"type": "string", "minLength": 1, "maxLength": 128},
                "to_agent": {"type": "string", "minLength": 1, "maxLength": 128},
                "content": JSON_VALUE_SCHEMA,
                "channel": {"type": "string", "maxLength": 128, "default": "default"},
                "kind": {"type": "string", "maxLength": 64, "default": "message"},
                "metadata": OBJECT_SCHEMA,
                "correlation_id": {"type": "string", "maxLength": 256},
                "reply_to": {"type": "integer", "minimum": 1},
                "idempotency_key": IDEMPOTENCY_PROPERTY,
            },
            ["from_agent", "to_agent", "content"],
        ),
        "annotations": {"idempotentHint": True},
    },
    {
        "name": "fetch_messages",
        "title": "Fetch team messages",
        "description": "Fetch direct and optionally broadcast messages after a stable message id.",
        "inputSchema": _object_schema(
            {
                "agent": {"type": "string", "minLength": 1, "maxLength": 128},
                "after_id": {"type": "integer", "minimum": 0, "default": 0},
                "limit": {"type": "integer", "minimum": 1, "maximum": 200, "default": 50},
                "channel": {"type": "string", "maxLength": 128},
                "include_broadcast": {"type": "boolean", "default": True},
            },
            ["agent"],
        ),
        "annotations": {"readOnlyHint": True, "idempotentHint": True},
    },
    {
        "name": "create_task",
        "title": "Create shared task",
        "description": "Create an unclaimed task with JSON payload and priority.",
        "inputSchema": _object_schema(
            {
                "created_by": {"type": "string", "minLength": 1, "maxLength": 128},
                "title": {"type": "string", "minLength": 1, "maxLength": 512},
                "description": {"type": "string", "maxLength": 20_000},
                "payload": JSON_VALUE_SCHEMA,
                "priority": {"type": "integer", "minimum": -100, "maximum": 100, "default": 0},
                "idempotency_key": IDEMPOTENCY_PROPERTY,
            },
            ["created_by", "title"],
        ),
        "annotations": {"idempotentHint": True},
    },
    {
        "name": "claim_task",
        "title": "Claim shared task",
        "description": "Atomically claim one open task for a registered agent.",
        "inputSchema": _object_schema(
            {
                "task_id": {"type": "integer", "minimum": 1},
                "agent": {"type": "string", "minLength": 1, "maxLength": 128},
                "idempotency_key": IDEMPOTENCY_PROPERTY,
            },
            ["task_id", "agent"],
        ),
        "annotations": {"idempotentHint": True},
    },
    {
        "name": "update_task",
        "title": "Update shared task",
        "description": "Update task status/result with optional optimistic version checking.",
        "inputSchema": _object_schema(
            {
                "task_id": {"type": "integer", "minimum": 1},
                "actor": {"type": "string", "minLength": 1, "maxLength": 128},
                "status": {"type": "string", "enum": sorted(TASK_STATUSES)},
                "result": JSON_VALUE_SCHEMA,
                "expected_version": {"type": "integer", "minimum": 1},
                "idempotency_key": IDEMPOTENCY_PROPERTY,
            },
            ["task_id", "actor"],
        ),
        "annotations": {"idempotentHint": True},
    },
    {
        "name": "list_tasks",
        "title": "List shared tasks",
        "description": "List tasks with status, owner, assignee, and cursor filters.",
        "inputSchema": _object_schema(
            {
                "status": {"type": "string", "enum": sorted(TASK_STATUSES)},
                "assigned_to": {"type": "string", "maxLength": 128},
                "created_by": {"type": "string", "maxLength": 128},
                "after_id": {"type": "integer", "minimum": 0, "default": 0},
                "limit": {"type": "integer", "minimum": 1, "maximum": 200, "default": 100},
            }
        ),
        "annotations": {"readOnlyHint": True, "idempotentHint": True},
    },
    {
        "name": "get_audit_log",
        "title": "Read audit log",
        "description": "Read ordered local audit entries without command output or message bodies.",
        "inputSchema": _object_schema(
            {
                "actor": {"type": "string", "maxLength": 128},
                "action": {"type": "string", "maxLength": 128},
                "success": {"type": "boolean"},
                "after_id": {"type": "integer", "minimum": 0, "default": 0},
                "limit": {"type": "integer", "minimum": 1, "maximum": 500, "default": 100},
            }
        ),
        "annotations": {"readOnlyHint": True, "idempotentHint": True},
    },
    {
        "name": "execute_command",
        "title": "Execute authorized local command",
        "description": (
            "Explicitly execute an arbitrary local command. String commands use the system "
            "shell by default; argv arrays do not. Output is bounded and every execution is audited."
        ),
        "inputSchema": _object_schema(
            {
                "actor": {"type": "string", "minLength": 1, "maxLength": 128},
                "command": {
                    "oneOf": [
                        {"type": "string", "minLength": 1, "maxLength": MAX_COMMAND_CHARS},
                        {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                            "maxItems": 256,
                        },
                    ]
                },
                "cwd": {"type": "string", "minLength": 1, "maxLength": 32_768},
                "timeout_seconds": {"type": "number", "minimum": 0.1, "maximum": 3600, "default": 120},
                "max_output_chars": {"type": "integer", "minimum": 1, "maximum": MAX_COMMAND_OUTPUT_CHARS, "default": 100_000},
                "shell": {"type": "boolean"},
                "idempotency_key": IDEMPOTENCY_PROPERTY,
            },
            ["actor", "command"],
        ),
        "annotations": {
            "destructiveHint": True,
            "idempotentHint": True,
            "openWorldHint": True,
        },
    },
]


TOOL_METHODS = {
    "ping": TeamBus.ping,
    "register_agent": TeamBus.register_agent,
    "post_message": TeamBus.post_message,
    "fetch_messages": TeamBus.fetch_messages,
    "create_task": TeamBus.create_task,
    "claim_task": TeamBus.claim_task,
    "update_task": TeamBus.update_task,
    "list_tasks": TeamBus.list_tasks,
    "get_audit_log": TeamBus.get_audit_log,
    "execute_command": TeamBus.execute_command,
}


def _tool_result(payload: Dict[str, Any], *, is_error: bool = False) -> Dict[str, Any]:
    return {
        "content": [{"type": "text", "text": _json_dumps(payload)}],
        "structuredContent": payload,
        "isError": is_error,
    }


def _jsonrpc_result(request_id: Any, result: Any) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _jsonrpc_error(
    request_id: Any, code: int, message: str, data: Any = None
) -> Dict[str, Any]:
    error: Dict[str, Any] = {"code": int(code), "message": message}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": "2.0", "id": request_id, "error": error}


class MCPServer:
    """Small synchronous MCP dispatcher for newline-delimited stdio JSON-RPC."""

    def __init__(self, bus: TeamBus):
        self.bus = bus
        self.initialized = False
        self.protocol_version = DEFAULT_PROTOCOL_VERSION

    def handle(self, message: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(message, dict):
            return _jsonrpc_error(None, -32600, "Invalid Request")
        request_id = message.get("id")
        has_id = "id" in message
        if has_id and (
            isinstance(request_id, bool)
            or not isinstance(request_id, (str, int, float, type(None)))
        ):
            return _jsonrpc_error(None, -32600, "Invalid Request")
        if message.get("jsonrpc") != "2.0" or not isinstance(
            message.get("method"), str
        ):
            return _jsonrpc_error(request_id if has_id else None, -32600, "Invalid Request")
        method = message["method"]
        params = message.get("params", {})
        if params is None:
            params = {}
        if not isinstance(params, dict):
            if not has_id:
                return None
            return _jsonrpc_error(request_id, -32602, "Invalid params")

        if not has_id:
            if method == "notifications/initialized":
                self.initialized = True
            return None

        try:
            if method == "initialize":
                requested = params.get("protocolVersion")
                if requested is not None and not isinstance(requested, str):
                    raise ProtocolError(-32602, "Invalid protocolVersion")
                self.protocol_version = (
                    requested
                    if requested in SUPPORTED_PROTOCOL_VERSIONS
                    else DEFAULT_PROTOCOL_VERSION
                )
                return _jsonrpc_result(
                    request_id,
                    {
                        "protocolVersion": self.protocol_version,
                        "capabilities": {"tools": {"listChanged": False}},
                        "serverInfo": {
                            "name": SERVER_NAME,
                            "title": SERVER_TITLE,
                            "version": SERVER_VERSION,
                        },
                        "instructions": (
                            "Register stable agent names, use idempotency keys for mutations, "
                            "and exchange work through messages/tasks. execute_command is explicit, "
                            "arbitrary, output-capped, and audited without storing command output."
                        ),
                    },
                )
            if method == "ping":
                return _jsonrpc_result(request_id, {})
            if method == "tools/list":
                cursor = params.get("cursor")
                if cursor not in (None, ""):
                    raise ProtocolError(-32602, "Invalid or expired tools cursor")
                return _jsonrpc_result(request_id, {"tools": TOOLS})
            if method == "tools/call":
                name = params.get("name")
                arguments = params.get("arguments", {})
                if not isinstance(name, str) or not name:
                    raise ProtocolError(-32602, "Tool name is required")
                if name not in TOOL_METHODS:
                    raise ProtocolError(-32602, f"Unknown tool: {name}")
                if arguments is None:
                    arguments = {}
                if not isinstance(arguments, dict):
                    raise ProtocolError(-32602, "Tool arguments must be an object")
                try:
                    payload = TOOL_METHODS[name](self.bus, arguments)
                    result = _tool_result(payload)
                except TeamBusError as exc:
                    result = _tool_result(
                        {
                            "error": {
                                "type": type(exc).__name__,
                                "message": str(exc),
                            }
                        },
                        is_error=True,
                    )
                return _jsonrpc_result(request_id, result)
            raise ProtocolError(-32601, "Method not found")
        except ProtocolError as exc:
            return _jsonrpc_error(request_id, exc.code, exc.message, exc.data)
        except Exception:
            return _jsonrpc_error(request_id, -32603, "Internal error")


def _write_message(message: Mapping[str, Any]) -> None:
    encoded = (_json_dumps(dict(message)) + "\n").encode("utf-8")
    sys.stdout.buffer.write(encoded)
    sys.stdout.buffer.flush()


def _drain_oversized_line(stream: Any) -> None:
    while True:
        remainder = stream.readline(MAX_JSON_INPUT_BYTES + 1)
        if not remainder or remainder.endswith(b"\n"):
            return


def serve_stdio() -> int:
    try:
        server = MCPServer(TeamBus())
    except Exception as exc:
        print(
            f"{SERVER_NAME}: falha ao inicializar ({type(exc).__name__}).",
            file=sys.stderr,
            flush=True,
        )
        return 1

    input_stream = sys.stdin.buffer
    while True:
        raw = input_stream.readline(MAX_JSON_INPUT_BYTES + 1)
        if not raw:
            return 0
        if len(raw) > MAX_JSON_INPUT_BYTES:
            if not raw.endswith(b"\n"):
                _drain_oversized_line(input_stream)
            _write_message(_jsonrpc_error(None, -32700, "Parse error: message too large"))
            continue
        if not raw.strip():
            continue
        try:
            decoded = raw.decode("utf-8")
            message = _json_loads(decoded)
        except (UnicodeDecodeError, ValueError, json.JSONDecodeError):
            _write_message(_jsonrpc_error(None, -32700, "Parse error"))
            continue
        response = server.handle(message)
        if response is not None:
            _write_message(response)


if __name__ == "__main__":
    raise SystemExit(serve_stdio())

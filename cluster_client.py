"""
cluster_client.py - optional local/LAN Aether cluster adapter.

This module does not call cloud APIs. It talks only to an explicitly configured
Aether Console HTTP endpoint, usually http://127.0.0.1:8080, and keeps a local
command policy before forwarding work to the cluster.
"""

from __future__ import annotations

import ipaddress
import json
import os
import socket
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen


TRUTHY = {"1", "true", "yes", "on", "sim"}
DEFAULT_DENIED_FRAGMENTS = (
    "rm -rf",
    "mkfs",
    "dd if=",
    "shutdown",
    "reboot",
    "format ",
    "del /f",
    "rmdir /s",
    "curl ",
    "wget ",
    "chmod 777",
    "chown ",
    "useradd",
    "passwd",
    ":(){",
)


class ClusterError(RuntimeError):
    """Base error for optional cluster integration."""


class ClusterDisabled(ClusterError):
    """Raised when the local cluster adapter is not enabled."""


class ClusterSecurityError(ClusterError):
    """Raised when a task violates the local Jarvis command policy."""


@dataclass(frozen=True)
class ClusterConfig:
    enabled: bool = False
    base_url: str = "http://127.0.0.1:8080"
    timeout_seconds: float = 4.0
    allow_remote: bool = False
    default_tags: List[str] = field(default_factory=list)
    allowed_prefixes: List[str] = field(
        default_factory=lambda: ["python", "py", "python.exe"]
    )
    denied_fragments: List[str] = field(
        default_factory=lambda: list(DEFAULT_DENIED_FRAGMENTS)
    )


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in TRUTHY


def _env_list(name: str, default: List[str]) -> List[str]:
    raw = os.getenv(name)
    if not raw:
        return list(default)
    return [item.strip() for item in raw.split(",") if item.strip()]


def _is_private_host(hostname: str) -> bool:
    host = hostname.strip("[]").lower()
    if host in {"localhost", "127.0.0.1", "::1"}:
        return True

    def is_private_ip(value: str) -> bool:
        ip = ipaddress.ip_address(value)
        return ip.is_loopback or ip.is_private or ip.is_link_local

    try:
        return is_private_ip(host)
    except ValueError:
        pass

    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror:
        return False

    addresses = {info[4][0] for info in infos}
    if not addresses:
        return False
    try:
        return all(is_private_ip(addr) for addr in addresses)
    except ValueError:
        return False


def _split_command_segments(command: str) -> List[str]:
    segments: List[str] = []
    for line in command.splitlines():
        for part in line.split(";"):
            for and_part in part.split("&&"):
                segments.extend(and_part.split("||"))
    return [segment.strip() for segment in segments if segment.strip()]


class ClusterClient:
    """Small HTTP client for Cluster-Aether's web console API."""

    def __init__(self, config: ClusterConfig):
        self.config = config
        parsed = urlparse(config.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ClusterSecurityError("JARVIS_CLUSTER_URL inválida.")
        if not config.allow_remote and not _is_private_host(parsed.hostname or ""):
            raise ClusterSecurityError(
                "Cluster remoto bloqueado. Use localhost/LAN ou defina "
                "JARVIS_CLUSTER_ALLOW_REMOTE=1 conscientemente."
            )

    @classmethod
    def from_env(cls) -> "ClusterClient":
        explicit_url = os.getenv("JARVIS_CLUSTER_URL")
        enabled_default = bool(explicit_url)
        cfg = ClusterConfig(
            enabled=_env_bool("JARVIS_CLUSTER_ENABLED", enabled_default),
            base_url=explicit_url or "http://127.0.0.1:8080",
            timeout_seconds=float(os.getenv("JARVIS_CLUSTER_TIMEOUT", "4")),
            allow_remote=_env_bool("JARVIS_CLUSTER_ALLOW_REMOTE", False),
            default_tags=_env_list("JARVIS_CLUSTER_DEFAULT_TAGS", []),
            allowed_prefixes=_env_list(
                "JARVIS_CLUSTER_ALLOWED_PREFIXES",
                ["python", "py", "python.exe"],
            ),
            denied_fragments=_env_list(
                "JARVIS_CLUSTER_DENIED_FRAGMENTS",
                list(DEFAULT_DENIED_FRAGMENTS),
            ),
        )
        return cls(cfg)

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def _ensure_enabled(self) -> None:
        if not self.config.enabled:
            raise ClusterDisabled(
                "Cluster Aether desativado. Configure JARVIS_CLUSTER_ENABLED=1 "
                "e JARVIS_CLUSTER_URL para liberar offload local/LAN."
            )

    def _url(self, path: str) -> str:
        return urljoin(self.config.base_url.rstrip("/") + "/", path.lstrip("/"))

    def _request_json(
        self,
        path: str,
        *,
        method: str = "GET",
        payload: Optional[Dict[str, Any]] = None,
    ) -> Any:
        self._ensure_enabled()
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(self._url(path), data=data, headers=headers, method=method)
        try:
            with urlopen(request, timeout=self.config.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ClusterError(f"Aether HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise ClusterError(f"Aether indisponível: {exc.reason}") from exc

        return json.loads(raw) if raw else {}

    def validate_command(self, command: str) -> None:
        stripped = command.strip()
        if not stripped:
            raise ClusterSecurityError("Comando vazio.")

        lower = stripped.lower()
        for fragment in self.config.denied_fragments:
            if fragment and fragment.lower() in lower:
                raise ClusterSecurityError(f"Fragmento bloqueado: {fragment}")

        allowed = [prefix.strip().lower() for prefix in self.config.allowed_prefixes if prefix.strip()]
        if not allowed:
            raise ClusterSecurityError("Nenhum prefixo permitido configurado.")

        for segment in _split_command_segments(stripped):
            normalized = segment.lower()
            if not any(
                normalized == prefix or normalized.startswith(prefix + " ")
                for prefix in allowed
            ):
                raise ClusterSecurityError(
                    f"Segmento fora da allowlist local do Jarvis: {segment}"
                )

    def status(self) -> Dict[str, Any]:
        if not self.config.enabled:
            return {
                "enabled": False,
                "base_url": self.config.base_url,
                "message": "Cluster Aether desativado.",
            }
        try:
            return {
                "enabled": True,
                "base_url": self.config.base_url,
                "cluster": self._request_json("/api/cluster/status"),
            }
        except ClusterError as exc:
            return {
                "enabled": True,
                "base_url": self.config.base_url,
                "available": False,
                "error": str(exc),
            }

    def workers(self) -> Dict[str, Any]:
        if not self.config.enabled:
            return {"enabled": False, "workers": []}
        try:
            return {"enabled": True, "workers": self._request_json("/api/workers")}
        except ClusterError as exc:
            return {"enabled": True, "workers": [], "error": str(exc)}

    def tasks(self) -> Dict[str, Any]:
        if not self.config.enabled:
            return {"enabled": False, "tasks": []}
        try:
            return {"enabled": True, "tasks": self._request_json("/api/tasks")}
        except ClusterError as exc:
            return {"enabled": True, "tasks": [], "error": str(exc)}

    def snapshot(self) -> Dict[str, Any]:
        return {
            "status": self.status(),
            "workers": self.workers().get("workers", []),
            "tasks": self.tasks().get("tasks", []),
            "policy": {
                "allowed_prefixes": self.config.allowed_prefixes,
                "default_tags": self.config.default_tags,
                "remote_allowed": self.config.allow_remote,
            },
        }

    def submit_task(
        self,
        command: str,
        *,
        required_tags: Optional[List[str]] = None,
        timeout_seconds: int = 120,
        priority: int = 5,
    ) -> Dict[str, Any]:
        self.validate_command(command)
        payload = {
            "command": command.strip(),
            "required_tags": required_tags or self.config.default_tags,
            "timeout_seconds": max(30, min(int(timeout_seconds), 24 * 60 * 60)),
            "priority": max(0, min(int(priority), 100)),
        }
        return {
            "submitted": True,
            "payload": payload,
            "response": self._request_json("/api/tasks", method="POST", payload=payload),
        }

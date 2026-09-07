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
import re
import socket
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

from jarvis_localhost.sovereign import POLICY, SovereignModeViolation


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
SHELL_EXECUTABLES = frozenset(
    {
        "bash",
        "cmd",
        "cmd.exe",
        "command.com",
        "csh",
        "dash",
        "fish",
        "ksh",
        "powershell",
        "powershell.exe",
        "pwsh",
        "pwsh.exe",
        "sh",
        "wsl",
        "wsl.exe",
        "zsh",
    }
)
EVAL_FLAGS = frozenset({"-c", "/c", "--command", "-e", "--eval", "-m"})
PYTHON_EXECUTABLES = frozenset({"python", "python.exe", "python3", "python3.exe", "py", "py.exe"})
SAFE_TOKEN_PUNCTUATION = frozenset("_./:\\=,+@-")


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


def _is_safe_command_token(token: str) -> bool:
    return bool(token) and all(
        character.isalnum() or character in SAFE_TOKEN_PUNCTUATION
        for character in token
    )


def _executable_basename(executable: str) -> str:
    return executable.replace("\\", "/").rsplit("/", 1)[-1].casefold()


class ClusterClient:
    """Small HTTP client for Cluster-Aether's web console API."""

    def __init__(self, config: ClusterConfig):
        self.config = config
        parsed = urlparse(config.base_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ClusterSecurityError("JARVIS_CLUSTER_URL inválida.")
        try:
            parsed.port
        except ValueError:
            raise ClusterSecurityError("JARVIS_CLUSTER_URL inválida.") from None
        try:
            POLICY.assert_url_allowed(config.base_url)
        except SovereignModeViolation:
            raise ClusterSecurityError(
                "Cluster externo bloqueado pela política soberana."
            ) from None
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
        return self.config.enabled and not POLICY.enabled

    def _ensure_enabled(self) -> None:
        if POLICY.enabled:
            raise ClusterDisabled(
                "Execução de comandos de cluster desativada no modo soberano."
            )
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
            raise ClusterError(f"Aether respondeu com HTTP {exc.code}.") from None
        except URLError:
            raise ClusterError("Aether indisponível.") from None

        try:
            return json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            raise ClusterError("Aether retornou uma resposta inválida.") from None

    def _validated_command(self, command: str) -> tuple[str, str, List[str]]:
        stripped = command.strip()
        if not stripped:
            raise ClusterSecurityError("Comando vazio.")
        if stripped != " ".join(stripped.split()) or any(
            character in stripped for character in "\r\n\t"
        ):
            raise ClusterSecurityError("Comando deve usar um formato simples e estruturado.")

        lower = stripped.lower()
        for fragment in self.config.denied_fragments:
            if fragment and fragment.lower() in lower:
                raise ClusterSecurityError(f"Fragmento bloqueado: {fragment}")

        allowed = [prefix.strip().lower() for prefix in self.config.allowed_prefixes if prefix.strip()]
        if not allowed:
            raise ClusterSecurityError("Nenhum prefixo permitido configurado.")

        tokens = stripped.split(" ")
        if not all(_is_safe_command_token(token) for token in tokens):
            raise ClusterSecurityError("Comando contém caracteres de shell bloqueados.")

        executable, args = tokens[0], tokens[1:]
        if executable.casefold() not in allowed:
            raise ClusterSecurityError("Executável fora da allowlist local do Jarvis.")

        executable_name = _executable_basename(executable)
        if executable_name in SHELL_EXECUTABLES:
            raise ClusterSecurityError("Interpretadores de shell não são permitidos.")
        if any(argument.casefold() in EVAL_FLAGS for argument in args):
            raise ClusterSecurityError("Execução dinâmica de código não é permitida.")
        if executable_name not in PYTHON_EXECUTABLES and any(
            argument.casefold().startswith(("-c", "-e", "--command=", "--eval="))
            for argument in args
        ):
            raise ClusterSecurityError("Execução dinâmica de código não é permitida.")

        if executable_name in PYTHON_EXECUTABLES:
            python_args = list(args)
            if executable_name in {"py", "py.exe"} and python_args:
                if re.fullmatch(r"-\d+(?:\.\d+)?(?:-\d+)?", python_args[0]):
                    python_args.pop(0)
            if not python_args:
                raise ClusterSecurityError("Um script Python explícito é obrigatório.")
            script = python_args[0]
            normalized_script = script.replace("\\", "/")
            if (
                not script.casefold().endswith(".py")
                or script.startswith("-")
                or normalized_script.startswith("/")
                or re.match(r"^[A-Za-z]:", normalized_script)
                or ".." in normalized_script.split("/")
            ):
                raise ClusterSecurityError(
                    "Somente scripts Python relativos e explícitos são permitidos."
                )

        canonical = " ".join(tokens)
        return canonical, executable, args

    def validate_command(self, command: str) -> None:
        self._validated_command(command)

    def status(self) -> Dict[str, Any]:
        if not self.enabled:
            return {
                "enabled": False,
                "base_url": self.config.base_url,
                "message": (
                    "Cluster Aether desativado no modo soberano."
                    if POLICY.enabled
                    else "Cluster Aether desativado."
                ),
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
        if not self.enabled:
            return {"enabled": False, "workers": []}
        try:
            return {"enabled": True, "workers": self._request_json("/api/workers")}
        except ClusterError as exc:
            return {"enabled": True, "workers": [], "error": str(exc)}

    def tasks(self) -> Dict[str, Any]:
        if not self.enabled:
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
                "sovereign_mode": POLICY.enabled,
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
        self._ensure_enabled()
        canonical, _, _ = self._validated_command(command)
        payload = {
            "command": canonical,
            "required_tags": required_tags or self.config.default_tags,
            "timeout_seconds": max(30, min(int(timeout_seconds), 24 * 60 * 60)),
            "priority": max(0, min(int(priority), 100)),
        }
        return {
            "submitted": True,
            "payload": payload,
            "response": self._request_json("/api/tasks", method="POST", payload=payload),
        }

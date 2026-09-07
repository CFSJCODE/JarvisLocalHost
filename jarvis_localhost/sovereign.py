"""Sovereign-mode policy and lineage assertions.

The policy distinguishes bootstrap (where the operator may use the internet to
install source dependencies) from the Jarvis runtime.  In sovereign runtime,
semantic state may only be initialized randomly and learned from the canonical
authorized corpus.
"""

from __future__ import annotations

import ipaddress
import os
from dataclasses import asdict, dataclass
from urllib.parse import urlparse


class SovereignModeViolation(RuntimeError):
    """Raised when a requested capability violates sovereign runtime rules."""


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value, got {value!r}")


@dataclass(frozen=True)
class SovereignPolicy:
    enabled: bool = True
    allow_pretrained_ocr: bool = False
    allow_external_weights: bool = False
    allow_model_downloads: bool = False
    allow_external_apis: bool = False
    allow_runtime_egress: bool = False
    allow_system_tts: bool = False

    @classmethod
    def from_env(cls) -> "SovereignPolicy":
        enabled = _env_bool("JARVIS_SOVEREIGN_MODE", True)
        policy = cls(
            enabled=enabled,
            allow_pretrained_ocr=_env_bool("JARVIS_ALLOW_PRETRAINED_OCR", False),
            allow_external_weights=_env_bool("JARVIS_ALLOW_EXTERNAL_WEIGHTS", False),
            allow_model_downloads=_env_bool("JARVIS_ALLOW_MODEL_DOWNLOADS", False),
            allow_external_apis=_env_bool("JARVIS_ALLOW_EXTERNAL_APIS", False),
            allow_runtime_egress=_env_bool("JARVIS_ALLOW_RUNTIME_EGRESS", False),
            allow_system_tts=_env_bool("JARVIS_ALLOW_SYSTEM_TTS", False),
        )
        policy.validate()
        return policy

    def validate(self) -> None:
        if not self.enabled:
            return
        forbidden = {
            "pretrained OCR": self.allow_pretrained_ocr,
            "external weights": self.allow_external_weights,
            "model downloads": self.allow_model_downloads,
            "external APIs": self.allow_external_apis,
            "runtime egress": self.allow_runtime_egress,
        }
        active = [name for name, allowed in forbidden.items() if allowed]
        if active:
            raise SovereignModeViolation(
                "Sovereign mode cannot enable: " + ", ".join(active)
            )

    def require_text_layer(self, page: int, has_text: bool) -> None:
        if self.enabled and not has_text:
            raise SovereignModeViolation(
                f"PDF page {page} has no usable text layer. Pretrained OCR is disabled "
                "in sovereign mode."
            )

    def assert_random_initialization(self, initialized_from: str) -> None:
        if self.enabled and initialized_from != "random":
            raise SovereignModeViolation(
                f"Model initialization must be random, got {initialized_from!r}."
            )

    def assert_url_allowed(self, url: str) -> None:
        """Allow loopback/local transports while rejecting external runtime URLs."""

        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https", "ws", "wss"}:
            raise SovereignModeViolation(f"Unsupported runtime URL scheme: {parsed.scheme}")
        host = parsed.hostname
        if not host:
            raise SovereignModeViolation("Runtime URL has no hostname.")
        is_local = host.lower() == "localhost"
        if not is_local:
            try:
                is_local = ipaddress.ip_address(host).is_loopback
            except ValueError:
                is_local = False
        if self.enabled and not self.allow_runtime_egress and not is_local:
            raise SovereignModeViolation(
                f"External runtime network access is disabled: {host}"
            )

    def to_manifest(self) -> dict:
        return asdict(self)


POLICY = SovereignPolicy.from_env()

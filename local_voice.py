"""
local_voice.py - offline voice utilities for Jarvis.

The module intentionally avoids cloud speech recognition. Text-to-speech uses
pyttsx3 when available and falls back to Windows SAPI through PowerShell.
"""

from __future__ import annotations

import base64
import os
import subprocess
from dataclasses import dataclass
from typing import Dict, Iterable, Optional


TRUTHY = {"1", "true", "yes", "on", "sim"}


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in TRUTHY


@dataclass
class WakeWordState:
    active: bool
    wake_word: str
    last_intent: str


class LocalWakeGate:
    """Small text wake/sleep state machine for local voice frontends."""

    def __init__(self, wake_word: str = "jarvis"):
        self.wake_word = wake_word.lower().strip() or "jarvis"
        self.active = False
        self.sleep_terms = {"dormir", "standby", "silencio", "silenciar", "sleep"}

    def update(self, text: str) -> WakeWordState:
        normalized = text.lower()
        if self.wake_word in normalized:
            self.active = True
            return WakeWordState(True, self.wake_word, "wake")
        if any(term in normalized for term in self.sleep_terms):
            self.active = False
            return WakeWordState(False, self.wake_word, "sleep")
        return WakeWordState(self.active, self.wake_word, "pass")


class LocalVoice:
    """Offline TTS adapter; disabled by default unless configured."""

    def __init__(self, enabled: bool = False, rate: int = 165, volume: float = 1.0):
        self.enabled = enabled
        self.rate = rate
        self.volume = volume
        self.wake_gate = LocalWakeGate(os.getenv("JARVIS_WAKE_WORD", "jarvis"))

    @classmethod
    def from_env(cls) -> "LocalVoice":
        return cls(
            enabled=_env_bool("JARVIS_VOICE_ENABLED", False),
            rate=int(os.getenv("JARVIS_VOICE_RATE", "165")),
            volume=float(os.getenv("JARVIS_VOICE_VOLUME", "1.0")),
        )

    def status(self) -> Dict:
        return {
            "enabled": self.enabled,
            "wake_word": self.wake_gate.wake_word,
            "active": self.wake_gate.active,
            "mode": "offline_tts",
        }

    def update_wake_state(self, text: str) -> Dict:
        state = self.wake_gate.update(text)
        return {
            "active": state.active,
            "wake_word": state.wake_word,
            "intent": state.last_intent,
        }

    def speak(self, text: str) -> Dict:
        text = (text or "").strip()
        if not text:
            return {"spoken": False, "reason": "Texto vazio."}
        if not self.enabled:
            return {
                "spoken": False,
                "reason": "Voz local desativada. Defina JARVIS_VOICE_ENABLED=1.",
            }

        pyttsx3_result = self._speak_with_pyttsx3(text)
        if pyttsx3_result["spoken"]:
            return pyttsx3_result

        if os.name == "nt":
            sapi_result = self._speak_with_windows_sapi(text)
            if sapi_result["spoken"]:
                return sapi_result

        return {
            "spoken": False,
            "reason": pyttsx3_result.get("reason", "Nenhum sintetizador offline disponível."),
        }

    def _speak_with_pyttsx3(self, text: str) -> Dict:
        try:
            import pyttsx3

            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
            engine.setProperty("volume", self.volume)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
            return {"spoken": True, "engine": "pyttsx3"}
        except Exception as exc:
            return {"spoken": False, "engine": "pyttsx3", "reason": str(exc)}

    def _speak_with_windows_sapi(self, text: str) -> Dict:
        encoded_text = base64.b64encode(text.encode("utf-8")).decode("ascii")
        script = f"""
$bytes = [Convert]::FromBase64String('{encoded_text}')
$text = [Text.Encoding]::UTF8.GetString($bytes)
Add-Type -AssemblyName System.Speech
$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer
$speaker.Rate = 0
$speaker.Volume = {max(0, min(int(self.volume * 100), 100))}
$speaker.Speak($text)
"""
        encoded_script = base64.b64encode(script.encode("utf-16le")).decode("ascii")
        try:
            completed = subprocess.run(
                ["powershell", "-NoProfile", "-EncodedCommand", encoded_script],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if completed.returncode == 0:
                return {"spoken": True, "engine": "windows_sapi"}
            return {
                "spoken": False,
                "engine": "windows_sapi",
                "reason": completed.stderr.strip() or completed.stdout.strip(),
            }
        except Exception as exc:
            return {"spoken": False, "engine": "windows_sapi", "reason": str(exc)}

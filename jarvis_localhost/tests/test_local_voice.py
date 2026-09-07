from __future__ import annotations

import unittest
from unittest.mock import patch

from jarvis_localhost.integrations.local_voice import LocalVoice


class LocalVoicePolicyTests(unittest.TestCase):
    def test_sovereign_policy_blocks_all_tts_engines(self) -> None:
        voice = LocalVoice(enabled=True, policy_blocked=True)

        with patch.object(voice, "_speak_with_pyttsx3") as pyttsx3, patch.object(
            voice, "_speak_with_windows_sapi"
        ) as sapi:
            result = voice.speak("segredo local")

        self.assertFalse(result["spoken"])
        self.assertIn("politica soberana", result["reason"])
        self.assertTrue(voice.status()["policy_blocked"])
        self.assertEqual(voice.status()["mode"], "disabled_by_sovereign_policy")
        pyttsx3.assert_not_called()
        sapi.assert_not_called()

    def test_explicitly_allowed_voice_keeps_offline_adapter_available(self) -> None:
        voice = LocalVoice(enabled=True, policy_blocked=False)
        with patch.object(
            voice,
            "_speak_with_pyttsx3",
            return_value={"spoken": True, "engine": "test"},
        ):
            result = voice.speak("teste")

        self.assertTrue(result["spoken"])
        self.assertEqual(result["engine"], "test")


if __name__ == "__main__":
    unittest.main()

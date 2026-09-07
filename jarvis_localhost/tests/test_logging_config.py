"""F5 audit fix: jarvis_localhost.logging_config provides real structured logging."""

from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path

from jarvis_localhost.logging_config import configure_logging, get_logger


class LoggingConfigTests(unittest.TestCase):
    def test_configure_logging_is_idempotent_and_sets_level(self) -> None:
        logger = configure_logging(level="DEBUG", log_file=None)
        handler_count_first = len(logger.handlers)
        logger_again = configure_logging(level="DEBUG", log_file=None)
        self.assertIs(logger, logger_again)
        self.assertEqual(len(logger_again.handlers), handler_count_first)
        self.assertEqual(logger.level, logging.DEBUG)

    def test_configure_logging_writes_to_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "test.log"
            try:
                logger = configure_logging(level="INFO", log_file=log_path, to_stdout=False)
                logger.info("hello from audit test")
                for handler in logger.handlers:
                    handler.flush()
                self.assertTrue(log_path.exists())
                self.assertIn("hello from audit test", log_path.read_text(encoding="utf-8"))
            finally:
                configure_logging(log_file=None, to_stdout=False)

    def test_reconfiguration_releases_previous_log_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "first.log"
            logger = configure_logging(log_file=path, to_stdout=False)
            old_handler = logger.handlers[0]
            configure_logging(log_file=None, to_stdout=False)
            self.assertIsNone(old_handler.stream)
            path.unlink()


class LoggerHierarchyTests(unittest.TestCase):
    """F5 follow-up fix: get_logger(__name__) must actually reach the
    handlers configure_logging() attaches to the "jarvis" logger, for real
    module names -- not just for names that already start with "jarvis."."""

    def test_real_module_name_resolves_under_the_jarvis_namespace(self) -> None:
        self.assertEqual(
            get_logger("jarvis_localhost.core.brain").name, "jarvis.core.brain"
        )
        self.assertEqual(
            get_logger("jarvis_localhost.processing.pdf_processor").name,
            "jarvis.processing.pdf_processor",
        )
        self.assertEqual(get_logger("jarvis").name, "jarvis")
        self.assertEqual(
            get_logger("jarvis.already.nested").name, "jarvis.already.nested"
        )
        # A name that merely starts with the same six letters as "jarvis"
        # (the actual historical bug) must NOT be treated as already inside
        # the "jarvis" tree.
        self.assertEqual(get_logger("jarvisx.other").name, "jarvis.jarvisx.other")

    def test_module_logger_messages_actually_reach_the_jarvis_handlers(self) -> None:
        # Configure the real "jarvis" logger, then attach an in-memory probe
        # handler to it directly -- exactly mirroring what
        # configure_logging() sets up for the stream/file handlers.
        root = configure_logging(level="INFO", log_file=None, to_stdout=False)
        captured: list[str] = []

        class _Probe(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                captured.append(record.getMessage())

        probe = _Probe()
        root.addHandler(probe)
        try:
            module_logger = get_logger("jarvis_localhost.core.brain")
            module_logger.info("[Brain] mensagem de teste do audit")
        finally:
            root.removeHandler(probe)

        self.assertIn("[Brain] mensagem de teste do audit", captured)


if __name__ == "__main__":
    unittest.main()

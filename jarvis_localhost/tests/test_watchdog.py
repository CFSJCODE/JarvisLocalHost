"""Synthetic watchdog regressions; no real HTTP, processes or corpus writes."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

if not __package__ and Path(__file__).with_name("watchdog.py").is_file():
    spec = importlib.util.spec_from_file_location(
        "watchdog_under_test", Path(__file__).with_name("watchdog.py")
    )
    watchdog = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(watchdog)
else:
    from jarvis_localhost.tools import watchdog


def status(*, training=False, trained=False, progress=None):
    return {"is_training": training, "is_trained": trained,
            "documents": 2, "progress": progress or {}}


class WatchdogTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(watchdog, "log"))

    def temporary_corpus(self):
        root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.stack.enter_context(patch.object(watchdog, "CORPUS", root))
        self.stack.enter_context(patch.object(watchdog, "MANIFEST", root / "corpus_manifest.json"))
        return root

    def mocked_http(self, response=None):
        client = MagicMock()
        client.__enter__.return_value = client
        if response is not None:
            client.get.return_value = response
        self.stack.enter_context(patch.dict(sys.modules, {
            "httpx": SimpleNamespace(Client=Mock(return_value=client)),
        }))
        return client

    def monitor_mocks(self, states):
        mocks = {}
        for name, kwargs in {
            "port_open": {"return_value": True},
            "train_status": {"side_effect": states},
            "process_running": {"return_value": False},
            "ingest_done": {"return_value": True},
            "start_training": {"return_value": True},
            "start_server": {}, "start_ingest": {}, "find_orphans": {"return_value": []},
        }.items():
            mocks[name] = self.stack.enter_context(patch.object(watchdog, name, **kwargs))
        self.stack.enter_context(patch.object(watchdog.time, "sleep"))
        return mocks

    def test_orphan_report_preserves_every_file(self):
        root = self.temporary_corpus()
        (root / "doc_known.txt").write_bytes(b"authorized")
        (root / "doc_unknown.txt").write_bytes(b"must survive")
        (root / "unrelated.bin").write_bytes(b"other")
        (root / "doc_directory").mkdir()
        watchdog.MANIFEST.write_text(json.dumps({"documents": {
            "known": {"corpus_file": "doc_known.txt"}
        }}), encoding="utf-8")
        before = {p.name: p.read_bytes() for p in root.iterdir() if p.is_file()}
        self.assertEqual(watchdog.find_orphans(), [root / "doc_unknown.txt"])
        after = {p.name: p.read_bytes() for p in root.iterdir() if p.is_file()}
        self.assertEqual(before, after)

    def test_absent_manifest_still_reports_and_preserves_artifacts(self):
        root = self.temporary_corpus()
        artifact = root / "doc_unregistered.txt"
        artifact.write_bytes(b"retained")
        self.assertEqual(watchdog.find_orphans(), [artifact])
        self.assertEqual(artifact.read_bytes(), b"retained")

    def test_invalid_manifest_does_not_delete_anything(self):
        root = self.temporary_corpus()
        artifact = root / "doc_a.txt"
        artifact.write_bytes(b"retained")
        watchdog.MANIFEST.write_text("broken", encoding="utf-8")
        with self.assertRaises(ValueError):
            watchdog.find_orphans()
        self.assertEqual(artifact.read_bytes(), b"retained")

    def test_missing_source_directory_is_not_completed_ingestion(self):
        root = self.temporary_corpus()
        with patch.object(watchdog, "INGEST_FOLDER", root / "unavailable"):
            with self.assertRaises(FileNotFoundError):
                watchdog.ingest_done()

    def test_http_error_is_not_a_training_status(self):
        response = Mock()
        response.raise_for_status.side_effect = RuntimeError("503 unavailable")
        self.mocked_http(response)
        with self.assertRaisesRegex(RuntimeError, "503"):
            watchdog.train_status()
        response.json.assert_not_called()

    def test_status_requires_real_boolean_state(self):
        for payload in ({"detail": "unavailable"}, {"is_training": "false", "is_trained": True}, []):
            with self.subTest(payload=payload):
                response = Mock()
                response.json.return_value = payload
                self.mocked_http(response)
                with self.assertRaises(ValueError):
                    watchdog.train_status()

    def test_session_http_error_never_posts_training(self):
        session = Mock()
        session.raise_for_status.side_effect = RuntimeError("403 forbidden")
        client = self.mocked_http(session)
        with self.assertRaises(RuntimeError):
            watchdog.start_training()
        client.post.assert_not_called()

    def test_accepted_training_requires_expected_server_payload(self):
        session = Mock()
        session.json.return_value = {"csrf_token": "synthetic-token"}
        client = self.mocked_http(session)
        for outcome, accepted in (("started", True), ("already_training", True), ("unknown", False)):
            client.post.return_value.json.return_value = {"status": outcome}
            self.assertEqual(watchdog.start_training(), accepted)
        client.post.assert_called_with(
            "/api/train/start", headers={"X-Jarvis-CSRF": "synthetic-token"}
        )
        client.post.return_value.raise_for_status.assert_called()

    def test_accepted_post_is_monitored_until_actual_completion(self):
        mocks = self.monitor_mocks([
            status(), status(training=True), status(training=True), status(trained=True),
        ])
        self.assertEqual(watchdog.monitor(), 0)
        self.assertEqual(mocks["train_status"].call_count, 4)
        mocks["start_training"].assert_called_once()
        mocks["start_server"].assert_not_called()
        mocks["start_ingest"].assert_not_called()

    def test_old_trained_flag_does_not_finish_current_active_training(self):
        mocks = self.monitor_mocks([
            status(training=True, trained=True), status(trained=True),
        ])
        self.assertEqual(watchdog.monitor(), 0)
        self.assertEqual(mocks["train_status"].call_count, 2)
        mocks["start_training"].assert_not_called()

    def test_training_only_starts_current_corpus_without_ingestion_checks(self):
        mocks = self.monitor_mocks([
            status(), status(training=True), status(trained=True),
        ])
        self.assertEqual(watchdog.monitor(training_only=True), 0)
        mocks["start_training"].assert_called_once()
        mocks["process_running"].assert_not_called()
        mocks["ingest_done"].assert_not_called()
        mocks["start_ingest"].assert_not_called()

    def test_training_only_recovers_server_without_ingesting_after_drop(self):
        mocks = self.monitor_mocks([
            status(training=True), status(), status(training=True), status(trained=True),
        ])
        mocks["port_open"].side_effect = [True, False, True, True, True]
        self.assertEqual(watchdog.monitor(training_only=True), 0)
        mocks["start_server"].assert_called_once()
        mocks["start_training"].assert_called_once()
        mocks["process_running"].assert_called_once_with("jarvis_localhost.server.app")
        mocks["ingest_done"].assert_not_called()
        mocks["start_ingest"].assert_not_called()

    def test_temporary_http_failure_retries_observation_without_duplicate_start(self):
        mocks = self.monitor_mocks([
            status(), RuntimeError("synthetic HTTP outage"),
            status(training=True), status(trained=True),
        ])
        self.assertEqual(watchdog.monitor(training_only=True), 0)
        mocks["start_training"].assert_called_once()
        mocks["start_ingest"].assert_not_called()

    def test_training_error_is_failure_even_when_old_model_remains(self):
        mocks = self.monitor_mocks([status(trained=True, progress={"error": "synthetic failure"})])
        self.assertEqual(watchdog.monitor(), 1)
        mocks["start_training"].assert_not_called()

    def test_stopped_training_without_pipeline_does_not_restart_forever(self):
        mocks = self.monitor_mocks([status(), status()])
        self.assertEqual(watchdog.monitor(), 1)
        mocks["start_training"].assert_called_once()

    def test_orphan_artifacts_block_automatic_restart(self):
        mocks = self.monitor_mocks([])
        mocks["port_open"].return_value = False
        mocks["find_orphans"].return_value = [Path("synthetic-artifact")]
        self.assertEqual(watchdog.monitor(), 1)
        mocks["start_server"].assert_not_called()
        mocks["start_ingest"].assert_not_called()
        mocks["start_training"].assert_not_called()

    def test_process_inspection_failure_does_not_mean_process_absent(self):
        module = SimpleNamespace(process_iter=Mock(return_value=[
            SimpleNamespace(info={"name": "python.exe", "cmdline": None}),
        ]))
        with patch.dict(sys.modules, {"psutil": module}):
            with self.assertRaises(RuntimeError):
                watchdog.process_running("jarvis_localhost.server.app")

    def test_process_matching_uses_exact_command_argument(self):
        module = SimpleNamespace(process_iter=Mock(return_value=[
            SimpleNamespace(info={"name": "pythonw.exe", "cmdline": [
                "pythonw.exe", "-m", "jarvis_localhost.server.app",
            ]}),
        ]))
        with patch.dict(sys.modules, {"psutil": module}):
            self.assertTrue(watchdog.process_running("jarvis_localhost.server.app"))
            self.assertFalse(watchdog.process_running("server.app"))

    def test_launch_closes_parent_log_handle(self):
        root = self.temporary_corpus()
        with patch.object(watchdog.subprocess, "Popen") as popen:
            watchdog._launch(["synthetic-python"], root / "child.log")
        self.assertTrue(popen.call_args.kwargs["stdout"].closed)
        if watchdog.os.name == "nt":
            self.assertTrue(
                popen.call_args.kwargs["creationflags"] & watchdog.subprocess.CREATE_NO_WINDOW
            )

    def test_second_watchdog_does_not_monitor_or_spawn(self):
        lock = Mock()
        lock.acquire.return_value = False
        module = SimpleNamespace(InterProcessFileLock=Mock(return_value=lock))
        with patch.dict(sys.modules, {
            "jarvis_localhost.integrations.process_lock": module,
        }), patch.object(watchdog, "monitor") as monitor:
            self.assertEqual(watchdog.main([]), 2)
        monitor.assert_not_called()

    def test_monitor_exception_releases_singleton_lock(self):
        lock = Mock()
        lock.acquire.return_value = True
        module = SimpleNamespace(InterProcessFileLock=Mock(return_value=lock))
        with patch.dict(sys.modules, {
            "jarvis_localhost.integrations.process_lock": module,
        }), patch.object(watchdog, "monitor", side_effect=RuntimeError("synthetic failure")):
            with self.assertRaises(RuntimeError):
                watchdog.main([])
        lock.release.assert_called_once()

    def test_training_only_cli_reaches_monitor(self):
        lock = Mock()
        lock.acquire.return_value = True
        module = SimpleNamespace(InterProcessFileLock=Mock(return_value=lock))
        with patch.dict(sys.modules, {
            "jarvis_localhost.integrations.process_lock": module,
        }), patch.object(watchdog, "monitor", return_value=0) as monitor:
            self.assertEqual(watchdog.main(["--training-only"]), 0)
        monitor.assert_called_once_with(training_only=True)
        lock.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()

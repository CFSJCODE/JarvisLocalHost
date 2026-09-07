from __future__ import annotations

import importlib
import importlib.util
import io
import re
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from jarvis_localhost.integrations import cluster_client
from jarvis_localhost.sovereign import SovereignPolicy


REPO_ROOT = Path(__file__).resolve().parents[2]
APP_PATH = REPO_ROOT / "jarvis_localhost" / "server" / "app.py"
HUD_PATH = REPO_ROOT / "jarvis_localhost" / "web" / "static" / "index.html"
CLUSTER_PATH = REPO_ROOT / "jarvis_localhost" / "integrations" / "cluster_client.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


WEB_TEST_DEPENDENCIES = ("fastapi", "uvicorn", "pydantic", "multipart", "httpx")
FASTAPI_AVAILABLE = all(
    importlib.util.find_spec(dependency) is not None
    for dependency in WEB_TEST_DEPENDENCIES
)


class StaticLocalApiSecurityTests(unittest.TestCase):
    def test_hud_has_no_html_assignment_sink(self) -> None:
        source = _read(HUD_PATH)
        self.assertNotRegex(source, r"\.\s*(?:innerHTML|outerHTML)\s*=")
        self.assertNotIn("insertAdjacentHTML", source)
        self.assertNotIn("document.write", source)
        self.assertNotRegex(source, r"\beval\s*\(")
        for safe_renderer in (
            "function addMessage",
            "function renderDocList",
            "function addProjectTag",
            "function addLog",
            "function renderInsightsList",
            "function appendTrainLog",
        ):
            self.assertIn(safe_renderer, source)
        self.assertIn("textContent", source)
        self.assertIn("replaceChildren", source)

    def test_hud_uses_session_csrf_for_every_post(self) -> None:
        source = _read(HUD_PATH)
        self.assertIn("headers.set('X-Jarvis-CSRF', token)", source)
        self.assertIn("credentials: 'same-origin'", source)
        direct_post_fetches = re.findall(
            r"(?<!secure)fetch\s*\([^;]{0,500}?method\s*:\s*['\"]POST['\"]",
            source,
            flags=re.DOTALL,
        )
        self.assertEqual(direct_post_fetches, [])
        self.assertGreaterEqual(source.count("secureFetch(API + '/api/"), 4)

    def test_server_declares_loopback_origin_and_csrf_guards(self) -> None:
        source = _read(APP_PATH)
        self.assertIn("async def local_request_guard", source)
        self.assertIn("_origin_matches_host", source)
        self.assertIn("hmac.compare_digest", source)
        self.assertIn("SESSION_COOKIE", source)
        self.assertIn('host="127.0.0.1"', source)
        self.assertIn("reload=False", source)
        self.assertNotIn("reload=True", source)
        self.assertNotIn("detail=str(", source)

    def test_pdf_pipeline_is_bounded_and_cleans_partial_files(self) -> None:
        source = _read(APP_PATH)
        self.assertIn('first_chunk.startswith(b"%PDF-")', source)
        self.assertIn("source.read(PDF_COPY_CHUNK_BYTES)", source)
        self.assertIn("total > max_bytes", source)
        self.assertIn("PDF_UPLOAD_SEMAPHORE", source)
        self.assertIn("destination.unlink(missing_ok=True)", source)
        self.assertIn("_sanitize_pdf_filename", source)
        self.assertIn("assert_runtime_path", source)

    def test_pydantic_collections_use_factories(self) -> None:
        source = _read(APP_PATH)
        self.assertGreaterEqual(source.count("Field(default_factory=list)"), 3)
        self.assertNotRegex(source, r"(?:sources|tags|required_tags):[^\n]+ = \[\]")

    def test_cluster_source_has_no_dynamic_python_allowlist_bypass(self) -> None:
        source = _read(CLUSTER_PATH)
        self.assertIn("EVAL_FLAGS", source)
        self.assertIn("PYTHON_EXECUTABLES", source)
        self.assertIn("POLICY.enabled", source)
        self.assertIn("Somente scripts Python relativos", source)


class ClusterCommandSecurityTests(unittest.TestCase):
    def _client(self, *, allowed: list[str] | None = None) -> cluster_client.ClusterClient:
        return cluster_client.ClusterClient(
            cluster_client.ClusterConfig(
                enabled=True,
                base_url="http://127.0.0.1:8080",
                allowed_prefixes=allowed or ["python"],
            )
        )

    def test_sovereign_mode_disables_submission_before_network(self) -> None:
        with patch.object(cluster_client, "POLICY", SovereignPolicy(enabled=True)):
            client = self._client()
            self.assertFalse(client.enabled)
            with patch.object(cluster_client, "urlopen") as opener:
                with self.assertRaises(cluster_client.ClusterDisabled):
                    client.submit_task("python jobs/index.py")
                opener.assert_not_called()

    def test_python_requires_an_explicit_relative_script(self) -> None:
        with patch.object(cluster_client, "POLICY", SovereignPolicy(enabled=False)):
            client = self._client(allowed=["python", "python3"])
            client.validate_command("python jobs/index.py --mode=scan input.pdf")
            client.validate_command("python3 jobs/index.py")
            for command in (
                "python -c print",
                "python -Ic print.py",
                "python -m http.server",
                "python -",
                "python ../escape.py",
                "python C:\\temp\\escape.py",
                "python /tmp/escape.py",
            ):
                with self.subTest(command=command):
                    with self.assertRaises(cluster_client.ClusterSecurityError):
                        client.validate_command(command)

    def test_shell_syntax_and_shell_executables_are_rejected(self) -> None:
        with patch.object(cluster_client, "POLICY", SovereignPolicy(enabled=False)):
            client = self._client()
            for command in (
                "python jobs/index.py;whoami",
                "python jobs/index.py && whoami",
                "python jobs/index.py | whoami",
                "python jobs/index.py > output.txt",
                "python jobs/index.py $(whoami)",
                "python jobs/index.py `whoami`",
                'python "jobs/my task.py"',
            ):
                with self.subTest(command=command):
                    with self.assertRaises(cluster_client.ClusterSecurityError):
                        client.validate_command(command)

            shell_client = self._client(allowed=["cmd", "powershell", "bash"])
            for command in ("cmd /c whoami", "powershell -c whoami", "bash run.py"):
                with self.subTest(command=command):
                    with self.assertRaises(cluster_client.ClusterSecurityError):
                        shell_client.validate_command(command)

            eval_client = self._client(allowed=["node"])
            with self.assertRaises(cluster_client.ClusterSecurityError):
                eval_client.validate_command("node --eval=process.exit")

    def test_cluster_url_rejects_embedded_credentials(self) -> None:
        with patch.object(cluster_client, "POLICY", SovereignPolicy(enabled=False)):
            with self.assertRaises(cluster_client.ClusterSecurityError):
                cluster_client.ClusterClient(
                    cluster_client.ClusterConfig(
                        enabled=True,
                        base_url="http://secret@example@127.0.0.1:8080",
                    )
                )


@unittest.skipUnless(
    FASTAPI_AVAILABLE,
    "FastAPI/TestClient/python-multipart dependencies are not installed",
)
class FastApiLocalSecurityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.brain_module_name = "jarvis_localhost.core.brain"
        cls.app_module_name = "jarvis_localhost.server.app"
        cls.previous_brain_module = sys.modules.get(cls.brain_module_name)
        cls.previous_app_module = sys.modules.get(cls.app_module_name)

        brain_module = types.ModuleType(cls.brain_module_name)

        class StubBrain:
            def __init__(self) -> None:
                self.is_training = False
                self.is_trained = False
                self.projects: list[dict] = []
                self.store: list[object] = []
                self.tokenizer = None

            async def chat(self, message: str) -> dict:
                return {"answer": f"eco:{message}", "sources": []}

            def get_metrics(self) -> dict:
                return {"cpu": {}, "memory": {}, "disk": {}, "network": {}}

            def save_metrics_snapshot(self, metrics: dict) -> None:
                return None

            def set_curiosity_callback(self, callback: object) -> None:
                self.curiosity_callback = callback

            def shutdown(self) -> None:
                return None

            def process_pdf(self, path: str) -> dict:
                return {"pages": 1, "words": 2, "tables": 0, "images": 0}

        brain_module.JarvisBrain = StubBrain
        sys.modules[cls.brain_module_name] = brain_module
        sys.modules.pop(cls.app_module_name, None)
        cls.app_module = importlib.import_module(cls.app_module_name)

        from fastapi.testclient import TestClient

        cls.TestClient = TestClient

    @classmethod
    def tearDownClass(cls) -> None:
        sys.modules.pop(cls.app_module_name, None)
        if cls.previous_app_module is not None:
            sys.modules[cls.app_module_name] = cls.previous_app_module
        if cls.previous_brain_module is None:
            sys.modules.pop(cls.brain_module_name, None)
        else:
            sys.modules[cls.brain_module_name] = cls.previous_brain_module

    def _session(self, client: object) -> str:
        root = client.get("/")
        self.assertEqual(root.status_code, 200)
        response = client.get("/api/session")
        self.assertEqual(response.status_code, 200)
        return response.json()["csrf_token"]

    @contextmanager
    def _client(self):
        """Close Starlette 0.37's memory-stream endpoints after lifespan exit."""

        client = self.TestClient(
            self.app_module.app,
            base_url="http://127.0.0.1:8000",
        )
        try:
            with client:
                yield client
        finally:
            # TestClient 0.37 closes its portal but retains both stapled AnyIO
            # streams. AnyIO 4.14 correctly reports those references as leaked.
            # Closing the synchronous memory endpoints here keeps the test
            # harness deterministic without changing the production runtime.
            for stream_name in ("stream_send", "stream_receive"):
                stream = getattr(client, stream_name, None)
                if stream is None:
                    continue
                for endpoint_name in ("send_stream", "receive_stream"):
                    endpoint = getattr(stream, endpoint_name, None)
                    close = getattr(endpoint, "close", None)
                    if close is not None:
                        close()

    def test_host_origin_session_and_csrf_are_enforced(self) -> None:
        with self._client() as client:
            token = self._session(client)
            payload = {"message": "teste"}

            self.assertEqual(client.post("/api/chat", json=payload).status_code, 403)
            self.assertEqual(
                client.post(
                    "/api/chat",
                    json=payload,
                    headers={"Origin": "http://127.0.0.1:8000"},
                ).status_code,
                403,
            )
            accepted = client.post(
                "/api/chat",
                json=payload,
                headers={
                    "Origin": "http://127.0.0.1:8000",
                    "X-Jarvis-CSRF": token,
                },
            )
            self.assertEqual(accepted.status_code, 200)
            self.assertEqual(accepted.json()["answer"], "eco:teste")

            self.assertEqual(
                client.get("/api/metrics", headers={"Host": "evil.example"}).status_code,
                400,
            )
            self.assertEqual(
                client.get("/api/metrics", headers={"Host": "["}).status_code,
                400,
            )
            self.assertEqual(
                client.get(
                    "/api/metrics",
                    headers={"Origin": "https://evil.example"},
                ).status_code,
                403,
            )

    def test_websocket_requires_same_origin_session(self) -> None:
        from starlette.websockets import WebSocketDisconnect

        with self._client() as client:
            self._session(client)
            session_cookie = client.cookies.get(self.app_module.SESSION_COOKIE)
            websocket_headers = {
                "Host": "127.0.0.1:8000",
                "Origin": "http://127.0.0.1:8000",
                "Cookie": f"{self.app_module.SESSION_COOKIE}={session_cookie}",
            }
            with client.websocket_connect(
                "/ws", headers=websocket_headers
            ) as websocket:
                self.assertEqual(websocket.receive_json()["type"], "metrics")

            with self.assertRaises(WebSocketDisconnect):
                with client.websocket_connect(
                    "/ws",
                    headers={**websocket_headers, "Origin": "https://evil.example"},
                ):
                    pass

    def test_pdf_helpers_validate_magic_size_name_and_cleanup(self) -> None:
        module = self.app_module
        self.assertEqual(module._sanitize_pdf_filename("../../relatório?.PDF"), "relatório.pdf")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid = root / "valid.pdf"
            copied = module._copy_validated_pdf(
                io.BytesIO(b"%PDF-1.7\nbody\n%%EOF"), valid, 1024
            )
            self.assertEqual(copied, valid.stat().st_size)

            bad_magic = root / "bad-magic.pdf"
            with self.assertRaises(module.PDFUploadRejected):
                module._copy_validated_pdf(io.BytesIO(b"not a pdf"), bad_magic, 1024)
            self.assertFalse(bad_magic.exists())

            oversized = root / "oversized.pdf"
            with self.assertRaises(module.PDFUploadRejected) as raised:
                module._copy_validated_pdf(
                    io.BytesIO(b"%PDF-" + b"x" * 64), oversized, 16
                )
            self.assertEqual(raised.exception.status_code, 413)
            self.assertFalse(oversized.exists())

    def test_pdf_route_never_keeps_a_rejected_upload(self) -> None:
        module = self.app_module
        with self._client() as client, tempfile.TemporaryDirectory() as directory:
            token = self._session(client)
            headers = {
                "Origin": "http://127.0.0.1:8000",
                "X-Jarvis-CSRF": token,
            }
            temporary_uploads = Path(directory)
            with (
                patch.object(module, "UPLOADS_ROOT", temporary_uploads),
                patch.object(module, "assert_runtime_path", side_effect=lambda path: path.resolve()),
                patch.object(module, "MAX_PDF_BYTES", 16),
            ):
                response = client.post(
                    "/api/pdf/upload",
                    headers=headers,
                    files={"file": ("../../escape.pdf", b"%PDF-" + b"x" * 64, "application/pdf")},
                )
            self.assertEqual(response.status_code, 413)
            self.assertEqual(list(temporary_uploads.iterdir()), [])

            with (
                patch.object(module, "UPLOADS_ROOT", temporary_uploads),
                patch.object(module, "assert_runtime_path", side_effect=lambda path: path.resolve()),
                patch.object(module, "MAX_PDF_BYTES", 1024),
            ):
                accepted = client.post(
                    "/api/pdf/upload",
                    headers=headers,
                    files={"file": ("../../seguro?.PDF", b"%PDF-1.7\n%%EOF", "application/pdf")},
                )
            self.assertEqual(accepted.status_code, 200)
            saved_files = list(temporary_uploads.iterdir())
            self.assertEqual(len(saved_files), 1)
            self.assertEqual(saved_files[0].parent, temporary_uploads)
            self.assertTrue(saved_files[0].name.endswith("_seguro.pdf"))


if __name__ == "__main__":
    unittest.main()

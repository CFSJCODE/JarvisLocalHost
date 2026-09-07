"""Integration tests for the dependency-free MCP team bus server."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SERVER_PATH = (
    REPOSITORY_ROOT / "jarvis_localhost" / "integrations" / "team_bus_server.py"
)


class MCPProcessClient:
    def __init__(self, database_path: Path):
        environment = os.environ.copy()
        environment["JARVIS_TEAM_BUS_DB"] = str(database_path)
        environment["PYTHONIOENCODING"] = "utf-8"
        self.process = subprocess.Popen(
            [sys.executable, "-u", str(SERVER_PATH)],
            cwd=str(REPOSITORY_ROOT),
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        self._request_id = 0
        self._lock = threading.Lock()
        initialized = self.request(
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "stdlib-test-client", "version": "1.0"},
            },
        )
        if initialized.get("protocolVersion") != "2025-06-18":
            raise AssertionError(f"Initialization failed: {initialized!r}")
        self.notify("notifications/initialized", {})

    def _write(self, payload: Dict[str, Any]) -> None:
        if self.process.stdin is None:
            raise AssertionError("MCP stdin is closed")
        self.process.stdin.write(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
        )
        self.process.stdin.flush()

    def request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        with self._lock:
            self._request_id += 1
            request_id = self._request_id
            self._write(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": method,
                    "params": params or {},
                }
            )
            if self.process.stdout is None:
                raise AssertionError("MCP stdout is closed")
            line = self.process.stdout.readline()
            if not line:
                stderr = ""
                if self.process.stderr is not None:
                    stderr = self.process.stderr.read()
                raise AssertionError(
                    f"MCP process exited without response; rc={self.process.poll()}, "
                    f"stderr={stderr!r}"
                )
            response = json.loads(line)
            if response.get("id") != request_id:
                raise AssertionError(
                    f"Unexpected response id: expected {request_id}, got {response!r}"
                )
            if "error" in response:
                raise AssertionError(f"JSON-RPC error: {response['error']!r}")
            return response["result"]

    def notify(self, method: str, params: Optional[Dict[str, Any]] = None) -> None:
        with self._lock:
            self._write(
                {"jsonrpc": "2.0", "method": method, "params": params or {}}
            )

    def tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.request(
            "tools/call", {"name": name, "arguments": arguments or {}}
        )

    def close(self) -> Dict[str, Any]:
        if self.process.stdin is not None and not self.process.stdin.closed:
            self.process.stdin.close()
        try:
            return_code = self.process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            self.process.kill()
            return_code = self.process.wait(timeout=5)
        remaining_stdout = self.process.stdout.read() if self.process.stdout else ""
        stderr = self.process.stderr.read() if self.process.stderr else ""
        if self.process.stdout is not None:
            self.process.stdout.close()
        if self.process.stderr is not None:
            self.process.stderr.close()
        return {
            "return_code": return_code,
            "remaining_stdout": remaining_stdout,
            "stderr": stderr,
        }


class TeamBusServerIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temp_directory.name) / "shared-team-bus.sqlite"
        self.client_a = MCPProcessClient(self.database_path)
        self.client_b = MCPProcessClient(self.database_path)

    def tearDown(self) -> None:
        diagnostics = []
        for client in (self.client_a, self.client_b):
            diagnostics.append(client.close())
        self.temp_directory.cleanup()
        for diagnostic in diagnostics:
            self.assertEqual(diagnostic["return_code"], 0, diagnostic)
            self.assertEqual(
                diagnostic["remaining_stdout"],
                "",
                "stdout must contain JSON-RPC responses only",
            )
            self.assertEqual(diagnostic["stderr"], "", diagnostic)

    def register_pair(self) -> None:
        first = self.client_a.tool(
            "register_agent",
            {
                "name": "codex",
                "role": "implementer",
                "capabilities": ["code", "review"],
                "idempotency_key": "register-codex",
            },
        )
        second = self.client_b.tool(
            "register_agent",
            {
                "name": "antigravity",
                "role": "orchestrator",
                "capabilities": ["browser", "qa"],
                "idempotency_key": "register-antigravity",
            },
        )
        self.assertFalse(first["isError"], first)
        self.assertFalse(second["isError"], second)

    @staticmethod
    def structured(tool_result: Dict[str, Any]) -> Dict[str, Any]:
        return tool_result["structuredContent"]

    def test_protocol_bidirectional_messages_tasks_idempotency_and_concurrency(self) -> None:
        listed = self.client_a.request("tools/list")
        names = {tool["name"] for tool in listed["tools"]}
        self.assertEqual(
            names,
            {
                "ping",
                "register_agent",
                "post_message",
                "fetch_messages",
                "create_task",
                "claim_task",
                "update_task",
                "list_tasks",
                "get_audit_log",
                "execute_command",
            },
        )
        self.assertEqual(self.client_a.request("ping"), {})
        health = self.client_b.tool("ping")
        self.assertFalse(health["isError"])
        self.assertEqual(self.structured(health)["journal_mode"], "wal")

        self.register_pair()

        a_to_b_args = {
            "from_agent": "codex",
            "to_agent": "antigravity",
            "content": {"text": "implementation ready", "sequence": 1},
            "correlation_id": "round-trip-1",
            "idempotency_key": "message-a-to-b",
        }
        posted_once = self.client_a.tool("post_message", a_to_b_args)
        posted_replayed = self.client_b.tool("post_message", a_to_b_args)
        self.assertFalse(posted_once["isError"], posted_once)
        self.assertEqual(
            self.structured(posted_once)["message"]["id"],
            self.structured(posted_replayed)["message"]["id"],
        )

        b_to_a = self.client_b.tool(
            "post_message",
            {
                "from_agent": "antigravity",
                "to_agent": "codex",
                "content": {"text": "QA accepted", "sequence": 2},
                "correlation_id": "round-trip-1",
                "idempotency_key": "message-b-to-a",
            },
        )
        self.assertFalse(b_to_a["isError"], b_to_a)

        fetched_by_b = self.structured(
            self.client_b.tool("fetch_messages", {"agent": "antigravity"})
        )
        fetched_by_a = self.structured(
            self.client_a.tool("fetch_messages", {"agent": "codex"})
        )
        self.assertEqual(fetched_by_b["messages"][0]["from_agent"], "codex")
        self.assertEqual(fetched_by_a["messages"][0]["from_agent"], "antigravity")

        task_a_args = {
            "created_by": "codex",
            "title": "Run browser QA",
            "payload": {"suite": "smoke"},
            "priority": 10,
            "idempotency_key": "task-from-codex",
        }
        task_a_once = self.client_a.tool("create_task", task_a_args)
        task_a_replayed = self.client_b.tool("create_task", task_a_args)
        task_a = self.structured(task_a_once)["task"]
        self.assertEqual(
            task_a["id"], self.structured(task_a_replayed)["task"]["id"]
        )
        claimed_by_b = self.client_b.tool(
            "claim_task",
            {
                "task_id": task_a["id"],
                "agent": "antigravity",
                "idempotency_key": "claim-a-by-b",
            },
        )
        claimed_task_a = self.structured(claimed_by_b)["task"]
        self.assertEqual(claimed_task_a["assigned_to"], "antigravity")
        completed_by_b = self.client_a.tool(
            "update_task",
            {
                "task_id": task_a["id"],
                "actor": "antigravity",
                "status": "completed",
                "result": {"passed": True},
                "expected_version": claimed_task_a["version"],
                "idempotency_key": "complete-a-by-b",
            },
        )
        self.assertEqual(
            self.structured(completed_by_b)["task"]["status"], "completed"
        )

        task_b = self.structured(
            self.client_b.tool(
                "create_task",
                {
                    "created_by": "antigravity",
                    "title": "Review QA findings",
                    "idempotency_key": "task-from-antigravity",
                },
            )
        )["task"]
        claimed_by_a = self.client_a.tool(
            "claim_task",
            {
                "task_id": task_b["id"],
                "agent": "codex",
                "idempotency_key": "claim-b-by-a",
            },
        )
        self.assertEqual(
            self.structured(claimed_by_a)["task"]["assigned_to"], "codex"
        )

        contested = self.structured(
            self.client_a.tool(
                "create_task",
                {
                    "created_by": "codex",
                    "title": "Exactly one claimant",
                    "idempotency_key": "contested-task",
                },
            )
        )["task"]

        barrier = threading.Barrier(2)

        def claim(client: MCPProcessClient, agent: str) -> Dict[str, Any]:
            barrier.wait(timeout=5)
            return client.tool(
                "claim_task",
                {
                    "task_id": contested["id"],
                    "agent": agent,
                    "idempotency_key": f"contested-{agent}",
                },
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(
                executor.map(
                    lambda item: claim(*item),
                    [(self.client_a, "codex"), (self.client_b, "antigravity")],
                )
            )
        successes = [result for result in results if not result["isError"]]
        conflicts = [result for result in results if result["isError"]]
        self.assertEqual(len(successes), 1, results)
        self.assertEqual(len(conflicts), 1, results)
        self.assertEqual(
            conflicts[0]["structuredContent"]["error"]["type"], "ConflictError"
        )

        visible_from_b = self.structured(self.client_b.tool("list_tasks", {}))["tasks"]
        by_id = {task["id"]: task for task in visible_from_b}
        self.assertEqual(by_id[task_a["id"]]["status"], "completed")
        self.assertEqual(by_id[task_b["id"]]["assigned_to"], "codex")
        self.assertIn(by_id[contested["id"]]["assigned_to"], {"codex", "antigravity"})

        mismatch = self.client_a.tool(
            "post_message",
            dict(a_to_b_args, content={"text": "different payload"}),
        )
        self.assertTrue(mismatch["isError"], mismatch)
        self.assertEqual(
            mismatch["structuredContent"]["error"]["type"], "ConflictError"
        )

        with sqlite3.connect(str(self.database_path)) as connection:
            self.assertEqual(
                connection.execute("PRAGMA journal_mode").fetchone()[0].lower(), "wal"
            )

    def test_command_execution_is_capped_idempotent_timed_and_audited(self) -> None:
        self.register_pair()
        work_directory = Path(self.temp_directory.name) / "command-work"
        work_directory.mkdir()
        side_effect_script = (
            "from pathlib import Path; "
            "p=Path('count.txt'); "
            "p.write_text((p.read_text(encoding='utf-8') if p.exists() else '')+'x', "
            "encoding='utf-8'); "
            "print('0123456789ABCDEFGHIJ')"
        )
        command_args = {
            "actor": "codex",
            "command": [sys.executable, "-c", side_effect_script],
            "cwd": str(work_directory),
            "timeout_seconds": 10,
            "max_output_chars": 8,
            "idempotency_key": "execute-once",
        }
        first = self.client_a.tool("execute_command", command_args)
        second = self.client_b.tool("execute_command", command_args)
        self.assertFalse(first["isError"], first)
        self.assertFalse(second["isError"], second)
        first_result = self.structured(first)
        second_result = self.structured(second)
        self.assertTrue(first_result["ok"], first_result)
        self.assertEqual(first_result, second_result)
        self.assertEqual(first_result["stdout"], "01234567")
        self.assertTrue(first_result["stdout_truncated"])
        self.assertEqual(
            (work_directory / "count.txt").read_text(encoding="utf-8"), "x"
        )

        mismatch = self.client_b.tool(
            "execute_command", dict(command_args, command=[sys.executable, "-c", "print(2)"])
        )
        self.assertTrue(mismatch["isError"], mismatch)
        self.assertEqual(
            mismatch["structuredContent"]["error"]["type"], "ConflictError"
        )

        timeout_result = self.client_b.tool(
            "execute_command",
            {
                "actor": "antigravity",
                "command": [
                    sys.executable,
                    "-c",
                    "import time; print('started', flush=True); time.sleep(5)",
                ],
                "cwd": str(work_directory),
                "timeout_seconds": 0.2,
                "max_output_chars": 100,
                "idempotency_key": "timeout-command",
            },
        )
        self.assertFalse(timeout_result["isError"], timeout_result)
        timeout_payload = self.structured(timeout_result)
        self.assertFalse(timeout_payload["ok"])
        self.assertTrue(timeout_payload["timed_out"])
        self.assertIn("started", timeout_payload["stdout"])

        audit = self.structured(
            self.client_a.tool(
                "get_audit_log", {"action": "execute_command", "limit": 20}
            )
        )["entries"]
        self.assertEqual(len(audit), 2, audit)
        by_actor = {entry["actor"]: entry for entry in audit}
        self.assertTrue(by_actor["codex"]["success"])
        self.assertFalse(by_actor["antigravity"]["success"])
        self.assertEqual(len(by_actor["codex"]["details"]["command_sha256"]), 64)
        self.assertNotIn("stdout", by_actor["codex"]["details"])
        self.assertNotIn("0123456789", json.dumps(audit))

        invalid = self.client_a.tool(
            "register_agent", {"name": "extra", "unexpected": True}
        )
        self.assertTrue(invalid["isError"], invalid)
        self.assertEqual(
            invalid["structuredContent"]["error"]["type"], "ValidationError"
        )


if __name__ == "__main__":
    unittest.main()

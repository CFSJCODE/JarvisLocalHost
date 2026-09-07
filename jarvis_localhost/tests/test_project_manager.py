"""Integration tests for the safe project scaffold generator."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path, PurePosixPath
from unittest.mock import patch

from jarvis_localhost.projects.project_manager import ProjectManager


PROJECT_IMPORTS = {
    "Software / IA": ("main", "src.data", "src.model", "src.train"),
    "IoT / Nuvem": ("backend.server",),
    "Pesquisa": ("experiments.config", "experiments.run"),
    "Infraestrutura": ("backend.server",),
}


class ProjectManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.output_dir = Path(self.temporary_directory.name) / "archives"
        self.manager = ProjectManager(self.output_dir)

    def test_every_project_type_has_safe_complete_structure_and_valid_python(self) -> None:
        for index, project_type in enumerate((*self.manager.TEMPLATES, "Tipo desconhecido")):
            with self.subTest(project_type=project_type):
                archive_path = Path(
                    self.manager.generate(
                        {
                            "id": f"PROJECT-{index}",
                            "name": f"Projeto {project_type}",
                            "type": project_type,
                            "priority": "BETA",
                            "description": "Scaffold de validação.",
                            "created_at": 1_700_000_000,
                        }
                    )
                )
                self.assertEqual(archive_path.parent, self.output_dir.resolve())
                self.assertTrue(archive_path.is_file())

                extracted_root = self._validate_and_extract(archive_path, index)
                expected_template = self.manager.TEMPLATES.get(
                    project_type, self.manager.GENERIC_TEMPLATE
                )
                expected_relative_paths = {path for path, _key in expected_template}
                actual_relative_paths = {
                    path.relative_to(extracted_root).as_posix()
                    for path in extracted_root.rglob("*")
                    if path.is_file()
                }
                self.assertEqual(actual_relative_paths, expected_relative_paths)
                self._compile_generated_python(extracted_root)
                self._import_generated_modules(
                    extracted_root, PROJECT_IMPORTS.get(project_type, ())
                )

                if project_type == "Software / IA":
                    completed = subprocess.run(
                        [sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests", "-v"],
                        cwd=extracted_root,
                        capture_output=True,
                        text=True,
                        timeout=30,
                        check=False,
                    )
                    self.assertEqual(
                        completed.returncode,
                        0,
                        msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
                    )

    def test_untrusted_names_are_slugged_and_archive_members_cannot_traverse(self) -> None:
        archive_path = Path(
            self.manager.generate(
                {
                    "id": "SAFE-ID",
                    "name": "../../CON/<script>alert(1)</script>",
                    "type": "<img src=x onerror=alert(1)>",
                    "priority": "<script>alert(2)</script>",
                    "description": "<script>alert(3)</script>",
                }
            )
        )
        self.assertRegex(archive_path.name, r"^[A-Za-z0-9_-]+_SAFE-ID\.zip$")
        self.assertNotIn("..", archive_path.name)

        with zipfile.ZipFile(archive_path) as archive:
            self.assertIsNone(archive.testzip())
            members = archive.namelist()
            for member in members:
                parsed = PurePosixPath(member)
                self.assertFalse(parsed.is_absolute())
                self.assertNotIn("..", parsed.parts)
                self.assertNotIn("\\", member)
            readme_name = next(name for name in members if name.endswith("/README.md"))
            readme = archive.read(readme_name).decode("utf-8")
            self.assertNotIn("<script>", readme)
            self.assertNotIn("<img", readme)
            self.assertIn("&lt;script&gt;", readme)

    def test_invalid_identifiers_and_control_characters_are_rejected(self) -> None:
        invalid_projects = (
            {"id": "../../escape", "name": "Project"},
            {"id": "A/B", "name": "Project"},
            {"id": "CON", "name": "Project"},
            {"id": "OK", "name": "Project\x00hidden"},
            {"id": "OK", "name": "Project", "description": "x" * 4_001},
        )
        for project in invalid_projects:
            with self.subTest(project=project), self.assertRaises(ValueError):
                self.manager.generate(project)
        self.assertEqual(list(self.output_dir.iterdir()), [])

        with self.assertRaises(TypeError):
            self.manager.generate([("name", "Project")])  # type: ignore[arg-type]

    def test_missing_identifier_uses_stable_sha256_derived_value(self) -> None:
        project = {
            "name": "Projeto estável",
            "type": "Pesquisa",
            "priority": "ALFA",
            "description": "Mesmo conteúdo, mesmo identificador.",
            "created_at": 1_700_000_000,
        }
        first = Path(self.manager.generate(project))
        first_bytes = first.read_bytes()
        second = Path(self.manager.generate(dict(project)))

        self.assertEqual(first, second)
        identifier = first.stem.rsplit("_", 1)[-1]
        self.assertRegex(identifier, r"^[0-9a-f]{16}$")
        self.assertTrue(first_bytes)
        with zipfile.ZipFile(second) as archive:
            self.assertIsNone(archive.testzip())

    def test_failed_render_does_not_replace_existing_archive_or_leave_temporary_files(self) -> None:
        project = {"id": "ATOMIC-1", "name": "Atomic", "type": "Software / IA"}
        archive_path = Path(self.manager.generate(project))
        original = archive_path.read_bytes()
        real_render = self.manager._render

        def failing_render(key: str, context: dict[str, str]) -> str:
            if key == "__model_ai__":
                raise RuntimeError("simulated rendering failure")
            return real_render(key, context)

        with patch.object(self.manager, "_render", side_effect=failing_render):
            with self.assertRaisesRegex(RuntimeError, "simulated"):
                self.manager.generate(project)

        self.assertEqual(archive_path.read_bytes(), original)
        self.assertEqual(list(self.output_dir.glob("*.tmp")), [])
        with zipfile.ZipFile(archive_path) as archive:
            self.assertIsNone(archive.testzip())

    def test_generated_scaffolds_contain_no_embedded_credentials_or_import_side_effects(self) -> None:
        forbidden_literals = (
            "SUA_SENHA",
            "changeme",
            "allow_origins=[\"*\"]",
            "torch.load(",
            "subprocess.Popen(",
        )
        for index, project_type in enumerate(self.manager.TEMPLATES):
            archive_path = Path(
                self.manager.generate(
                    {"id": f"SECRET-{index}", "name": "No secrets", "type": project_type}
                )
            )
            with zipfile.ZipFile(archive_path) as archive:
                text = "\n".join(
                    archive.read(name).decode("utf-8", errors="replace")
                    for name in archive.namelist()
                )
            for forbidden in forbidden_literals:
                self.assertNotIn(forbidden, text, msg=f"{forbidden!r} found in {project_type}")

    def test_generated_firmware_cpp_is_syntactically_valid(self) -> None:
        compiler = shutil.which("g++")
        if compiler is None:
            self.skipTest("g++ is not installed")

        stub_include = Path(self.temporary_directory.name) / "arduino-stubs"
        stub_include.mkdir()
        (stub_include / "Arduino.h").write_text(
            """#pragma once
#include <stdint.h>
#define INPUT 0
#define OUTPUT 1
#define HIGH 1
#define LOW 0
#define constrain(value, low, high) ((value) < (low) ? (low) : ((value) > (high) ? (high) : (value)))
struct SerialStub {
    void begin(uint32_t) {}
    template <typename T> void println(const T&) {}
};
extern SerialStub Serial;
inline void pinMode(uint8_t, int) {}
inline int analogRead(uint8_t) { return 0; }
inline void analogWrite(uint8_t, int) {}
inline void digitalWrite(uint8_t, int) {}
inline void delay(uint32_t) {}
inline void yield() {}
inline int abs(int value) { return value < 0 ? -value : value; }
""",
            encoding="utf-8",
        )
        (stub_include / "Wire.h").write_text(
            """#pragma once
#include <stdint.h>
struct WireStub { void begin(uint8_t, uint8_t) {} };
extern WireStub Wire;
""",
            encoding="utf-8",
        )

        firmware_types = ("Firmware / Embarcado", "Robótica", "IoT / Nuvem")
        for index, project_type in enumerate(firmware_types, start=20):
            with self.subTest(project_type=project_type):
                archive_path = Path(
                    self.manager.generate(
                        {"id": f"CPP-{index}", "name": "Cpp syntax", "type": project_type}
                    )
                )
                project_root = self._validate_and_extract(archive_path, index)
                for source in project_root.rglob("*.cpp"):
                    completed = subprocess.run(
                        [
                            compiler,
                            "-std=c++17",
                            "-fsyntax-only",
                            f"-I{stub_include}",
                            f"-I{source.parent}",
                            str(source),
                        ],
                        capture_output=True,
                        text=True,
                        timeout=20,
                        check=False,
                    )
                    self.assertEqual(
                        completed.returncode,
                        0,
                        msg=f"{source}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
                    )

    def _validate_and_extract(self, archive_path: Path, index: int) -> Path:
        destination = Path(self.temporary_directory.name) / f"extracted-{index}"
        destination.mkdir()
        with zipfile.ZipFile(archive_path) as archive:
            self.assertIsNone(archive.testzip())
            members = archive.namelist()
            self.assertEqual(len(members), len(set(members)))
            roots = {PurePosixPath(member).parts[0] for member in members}
            self.assertEqual(len(roots), 1)
            for member in members:
                parsed = PurePosixPath(member)
                self.assertFalse(parsed.is_absolute())
                self.assertNotIn("..", parsed.parts)
                self.assertNotIn("\\", member)
            archive.extractall(destination)
        return destination / roots.pop()

    def _compile_generated_python(self, project_root: Path) -> None:
        for python_file in project_root.rglob("*.py"):
            source = python_file.read_text(encoding="utf-8")
            compile(source, str(python_file), "exec")

    def _import_generated_modules(self, project_root: Path, modules: tuple[str, ...]) -> None:
        if not modules:
            return
        script = "; ".join(f"import {module}" for module in modules)
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["PYTHONPATH"] = str(project_root)
        completed = subprocess.run(
            [sys.executable, "-B", "-c", script],
            cwd=project_root,
            env=environment,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )


if __name__ == "__main__":
    unittest.main()

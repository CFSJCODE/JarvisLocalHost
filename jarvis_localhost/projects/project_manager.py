"""Safe, atomic project scaffolding for J.A.R.V.I.S.

The public contract intentionally remains small: instantiate ``ProjectManager``
and call :meth:`ProjectManager.generate` with the project mapping returned by
``JarvisDB.save_project``. Generated archives contain inert, local-first
scaffolds; importing a generated Python module never starts a server, opens a
socket, downloads a model, or reads a secret.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import tempfile
import unicodedata
import zipfile
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Mapping, Tuple

from jarvis_localhost.paths import PROJECTS_ROOT


TemplateSpec = List[Tuple[str, str]]

_CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}


class ProjectManager:
    """Generate a validated project scaffold and return its ZIP path."""

    OUTPUT_DIR = PROJECTS_ROOT

    TEMPLATES: Dict[str, TemplateSpec] = {
        "Software / IA": [
            ("README.md", "__readme__"),
            ("requirements.txt", "__requirements_ai__"),
            ("main.py", "__main_ai__"),
            ("src/__init__.py", ""),
            ("src/model.py", "__model_ai__"),
            ("src/data.py", "__data_ai__"),
            ("src/train.py", "__train_ai__"),
            ("tests/__init__.py", ""),
            ("tests/test_model.py", "__test_ai__"),
            (".gitignore", "__gitignore__"),
            ("docs/architecture.md", "__arch_doc__"),
        ],
        "Firmware / Embarcado": [
            ("README.md", "__readme__"),
            ("firmware/main.cpp", "__firmware_main__"),
            ("firmware/config.h", "__firmware_config__"),
            ("firmware/sensors.cpp", "__firmware_sensors__"),
            ("firmware/sensors.h", "__firmware_sensors_h__"),
            ("firmware/comms.cpp", "__firmware_comms__"),
            ("firmware/comms.h", "__firmware_comms_h__"),
            ("docs/pinout.md", "__pinout_doc__"),
            ("platformio.ini", "__platformio__"),
            (".gitignore", "__gitignore__"),
        ],
        "Robótica": [
            ("README.md", "__readme__"),
            ("firmware/main.cpp", "__firmware_main__"),
            ("firmware/config.h", "__firmware_config__"),
            ("firmware/sensors.cpp", "__firmware_sensors__"),
            ("firmware/sensors.h", "__firmware_sensors_h__"),
            ("firmware/comms.cpp", "__firmware_comms__"),
            ("firmware/comms.h", "__firmware_comms_h__"),
            ("firmware/motors.cpp", "__firmware_motors__"),
            ("firmware/motors.h", "__firmware_motors_h__"),
            ("firmware/navigation.cpp", "__firmware_nav__"),
            ("firmware/navigation.h", "__firmware_nav_h__"),
            ("docs/hardware.md", "__hw_doc__"),
            ("docs/wiring.md", "__wiring_doc__"),
            ("platformio.ini", "__platformio__"),
            (".gitignore", "__gitignore__"),
        ],
        "IoT / Nuvem": [
            ("README.md", "__readme__"),
            ("device/main.cpp", "__firmware_main__"),
            ("device/config.h", "__firmware_config__"),
            ("device/sensors.cpp", "__firmware_sensors__"),
            ("device/sensors.h", "__firmware_sensors_h__"),
            ("device/mqtt.cpp", "__iot_mqtt__"),
            ("device/mqtt.h", "__iot_mqtt_h__"),
            ("backend/__init__.py", ""),
            ("backend/server.py", "__iot_backend__"),
            ("backend/requirements.txt", "__requirements_iot__"),
            ("dashboard/index.html", "__iot_dashboard__"),
            ("dashboard/app.js", "__iot_dashboard_js__"),
            ("docs/architecture.md", "__arch_doc__"),
            (".gitignore", "__gitignore__"),
        ],
        "Pesquisa": [
            ("README.md", "__readme__"),
            ("paper/draft.md", "__paper_draft__"),
            ("experiments/__init__.py", ""),
            ("experiments/run.py", "__exp_run__"),
            ("experiments/config.py", "__exp_config__"),
            ("data/.gitkeep", ""),
            ("results/.gitkeep", ""),
            ("requirements.txt", "__requirements_research__"),
            ("docs/methodology.md", "__methodology__"),
            (".gitignore", "__gitignore__"),
        ],
        "Infraestrutura": [
            ("README.md", "__readme__"),
            ("backend/__init__.py", ""),
            ("backend/server.py", "__infra_backend__"),
            ("requirements.txt", "__requirements_infra__"),
            ("docker-compose.yml", "__docker_compose__"),
            ("Dockerfile", "__dockerfile__"),
            ("config/nginx.conf", "__nginx__"),
            ("scripts/setup.sh", "__setup_sh__"),
            ("scripts/deploy.sh", "__deploy_sh__"),
            ("docs/runbook.md", "__runbook__"),
            (".gitignore", "__gitignore__"),
        ],
    }

    GENERIC_TEMPLATE: TemplateSpec = [
        ("README.md", "__readme__"),
        ("docs/spec.md", "__spec_doc__"),
        (".gitignore", "__gitignore__"),
    ]

    def __init__(self, output_dir: str | os.PathLike[str] | None = None):
        selected_output = Path(output_dir) if output_dir is not None else Path(self.OUTPUT_DIR)
        self.output_dir = selected_output.expanduser().resolve(strict=False)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, project: Mapping[str, Any]) -> str:
        """Create one complete ZIP atomically and return its absolute path."""

        context = self._build_context(project)
        template = self.TEMPLATES.get(context["type"], self.GENERIC_TEMPLATE)
        archive_path = self._output_path(f"{context['name']}_{context['pid']}.zip")
        temporary_path = self._temporary_path(archive_path.name)

        try:
            self._write_archive(temporary_path, template, context)
            with temporary_path.open("r+b") as archive_file:
                os.fsync(archive_file.fileno())
            os.replace(temporary_path, archive_path)
        finally:
            temporary_path.unlink(missing_ok=True)

        return str(archive_path)

    def _build_context(self, project: Mapping[str, Any]) -> Dict[str, str]:
        if not isinstance(project, Mapping):
            raise TypeError("project must be a mapping")

        original_name = self._bounded_text(project.get("name", "Projeto"), "name", 128)
        name = self._safe_project_name(original_name)
        project_type = self._bounded_inline_text(project.get("type", "Genérico"), "type", 80)
        priority = self._bounded_inline_text(project.get("priority", "BETA"), "priority", 32)
        description = self._bounded_text(project.get("description", ""), "description", 4_000)
        author = self._bounded_inline_text(project.get("author", "Equipe do projeto"), "author", 120)
        organization = self._bounded_inline_text(
            project.get("organization", "Projeto local"), "organization", 120
        )

        identifier = project.get("id")
        if identifier is None or str(identifier).strip() == "":
            identifier = self._stable_project_id(name, project_type, priority, description)
        pid = self._validated_identifier(identifier)

        return {
            "name": name,
            "type": project_type,
            "priority": priority,
            "desc": html.escape(description, quote=False),
            "pid": pid,
            "created": self._format_created_at(project.get("created_at")),
            "author": html.escape(author, quote=False),
            "org": html.escape(organization, quote=False),
        }

    @staticmethod
    def _bounded_text(value: Any, field: str, maximum: int) -> str:
        if value is None:
            return ""
        if not isinstance(value, (str, int, float)):
            raise TypeError(f"{field} must be text-compatible")
        text = unicodedata.normalize("NFKC", str(value)).strip()
        if _CONTROL_CHARACTERS.search(text):
            raise ValueError(f"{field} contains control characters")
        if len(text) > maximum:
            raise ValueError(f"{field} exceeds {maximum} characters")
        return text

    @classmethod
    def _bounded_inline_text(cls, value: Any, field: str, maximum: int) -> str:
        text = cls._bounded_text(value, field, maximum)
        return " ".join(text.splitlines())

    @staticmethod
    def _safe_project_name(value: str) -> str:
        ascii_name = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
        safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", ascii_name).strip("._-")
        safe_name = re.sub(r"_+", "_", safe_name)[:64]
        if not safe_name:
            safe_name = "Projeto"
        if safe_name.upper() in _WINDOWS_RESERVED_NAMES:
            safe_name = f"Projeto_{safe_name}"
        return safe_name

    @staticmethod
    def _validated_identifier(value: Any) -> str:
        identifier = unicodedata.normalize("NFKC", str(value)).strip()
        if not _SAFE_IDENTIFIER.fullmatch(identifier):
            raise ValueError("id must contain only letters, digits, underscores, or hyphens")
        if identifier.upper() in _WINDOWS_RESERVED_NAMES:
            raise ValueError("id is a reserved Windows filename")
        return identifier

    @staticmethod
    def _stable_project_id(name: str, project_type: str, priority: str, description: str) -> str:
        canonical = json.dumps(
            {"description": description, "name": name, "priority": priority, "type": project_type},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _format_created_at(value: Any) -> str:
        if isinstance(value, (int, float)):
            try:
                return datetime.fromtimestamp(value).strftime("%Y-%m-%d")
            except (OverflowError, OSError, ValueError):
                raise ValueError("created_at is outside the supported range") from None
        return datetime.now().strftime("%Y-%m-%d")

    def _output_path(self, filename: str) -> Path:
        candidate = (self.output_dir / filename).resolve(strict=False)
        if candidate.parent != self.output_dir:
            raise ValueError("archive path escapes the configured output directory")
        return candidate

    def _temporary_path(self, archive_name: str) -> Path:
        descriptor, raw_path = tempfile.mkstemp(
            dir=self.output_dir, prefix=f".{archive_name}.", suffix=".tmp"
        )
        os.close(descriptor)
        return Path(raw_path)

    def _write_archive(
        self, archive_path: Path, template: TemplateSpec, context: Dict[str, str]
    ) -> None:
        seen: set[str] = set()
        with zipfile.ZipFile(
            archive_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for relative_path, template_key in template:
                member_name = self._safe_member_name(context["name"], relative_path)
                if member_name in seen:
                    raise ValueError(f"duplicate archive member: {member_name}")
                seen.add(member_name)
                content = self._render(template_key, context)
                info = zipfile.ZipInfo(member_name, date_time=datetime.now().timetuple()[:6])
                info.compress_type = zipfile.ZIP_DEFLATED
                mode = 0o755 if relative_path.startswith("scripts/") else 0o644
                info.external_attr = mode << 16
                archive.writestr(info, content.encode("utf-8"))

        with zipfile.ZipFile(archive_path, mode="r") as archive:
            corrupt_member = archive.testzip()
            if corrupt_member is not None:
                raise OSError(f"generated ZIP failed CRC validation: {corrupt_member}")

    @staticmethod
    def _safe_member_name(project_name: str, relative_path: str) -> str:
        path = PurePosixPath(relative_path)
        if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
            raise ValueError(f"unsafe template path: {relative_path}")
        if any("\\" in part or ":" in part for part in path.parts):
            raise ValueError(f"non-portable template path: {relative_path}")
        return PurePosixPath(project_name, path).as_posix()

    def _render(self, key: str, context: Dict[str, str]) -> str:
        if not key:
            return ""
        renderer = getattr(self, f"_tpl{key}", None)
        if renderer is None or not callable(renderer):
            raise KeyError(f"unknown project template: {key}")
        content = renderer(context)
        if not isinstance(content, str):
            raise TypeError(f"template {key} did not return text")
        if "\x00" in content:
            raise ValueError(f"template {key} contains a NUL byte")
        return content.replace("\r\n", "\n")

    @staticmethod
    def _python_string(value: str) -> str:
        return json.dumps(value, ensure_ascii=False)

    def _tpl__readme__(self, c: Dict[str, str]) -> str:
        return f"""# {c['name']}

| Campo | Valor |
|---|---|
| Tipo | {html.escape(c['type'], quote=False)} |
| Prioridade | {html.escape(c['priority'], quote=False)} |
| ID | `{c['pid']}` |
| Criado em | {c['created']} |

## Descrição

{c['desc'] or 'Descrição do projeto a ser preenchida.'}

## Princípios do scaffold

- execução local por padrão;
- nenhum segredo incluído no repositório;
- nenhum serviço ou download iniciado durante importação;
- validação e testes antes de publicação.

## Primeiros passos

1. Revise a documentação em `docs/`.
2. Crie segredos somente em arquivos locais ignorados pelo Git ou em variáveis de ambiente.
3. Execute os testes antes de iniciar qualquer serviço.

---

Responsável: {c['author']}
Organização: {c['org']}
"""

    def _tpl__gitignore__(self, _c: Dict[str, str]) -> str:
        return """# Python
__pycache__/
*.py[cod]
.venv/
venv/
dist/
build/

# C/C++ and PlatformIO
*.o
*.elf
*.bin
*.hex
.pio/

# Runtime data
data/
results/*.json
*.db
*.sqlite*
*.log

# Secrets and local overrides
.env
.env.*
!.env.example
secrets.h
secrets.json
credentials.json
config.local.*

# Editors
.vscode/
.idea/
*.swp
"""

    def _tpl__requirements_ai__(self, _c: Dict[str, str]) -> str:
        return """# The starter implementation uses only the Python standard library.
# Add reviewed, pinned dependencies here when the project requires them.
"""

    def _tpl__main_ai__(self, c: Dict[str, str]) -> str:
        project_name = self._python_string(c["name"])
        return f'''"""Command-line entry point for {c["name"]}."""

from __future__ import annotations

import argparse

from src.data import sample_dataset
from src.model import LinearClassifier, ModelConfig
from src.train import evaluate, train


PROJECT_NAME = {project_name}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=PROJECT_NAME)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.epochs < 1 or args.learning_rate <= 0:
        raise SystemExit("epochs and learning-rate must be positive")
    dataset = sample_dataset()
    model = LinearClassifier(ModelConfig(input_dim=2, classes=2))
    train(model, dataset, epochs=args.epochs, learning_rate=args.learning_rate)
    print({{"project": PROJECT_NAME, "accuracy": evaluate(model, dataset)}})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''

    def _tpl__model_ai__(self, c: Dict[str, str]) -> str:
        return f'''"""Small dependency-free model scaffold for {c["name"]}."""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class ModelConfig:
    input_dim: int = 2
    classes: int = 2

    def __post_init__(self) -> None:
        if self.input_dim < 1 or self.classes < 2:
            raise ValueError("input_dim must be positive and classes must be at least two")


class LinearClassifier:
    def __init__(self, config: ModelConfig | None = None):
        self.config = config or ModelConfig()
        self.weights = [[0.0] * self.config.input_dim for _ in range(self.config.classes)]
        self.bias = [0.0 for _ in range(self.config.classes)]

    def scores(self, features: Sequence[float]) -> list[float]:
        if len(features) != self.config.input_dim:
            raise ValueError("feature vector has the wrong size")
        return [
            sum(weight * float(value) for weight, value in zip(row, features)) + bias
            for row, bias in zip(self.weights, self.bias)
        ]

    def predict(self, features: Sequence[float]) -> int:
        scores = self.scores(features)
        return max(range(len(scores)), key=scores.__getitem__)

    def probabilities(self, features: Sequence[float]) -> list[float]:
        scores = self.scores(features)
        offset = max(scores)
        exponentials = [math.exp(score - offset) for score in scores]
        total = sum(exponentials)
        return [value / total for value in exponentials]

    def save(self, path: str | os.PathLike[str]) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {{"config": asdict(self.config), "weights": self.weights, "bias": self.bias}}
        descriptor, raw_temp = tempfile.mkstemp(dir=destination.parent, prefix=".model-", suffix=".tmp")
        os.close(descriptor)
        temporary = Path(raw_temp)
        try:
            temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "LinearClassifier":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        model = cls(ModelConfig(**payload["config"]))
        model.weights = _validated_matrix(payload["weights"], model.config)
        model.bias = _validated_bias(payload["bias"], model.config)
        return model


def _validated_matrix(value: object, config: ModelConfig) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != config.classes:
        raise ValueError("invalid weight matrix")
    matrix = []
    for row in value:
        if not isinstance(row, list) or len(row) != config.input_dim:
            raise ValueError("invalid weight row")
        matrix.append([float(item) for item in row])
    return matrix


def _validated_bias(value: object, config: ModelConfig) -> list[float]:
    if not isinstance(value, list) or len(value) != config.classes:
        raise ValueError("invalid bias vector")
    return [float(item) for item in value]
'''

    def _tpl__data_ai__(self, c: Dict[str, str]) -> str:
        return f'''"""Validated data helpers for {c["name"]}."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TypeAlias


DataPoint: TypeAlias = tuple[tuple[float, ...], int]


def sample_dataset() -> list[DataPoint]:
    return [((-1.0, -1.0), 0), ((-0.8, -0.4), 0), ((0.7, 0.4), 1), ((1.0, 1.0), 1)]


def load_csv(path: str | Path, input_dim: int) -> list[DataPoint]:
    if input_dim < 1:
        raise ValueError("input_dim must be positive")
    rows: list[DataPoint] = []
    with Path(path).open("r", encoding="utf-8", newline="") as source:
        for line_number, row in enumerate(csv.reader(source), start=1):
            if len(row) != input_dim + 1:
                raise ValueError(f"line {{line_number}} has the wrong number of columns")
            rows.append((tuple(float(value) for value in row[:-1]), int(row[-1])))
    if not rows:
        raise ValueError("dataset is empty")
    return rows
'''

    def _tpl__train_ai__(self, c: Dict[str, str]) -> str:
        return f'''"""Training helpers for {c["name"]}."""

from __future__ import annotations

from collections.abc import Sequence

from src.data import DataPoint
from src.model import LinearClassifier


def train(
    model: LinearClassifier,
    dataset: Sequence[DataPoint],
    *,
    epochs: int = 20,
    learning_rate: float = 0.05,
) -> list[float]:
    if not dataset or epochs < 1 or learning_rate <= 0:
        raise ValueError("dataset, epochs, and learning_rate must be valid")
    history: list[float] = []
    for _ in range(epochs):
        mistakes = 0
        for features, expected in dataset:
            if expected < 0 or expected >= model.config.classes:
                raise ValueError("class label is outside the model range")
            predicted = model.predict(features)
            if predicted == expected:
                continue
            mistakes += 1
            for index, value in enumerate(features):
                delta = learning_rate * float(value)
                model.weights[expected][index] += delta
                model.weights[predicted][index] -= delta
            model.bias[expected] += learning_rate
            model.bias[predicted] -= learning_rate
        history.append(mistakes / len(dataset))
    return history


def evaluate(model: LinearClassifier, dataset: Sequence[DataPoint]) -> float:
    if not dataset:
        raise ValueError("dataset is empty")
    correct = sum(model.predict(features) == expected for features, expected in dataset)
    return correct / len(dataset)
'''

    def _tpl__test_ai__(self, c: Dict[str, str]) -> str:
        return f'''"""Tests for {c["name"]}."""

import tempfile
import unittest
from pathlib import Path

from src.data import sample_dataset
from src.model import LinearClassifier, ModelConfig
from src.train import evaluate, train


class ModelTests(unittest.TestCase):
    def test_training_and_round_trip(self) -> None:
        model = LinearClassifier(ModelConfig(input_dim=2, classes=2))
        history = train(model, sample_dataset(), epochs=20)
        self.assertEqual(len(history), 20)
        self.assertGreaterEqual(evaluate(model, sample_dataset()), 0.75)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            model.save(path)
            loaded = LinearClassifier.load(path)
            self.assertEqual(loaded.predict((1.0, 1.0)), model.predict((1.0, 1.0)))


if __name__ == "__main__":
    unittest.main()
'''

    def _tpl__firmware_main__(self, c: Dict[str, str]) -> str:
        return f'''/** Local firmware entry point for {c["name"]}. */
#include <Arduino.h>
#include <Wire.h>

#include "config.h"
#include "sensors.h"


void setup() {{
    Serial.begin(SERIAL_BAUD);
    Wire.begin(PIN_SDA, PIN_SCL);
    sensors_init();
    Serial.println("[{c['name']}] ready");
}}


void loop() {{
    sensors_read_all();
    delay(SENSOR_PERIOD_MS);
}}
'''

    def _tpl__firmware_config__(self, c: Dict[str, str]) -> str:
        return f'''/** Non-secret defaults for {c["name"]}. */
#pragma once

#include <stdint.h>

constexpr uint32_t SERIAL_BAUD = 115200;
constexpr uint8_t PIN_SDA = 21;
constexpr uint8_t PIN_SCL = 22;
constexpr uint8_t PIN_MQ02_AO = 34;
constexpr uint32_t SENSOR_PERIOD_MS = 200;
constexpr uint16_t MQ02_THRESHOLD = 300;
constexpr uint8_t PIN_MOTOR_LEFT_PWM = 18;
constexpr uint8_t PIN_MOTOR_LEFT_DIR = 19;
constexpr uint8_t PIN_MOTOR_RIGHT_PWM = 5;
constexpr uint8_t PIN_MOTOR_RIGHT_DIR = 4;
'''

    def _tpl__firmware_sensors__(self, _c: Dict[str, str]) -> str:
        return '''#include "sensors.h"

#include <Arduino.h>
#include "config.h"


SensorData g_sensors = {0, false};


void sensors_init() {
    pinMode(PIN_MQ02_AO, INPUT);
}


void sensors_read_all() {
    g_sensors.gas_raw = static_cast<uint16_t>(analogRead(PIN_MQ02_AO));
    g_sensors.gas_alert = g_sensors.gas_raw > MQ02_THRESHOLD;
}
'''

    def _tpl__firmware_sensors_h__(self, _c: Dict[str, str]) -> str:
        return '''#pragma once

#include <stdbool.h>
#include <stdint.h>


struct SensorData {
    uint16_t gas_raw;
    bool gas_alert;
};

extern SensorData g_sensors;

void sensors_init();
void sensors_read_all();
'''

    def _tpl__firmware_comms__(self, _c: Dict[str, str]) -> str:
        return '''#include "comms.h"

#include <Arduino.h>


void comms_init() {
    // Add an explicitly configured transport here. No credentials are embedded.
}


void comms_poll() {
    yield();
}
'''

    def _tpl__firmware_comms_h__(self, _c: Dict[str, str]) -> str:
        return """#pragma once

void comms_init();
void comms_poll();
"""

    def _tpl__firmware_motors__(self, _c: Dict[str, str]) -> str:
        return '''#include "motors.h"

#include <Arduino.h>
#include "config.h"


void motors_init() {
    pinMode(PIN_MOTOR_LEFT_PWM, OUTPUT);
    pinMode(PIN_MOTOR_LEFT_DIR, OUTPUT);
    pinMode(PIN_MOTOR_RIGHT_PWM, OUTPUT);
    pinMode(PIN_MOTOR_RIGHT_DIR, OUTPUT);
    motors_stop();
}


void motors_set(int16_t left, int16_t right) {
    const int16_t bounded_left = constrain(left, -255, 255);
    const int16_t bounded_right = constrain(right, -255, 255);
    digitalWrite(PIN_MOTOR_LEFT_DIR, bounded_left >= 0 ? HIGH : LOW);
    digitalWrite(PIN_MOTOR_RIGHT_DIR, bounded_right >= 0 ? HIGH : LOW);
    analogWrite(PIN_MOTOR_LEFT_PWM, abs(bounded_left));
    analogWrite(PIN_MOTOR_RIGHT_PWM, abs(bounded_right));
}


void motors_stop() {
    analogWrite(PIN_MOTOR_LEFT_PWM, 0);
    analogWrite(PIN_MOTOR_RIGHT_PWM, 0);
}
'''

    def _tpl__firmware_motors_h__(self, _c: Dict[str, str]) -> str:
        return """#pragma once

#include <stdint.h>

void motors_init();
void motors_set(int16_t left, int16_t right);
void motors_stop();
"""

    def _tpl__firmware_nav__(self, _c: Dict[str, str]) -> str:
        return '''#include "navigation.h"


NavState navigation_decide(float front_mm, float left_mm, float right_mm) {
    constexpr float safe_distance_mm = 350.0F;
    if (front_mm >= safe_distance_mm) {
        return NavState::Forward;
    }
    return left_mm >= right_mm ? NavState::TurnLeft : NavState::TurnRight;
}
'''

    def _tpl__firmware_nav_h__(self, _c: Dict[str, str]) -> str:
        return """#pragma once

enum class NavState { Stop, Forward, TurnLeft, TurnRight };

NavState navigation_decide(float front_mm, float left_mm, float right_mm);
"""

    def _tpl__platformio__(self, c: Dict[str, str]) -> str:
        return f"""[env:esp32dev]
platform = espressif32
board = esp32dev
framework = arduino
monitor_speed = 115200

; {c['name']} - generated {c['created']}
"""

    def _tpl__iot_mqtt__(self, c: Dict[str, str]) -> str:
        topic = f"{c['name'].lower()}/sensors"
        return f'''#include "mqtt.h"

#include <Arduino.h>


constexpr const char* TELEMETRY_TOPIC = "{topic}";


bool mqtt_publish_sensor(uint16_t gas_raw, bool gas_alert) {{
    // Wire an authenticated MQTT client here after loading credentials from a
    // local, ignored secrets.h file. The starter performs no network action.
    (void)gas_raw;
    (void)gas_alert;
    return false;
}}
'''

    def _tpl__iot_mqtt_h__(self, _c: Dict[str, str]) -> str:
        return """#pragma once

#include <stdbool.h>
#include <stdint.h>

bool mqtt_publish_sensor(uint16_t gas_raw, bool gas_alert);
"""

    def _tpl__iot_backend__(self, c: Dict[str, str]) -> str:
        project_name = self._python_string(c["name"])
        return f'''"""Loopback-only telemetry API for {c["name"]}."""

from __future__ import annotations

import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


PROJECT_NAME = {project_name}
MAX_BODY_BYTES = 64 * 1024
_latest: dict[str, Any] = {{}}
_lock = threading.Lock()
DASHBOARD_PATH = Path(__file__).resolve().parents[1] / "dashboard" / "index.html"


class TelemetryHandler(BaseHTTPRequestHandler):
    server_version = "LocalTelemetry/1.0"

    def do_GET(self) -> None:
        if self.path == "/":
            self._static_response(DASHBOARD_PATH, "text/html; charset=utf-8")
            return
        if self.path == "/app.js":
            self._static_response(DASHBOARD_PATH.with_name("app.js"), "text/javascript; charset=utf-8")
            return
        if self.path != "/api/latest":
            self._json_response(HTTPStatus.NOT_FOUND, {{"error": "not found"}})
            return
        with _lock:
            payload = dict(_latest)
        self._json_response(HTTPStatus.OK, payload)

    def do_POST(self) -> None:
        if self.path != "/api/readings":
            self._json_response(HTTPStatus.NOT_FOUND, {{"error": "not found"}})
            return
        if self.headers.get_content_type() != "application/json":
            self._json_response(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {{"error": "JSON required"}})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._json_response(HTTPStatus.BAD_REQUEST, {{"error": "invalid length"}})
            return
        if length < 1 or length > MAX_BODY_BYTES:
            self._json_response(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {{"error": "invalid body size"}})
            return
        try:
            payload = json.loads(self.rfile.read(length))
            reading = validate_reading(payload)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            self._json_response(HTTPStatus.BAD_REQUEST, {{"error": str(error)}})
            return
        with _lock:
            _latest.clear()
            _latest.update(reading)
        self._json_response(HTTPStatus.ACCEPTED, {{"accepted": True}})

    def log_message(self, _format: str, *args: object) -> None:
        return

    def _json_response(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(encoded)

    def _static_response(self, path: Path, content_type: str) -> None:
        encoded = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(encoded)


def validate_reading(payload: object) -> dict[str, float | int | bool]:
    if not isinstance(payload, dict):
        raise ValueError("reading must be an object")
    allowed = {{"temperature", "humidity", "gas_raw", "gas_alert"}}
    if not payload or set(payload) - allowed:
        raise ValueError("reading contains unsupported fields")
    result: dict[str, float | int | bool] = {{}}
    for key, value in payload.items():
        if key == "gas_alert":
            if not isinstance(value, bool):
                raise ValueError("gas_alert must be boolean")
            result[key] = value
        elif not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{{key}} must be numeric")
        else:
            result[key] = float(value)
    return result


def create_server(host: str = "127.0.0.1", port: int = 8001) -> ThreadingHTTPServer:
    if host not in {{"127.0.0.1", "::1", "localhost"}}:
        raise ValueError("the starter server is restricted to loopback")
    return ThreadingHTTPServer((host, port), TelemetryHandler)


def main() -> None:
    server = create_server()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
'''

    def _tpl__requirements_iot__(self, _c: Dict[str, str]) -> str:
        return """# The local telemetry API uses only the Python standard library.
"""

    def _tpl__iot_dashboard__(self, c: Dict[str, str]) -> str:
        title = html.escape(c["name"], quote=True)
        return f'''<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title} - Telemetria local</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #14213d; }}
    dl {{ display: grid; grid-template-columns: max-content 1fr; gap: .5rem 1rem; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <p id="status">Aguardando a API local.</p>
  <dl id="readings"></dl>
  <script src="/app.js" defer></script>
</body>
</html>
'''

    def _tpl__iot_dashboard_js__(self, _c: Dict[str, str]) -> str:
        return '''"use strict";

const statusNode = document.getElementById("status");
const readingsNode = document.getElementById("readings");

async function refresh() {
  try {
    const response = await fetch("/api/latest", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const reading = await response.json();
    readingsNode.replaceChildren();
    Object.entries(reading).forEach(([key, value]) => {
      const term = document.createElement("dt");
      const detail = document.createElement("dd");
      term.textContent = key;
      detail.textContent = String(value);
      readingsNode.append(term, detail);
    });
    statusNode.textContent = "API local conectada.";
  } catch (error) {
    statusNode.textContent = `API indisponível: ${error.message}`;
  }
}

refresh();
setInterval(refresh, 2000);
'''

    def _tpl__paper_draft__(self, c: Dict[str, str]) -> str:
        return f"""# {c['name']} - rascunho

**Autores:** {c['author']}
**Data:** {c['created']}

## Resumo

Descreva objetivo, método, resultados e conclusão.

## 1. Introdução

{c['desc'] or 'Contextualize o problema e a hipótese.'}

## 2. Trabalhos relacionados

Registre somente fontes verificadas e mantenha citações rastreáveis.

## 3. Metodologia

Defina conjunto de dados, protocolo experimental e critérios de exclusão.

## 4. Resultados

Reporte resultados positivos, negativos e limitações.

## 5. Conclusão

Compare os resultados com a hipótese original.

## Referências

1. Adicione referências verificadas.
"""

    def _tpl__exp_run__(self, c: Dict[str, str]) -> str:
        return f'''"""Reproducible experiment runner for {c["name"]}."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.config import EXPERIMENTS


def run_experiment(config: dict[str, Any]) -> dict[str, Any]:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return {{
        "config": dict(config),
        "experiment_id": hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16],
        "metrics": {{}},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }}


def save_results(results: list[dict[str, Any]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temp = tempfile.mkstemp(dir=destination.parent, prefix=".results-", suffix=".tmp")
    os.close(descriptor)
    temporary = Path(raw_temp)
    try:
        temporary.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    results = [run_experiment(config) for config in EXPERIMENTS]
    save_results(results, Path("results") / "latest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''

    def _tpl__exp_config__(self, c: Dict[str, str]) -> str:
        return f'''"""Reviewed experiment definitions for {c["name"]}."""

EXPERIMENTS = (
    {{"name": "baseline", "learning_rate": 0.001, "epochs": 10}},
    {{"name": "candidate", "learning_rate": 0.0003, "epochs": 20}},
)
'''

    def _tpl__requirements_research__(self, _c: Dict[str, str]) -> str:
        return """# The reproducibility scaffold uses only the Python standard library.
"""

    def _tpl__methodology__(self, c: Dict[str, str]) -> str:
        return f"""# Metodologia - {c['name']}

## Hipótese

Declare uma hipótese falsificável.

## Dados e proveniência

Registre origem, licença, hash SHA-256 e critérios de inclusão de cada fonte.

## Protocolo experimental

1. congele a configuração;
2. execute o baseline;
3. altere uma variável por vez;
4. preserve resultados negativos;
5. reporte incerteza e limitações.

## Critério de parada

Defina-o antes do primeiro experimento.
"""

    def _tpl__infra_backend__(self, c: Dict[str, str]) -> str:
        project_name = self._python_string(c["name"])
        return f'''"""Minimal loopback-ready health service for {c["name"]}."""

from __future__ import annotations

import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


PROJECT_NAME = {project_name}


class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path not in {{"/", "/health"}}:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        payload = json.dumps({{"project": PROJECT_NAME, "status": "ok"}}).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, _format: str, *args: object) -> None:
        return


def create_server(host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    loopback = {{"127.0.0.1", "::1", "localhost"}}
    explicitly_containerized = host == "0.0.0.0" and os.getenv("ALLOW_NON_LOOPBACK") == "1"
    if host not in loopback and not explicitly_containerized:
        raise ValueError("bind to a non-loopback host only after an explicit security review")
    return ThreadingHTTPServer((host, port), HealthHandler)


def main() -> None:
    server = create_server(
        host=os.getenv("APP_HOST", "127.0.0.1"),
        port=int(os.getenv("APP_PORT", "8000")),
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
'''

    def _tpl__requirements_infra__(self, _c: Dict[str, str]) -> str:
        return """# The health service uses only the Python standard library.
"""

    def _tpl__docker_compose__(self, _c: Dict[str, str]) -> str:
        return """services:
  backend:
    build:
      context: .
    environment:
      APP_HOST: "0.0.0.0"
      APP_PORT: "8000"
      ALLOW_NON_LOOPBACK: "1"
    expose:
      - "8000"
    read_only: true
    tmpfs:
      - /tmp
    security_opt:
      - no-new-privileges:true
    restart: unless-stopped

  nginx:
    image: nginx:1.27-alpine
    depends_on:
      - backend
    ports:
      - "127.0.0.1:8080:8080"
    volumes:
      - ./config/nginx.conf:/etc/nginx/nginx.conf:ro
    read_only: true
    tmpfs:
      - /var/cache/nginx
      - /var/run
    security_opt:
      - no-new-privileges:true
    restart: unless-stopped
"""

    def _tpl__dockerfile__(self, _c: Dict[str, str]) -> str:
        return """FROM python:3.11.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \\
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY backend ./backend

USER 65532:65532
EXPOSE 8000
CMD ["python", "-m", "backend.server"]
"""

    def _tpl__nginx__(self, _c: Dict[str, str]) -> str:
        return """events {
  worker_connections 256;
}

http {
  server_tokens off;
  server {
    listen 8080;
    location / {
      proxy_pass http://backend:8000;
      proxy_set_header Host $host;
      proxy_set_header X-Forwarded-Proto $scheme;
      proxy_connect_timeout 5s;
      proxy_read_timeout 30s;
    }
  }
}
"""

    def _tpl__setup_sh__(self, c: Dict[str, str]) -> str:
        return f'''#!/usr/bin/env sh
set -eu

echo "Preparing {c['name']} locally"
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --requirement requirements.txt
echo "Run: . .venv/bin/activate && python -m backend.server"
'''

    def _tpl__deploy_sh__(self, c: Dict[str, str]) -> str:
        return f'''#!/usr/bin/env sh
set -eu

echo "Building {c['name']} for loopback-only access"
docker compose config --quiet
docker compose up --build --detach
docker compose ps
'''

    def _tpl__runbook__(self, c: Dict[str, str]) -> str:
        return f"""# Runbook - {c['name']}

## Validação

```sh
docker compose config --quiet
python -m unittest discover -v
```

## Inicialização local

```sh
docker compose up --build --detach
curl --fail http://127.0.0.1:8080/health
```

## Parada recuperável

```sh
docker compose stop
```

## Rollback

Mantenha imagens versionadas e restaure uma tag previamente validada. Não use
comandos que descartem alterações locais como etapa automática de rollback.
"""

    def _tpl__arch_doc__(self, c: Dict[str, str]) -> str:
        return f"""# Arquitetura - {c['name']}

## Objetivo

{c['desc'] or 'Documente o objetivo e os limites do sistema.'}

## Limites de confiança

- dados externos são tratados como entrada não confiável;
- segredos vêm de armazenamento local ignorado ou variáveis de ambiente;
- serviços escutam apenas em loopback até revisão explícita;
- importações não iniciam rede, processos ou treinamento.

## Fluxo sugerido

```text
entrada validada -> domínio -> persistência atômica -> saída observável
```
"""

    def _tpl__hw_doc__(self, c: Dict[str, str]) -> str:
        return f"""# Hardware - {c['name']}

## Inventário

| Componente | Modelo | Tensão | Quantidade |
|---|---|---:|---:|
| Controlador | ESP32 | 3,3 V | 1 |
| Sensor | A definir | A verificar | 1 |
| Atuador | A definir | A verificar | 1 |

## Gate de segurança

Confirme pinagem, corrente, tensão e estado seguro dos atuadores antes de
energizar. O scaffold não autoriza upload de firmware para hardware conectado.
"""

    def _tpl__wiring_doc__(self, c: Dict[str, str]) -> str:
        return f"""# Ligações - {c['name']}

Registre cada conexão somente após conferir o datasheet do componente real.

| Origem | Destino | Nível lógico | Verificado por |
|---|---|---:|---|
| ESP32 GND | GND comum | 0 V | Pendente |
| ESP32 SDA | Sensor SDA | 3,3 V | Pendente |
| ESP32 SCL | Sensor SCL | 3,3 V | Pendente |
"""

    def _tpl__pinout_doc__(self, c: Dict[str, str]) -> str:
        return self._tpl__hw_doc__(c)

    def _tpl__spec_doc__(self, c: Dict[str, str]) -> str:
        return f"""# Especificação técnica - {c['name']}

## Contexto

{c['desc'] or 'Descreva o problema, os usuários e os limites do projeto.'}

## Requisitos funcionais

- RF01: definir comportamento observável.

## Requisitos não funcionais

- RNF01: validar entradas antes de qualquer efeito colateral.
- RNF02: gravar artefatos importantes de forma atômica.
- RNF03: manter segredos fora do código e dos logs.

## Critérios de aceite

- testes automatizados passam em ambiente limpo;
- caminhos permanecem confinados ao diretório do projeto;
- nenhuma importação inicia rede ou processo.
"""

"""
project_manager.py — J.A.R.V.I.S Project File Generator
Gera estruturas reais de arquivos para novos projetos:
README, estrutura de pastas, arquivos de configuração,
esqueleto de código, documentação técnica.
"""

import os
import re
import time
import json
import zipfile
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime


class ProjectManager:
    """
    Gera arquivos reais de projeto baseados no tipo selecionado.
    Retorna um .zip para download.
    """

    OUTPUT_DIR = Path("data/projects")

    TEMPLATES: Dict[str, List[Tuple[str, str]]] = {

        "Software / IA": [
            ("README.md",             "__readme__"),
            ("requirements.txt",      "__requirements_ai__"),
            ("main.py",               "__main_ai__"),
            ("src/__init__.py",       ""),
            ("src/model.py",          "__model_ai__"),
            ("src/data.py",           "__data_ai__"),
            ("src/train.py",          "__train_ai__"),
            ("tests/__init__.py",     ""),
            ("tests/test_model.py",   "__test_ai__"),
            (".gitignore",            "__gitignore__"),
            ("docs/architecture.md",  "__arch_doc__"),
        ],

        "Firmware / Embarcado": [
            ("README.md",             "__readme__"),
            ("firmware/main.cpp",     "__firmware_main__"),
            ("firmware/config.h",     "__firmware_config__"),
            ("firmware/sensors.cpp",  "__firmware_sensors__"),
            ("firmware/sensors.h",    "__firmware_sensors_h__"),
            ("firmware/comms.cpp",    "__firmware_comms__"),
            ("docs/pinout.md",        "__pinout_doc__"),
            ("platformio.ini",        "__platformio__"),
            (".gitignore",            "__gitignore__"),
        ],

        "Robótica": [
            ("README.md",             "__readme__"),
            ("firmware/main.cpp",     "__firmware_main__"),
            ("firmware/config.h",     "__firmware_config__"),
            ("firmware/motors.cpp",   "__firmware_motors__"),
            ("firmware/sensors.cpp",  "__firmware_sensors__"),
            ("firmware/navigation.cpp","__firmware_nav__"),
            ("docs/hardware.md",      "__hw_doc__"),
            ("docs/wiring.md",        "__wiring_doc__"),
            ("platformio.ini",        "__platformio__"),
            (".gitignore",            "__gitignore__"),
        ],

        "IoT / Nuvem": [
            ("README.md",             "__readme__"),
            ("device/main.cpp",       "__firmware_main__"),
            ("device/config.h",       "__firmware_config__"),
            ("device/mqtt.cpp",       "__iot_mqtt__"),
            ("backend/server.py",     "__iot_backend__"),
            ("backend/requirements.txt","__requirements_iot__"),
            ("dashboard/index.html",  "__iot_dashboard__"),
            ("docs/architecture.md",  "__arch_doc__"),
            (".gitignore",            "__gitignore__"),
        ],

        "Pesquisa": [
            ("README.md",             "__readme__"),
            ("paper/draft.md",        "__paper_draft__"),
            ("experiments/run.py",    "__exp_run__"),
            ("experiments/config.py", "__exp_config__"),
            ("data/.gitkeep",         ""),
            ("results/.gitkeep",      ""),
            ("requirements.txt",      "__requirements_ai__"),
            ("docs/methodology.md",   "__methodology__"),
        ],

        "Infraestrutura": [
            ("README.md",             "__readme__"),
            ("docker-compose.yml",    "__docker_compose__"),
            ("Dockerfile",            "__dockerfile__"),
            ("config/nginx.conf",     "__nginx__"),
            ("scripts/setup.sh",      "__setup_sh__"),
            ("scripts/deploy.sh",     "__deploy_sh__"),
            ("docs/runbook.md",       "__runbook__"),
            (".gitignore",            "__gitignore__"),
        ],
    }

    # Fallback genérico para tipos não mapeados
    GENERIC_TEMPLATE = [
        ("README.md",   "__readme__"),
        ("docs/spec.md","__spec_doc__"),
        (".gitignore",  "__gitignore__"),
    ]

    def __init__(self):
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def generate(self, project: Dict) -> str:
        """
        Gera todos os arquivos do projeto e empacota em .zip.
        Retorna o path do .zip gerado.
        """
        name     = re.sub(r'[^\w\-]', '_', project.get("name", "Projeto"))
        ptype    = project.get("type", "Genérico")
        priority = project.get("priority", "BETA")
        desc     = project.get("description", "")
        pid      = project.get("id", "0001")
        created  = datetime.now().strftime("%d/%m/%Y")

        template = self.TEMPLATES.get(ptype, self.GENERIC_TEMPLATE)
        ctx = {
            "name": name, "type": ptype, "priority": priority,
            "desc": desc, "pid": pid, "created": created,
            "author": "Cláudio Francisco Dos Santos Júnior",
            "org": "CFSJ TECH / Stark Industries",
        }

        zip_path = self.OUTPUT_DIR / f"{name}_{pid}.zip"
        with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_DEFLATED) as zf:
            for rel_path, template_key in template:
                content = self._render(template_key, ctx)
                zf.writestr(f"{name}/{rel_path}", content)

        print(f"[ProjectManager] Generated -> {zip_path}")
        return str(zip_path)

    # ─── Template Renderer ────────────────────────────────────────────────────

    def _render(self, key: str, ctx: Dict) -> str:
        if not key:
            return ""
        fn = getattr(self, f"_tpl{key}", None)
        return fn(ctx) if fn else f"# {key}\n"

    def _tpl__readme__(self, c):
        return f"""# {c['name']}
> **Tipo:** {c['type']} | **Prioridade:** {c['priority']} | **ID:** {c['pid']}

## Descrição
{c['desc'] or 'Descrição do projeto a ser preenchida.'}

## Equipe
- **Responsável:** {c['author']}
- **Organização:** {c['org']}
- **Criado em:** {c['created']}

## Estrutura do Projeto
```
{c['name']}/
├── README.md
├── docs/
├── src/ (ou firmware/)
└── tests/
```

## Como Usar
```bash
# Clone o repositório
git clone <url>
cd {c['name']}

# Instale as dependências
pip install -r requirements.txt   # Python
# ou: pio run                     # PlatformIO (Embarcado)
```

## Status
- [ ] Levantamento de requisitos
- [ ] Prototipagem
- [ ] Desenvolvimento
- [ ] Testes
- [ ] Validação final

---
*Gerado por J.A.R.V.I.S — Stark Industries © 2026*
"""

    def _tpl__gitignore__(self, c):
        return """# Python
__pycache__/
*.pyc
*.pyo
.venv/
venv/
*.egg-info/
dist/
build/

# C/C++
*.o
*.elf
*.bin
*.hex
.pio/

# Data
data/
*.db
*.sqlite
*.log

# Secrets
.env
secrets.json
config.local.*

# IDE
.vscode/
.idea/
*.swp
"""

    def _tpl__requirements_ai__(self, c):
        return """# Core ML
torch>=2.0.0
numpy>=1.26.0
scipy>=1.13.0

# Data
pandas>=2.2.0
scikit-learn>=1.4.0

# Visualization
matplotlib>=3.8.0
seaborn>=0.13.0

# Utils
tqdm>=4.66.0
pydantic>=2.7.0
python-dotenv>=1.0.0
loguru>=0.7.0

# API (optional)
fastapi>=0.111.0
uvicorn[standard]>=0.29.0
"""

    def _tpl__main_ai__(self, c):
        return f'''"""
{c['name']} — Main Entry Point
Projeto: {c['type']}
Criado:  {c['created']}
Autor:   {c['author']}
"""

import argparse
from pathlib import Path
from src.model  import Model
from src.data   import load_dataset
from src.train  import Trainer


def main():
    parser = argparse.ArgumentParser(description="{c['name']}")
    parser.add_argument("--mode",   choices=["train","eval","infer"], default="train")
    parser.add_argument("--data",   default="data/",   help="Diretório de dados")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr",     type=float, default=3e-4)
    parser.add_argument("--ckpt",   default="checkpoints/model.pt")
    args = parser.parse_args()

    print(f"[{c['name']}] Modo: {{args.mode}}")

    if args.mode == "train":
        dataset = load_dataset(args.data)
        model   = Model()
        trainer = Trainer(model, dataset, lr=args.lr)
        trainer.train(epochs=args.epochs, ckpt_path=args.ckpt)

    elif args.mode == "eval":
        model = Model.load(args.ckpt)
        # TODO: avaliação


if __name__ == "__main__":
    main()
'''

    def _tpl__model_ai__(self, c):
        return f'''"""
src/model.py — Definição do Modelo
{c['name']} — {c['type']}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class ModelConfig:
    input_dim:  int   = 128
    hidden_dim: int   = 256
    output_dim: int   = 10
    dropout:    float = 0.1
    num_layers: int   = 3


class Model(nn.Module):
    """Modelo principal para {c['name']}."""

    def __init__(self, cfg: ModelConfig = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        cfg = self.cfg

        layers = []
        in_dim = cfg.input_dim
        for _ in range(cfg.num_layers):
            layers += [
                nn.Linear(in_dim, cfg.hidden_dim),
                nn.LayerNorm(cfg.hidden_dim),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
            ]
            in_dim = cfg.hidden_dim
        layers.append(nn.Linear(cfg.hidden_dim, cfg.output_dim))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def save(self, path: str):
        torch.save({{"state_dict": self.state_dict(), "cfg": self.cfg}}, path)

    @classmethod
    def load(cls, path: str) -> "Model":
        data  = torch.load(path, map_location="cpu")
        model = cls(data["cfg"])
        model.load_state_dict(data["state_dict"])
        return model
'''

    def _tpl__data_ai__(self, c):
        return f'''"""src/data.py — Dataset e Pipeline de Dados — {c['name']}"""
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class ProjectDataset(Dataset):
    def __init__(self, data_dir: str, split: str = "train"):
        self.path  = Path(data_dir) / split
        self.items = []   # TODO: carregar dados
        print(f"[Dataset] {{len(self.items)}} amostras - {{split}}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        x, y = self.items[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def load_dataset(data_dir: str, batch_size: int = 32):
    train = ProjectDataset(data_dir, "train")
    val   = ProjectDataset(data_dir, "val")
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True),
        DataLoader(val,   batch_size=batch_size),
    )
'''

    def _tpl__train_ai__(self, c):
        return f'''"""src/train.py — Loop de Treinamento — {c['name']}"""
import torch
import torch.nn as nn
from tqdm import tqdm


class Trainer:
    def __init__(self, model, dataset, lr: float = 3e-4):
        self.model   = model
        self.train_loader, self.val_loader = dataset
        self.optim   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
        self.loss_fn = nn.CrossEntropyLoss()
        self.history = []

    def train(self, epochs: int = 10, ckpt_path: str = "model.pt"):
        for epoch in range(1, epochs + 1):
            train_loss = self._run_epoch(self.train_loader, train=True)
            val_loss   = self._run_epoch(self.val_loader,   train=False)
            self.history.append({{"epoch": epoch, "train": train_loss, "val": val_loss}})
            print(f"Epoch {{epoch:3d}} | train={{train_loss:.4f}} val={{val_loss:.4f}}")
        self.model.save(ckpt_path)

    def _run_epoch(self, loader, train: bool) -> float:
        self.model.train(train)
        total = 0.0
        with torch.set_grad_enabled(train):
            for x, y in tqdm(loader, leave=False):
                pred = self.model(x)
                loss = self.loss_fn(pred, y)
                if train:
                    self.optim.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optim.step()
                total += loss.item()
        return total / len(loader)
'''

    def _tpl__test_ai__(self, c):
        return f'''"""tests/test_model.py — Testes Unitários — {c['name']}"""
import torch
import pytest
from src.model import Model, ModelConfig


def test_model_forward():
    cfg   = ModelConfig(input_dim=32, hidden_dim=64, output_dim=4)
    model = Model(cfg)
    x     = torch.randn(8, 32)
    out   = model(x)
    assert out.shape == (8, 4), f"Shape esperado (8,4), obtido {{out.shape}}"


def test_model_save_load(tmp_path):
    model = Model()
    path  = str(tmp_path / "model.pt")
    model.save(path)
    loaded = Model.load(path)
    x      = torch.randn(2, 128)
    assert torch.allclose(model(x), loaded(x), atol=1e-5)


def test_no_nan():
    model = Model()
    x = torch.randn(16, 128)
    out = model(x)
    assert not torch.isnan(out).any()
'''

    def _tpl__firmware_main__(self, c):
        return f"""/**
 * @file    main.cpp
 * @brief   {c['name']} — Firmware Principal
 * @author  {c['author']}
 * @date    {c['created']}
 * @version 1.0.0
 *
 * Plataforma: ESP32 (RoboCore Vespa)
 * Framework:  Arduino / FreeRTOS
 */

#include <Arduino.h>
#include <Wire.h>
#include "config.h"
#include "sensors.h"

// ─── FreeRTOS Task Handles ────────────────────────────────────────────────────
TaskHandle_t hTaskSensors  = nullptr;
TaskHandle_t hTaskControl  = nullptr;
TaskHandle_t hTaskComms    = nullptr;

// ─── Task: Sensoriamento ──────────────────────────────────────────────────────
void vTaskSensors(void* pvParam) {{
    TickType_t xLastWake = xTaskGetTickCount();
    for (;;) {{
        sensors_read_all();
        vTaskDelayUntil(&xLastWake, pdMS_TO_TICKS(SENSOR_PERIOD_MS));
    }}
}}

// ─── Task: Controle ───────────────────────────────────────────────────────────
void vTaskControl(void* pvParam) {{
    for (;;) {{
        // TODO: lógica de controle
        vTaskDelay(pdMS_TO_TICKS(50));
    }}
}}

// ─── Task: Comunicação ────────────────────────────────────────────────────────
void vTaskComms(void* pvParam) {{
    for (;;) {{
        // TODO: WebSocket / MQTT / HTTP
        vTaskDelay(pdMS_TO_TICKS(100));
    }}
}}

// ─── Setup ────────────────────────────────────────────────────────────────────
void setup() {{
    Serial.begin(SERIAL_BAUD);
    Wire.begin(PIN_SDA, PIN_SCL);
    Serial.println("[{c['name']}] Boot OK");

    sensors_init();

    xTaskCreatePinnedToCore(vTaskSensors, "Sensors", 4096, nullptr, 3, &hTaskSensors, 0);
    xTaskCreatePinnedToCore(vTaskControl, "Control", 8192, nullptr, 2, &hTaskControl, 1);
    xTaskCreatePinnedToCore(vTaskComms,   "Comms",   8192, nullptr, 1, &hTaskComms,   1);
}}

void loop() {{
    // Todas as tarefas rodam via FreeRTOS
    vTaskDelay(portMAX_DELAY);
}}
"""

    def _tpl__firmware_config__(self, c):
        return f"""/**
 * config.h — Configurações Globais — {c['name']}
 */
#pragma once

// ─── Serial ──────────────────────────────────────────────────────────────────
#define SERIAL_BAUD         115200

// ─── I2C ─────────────────────────────────────────────────────────────────────
#define PIN_SDA             21
#define PIN_SCL             22

// ─── Sensores ────────────────────────────────────────────────────────────────
#define SENSOR_PERIOD_MS    200     // 5 Hz
#define PIN_MQ02_AO         34      // ADC1_CH6 (somente leitura)
#define MQ02_THRESHOLD      300     // valor ADC de alerta

// ─── Atuadores ───────────────────────────────────────────────────────────────
#define PIN_FAN             25
#define PIN_BUZZER          26
#define PIN_LED_STATUS      2

// ─── WiFi ────────────────────────────────────────────────────────────────────
#define WIFI_SSID           "SUA_REDE"
#define WIFI_PASS           "SUA_SENHA"

// ─── Timing ──────────────────────────────────────────────────────────────────
#define WATCHDOG_TIMEOUT_MS 5000
"""

    def _tpl__firmware_sensors__(self, c):
        return f"""/**
 * sensors.cpp — Drivers de Sensores — {c['name']}
 */
#include "sensors.h"
#include <Wire.h>
#include <Arduino.h>

SensorData g_sensors = {{}};

void sensors_init() {{
    // AHT10
    // aht.begin();
    Serial.println("[Sensors] Initialized");
}}

void sensors_read_all() {{
    // Leitura MQ-02
    g_sensors.gas_raw = analogRead(PIN_MQ02_AO);
    g_sensors.gas_alert = g_sensors.gas_raw > MQ02_THRESHOLD;

    // Leitura AHT10 (I2C)
    // TempAndHumidity th = aht.readTempAndHumidity();
    // g_sensors.temp = th.Temperature;
    // g_sensors.humidity = th.Humidity;
}}
"""

    def _tpl__firmware_sensors_h__(self, c):
        return """/**
 * sensors.h — Declarações de Sensores
 */
#pragma once
#include <stdint.h>
#include <stdbool.h>
#include "config.h"

struct SensorData {
    float    temp;
    float    humidity;
    uint16_t gas_raw;
    bool     gas_alert;
};

extern SensorData g_sensors;

void sensors_init();
void sensors_read_all();
"""

    def _tpl__firmware_comms__(self, c):
        return f"""/**
 * comms.cpp — Comunicação WiFi/WebSocket — {c['name']}
 */
#include "config.h"
#include <WiFi.h>
#include <WebSocketsServer.h>
#include <ArduinoJson.h>

WebSocketsServer ws(81);

void comms_init() {{
    WiFi.begin(WIFI_SSID, WIFI_PASS);
    while (WiFi.status() != WL_CONNECTED) {{
        delay(500);
    }}
    Serial.print("[Comms] IP: ");
    Serial.println(WiFi.localIP());
    ws.begin();
}}

void comms_send_sensors(const SensorData& s) {{
    StaticJsonDocument<128> doc;
    doc["temp"]      = s.temp;
    doc["humidity"]  = s.humidity;
    doc["gas_raw"]   = s.gas_raw;
    doc["gas_alert"] = s.gas_alert;
    String out;
    serializeJson(doc, out);
    ws.broadcastTXT(out);
}}
"""

    def _tpl__firmware_motors__(self, c):
        return f"""/**
 * motors.cpp — Controle de Motores — {c['name']}
 * Driver: RoboCore DinoV2 (ou ponte H TB6612)
 */
#include "config.h"
#include <Arduino.h>

#define PIN_M_LEFT_PWM   18
#define PIN_M_LEFT_DIR   19
#define PIN_M_RIGHT_PWM  5
#define PIN_M_RIGHT_DIR  4
#define PWM_CHANNEL_L    0
#define PWM_CHANNEL_R    1
#define PWM_FREQ         20000
#define PWM_BITS         8

void motors_init() {{
    ledcSetup(PWM_CHANNEL_L, PWM_FREQ, PWM_BITS);
    ledcSetup(PWM_CHANNEL_R, PWM_FREQ, PWM_BITS);
    ledcAttachPin(PIN_M_LEFT_PWM,  PWM_CHANNEL_L);
    ledcAttachPin(PIN_M_RIGHT_PWM, PWM_CHANNEL_R);
    pinMode(PIN_M_LEFT_DIR,  OUTPUT);
    pinMode(PIN_M_RIGHT_DIR, OUTPUT);
}}

/**
 * @param left   -255 a +255 (negativo = ré)
 * @param right  -255 a +255
 */
void motors_set(int left, int right) {{
    digitalWrite(PIN_M_LEFT_DIR,  left  >= 0 ? HIGH : LOW);
    digitalWrite(PIN_M_RIGHT_DIR, right >= 0 ? HIGH : LOW);
    ledcWrite(PWM_CHANNEL_L, abs(constrain(left,  -255, 255)));
    ledcWrite(PWM_CHANNEL_R, abs(constrain(right, -255, 255)));
}}

void motors_stop()    {{ motors_set(0, 0); }}
void motors_forward(int v)  {{ motors_set(v, v);   }}
void motors_backward(int v) {{ motors_set(-v, -v); }}
void motors_turn_left(int v)  {{ motors_set(-v, v); }}
void motors_turn_right(int v) {{ motors_set(v, -v); }}
"""

    def _tpl__firmware_nav__(self, c):
        return f"""/**
 * navigation.cpp — Navegação Autônoma — {c['name']}
 * Algoritmo: wall-following + obstacle avoidance com LiDAR LD14P
 */
#include "config.h"
#include <Arduino.h>

#define LIDAR_SERIAL        Serial2
#define LIDAR_BAUD          230400
#define SAFE_DIST_MM        400
#define TURN_SPEED          150
#define FWD_SPEED           200

enum NavState {{ NAV_FORWARD, NAV_TURN_LEFT, NAV_TURN_RIGHT, NAV_STOP }};
static NavState s_state = NAV_FORWARD;

// Distâncias LiDAR por setor (mm), 0=frente, 90=direita, 180=trás, 270=esquerda
static float s_dist[360] = {{}};

void nav_update_lidar(uint16_t angle_deg, float dist_mm) {{
    if (angle_deg < 360) s_dist[angle_deg] = dist_mm;
}}

float nav_sector_min(int center, int half_width) {{
    float mn = 9999;
    for (int a = center - half_width; a <= center + half_width; a++) {{
        float d = s_dist[(a + 360) % 360];
        if (d > 10 && d < mn) mn = d;
    }}
    return mn;
}}

NavState nav_decide() {{
    float front = nav_sector_min(0,   30);
    float left  = nav_sector_min(270, 30);
    float right = nav_sector_min(90,  30);
    if (front < SAFE_DIST_MM) {{
        return (left > right) ? NAV_TURN_LEFT : NAV_TURN_RIGHT;
    }}
    return NAV_FORWARD;
}}
"""

    def _tpl__platformio__(self, c):
        return f"""[env:vespa]
platform  = espressif32
board     = esp32dev
framework = arduino
monitor_speed = 115200
build_flags =
    -DCORE_DEBUG_LEVEL=1

lib_deps =
    ArduinoJson@^7.0.0
    bblanchon/ArduinoJson
    links2004/WebSockets@^2.4.0
    adafruit/Adafruit AHTX0@^2.0.5

; {c['name']} — {c['created']}
"""

    def _tpl__arch_doc__(self, c):
        return f"""# Arquitetura do Sistema — {c['name']}

## Visão Geral
{c['desc'] or 'Descrição da arquitetura aqui.'}

## Diagrama de Componentes
```
┌─────────────────────────────────────────────┐
│              {c['name']}                     │
├──────────┬──────────────┬───────────────────┤
│ Sensores │  Processador │    Atuadores       │
│  MQ-02   │  ESP32 Vespa │  Fan 12V           │
│  AHT10   │  FreeRTOS    │  Sirene 105dB      │
│  LiDAR   │  WiFi/BLE    │  LEDs              │
└──────────┴──────────────┴───────────────────┘
          ↕ WebSocket / MQTT
┌─────────────────────────────────────────────┐
│            Backend (Python/FastAPI)          │
│            Banco de Dados (SQLite/Firebase)  │
│            Dashboard Web / App Mobile        │
└─────────────────────────────────────────────┘
```

## Fluxo de Dados
1. Sensores → ESP32 (leitura a 5Hz via FreeRTOS)
2. ESP32 → Backend via WebSocket/MQTT
3. Backend → Banco de Dados (persistência)
4. Backend → Dashboard (visualização em tempo real)

## Stack Tecnológico
| Camada     | Tecnologia                  |
|------------|-----------------------------|
| Firmware   | C++ / Arduino / FreeRTOS    |
| Hardware   | ESP32 RoboCore Vespa        |
| Backend    | Python / FastAPI            |
| Banco      | SQLite / Firebase           |
| Frontend   | HTML5 / JavaScript          |

---
*{c['org']} — {c['created']}*
"""

    def _tpl__hw_doc__(self, c):
        return f"""# Hardware — {c['name']}

## Lista de Componentes
| Componente                | Modelo / Ref         | Qtd |
|---------------------------|----------------------|-----|
| Microcontrolador          | RoboCore Vespa (ESP32)| 1  |
| Sensor LiDAR              | WayPonDEV LD14P      | 1   |
| Sensor Gás                | MQ-02                | 1   |
| Sensor Temp/Umidade       | AHT10                | 1   |
| Fan Brushless             | 12V                  | 1   |
| Sirene Piezoelétrica      | Intelbras SIR 1000   | 1   |
| Chassi                    | RoboCore Rocket Tank | 1   |
| Driver Motores            | RoboCore DinoV2      | 1   |
| Bateria Li-Ion 18650      | Suporte 4x           | 1   |

## Pinout ESP32 (RoboCore Vespa)
| GPIO | Função            | Componente   |
|------|-------------------|--------------|
| 34   | ADC (somente ent) | MQ-02 AO     |
| 21   | SDA               | AHT10 / I2C  |
| 22   | SCL               | AHT10 / I2C  |
| 16   | RX2               | LiDAR LD14P  |
| 17   | TX2               | LiDAR LD14P  |
| 18   | PWM Motor E       | DinoV2       |
| 19   | DIR Motor E       | DinoV2       |
| 5    | PWM Motor D       | DinoV2       |
| 4    | DIR Motor D       | DinoV2       |
| 25   | Fan Control       | Fan 12V      |
| 26   | Buzzer            | Sirene       |

---
*{c['org']} — {c['created']}*
"""

    def _tpl__wiring_doc__(self, c):
        return f"""# Esquema de Ligação — {c['name']}

## Alimentação
```
Bateria 4x 18650 (~16.8V)
  └── VBAT ──► DinoV2 (Vin)
              └── 5V reg ──► ESP32 Vespa (5V)
              └── 12V ──────► Fan Brushless
              └── Motor L ──► Esteira Esquerda
              └── Motor R ──► Esteira Direita
```

## I2C Bus (3.3V)
```
ESP32 SDA (21) ──► AHT10 SDA
ESP32 SCL (22) ──► AHT10 SCL
                  Pull-up 4.7kΩ para 3.3V
```

## LiDAR LD14P (UART2)
```
ESP32 RX2 (16) ──► LD14P TX
ESP32 TX2 (17) ──► LD14P RX (cmd)
ESP32 5V  ──────► LD14P VCC
ESP32 GND ──────► LD14P GND
```

## MQ-02 (ADC)
```
3.3V ──► MQ-02 VCC
GND  ──► MQ-02 GND
MQ-02 AO ──► GPIO 34 (ADC, divisor 1/2 se necessário)
```

---
*Nota: Verificar tensões antes de conectar. ESP32 I/O: 3.3V.*
"""

    def _tpl__iot_mqtt__(self, c):
        return f"""/**
 * mqtt.cpp — Cliente MQTT — {c['name']}
 */
#include "config.h"
#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>

#ifndef MQTT_HOST
#define MQTT_HOST   "127.0.0.1"
#endif
#ifndef MQTT_PORT
#define MQTT_PORT   1883
#endif
#define MQTT_TOPIC  "{c['name'].lower()}/sensors"

WiFiClient   wifiClient;
PubSubClient mqtt(wifiClient);

void mqtt_connect() {{
    while (!mqtt.connected()) {{
        mqtt.connect("{c['name']}_device");
        delay(500);
    }}
}}

void mqtt_publish_sensors(float temp, float hum, int gas) {{
    if (!mqtt.connected()) mqtt_connect();
    StaticJsonDocument<128> doc;
    doc["temp"]  = temp;
    doc["hum"]   = hum;
    doc["gas"]   = gas;
    char buf[128];
    serializeJson(doc, buf);
    mqtt.publish(MQTT_TOPIC, buf);
    mqtt.loop();
}}
"""

    def _tpl__iot_backend__(self, c):
        return f'''"""
backend/server.py — Backend IoT — {c['name']}
"""
import asyncio
import json
import os
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import paho.mqtt.client as mqtt

app = FastAPI(title="{c['name']}")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

ws_clients = []
latest_data = {{}}

# MQTT
def on_message(client, userdata, msg):
    global latest_data
    latest_data = json.loads(msg.payload)
    asyncio.run_coroutine_threadsafe(broadcast(latest_data), loop)

mqtt_client = mqtt.Client()
mqtt_client.on_message = on_message
MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
mqtt_client.connect(MQTT_HOST, MQTT_PORT)
mqtt_client.subscribe("{c['name'].lower()}/sensors")
mqtt_client.loop_start()

async def broadcast(data):
    for ws in ws_clients:
        try:
            await ws.send_json(data)
        except Exception:
            ws_clients.remove(ws)

@app.get("/api/latest")
async def latest():
    return latest_data

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    ws_clients.append(ws)
    await ws.send_json(latest_data)
    try:
        while True:
            await ws.receive_text()
    except Exception:
        ws_clients.remove(ws)

if __name__ == "__main__":
    import uvicorn
    loop = asyncio.get_event_loop()
    uvicorn.run(app, host="0.0.0.0", port=8001)
'''

    def _tpl__iot_dashboard__(self, c):
        return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<title>{c['name']} — Dashboard</title>
<style>
  body {{ font-family: monospace; background:#07131d; color:#f8fafc; padding:20px; }}
  h1   {{ letter-spacing:4px; margin-bottom:20px; }}
  .cards {{ display:flex; gap:16px; flex-wrap:wrap; margin-bottom:24px; }}
  .card  {{ background:#0f2d40; border:1px solid rgba(34,211,238,.42); padding:16px; min-width:140px; text-align:center; }}
  .card-val {{ font-size:28px; font-weight:bold; color:#22d3ee; }}
  .card-lbl {{ font-size:11px; color:#dbeafe; margin-top:4px; letter-spacing:2px; }}
  canvas {{ background:#0f2d40; border:1px solid rgba(241,136,33,.45); border-radius:4px; }}
</style>
</head>
<body>
<h1>⬡ {c['name'].upper()}</h1>
<div class="cards">
  <div class="card"><div class="card-val" id="temp">--</div><div class="card-lbl">TEMP °C</div></div>
  <div class="card"><div class="card-val" id="hum">--</div><div class="card-lbl">UMIDADE %</div></div>
  <div class="card"><div class="card-val" id="gas">--</div><div class="card-lbl">GÁS ADC</div></div>
  <div class="card"><div class="card-val" id="conn" style="color:#00e676">●</div><div class="card-lbl">STATUS</div></div>
</div>
<canvas id="chart" width="600" height="200"></canvas>
<script>
const ws = new WebSocket('ws://localhost:8001/ws');
const data = {{ temp:[], hum:[], labels:[] }};
const canvas = document.getElementById('chart');
const ctx = canvas.getContext('2d');
function drawSeries(values, color, max) {{
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  values.forEach((value, index) => {{
    const x = 30 + index * ((canvas.width - 45) / Math.max(1, values.length - 1));
    const y = canvas.height - 20 - ((value || 0) / Math.max(1, max)) * (canvas.height - 40);
    if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }});
  ctx.stroke();
}}
function drawChart() {{
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = 'rgba(34,211,238,.12)';
  for (let i = 0; i <= 4; i++) {{
    const y = 12 + i * ((canvas.height - 32) / 4);
    ctx.beginPath(); ctx.moveTo(30, y); ctx.lineTo(canvas.width - 10, y); ctx.stroke();
  }}
  const max = Math.max(100, ...data.temp, ...data.hum);
  drawSeries(data.temp, '#22d3ee', max);
  drawSeries(data.hum, '#f18821', max);
}}
const chart = {{ update: drawChart }};
ws.onmessage = e => {{
  const d = JSON.parse(e.data);
  document.getElementById('temp').textContent = (d.temp||0).toFixed(1);
  document.getElementById('hum').textContent  = (d.hum||0).toFixed(1);
  document.getElementById('gas').textContent  = d.gas || '--';
  data.labels.push(new Date().toLocaleTimeString());
  data.temp.push(d.temp||0); data.hum.push(d.hum||0);
  if (data.labels.length > 30) {{ data.labels.shift(); data.temp.shift(); data.hum.shift(); }}
  chart.update();
}};
</script>
</body>
</html>
"""

    def _tpl__paper_draft__(self, c):
        return f"""# {c['name']} — Rascunho do Paper

**Autores:** {c['author']}
**Data:** {c['created']}

## Abstract
*Escreva aqui um resumo de 150-250 palavras.*

## 1. Introdução
{c['desc'] or '...'}

## 2. Trabalhos Relacionados
...

## 3. Metodologia
### 3.1 Dataset
### 3.2 Modelo
### 3.3 Treinamento

## 4. Experimentos
### 4.1 Configuração
### 4.2 Resultados

| Método   | Métrica 1 | Métrica 2 |
|----------|-----------|-----------|
| Baseline | --        | --        |
| Proposto | --        | --        |

## 5. Conclusão
...

## Referências
1. ...
"""

    def _tpl__exp_run__(self, c):
        return f'''"""experiments/run.py — Runner de Experimentos — {c['name']}"""
import json
from pathlib import Path
from datetime import datetime


def run_experiment(config: dict) -> dict:
    """Executa um experimento com o config fornecido e retorna métricas."""
    print(f"[Exp] Iniciando: {{config}}")
    results = {{"config": config, "metrics": {{}}, "ts": datetime.now().isoformat()}}
    # TODO: implementar experimento
    return results


if __name__ == "__main__":
    from exp_config import EXPERIMENTS
    all_results = []
    for cfg in EXPERIMENTS:
        res = run_experiment(cfg)
        all_results.append(res)
        print(f"  -> {{res['metrics']}}")
    out = Path("results") / f"run_{{datetime.now().strftime('%Y%m%d_%H%M')}}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(all_results, indent=2))
    print(f"[Exp] Resultados -> {{out}}")
'''

    def _tpl__exp_config__(self, c):
        return f"""# experiments/config.py — Configurações de Experimentos — {c['name']}

EXPERIMENTS = [
    {{"name": "baseline",  "lr": 1e-3, "epochs": 10, "hidden": 128}},
    {{"name": "exp_large", "lr": 3e-4, "epochs": 20, "hidden": 256}},
    {{"name": "exp_deep",  "lr": 1e-4, "epochs": 30, "hidden": 512}},
]
"""

    def _tpl__methodology__(self, c):
        return f"""# Metodologia — {c['name']}

## Hipótese de Pesquisa
...

## Protocolo Experimental
1. Coleta de dados
2. Pré-processamento
3. Treinamento do modelo
4. Avaliação
5. Análise estatística

## Métricas de Avaliação
- Métrica 1: ...
- Métrica 2: ...

## Baseline
...

## Critério de Parada
...

---
*{c['org']} — {c['created']}*
"""

    def _tpl__docker_compose__(self, c):
        return f"""version: '3.9'
services:
  backend:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
    environment:
      - ENV=production
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./config/nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - backend
    restart: unless-stopped

# {c['name']} — {c['created']}
"""

    def _tpl__dockerfile__(self, c):
        return f"""FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "backend.server:app", "--host", "0.0.0.0", "--port", "8000"]
# {c['name']} — {c['author']}
"""

    def _tpl__nginx__(self, c):
        return """events { worker_connections 1024; }
http {
  server {
    listen 80;
    location /api { proxy_pass http://backend:8000; }
    location /ws  {
      proxy_pass http://backend:8000;
      proxy_http_version 1.1;
      proxy_set_header Upgrade $http_upgrade;
      proxy_set_header Connection "upgrade";
    }
    location / { root /usr/share/nginx/html; try_files $uri /index.html; }
  }
}
"""

    def _tpl__setup_sh__(self, c):
        return f"""#!/bin/bash
# setup.sh — Setup inicial — {c['name']}
set -e
echo "[Setup] Iniciando {c['name']}..."
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdir -p data logs
echo "[Setup] OK. Execute: source .venv/bin/activate && python backend/server.py"
"""

    def _tpl__deploy_sh__(self, c):
        return f"""#!/bin/bash
# deploy.sh — Deploy via Docker — {c['name']}
set -e
echo "[Deploy] {c['name']}..."
docker-compose down
docker-compose build --no-cache
docker-compose up -d
echo "[Deploy] Serviços ativos:"
docker-compose ps
"""

    def _tpl__runbook__(self, c):
        return f"""# Runbook — {c['name']}

## Deploy
```bash
chmod +x scripts/setup.sh scripts/deploy.sh
./scripts/setup.sh
./scripts/deploy.sh
```

## Monitoramento
```bash
docker-compose logs -f backend
```

## Rollback
```bash
docker-compose down
git checkout HEAD~1
./scripts/deploy.sh
```

## Troubleshooting
| Sintoma | Causa Provável | Solução |
|---------|---------------|---------|
| 502 Bad Gateway | Backend offline | `docker-compose restart backend` |
| DB locked | Múltiplas escritas | Verificar WAL mode |

---
*{c['org']} — {c['created']}*
"""

    def _tpl__pinout_doc__(self, c):
        return self._tpl__hw_doc__(c)

    def _tpl__spec_doc__(self, c):
        return f"""# Especificação Técnica — {c['name']}

## Requisitos Funcionais
- RF01: ...
- RF02: ...

## Requisitos Não-Funcionais
- RNF01: Latência < 200ms
- RNF02: Disponibilidade > 99%

## Restrições
- Plataforma: ...
- Linguagem: ...

## Casos de Uso
### UC01: ...

---
*{c['org']} — {c['created']}*
"""

    def _tpl__requirements_iot__(self, c):
        return """fastapi>=0.111.0
uvicorn[standard]>=0.29.0
paho-mqtt>=2.0.0
pydantic>=2.7.0
"""

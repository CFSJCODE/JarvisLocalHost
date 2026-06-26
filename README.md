# J.A.R.V.I.S. LocalHost

J.A.R.V.I.S. LocalHost e um assistente de IA local servido por FastAPI, com HUD web, WebSocket para telemetria, ingestao de PDFs, RAG, treino local em PyTorch, memoria em SQLite e modulos opcionais de voz/offload em rede local.

> Status: projeto em desenvolvimento. Algumas funcionalidades podem apresentar bugs, erros, resultados instaveis ou depender de componentes locais ainda nao configurados, como Tesseract, PyTorch, drivers de GPU, voz offline ou workers LAN.

## Principais Recursos

- Backend FastAPI com endpoints REST e WebSocket.
- Frontend local em `static/index.html`.
- Pipeline de upload e processamento de PDFs com extracao de texto, tabelas, imagens e OCR.
- RAG local com corpus em `data/embeddings`.
- Transformer decoder-only e tokenizador proprio em `neural.py`.
- Banco SQLite local para historico, documentos, metricas, projetos e treinamento.
- Curiosidade autonoma para gerar insights a partir do corpus local.
- Monitoramento de CPU, RAM, disco e rede.
- Voz offline opcional via `pyttsx3`/SAPI.
- Offload opcional para workers Aether em `localhost` ou LAN.

## Estrutura Do Repositorio

```text
.
├── app.py                    # Servidor FastAPI e rotas HTTP/WebSocket
├── brain.py                  # Orquestrador central do Jarvis
├── neural.py                 # Modelo, tokenizador, treino, RAG e curiosidade
├── database.py               # Persistencia SQLite
├── pdf_processor.py          # Processamento de PDFs e OCR
├── project_manager.py        # Geracao de projetos locais
├── cluster_client.py         # Cliente de offload LAN opcional
├── local_voice.py            # Voz offline opcional
├── system_monitor.py         # Telemetria local
├── static/index.html         # HUD principal
├── docs/                     # Documentacao tecnica e legado
├── scripts/                  # Automacao local de setup/run/validacao
├── data/                     # Estado local gerado em runtime, ignorado pelo Git
├── uploads/                  # Uploads locais, ignorados pelo Git
└── requirements.txt          # Dependencias Python
```

Os diretorios `data/` e `uploads/` sao mantidos no repositorio apenas com `.gitkeep`. Bancos, modelos, embeddings, PDFs enviados, imagens extraidas e checkpoints nao devem ser versionados.

## Requisitos

- Windows 10/11 ou Linux com Python compativel com as dependencias.
- Python 3.10 ou 3.11 recomendado.
- Tesseract OCR instalado no sistema para OCR de imagens em PDFs.
- GPU/CPU compatível com PyTorch caso o treino neural seja usado.

## Setup Rapido No Windows

```powershell
cd C:\caminho\para\JarvisLocalHost
.\scripts\setup.ps1
.\scripts\run.ps1
```

Depois abra:

```text
http://127.0.0.1:8000
```

Setup manual:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

## Validacao Local

```powershell
.\scripts\validate.ps1
```

A validacao atual verifica:

- arquivos essenciais do projeto;
- presenca do frontend principal;
- compilacao sintatica dos arquivos Python versionados;
- ausencia de artefatos de runtime rastreados em `data/` e `uploads/`, exceto `.gitkeep`.

## Variaveis De Ambiente

Copie `.env.example` para `.env` quando quiser customizar a execucao local:

```powershell
Copy-Item .env.example .env
```

Principais chaves:

- `JARVIS_CLUSTER_ENABLED`: habilita ou desabilita offload para cluster local/LAN.
- `JARVIS_CLUSTER_URL`: URL do worker/controlador Aether.
- `JARVIS_CLUSTER_ALLOWED_PREFIXES`: allowlist de comandos permitidos para offload.
- `JARVIS_CLUSTER_DEFAULT_TAGS`: tags usadas para selecionar workers.
- `JARVIS_VOICE_ENABLED`: habilita voz offline local.

## Endpoints Principais

- `GET /`
- `POST /api/chat`
- `POST /api/inferencia`
- `POST /api/pdf/upload`
- `POST /api/train/start`
- `GET /api/train/status`
- `GET /api/documents`
- `GET /api/curiosity/stats`
- `GET /api/curiosity/insights`
- `GET /api/curiosity/topics`
- `GET /api/metrics`
- `GET /api/metrics/history`
- `GET /api/projects`
- `POST /api/project/save`
- `WS /ws`

## Dados Locais E Privacidade

O projeto foi preparado para execucao local. Os dados sensiveis e gerados pelo usuario ficam fora do Git por padrao:

- `data/jarvis.db`
- `data/models/*.pt`
- `data/embeddings/*`
- `data/extracted_images/*`
- `data/curiosity/*`
- `data/projects/*`
- `uploads/*`

Se for necessario publicar um exemplo, use arquivos anonimizados e pequenos dentro de uma pasta de documentacao apropriada.

## Documentacao

- [Arquitetura](docs/ARCHITECTURE.md)
- [Desenvolvimento](docs/DEVELOPMENT.md)
- [Pagina HTML legada](docs/legacy/homepage.html)

## Licenca

Este projeto usa a licenca presente em `LICENSE`.

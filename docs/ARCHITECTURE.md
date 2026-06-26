# Arquitetura

O J.A.R.V.I.S. LocalHost e uma aplicacao local composta por um backend FastAPI, um HUD estatico e um conjunto de modulos Python responsaveis por memoria, processamento de documentos, RAG, treino neural e telemetria.

## Visao Geral

```text
static/index.html
        |
        | HTTP REST + WebSocket
        v
app.py  FastAPI
        |
        v
brain.py  JarvisBrain
        |
        +-- neural.py          Transformer, tokenizador, treino, RAG e curiosidade
        +-- pdf_processor.py   Extracao de PDF, tabelas, imagens e OCR
        +-- curiosity.py       Insights locais por corpus
        +-- database.py        SQLite: chat, docs, metricas, projetos e treino
        +-- system_monitor.py  CPU, RAM, disco e rede
        +-- project_manager.py Geracao de projetos locais
        +-- engine_ai.py       Inferencia direta local
        +-- cluster_client.py  Offload opcional para LAN
        +-- local_voice.py     Voz offline opcional
```

## Backend

`app.py` inicializa a API, configura CORS local, monta `/static`, cria os diretorios de runtime e expõe os endpoints HTTP/WebSocket. O servidor usa `JarvisBrain` como fachada principal para manter o contrato das rotas pequeno.

`brain.py` concentra o estado de execucao: memoria conversacional curta, modelo, tokenizador, RAG, treinamento, documentos, curiosidade, projetos e integrações opcionais. Esse modulo tambem faz o carregamento de modelos e tokenizer existentes quando os arquivos locais estao presentes.

## Dados De Runtime

Os caminhos abaixo sao gerados em execucao e ficam ignorados pelo Git:

- `data/jarvis.db`
- `data/models/`
- `data/embeddings/`
- `data/extracted_images/`
- `data/curiosity/`
- `data/projects/`
- `uploads/`

Essa separacao evita publicar PDFs, bancos locais, imagens extraidas, checkpoints e vetores de embeddings no GitHub.

## Frontend

O frontend principal e `static/index.html`. Ele conversa com o backend local por endpoints REST e recebe metricas via WebSocket em `/ws`.

`docs/legacy/homepage.html` foi preservado como referencia visual legada e nao participa da execucao principal.

## Modulos Opcionais

O offload por cluster fica desativado por padrao e deve ser habilitado por variaveis `JARVIS_CLUSTER_*`. A voz offline tambem fica desativada por padrao e usa `JARVIS_VOICE_ENABLED`.

Esses recursos devem falhar de forma controlada quando a dependencia local nao existir.

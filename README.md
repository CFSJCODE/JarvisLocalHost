# JarvisLocalHost — LLM + RAG Pipeline treinado do zero em documentos locais (Python · FastAPI · DirectML · PyTorch)

**Problema:** modelos de linguagem comerciais expõem dados privados a terceiros e não permitem controle total do corpus de treinamento.  
**Solução:** pipeline completo de NLP local — tokenizador BPE, Transformer LM, retriever contrastivo InfoNCE e RAG extrativo — treinados exclusivamente nos documentos do usuário, sem nenhum peso pré-treinado externo.  
**Resultado:** assistente soberano capaz de responder perguntas com recuperação de evidências sobre um corpus de 269 documentos e 233 k chunks, treinado integralmente na APU local.

---

## Tecnologias

### Linguagens

<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PowerShell-5391FE?style=flat-square&logo=powershell&logoColor=white" alt="PowerShell">
</p>

### IA & Deep Learning

<p>
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/DirectML-0078D4?style=flat-square&logo=microsoft&logoColor=white" alt="DirectML">
  <img src="https://img.shields.io/badge/Transformer_LM-custom-6C3483?style=flat-square" alt="Transformer LM">
  <img src="https://img.shields.io/badge/BPE_Tokenizer-custom-1A5276?style=flat-square" alt="BPE Tokenizer">
  <img src="https://img.shields.io/badge/InfoNCE_Retriever-contrastive-117A65?style=flat-square" alt="InfoNCE Retriever">
  <img src="https://img.shields.io/badge/RAG-extrativo-1F618D?style=flat-square" alt="RAG Extrativo">
</p>

### API & Backend

<p>
  <img src="https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Uvicorn-499848?style=flat-square&logo=gunicorn&logoColor=white" alt="Uvicorn">
  <img src="https://img.shields.io/badge/WebSocket-010101?style=flat-square&logo=socketdotio&logoColor=white" alt="WebSocket">
</p>

### Dados & Qualidade

<p>
  <img src="https://img.shields.io/badge/SQLite-003B57?style=flat-square&logo=sqlite&logoColor=white" alt="SQLite">
  <img src="https://img.shields.io/badge/pytest-0A9EDC?style=flat-square&logo=pytest&logoColor=white" alt="pytest">
  <img src="https://img.shields.io/badge/Conventional_Commits-FE5196?style=flat-square&logo=conventionalcommits&logoColor=white" alt="Conventional Commits">
</p>

### Plataforma

<p>
  <img src="https://img.shields.io/badge/Windows-0078D4?style=flat-square&logo=windows&logoColor=white" alt="Windows">
  <img src="https://img.shields.io/badge/Git-F05032?style=flat-square&logo=git&logoColor=white" alt="Git">
  <img src="https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white" alt="GitHub">
</p>

---

## O que este projeto demonstra

- **Treinamento de LLM do zero** — arquitetura Transformer implementada em PyTorch sem uso de Hugging Face ou pesos externos
- **Pipeline NLP completo** — tokenizador BPE → LM → retriever contrastivo InfoNCE → índice vetorial → RAG extrativo
- **Engenharia de dados** — ingestão, validação e integridade de corpus com hash SHA-256; pipeline de PDFs com camada de texto
- **API REST + WebSocket** — backend FastAPI com autenticação CSRF, session management e streaming de progresso em tempo real
- **Otimização de hardware** — backend DirectML para aceleração GPU em Windows (AMD 4600G APU), fallback automático para CPU
- **Resiliência** — retomada de checkpoint por lote/step; watchdog de processo; validação de linhagem do corpus
- **Qualidade de código** — 209 testes (pytest), 95 arquivos em validação estática, Conventional Commits com hook automático

---

## Características principais

- **Modo soberano** — corpus fechado; sem downloads em runtime; zero dados enviados a terceiros
- **Checkpoint granular** — retomada por lote (retriever) e por step (LM) após qualquer interrupção
- **Ingestão de PDFs** — upload via API local; rejeição de scans sem camada de texto (comportamento intencional)
- **Hardware adaptativo** — DirectML (GPU Windows) → CUDA → CPU; detecção automática com probe de hardware

---

## Requisitos

- Windows 10/11 (64-bit)
- Python 3.10
- 8 GB RAM (mínimo); 16 GB+ recomendado
- GPU com suporte a DirectX 12 (opcional; CPU funciona)

---

## Início rápido

```powershell
# 1. Clonar o repositório
git clone https://github.com/CFSJCODE/JarvisLocalHost.git
cd JarvisLocalHost

# 2. Criar o ambiente virtual e instalar dependências
./jarvis_localhost/tools/setup.ps1

# 3. Copiar e ajustar a configuração
copy .env.example .env

# 4. Iniciar o servidor
./jarvis_localhost/tools/run.ps1
```

O painel local estará disponível em `http://127.0.0.1:8008`.

---

## Fluxo de uso

1. Acesse o painel e faça upload dos PDFs desejados via **Carregar documento**
2. Clique em **Iniciar treinamento** — BPE + LM + retriever treinam do zero no corpus enviado
3. Quando `is_trained = true`, o chat utiliza RAG extrativo com evidências dos documentos

---

## Estrutura principal

```
jarvis_localhost/
├── ai/           — trainer, LM Transformer, tokenizador BPE, dataset
├── core/         — orquestrador do pipeline (brain.py)
├── retrieval/    — retriever contrastivo InfoNCE
├── rag/          — geração aumentada por recuperação extrativa
├── corpus/       — manifesto e integridade SHA-256 do corpus
├── hardware/     — seleção de backend (DirectML/CUDA/CPU)
├── server/       — API FastAPI + WebSocket + CSRF
├── storage/      — banco SQLite local
├── curiosity/    — agentes de curiosidade (PPO + ICM)
└── tools/        — scripts de setup, validação e monitoramento
```

---

## Desenvolvimento

```powershell
# Executar a suite de testes (209 testes)
./jarvis_localhost/tools/validate.ps1

# Verificar o hardware disponível
.venv\Scripts\python.exe -m jarvis_localhost.tools.hardware_probe
```

### Conventional Commits

Este projeto adota [Conventional Commits](https://www.conventionalcommits.org/pt-br/v1.0.0/).
O hook `commit-msg` valida automaticamente as mensagens via commitlint.

Formato: `tipo(escopo opcional): descrição`

Exemplos:
```
feat(retrieval): adiciona checkpoint por lote no retriever contrastivo
fix(trainer): retomada com step == max_steps nao descarta LM concluido
docs: atualiza README com fluxo de uso
ci: adiciona job de validacao de commits no PR
```

---

## Licença

[PolyForm Noncommercial License 1.0.0](LICENSE) —
`SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0`

Copyright © 2026 Cláudio Francisco Dos Santos Júnior

Uso pessoal, educacional e acadêmico permitidos. Uso comercial requer
autorização separada. Veja [`.github/USAGE_POLICY.md`](.github/USAGE_POLICY.md).

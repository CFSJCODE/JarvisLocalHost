# JarvisLocalHost

Assistente local soberano para leitura de PDFs, recuperação de evidências e
treinamento neural a partir de um corpus autorizado pelo usuário.

O sistema não baixa pesos, tokenizadores, embeddings ou checkpoints externos.
O modelo de linguagem, o retriever contrastivo e os agentes de curiosidade
partem do zero e aprendem exclusivamente dos documentos locais.

---

## Características

- **Modo soberano** — sem modelos pré-treinados externos; corpus fechado com
  verificação de integridade SHA-256
- **Pipeline completo** — tokenizador BPE → LM Transformer → retriever
  contrastivo InfoNCE → índice vetorial → RAG extrativo
- **Backend DirectML** — aceleração GPU via DirectML no Windows; fallback
  automático para CPU
- **Ingestão de PDFs** — upload via API local; documentos sem camada de texto
  são rejeitados (modo soberano, comportamento intencional)
- **Retomada de checkpoint** — treinamento pode ser retomado após interrupção
  desde que corpus e tokenizador não mudem

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
2. Clique em **Iniciar treinamento** — o sistema treina do zero no corpus enviado
3. Quando `is_trained = true`, o chat utiliza RAG extrativo sobre os documentos

---

## Estrutura principal

```
jarvis_localhost/
├── ai/           — trainer, LM, tokenizador BPE, dataset
├── core/         — orquestrador do pipeline (brain.py)
├── retrieval/    — retriever contrastivo InfoNCE
├── rag/          — geração aumentada por recuperação extrativa
├── corpus/       — manifesto e integridade do corpus
├── hardware/     — seleção de backend (DirectML/CUDA/CPU)
├── server/       — API FastAPI + WebSocket
├── storage/      — banco SQLite local
├── curiosity/    — agentes de curiosidade (PPO + ICM)
└── tools/        — scripts de setup, validação e monitoramento
```

---

## Desenvolvimento

```powershell
# Executar a suite de testes
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

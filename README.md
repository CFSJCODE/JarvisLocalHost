# J.A.R.V.I.S. — Just A Rather Very Intelligent System

> **IA 100% Local · Transformer GPT do Zero · PyTorch · FastAPI · RAG · OCR**

---

## Visão Geral

**J.A.R.V.I.S.** é um assistente de inteligência artificial completamente local, construído do zero sem dependência de APIs externas ou modelos proprietários. O núcleo é um **modelo Transformer decoder-only** (estilo GPT) implementado em PyTorch puro, treinado em tempo real com documentos carregados pelo próprio usuário.

O sistema roda inteiramente na máquina do usuário — os dados nunca saem do hardware local.

---

## Arquitetura do Sistema

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend HUD                             │
│              (index.html · WebSockets · Telemetria)             │
└────────────────────────────┬────────────────────────────────────┘
                             │ REST + WebSocket
┌────────────────────────────▼────────────────────────────────────┐
│                    FastAPI Server (app.py)                       │
│         /api/chat · /api/train · /api/pdf · /ws                 │
└──────┬──────────────┬──────────────┬───────────────┬────────────┘
       │              │              │               │
┌──────▼──────┐ ┌─────▼──────┐ ┌────▼─────┐ ┌──────▼──────┐
│  JarvisBrain│ │PDFProcessor│ │ AITrainer│ │SystemMonitor│
│  (brain.py) │ │(pdf_proc.) │ │(trainer.)│ │(sys_mon.)   │
└──────┬──────┘ └─────┬──────┘ └────┬─────┘ └─────────────┘
       │              │              │
┌──────▼──────────────▼──────────────▼──────────────────────────┐
│                     Núcleo Neural Local                        │
│   JarvisTransformer · BasicTokenizer · RAGEngine · VectorStore │
│        (model.py)   · (tokenizer.py) · (embeddings.py)        │
└────────────────────────────────────────────────────────────────┘
```

---

## Funcionalidades

### Modelo Transformer (do zero)
- Arquitetura **decoder-only GPT** implementada puramente em PyTorch
- Multi-Head Causal Self-Attention com máscara triangular inferior
- Codificação posicional **Sinusoidal** (Attention Is All You Need)
- Feed-Forward com ativação **GELU** e conexões residuais
- **Weight tying** entre embedding de entrada e projeção de saída
- Inicialização com `std=0.02` e **Layer Normalization** pre-norm

### Pipeline RAG (Retrieval-Augmented Generation)
- Geração de **embeddings densos** a partir do próprio modelo treinado
- Busca vetorial via **FAISS** (com fallback NumPy)
- Chunking de documentos com sobreposição configurável (`CHUNK_OVERLAP = 40`)
- Geração autoregressiva com **top-k sampling** e controle de temperatura
- Memória de curto prazo: respostas são re-indexadas automaticamente

### Processamento Multimodal de PDFs
- Extração de texto por spans tipográficos via **PyMuPDF (fitz)**
- Detecção e extração de tabelas via **pdfplumber** → formato Markdown
- Extração de imagens embutidas com **OCR via Tesseract**
- Mesclagem inteligente de chunks curtos por página
- Detecção automática de idioma (pt-BR / en)
- Exportação de corpus limpo `.txt` + metadados `.json`

### Motor de Treino Causal
- **Causal Language Modeling** com Cross-Entropy Loss
- Otimizador **AdamW** com gradient clipping (`max_norm=1.0`)
- Construção de vocabulário dinâmico por frequência (`max_vocab=10.000`)
- Progresso de treino transmitido em tempo real via WebSocket
- Execução em thread separada (não bloqueia o servidor)

### Telemetria de Hardware
- Monitoramento em tempo real: **CPU, RAM, Disco, Rede**
- Temperatura da CPU (quando disponível via `psutil`)
- Alertas automáticos de uso crítico de recursos
- Top 5 processos mais pesados por CPU
- Push de métricas via WebSocket a cada 2 segundos

---

## Estrutura do Projeto

```
jarvis/
├── app.py              # Servidor FastAPI: REST + WebSocket
├── brain.py            # Controlador mestre (orquestrador)
├── model.py            # JarvisTransformer (GPT do zero)
├── tokenizer.py        # BasicTokenizer (vocabulário local)
├── trainer.py          # AITrainer (loop de treino causal)
├── embeddings.py       # EmbeddingEngine + VectorStore + RAGEngine
├── pdf_processor.py    # Extração multimodal de PDFs
├── system_monitor.py   # Telemetria de hardware via psutil
├── static/
│   └── index.html      # HUD interativo (frontend)
├── data/
│   ├── embeddings/     # Corpus e metadados extraídos dos PDFs
│   └── extracted_images/
└── uploads/            # PDFs enviados pelo usuário
```

---

## Instalação

### Pré-requisitos

- Python **3.11+**
- Recomendado: AMD Ryzen 5 / Intel Core i5 (6+ cores) — sem necessidade de GPU
- Tesseract OCR instalado no sistema (opcional, para OCR de imagens)

### Dependências Python

```bash
pip install fastapi uvicorn torch pydantic psutil \
            pymupdf pdfplumber pillow pytesseract \
            numpy faiss-cpu
```

### Instalação do Tesseract (opcional)

```bash
# Ubuntu/Debian
sudo apt install tesseract-ocr tesseract-ocr-por

# macOS
brew install tesseract
```

### Inicialização

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/jarvis-local.git
cd jarvis-local

# Instale as dependências
pip install -r requirements.txt

# Inicie o servidor
python app.py
```

O servidor iniciará em `http://localhost:8000`.

---

## Uso

### 1. Carregar Documentos

Acesse o HUD em `http://localhost:8000` e faça upload de arquivos PDF. O sistema extrai automaticamente texto, tabelas e imagens (OCR) e salva o corpus em `data/embeddings/`.

### 2. Treinar o Modelo

Clique em **"Iniciar Treino"** no HUD. O pipeline executa:

```
Sincronizar Datalake → Construir Vocabulário → Instanciar Transformer
    → Treino Causal (10 épocas) → Indexar RAG → Modelo Pronto
```

O progresso é transmitido em tempo real via WebSocket.

### 3. Chat Local

Com o modelo treinado, envie mensagens no chat. O sistema:
1. Gera o embedding da pergunta
2. Recupera os chunks mais relevantes do VectorStore (FAISS)
3. Monta o prompt com contexto
4. Gera a resposta autoregressivamente (top-k sampling)
5. Re-indexa a resposta na memória de curto prazo

### 4. API REST

```bash
# Métricas do sistema
GET  /api/metrics

# Chat com o assistente
POST /api/chat
     {"message": "Explique a arquitetura do modelo"}

# Upload de PDF
POST /api/pdf/upload   (multipart/form-data)

# Status do treino
GET  /api/train/status

# Iniciar treino
POST /api/train/start

# Listar documentos indexados
GET  /api/documents

# Tokenizar texto (debug)
POST /api/tokenize
     {"text": "texto de exemplo"}
```

---

## Configuração do Modelo

Edite `JarvisConfig` em `model.py` para ajustar a arquitetura:

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `vocab_size` | 8.000 | Tamanho do vocabulário |
| `context_len` | 512 | Comprimento máximo da sequência |
| `embed_dim` | 256 | Dimensão dos embeddings |
| `num_heads` | 8 | Cabeças de atenção |
| `num_layers` | 6 | Blocos Transformer |
| `ff_dim` | 1.024 | Dimensão interna do FFN |
| `dropout` | 0.1 | Taxa de dropout |

---

## Detalhes Técnicos

### Geração de Texto
O método `generate()` implementa **top-k sampling** com truncagem ao `context_len`. A geração para automaticamente ao encontrar o token `<EOS>` (id=3).

### Tokenizador
O `BasicTokenizer` opera sobre unigramas extraídos via regex (`\b\w+\b|[^\w\s]`), constrói vocabulário por frequência e suporta tokens especiais: `<PAD>` (0), `<UNK>` (1), `<EOS>` (2).

### VectorStore
Utiliza `faiss.IndexFlatIP` (produto interno / cosine similarity após normalização L2). Fallback para NumPy quando FAISS não está disponível.

### Concorrência
O treino roda em uma `daemon thread` separada. O broadcast de progresso para os clientes WebSocket é feito via `asyncio.run_coroutine_threadsafe()`, garantindo segurança entre threads.

---

## Limitações Conhecidas

- O modelo é treinado do zero a cada sessão (sem persistência de pesos por padrão)
- A qualidade das respostas depende diretamente do volume e qualidade dos PDFs carregados
- OCR requer Tesseract instalado no sistema operacional
- FAISS não está disponível em todos os ambientes (fallback ativo)

---

## Roadmap

- [ ] Persistência dos pesos do modelo treinado (`model.save()` já implementado)
- [ ] Suporte a múltiplos idiomas no tokenizador (BPE)
- [ ] Interface de anotação para fine-tuning supervisionado
- [ ] Aceleração GPU via CUDA (estrutura já compatível)
- [ ] Exportação de modelo para ONNX
- [ ] Suporte a arquivos `.txt`, `.docx`, `.epub`

---

## Licença

Este projeto é distribuído sob a licença **MIT**. Consulte o arquivo `LICENSE` para mais detalhes.

---

<div align="center">

**J.A.R.V.I.S. · 100% Local · 100% Seu**

*"A mente mais brilhante não precisa de nuvem para pensar."*

</div>

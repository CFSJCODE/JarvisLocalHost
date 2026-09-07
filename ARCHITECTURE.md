# Arquitetura soberana do J.A.R.V.I.S. LocalHost

## Objetivo e invariantes

O J.A.R.V.I.S. é um sistema local de PDF RAG em que o corpus autorizado é a única fonte de conhecimento semântico aprendido. O desenho preserva cinco invariantes:

1. Nenhum peso, tokenizador, embedding, OCR ou checkpoint semântico pré-treinado entra no runtime soberano.
2. Toda evidência recuperável mantém identidade do documento, hash, página, região e chunk.
3. Nenhuma resposta é apresentada como sustentada quando a recuperação ou o grounding falham.
4. Todo artefato treinado fica vinculado, por hashes, ao corpus e ao tokenizador que o produziram.
5. Dados privados e estado de execução permanecem fora do Git e, no modo soberano, fora da rede.

Dependências como PyTorch, FastAPI, PyMuPDF e `torch-directml` fornecem execução numérica, HTTP ou parsing; elas não fornecem conhecimento semântico ao modelo.

## Fronteiras de confiança

```text
                         BOOTSTRAP (rede permitida pelo operador)
       Git / PyPI / documentação oficial -> código e dependências locais
                                      |
                                      v
┌──────────────────────────── RUNTIME SOBERANO ────────────────────────────┐
│                                                                          │
│  navegador loopback -> FastAPI -> PDF processor -> corpus canônico       │
│                              |                 |                         │
│                              |                 +-> hashes/proveniência   │
│                              v                                           │
│                    treino local do zero                                  │
│                 tokenizer + LM + retriever                               │
│                              |                                           │
│                              v                                           │
│       pergunta -> recuperação -> citações -> strict / RAG aterrado        │
│                              |                                           │
│                              +-> SQLite/checkpoints locais                │
│                                                                          │
└──────────────────────────── sem egress/API externa ──────────────────────┘
```

`jarvis_localhost/sovereign.py` materializa essa fronteira. Com `JARVIS_SOVEREIGN_MODE=1`, flags de OCR pré-treinado, pesos externos, downloads, APIs, egress e TTS do sistema não podem ser liberadas ao mesmo tempo. O cluster de execução também permanece indisponível.

## Fluxo de dados e linhagem

### 1. Ingresso do PDF

`server/app.py` recebe o upload somente por uma sessão local válida. Antes do processamento:

- normaliza o nome sem aceitar caminhos do cliente;
- verifica extensão e assinatura `%PDF-`;
- impõe limite de bytes e concorrência;
- grava com criação exclusiva dentro do diretório de runtime;
- remove o parcial em qualquer erro ou cancelamento.

`processing/pdf_processor.py` abre o documento local, extrai texto por página e constrói `PageSpan`. No modo soberano, uma página sem texto não dispara OCR com pesos externos.

### 2. Corpus canônico

`corpus/chunker.py` converte spans em `CanonicalChunk`. Cada chunk preserva:

- `document_id` e SHA-256 do PDF;
- nome de origem sanitizado;
- ordinal e `chunk_id` determinístico;
- página inicial/final;
- caixas delimitadoras quando disponíveis;
- texto e hash do texto.

Chunks são gravados como JSONL. `corpus/manifest.py` agrega documentos, páginas, bytes, palavras e chunks, vincula hash/tamanho/quantidade de registros de cada artefato e define um mundo fechado: arquivos órfãos, ausentes, adulterados ou fora do manifesto são recusados. `corpus/provenance.py` calcula o digest determinístico do corpus. Reprocessar o mesmo conteúdo deve produzir a mesma identidade; uma colisão com conteúdo divergente é um erro.

### 3. Estado aprendido

O treinamento usa somente o snapshot canônico:

- `ai/tokenizer.py`: aprende o vocabulário BPE no corpus local.
- `ai/language_model.py`: cria o Transformer decoder-only a partir de seed/configuração local.
- `ai/dataset.py` e `ai/trainer.py`: constroem amostras causais e executam treino/checkpoints.
- `retrieval/encoder.py`: cria um encoder separado para busca.
- `retrieval/contrastive.py`: treina pares positivos/negativos com InfoNCE.
- `retrieval/vector_store.py`: persiste vetores e metadados canônicos.
- `retrieval/retriever.py`: combina o caminho denso treinado com recuperação lexical local.

O manifesto `data/models/jarvis_pipeline.json` referencia artefatos por caminho relativo, tamanho e SHA-256, além do hash do corpus. O carregamento verifica containment, checksum e lineage antes de ativar modelo, tokenizador, retriever e vector store como um conjunto.

### 4. Resposta baseada em evidência

```text
pergunta
   |
   v
SovereignRetriever ---- baixa confiança/sem passagem ----> abstenção
   |
   v
Citation[E1..En] -> página, bbox, chunk, hash, trecho
   |
   +---- mode=strict ----> resposta extrativa -> grounding report
   |
   +---- mode=rag -------> prompt com pergunta reservada
                              |
                              v
                         geração local
                              |
                    grounding aprovado?
                       |              |
                      sim            não
                       |              |
                       v              v
                    resposta     fallback extrativo
```

`rag/citations.py` transforma resultados em citações rastreáveis. `rag/grounding.py` mede suporte da resposta nas evidências e produz um relatório. `rag/engine.py` nunca deixa a geração ignorar esses gates.

| Modo | Modelo generativo | Comportamento | Uso recomendado |
|---|---:|---|---|
| `strict` | não exigido | Extrai resposta da evidência; abstém sem suporte | padrão, auditoria e pré-treino |
| `rag` | exigido | Gera localmente; exige grounding; fallback extrativo | após avaliação do checkpoint local |

O prompt reserva primeiro a pergunta e o sufixo de resposta; somente o orçamento restante recebe evidências. Isso impede truncamento silencioso da pergunta em contextos grandes.

## Perfil de hardware e execução

`hardware/device.py` detecta CPU/RAM/DirectX e tenta backends na ordem apropriada à plataforma. Um backend acelerado só é aceito depois de um smoke test de tensor. Falhas são registradas e resultam em fallback para CPU.

`hardware/profiles.py` deriva do corpus e do host:

- vocabulário e dimensões do modelo;
- contexto e formato do Transformer;
- batch, acumulação e quantidade de passos;
- intervalos de avaliação/checkpoint;
- threads e workers;
- orçamento de RAM e memória de acelerador.

Os caps preservam responsividade do desktop: no máximo 16 GiB de orçamento de host, 6 GiB de acelerador, contexto 512, acumulação 16 e 10.000 passos, mesmo quando uma variável pede mais.

Para o Ryzen 5 4600G no Windows, DirectML é o caminho esperado. Os
aproximadamente 8 GB reservados para a Radeon integrada são memória UMA, não
VRAM discreta. Como DirectML está em modo de manutenção, a combinação de
dependências é fixada e cada operação neural crítica entra no smoke da pilha.
O probe é a autoridade: se DirectML falhar, o processo usa CPU em vez de fingir
aceleração.

## Curiosidade, ICM, PPO e currículo

O subsistema `curiosity/` é separado do RAG de resposta:

- `environment.py` expõe estados e ações derivados do corpus, sem semântica de domínio codificada manualmente.
- `encoder.py` representa observações locais.
- `icm.py` calcula erro de dinâmica e recompensa intrínseca.
- `policy.py` contém actor-critic.
- `ppo.py` aplica a atualização clipped PPO.
- `curriculum.py` pondera a próxima amostragem com perda causal e incerteza contrastiva realmente medidas por chunk, além da recompensa intrínseca e cobertura.
- `memory.py` e `engine.py` persistem gerações imutáveis com ICM, PPO, estados dos otimizadores, RNG, currículo, sinais, pesos de amostragem e checksums ligados ao SHA-256 do corpus.

Esse agente pode sugerir relações e tópicos, mas sua recompensa não constitui prova factual. Insights precisam conservar vínculo com o corpus e não podem executar shell, instalar software ou editar o repositório.

## API local e controles de segurança

`server/app.py` limita o serviço a `127.0.0.1`. A camada HTTP exige:

- autoridade local conhecida para impedir DNS rebinding;
- `Origin` igual ao `Host` em mutações;
- cookie de sessão e token CSRF com comparação segura;
- sessão válida e origem local no upgrade WebSocket;
- headers de isolamento (`nosniff`, frame deny, referrer e CSP de enquadramento);
- modelos Pydantic com limites de tamanho;
- operações pesadas fora do event loop;
- mensagens de erro genéricas, sem eco de exceções internas.

`web/static/index.html` usa APIs DOM seguras para conteúdo não confiável. `projects/project_manager.py` valida identificadores, contém caminhos no diretório de projetos e grava ZIPs de forma atômica. `integrations/cluster_client.py` fica desligado por padrão, limita destinos a loopback/LAN quando relaxado e rejeita shells/argumentos perigosos.

Esses controles assumem uma única conta local confiável. Eles não implementam autenticação multiusuário nem TLS público.

## Colaboração Codex–Antigravity

Há dois planos distintos:

```text
Antigravity -- MCP codex ----------> Codex/ChatGPT Desktop
     |                                    |
     +------ MCP jarvis-team-bus ---------+
                    |
                    v
      SQLite WAL: agentes, mensagens, tarefas,
       idempotência, versões, comandos e auditoria
```

O servidor `integrations/team_bus_server.py` usa transporte MCP/JSON-RPC por stdio. As ferramentas são `ping`, `register_agent`, `post_message`, `fetch_messages`, `create_task`, `claim_task`, `update_task`, `list_tasks`, `get_audit_log` e `execute_command`.

Claims e atualizações usam transações/versões; mutações aceitam chaves idempotentes; comandos recebem argv estruturado, timeout, limite de saída e auditoria. O barramento coordena trabalho, mas não substitui a regra de um único escritor por arquivo.

`.agents/launch-codex-mcp.ps1` encontra o executável atual do Codex mesmo após atualização do ChatGPT Desktop. `.agents/launch-team-bus.ps1` prefere a venv do projeto e define o banco local. O instalador/rollback preserva configurações globais por backup e hash.

## Persistência

| Local | Conteúdo | Versionado? |
|---|---|---:|
| `data/jarvis.db*` | aplicação, lineage e métricas | não |
| `data/embeddings/` | chunks, manifestos e vector store | não |
| `data/models/` | tokenizador, LM, retriever e manifesto | não |
| `data/curiosity/` | estado/insights de curiosidade | não |
| `data/integrations/` | team bus e manifesto de instalação | não |
| `uploads/` | PDFs privados recebidos | não |
| `tests/` | dados sintéticos e contratos | sim |

SQLite usa WAL, foreign keys e busy timeout. Escritas de controle e artefatos críticos usam temporários e substituição atômica quando aplicável.

## Limites conhecidos

- Não há modelo de produção distribuído nem métrica de qualidade prometida.
- O treino do zero em um corpus pequeno pode memorizar, ficar incoerente ou não generalizar.
- DirectML pode executar algumas operações na CPU; desempenho deve ser medido no host real.
- PDF escaneado sem camada de texto exige relaxar conscientemente a política e usar OCR externo/pré-treinado; isso deixa de ser o caminho soberano estrito.
- O modo `rag` reduz alucinação por gates, mas a saída precisa de avaliação humana proporcional ao risco.
- Citações comprovam a passagem recuperada, não garantem que o documento original esteja correto.

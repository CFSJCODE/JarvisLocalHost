# J.A.R.V.I.S. LocalHost

Assistente local para leitura de PDFs, recuperação de evidências e treinamento neural a partir de um corpus autorizado. O caminho soberano não baixa modelos, tokenizadores, embeddings, OCR ou checkpoints semânticos: tokenizador, modelo de linguagem, retriever e agentes de curiosidade começam sem conhecimento pré-treinado e aprendem somente com os documentos locais.

> **Estado honesto:** a arquitetura, os testes automatizados e os smoke tests locais estão implementados. O repositório não inclui PDFs privados, pesos treinados nem um modelo de produção. Qualidade, tempo de treinamento e capacidade de generalização dependem do corpus que o operador importar e precisam ser medidos nesse corpus.

## O que está implementado

- Upload e processamento local de PDFs com limite de tamanho, validação da assinatura `%PDF-`, nome seguro, concorrência limitada e limpeza de arquivos parciais.
- Corpus canônico com identidade por SHA-256, chunks determinísticos e proveniência até documento, página e região da página.
- Tokenizador BPE próprio, Transformer decoder-only e encoder de recuperação inicializados localmente, sem pesos semânticos externos.
- Treino do modelo de linguagem e treino contrastivo do retriever, com perfil calculado a partir do corpus e do hardware detectado.
- RAG com dois modos: `strict`, extrativo e padrão; `rag`, generativo com verificação obrigatória de grounding e fallback extrativo.
- Citações rastreáveis, limiar mínimo de recuperação e abstenção quando o corpus não sustenta a resposta.
- Curiosidade baseada em ambiente de corpus, ICM, PPO e currículo progressivo. Esse subsistema explora relações nos documentos; ele não executa comandos nem modifica o código-fonte.
- Backend FastAPI e HUD local protegidos contra DNS rebinding, requisições cross-origin, CSRF, WebSocket sem sessão e uploads abusivos.
- Persistência SQLite local com WAL para histórico, documentos, treinamento, curiosidade e barramento de colaboração.
- Seleção de CPU, DirectML, CUDA/ROCm ou MPS somente após smoke test real do backend, sempre com fallback seguro para CPU.
- Barramento MCP bidirecional para Codex/ChatGPT Desktop e Google Antigravity, com mensagens, tarefas, claims atômicos, idempotência e auditoria local.

## Arquitetura em uma frase

```text
PDF autorizado
  -> texto e páginas
  -> chunks canônicos + hashes + proveniência
  -> tokenizador/LM/retriever treinados do zero
  -> recuperação
  -> resposta extrativa ou geração aterrada
  -> citações por documento e página
```

O manifesto do pipeline liga cada artefato ao hash do corpus e do tokenizador. Um checkpoint incompatível, alterado ou proveniente de outro corpus não é ativado silenciosamente.

Leia [ARCHITECTURE.md](ARCHITECTURE.md) para o fluxo completo e [DEVELOPMENT.md](DEVELOPMENT.md) para o procedimento de desenvolvimento e validação.

## Requisitos

- Windows 10/11 para a configuração DirectML descrita aqui.
- Python **3.10 x64**.
- PowerShell 5.1 ou PowerShell 7.
- Espaço local para o corpus, vector store e checkpoints.
- PDFs com camada de texto no modo soberano estrito.

O bootstrap pode usar a internet para instalar dependências de código. Depois disso, o runtime soberano bloqueia downloads de modelos, APIs externas e saída de rede. As bibliotecas instaladas não substituem nem fornecem conhecimento semântico ao modelo.

## Início rápido no Windows

```powershell
Set-Location E:\SoftwareProjects\JarvisLocalHost

# Cria jarvis_localhost\.venv com Python 3.10 e instala DirectML no Windows.
.\jarvis_localhost\tools\setup.ps1

# Configuração soberana recomendada. O arquivo .env permanece fora do Git.
Copy-Item .\jarvis_localhost\.env.example .\.env

# Compila o código, executa todos os testes e verifica a higiene do repositório.
.\jarvis_localhost\tools\validate.ps1

# Inicia somente em loopback.
.\jarvis_localhost\tools\run.ps1
```

Abra `http://127.0.0.1:8000`. O servidor não deve ser exposto diretamente à internet ou a uma interface LAN.

Para uma instalação deliberadamente apenas em CPU:

```powershell
.\jarvis_localhost\tools\setup.ps1 -NoDirectML
$env:JARVIS_COMPUTE_BACKEND = "cpu"
.\jarvis_localhost\tools\run.ps1
```

## AMD Ryzen 5 4600G e DirectML

No Windows, o backend recomendado para a Radeon integrada do Ryzen 5 4600G é DirectML. O setup fixa uma combinação compatível de `torch`, `torchvision` e `torch-directml`; o detector só aceita o acelerador depois de alocação, cálculo, backward e ida/volta para a memória do host.

```powershell
$python = ".\jarvis_localhost\.venv\Scripts\python.exe"

& $python .\jarvis_localhost\tools\hardware_probe.py `
  --corpus .\jarvis_localhost\data\embeddings `
  --configure-cpu-threads

& $python .\jarvis_localhost\tools\directml_smoke.py `
  --require-backend directml `
  --require-accelerator
```

O segundo comando exercita forward/backward e atualização do modelo de linguagem, retriever contrastivo, ICM/policy e save/load de checkpoint. Os dois argumentos tornam o ensaio uma prova de aceitação: ele falha se o backend não for `directml`, se não estiver acelerado ou se houver queda silenciosa do backend inteiro para CPU.

### A ressalva dos “8 GB dedicados”

A Radeon do 4600G é uma GPU **integrada (UMA)**. O Windows/BIOS pode reportar aproximadamente 8 GB como memória reservada/dedicada, mas isso continua sendo RAM do sistema compartilhada fisicamente com a CPU; não equivale a uma placa de vídeo discreta com 8 GB de VRAM. O perfil do Jarvis é conservador: preserva memória para o Windows, limita o orçamento de RAM e usa no máximo 6 GiB como orçamento do acelerador, reduzindo batch/contexto quando necessário.

Algumas operações de versões atuais do `torch-directml` podem cair para CPU.
Isso pode reduzir desempenho sem invalidar a correção do resultado. O projeto
DirectML encontra-se em modo de manutenção, por isso as versões são fixadas e
cada caminho crítico precisa de smoke real. Para esse hardware, não trate ROCm
como requisito nem como backend garantido; use o backend que o probe
efetivamente validar.

## Ingestão e treinamento

1. Inicie o servidor e envie PDFs pelo HUD.
2. O processador extrai a camada de texto e grava chunks canônicos e metadados no diretório local do corpus.
3. Confira os documentos e páginas antes de iniciar o treino.
4. Inicie o treinamento pelo HUD ou por `POST /api/train/start` a partir de uma sessão local válida.
5. Acompanhe `GET /api/train/status` e os eventos do WebSocket.
6. Só promova um checkpoint depois de validar respostas e citações no seu próprio conjunto de avaliação.

Se não houver corpus canônico, não há base legítima para treinamento. Um PDF escaneado sem camada de texto é rejeitado no modo soberano em vez de acionar OCR pré-treinado. O arquivo `requirements-optional-ocr.txt` existe apenas para um modo conscientemente relaxado e não deve ser instalado ou usado quando `JARVIS_SOVEREIGN_MODE=1`.

## Modos de resposta

### `strict` — padrão recomendado

- Recupera evidências canônicas.
- Produz uma resposta extrativa a partir das passagens localizadas.
- Retorna documento, página, região, chunk e trecho citado.
- Abstém quando a recuperação fica abaixo do limiar.
- Continua disponível antes de existir um modelo generativo treinado.

### `rag` — geração local com gate

- Exige o modelo e o retriever treinados para o mesmo corpus.
- Reserva espaço para a pergunta antes de montar o contexto.
- Gera usando apenas as evidências recuperadas.
- Verifica grounding depois da geração.
- Usa fallback extrativo quando a geração não é sustentada.

Ative explicitamente somente após validar um checkpoint local:

```powershell
$env:JARVIS_RAG_MODE = "rag"
.\jarvis_localhost\tools\run.ps1
```

## Variáveis de ambiente

O exemplo canônico está em `jarvis_localhost/.env.example` e possui um espelho
idêntico em `.env.example` na raiz. A validação compara os hashes para impedir
divergência. Copie um deles para `.env` na raiz. As chaves mais importantes são:

- `JARVIS_SOVEREIGN_MODE=1`: ativa a política soberana.
- `JARVIS_ALLOW_*`: permanecem `0` em modo soberano; combinações contraditórias são rejeitadas.
- `JARVIS_RAG_MODE=strict`: seleciona o caminho extrativo padrão.
- `JARVIS_COMPUTE_BACKEND=auto`: tenta backends acelerados e aceita apenas um que passe o smoke test.
- `JARVIS_MAX_RAM_GB` e `JARVIS_MAX_GPU_MEMORY_GB`: limites, não promessas de alocação.
- `JARVIS_CPU_THREADS` e `JARVIS_INTEROP_THREADS`: mantêm o desktop responsivo durante o treino.
- `JARVIS_CLUSTER_ENABLED=0` e `JARVIS_VOICE_ENABLED=0`: integrações opcionais ficam desligadas.

Os limites de batch, contexto, acumulação e passos podem ser sobrescritos, mas são validados e limitados pelos caps internos. Comece com os valores automáticos.

## Segurança da API local

O HUD obtém uma sessão e um token CSRF automaticamente. Clientes próprios precisam respeitar o mesmo contrato:

- `Host` deve ser exatamente `127.0.0.1:8000` ou `localhost:8000`.
- `Origin` deve corresponder ao host em operações mutáveis e no WebSocket.
- Use primeiro `GET /` ou `GET /api/session`, mantenha o cookie local e envie `X-Jarvis-CSRF` nos métodos mutáveis.
- Uploads aceitam somente PDF, têm limite configurável e nunca preservam arquivo parcial após falha/cancelamento.
- O frontend renderiza conteúdo como texto, sem injetar respostas do modelo como HTML.
- Cluster e voz ficam desativados por padrão; execução de cluster é bloqueada no modo soberano.

Essas proteções reduzem o risco local, mas não transformam o servidor em um serviço público multiusuário.

## Antigravity + ChatGPT Desktop/Codex

O repositório fornece dois servidores MCP:

- `codex`: expõe o Codex instalado com o ChatGPT Desktop ao Antigravity por um launcher estável.
- `jarvis-team-bus`: barramento SQLite local para registro de agentes, mensagens bidirecionais, tarefas, claims atômicos, idempotência e auditoria.

O instalador é transacional, preserva configurações existentes, cria backups, registra hashes em um manifesto e faz dry-run por padrão:

```powershell
# Inspeciona o plano, sem alterar configuração global.
.\jarvis_localhost\tools\install_antigravity_chatgpt.ps1

# Aplica a configuração depois da revisão.
.\jarvis_localhost\tools\install_antigravity_chatgpt.ps1 -Apply
```

Para reverter, o rollback valida cada destino. Alterações posteriores fora do bloco gerenciado do Codex são preservadas por uma remoção seletiva; qualquer mudança não reconhecida interrompe a operação:

```powershell
.\jarvis_localhost\tools\rollback_antigravity_chatgpt.ps1
.\jarvis_localhost\tools\rollback_antigravity_chatgpt.ps1 -Apply
```

O banco do barramento fica em `jarvis_localhost/data/integrations/team_bus.sqlite` e não deve ser versionado. Use nomes estáveis de agente (`codex`, `antigravity`), chaves de idempotência em toda mutação e uma única responsabilidade de escrita por arquivo. Permissão ampla das ferramentas não autoriza publicação de dados, exposição de segredos, `push`, deploy ou exclusões fora da tarefa concreta.

Detalhes operacionais ficam em [DEVELOPMENT.md](DEVELOPMENT.md) e o contrato entre agentes em [AGENTS.md](AGENTS.md).

## Validação

```powershell
# Validação padrão: estrutura, Python 3.10, compileall, testes e git diff --check.
.\jarvis_localhost\tools\validate.ps1

# Exige DirectML acelerado e exercita toda a pilha neural crítica.
.\jarvis_localhost\tools\validate.ps1 -DirectMLSmoke
```

O GitHub Actions instala apenas `requirements.txt` com Python 3.10 e executa a suíte em CPU. DirectML depende de Windows/driver/hardware reais e por isso é validado localmente, não no CI hospedado.

## Dados locais

Estes artefatos são runtime privado e ficam fora do Git:

- `jarvis_localhost/uploads/`
- `jarvis_localhost/data/jarvis.db*`
- `jarvis_localhost/data/embeddings/`
- `jarvis_localhost/data/models/`
- `jarvis_localhost/data/curiosity/`
- `jarvis_localhost/data/projects/`
- `jarvis_localhost/data/integrations/`
- `jarvis_localhost/.venv/`
- `.env`

Não coloque PDFs, prompts privados, tokens, cookies, chaves, bancos ou checkpoints em issues, logs, commits ou mensagens entre agentes.

## Estrutura principal

```text
.
├── .agents/                         # Launchers/config MCP do projeto
├── .github/workflows/validate.yml  # CI Python 3.10 em CPU
├── AGENTS.md                        # Contrato de colaboração
├── ARCHITECTURE.md                  # Arquitetura e fronteiras de confiança
├── DEVELOPMENT.md                   # Setup, testes e checklist
└── jarvis_localhost/
    ├── ai/                          # Tokenizador, LM, dataset e trainer
    ├── corpus/                      # Chunks, manifestos e proveniência
    ├── curiosity/                   # Ambiente, ICM, PPO e currículo
    ├── hardware/                    # Probe, seleção de device e perfis
    ├── rag/                         # Citações, grounding e resposta
    ├── retrieval/                   # Encoder, InfoNCE e vector store
    ├── processing/                  # Processamento determinístico de PDF
    ├── integrations/                # Team bus, voz e cluster opcionais
    ├── server/                      # FastAPI local endurecida
    ├── storage/                     # SQLite e lineage de treinamento
    ├── tests/                       # Testes unitários e de integração local
    └── tools/                       # Setup, probe, smoke, validação e rollback
```

## Licença

Consulte [LICENSE](LICENSE).

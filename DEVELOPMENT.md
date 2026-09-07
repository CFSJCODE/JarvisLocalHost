# Desenvolvimento e validação

Este guia descreve o fluxo reproduzível para alterar o J.A.R.V.I.S. sem romper soberania, proveniência, segurança local ou compatibilidade com o Ryzen 5 4600G.

## Preparação do ambiente

Use Python 3.10 x64. No Windows:

```powershell
Set-Location E:\SoftwareProjects\JarvisLocalHost
.\jarvis_localhost\tools\setup.ps1
Copy-Item .\jarvis_localhost\.env.example .\.env
```

`setup.ps1` cria `jarvis_localhost/.venv`, instala `requirements.txt` e, por padrão no Windows, a pilha fixada em `requirements-directml.txt`. Use `-NoDirectML` somente quando desejar CPU deliberadamente.

Dependências opcionais ficam separadas:

- `requirements-optional-ocr.txt`: incompatível com a promessa zero-pretrained em modo soberano.
- `requirements-optional-voice.txt`: TTS local opcional; desativado por padrão e sujeito à política.

Não adicione bibliotecas que façam download automático de pesos no import ou na primeira execução.

## Comandos de validação

```powershell
# Padrão obrigatório antes de entrega.
.\jarvis_localhost\tools\validate.ps1

# Exige DirectML acelerado; falha se o backend cair para CPU.
.\jarvis_localhost\tools\validate.ps1 -DirectMLSmoke

# Útil somente para diagnóstico estrutural quando dependências ainda não existem.
.\jarvis_localhost\tools\validate.ps1 -SkipTests
```

A validação padrão verifica:

1. arquivos e módulos essenciais;
2. Python 3.10;
3. sintaxe de todo o pacote com `compileall`;
4. descoberta completa de testes em `jarvis_localhost/tests`;
5. ausência de runtime privado rastreado em `data/` e `uploads/`;
6. whitespace inválido por `git diff --check`.

O smoke DirectML adiciona forward/backward/update do LM e retriever, ICM/policy e save/load de checkpoint. Com `-DirectMLSmoke`, a validação chama o ensaio com `--require-backend directml --require-accelerator`; sucesso em CPU é recusado em vez de ser descrito como aceleração GPU.

Comandos equivalentes para diagnóstico:

```powershell
$python = ".\jarvis_localhost\.venv\Scripts\python.exe"

& $python -m compileall -q .\jarvis_localhost
& $python -m unittest discover -s .\jarvis_localhost\tests -t . -v
& $python .\jarvis_localhost\tools\hardware_probe.py --configure-cpu-threads
& $python .\jarvis_localhost\tools\directml_smoke.py
git diff --check
```

## CI

`.github/workflows/validate.yml` usa Windows e Python 3.10, instala apenas as dependências-base e executa `validate.ps1`. O CI hospedado não instala DirectML porque não representa o driver, a Radeon UMA nem a configuração real do operador.

Uma mudança no código neural deve passar em CPU no CI e no smoke do backend real antes de ser chamada de compatível com DirectML.

## Regras para o pipeline soberano

Ao alterar ingestão, treino, recuperação ou loading:

- inicialize pesos aleatoriamente e registre a seed/configuração;
- derive vocabulário e semântica somente do corpus canônico;
- nunca use nomes de domínio ou listas codificadas para simular compreensão;
- não introduza chamadas a hubs, APIs, endpoints de embedding ou download de checkpoints;
- mantenha `weights_only=True` ou formatos próprios seguros no loading PyTorch;
- valide hash do corpus, tokenizador e artefato antes de carregar;
- grave controle/checkpoints de forma atômica;
- preserve a pergunta ao truncar o prompt;
- abstenha quando a recuperação não alcançar o limiar;
- aplique grounding e fallback extrativo à geração.

Qualquer modo relaxado precisa ser opt-in explícito, documentado como não soberano e testado separadamente.

## Alterando o corpus e a proveniência

`CanonicalChunk` é o contrato entre PDF, treinamento, retriever, RAG e citações. Uma alteração nesse tipo exige testes para:

- determinismo do `document_id` e `chunk_id`;
- hash estável do corpus independentemente da ordem de descoberta;
- página e `bbox` corretos;
- overlap sem perda ou duplicação conflitante;
- round-trip JSONL;
- rejeição de registros inválidos/conflitantes;
- manifestação e estatísticas agregadas corretas.

Não grave caminhos absolutos privados em checkpoints ou manifestos portáveis. Dentro do manifesto de pipeline, use caminhos relativos contidos no diretório de modelos.

## Alterando o RAG

Teste os dois modos:

- `strict` precisa funcionar com `model=None` e responder somente por evidência.
- `rag` precisa recusar ou retornar ao extrativo quando a geração não passa no grounding.

Inclua casos de:

- resultado vazio;
- confiança abaixo do limiar;
- pergunta maior que o orçamento normal;
- evidência maior que o contexto;
- múltiplas páginas/documentos;
- trecho que não sustenta a resposta;
- citações e hashes após persistência/reload.

Nunca trate score de similaridade como probabilidade calibrada sem avaliação específica.

## Alterando curiosidade/RL

O ambiente deve observar somente artefatos derivados do corpus. A política e o ICM não podem receber “verdades” de domínio codificadas no código.

Mantenha testes para:

- dimensões de observação/ação;
- recompensa intrínseca finita;
- atualização PPO clipped;
- avanço e persistência do currículo;
- IDs/hashes determinísticos;
- memória limitada;
- execução em CPU e no device selecionado.

Um insight é hipótese exploratória. Não o promova automaticamente a fato nem a evidência de resposta.

## Segurança da API

Mudanças em rotas mutáveis devem conservar sessão, CSRF e same-origin. Novos uploads precisam de:

- validação por conteúdo, não apenas extensão;
- limite de bytes antes e durante a cópia;
- nome gerado no servidor;
- containment no diretório de runtime;
- concorrência limitada;
- remoção do parcial em falha/cancelamento;
- trabalho pesado fora do event loop.

Não use `innerHTML` com mensagens, nomes de arquivo, respostas ou metadados. Não exponha stack traces, caminhos, tokens ou conteúdo de documento em erros/logs.

O servidor deve continuar em loopback. Alterar para `0.0.0.0`, abrir CORS ou remover o gate de `Host` não é uma refatoração neutra; muda a fronteira de segurança e exige projeto próprio.

## Hardware e orçamento

Não selecione um backend somente porque a biblioteca importa. `select_compute_device` exige smoke real e registra tentativas. Toda nova operação crítica do treino deve entrar em `tools/directml_smoke.py` se houver risco de incompatibilidade DirectML.

Para o 4600G:

- mantenha DirectML como tentativa acelerada no Windows;
- aceite fallback para CPU;
- não confunda memória UMA reservada com VRAM discreta;
- deixe pelo menos parte da RAM/SMT para Windows, FastAPI e ingestão;
- prefira reduzir batch/contexto e usar acumulação a causar paginação do sistema.

Overrides de ambiente são limites de operador, mas continuam sujeitos aos caps internos. Documente qualquer mudança nesses caps com benchmark de memória e tempo.

## Antigravity, Codex e team bus

Antes de aplicar configuração global, execute o dry-run:

```powershell
.\jarvis_localhost\tools\install_antigravity_chatgpt.ps1
```

Depois da revisão:

```powershell
.\jarvis_localhost\tools\install_antigravity_chatgpt.ps1 -Apply
```

O instalador deve preservar servidores/preferências existentes, alterar somente chaves gerenciadas, criar backups antes da escrita, validar JSON/TOML e registrar hashes no manifesto local. Nunca imprima o conteúdo completo das configurações, pois elas podem conter segredos de outros MCPs.

Fluxo de colaboração recomendado:

1. Registre `codex` e `antigravity` no team bus.
2. Crie uma tarefa autocontida com chave de idempotência.
3. O agente responsável faz claim atômico antes de escrever.
4. Registre arquivos sob responsabilidade e resultado esperado.
5. Apenas um agente edita cada arquivo por vez.
6. Um segundo agente revisa o diff em modo somente leitura.
7. Poste resultados de testes e limitações; não apenas “concluído”.
8. Use `execute_command` somente com argv explícito, timeout e necessidade concreta.

Para testar diretamente o servidor MCP, os testes de integração iniciam dois processos stdio sobre um SQLite WAL temporário e comprovam mensagens nos dois sentidos, tarefas concorrentes, idempotência, timeout e auditoria:

```powershell
$python = ".\jarvis_localhost\.venv\Scripts\python.exe"
& $python -m unittest jarvis_localhost.tests.test_team_bus_server -v
```

Rollback também começa em dry-run:

```powershell
.\jarvis_localhost\tools\rollback_antigravity_chatgpt.ps1
.\jarvis_localhost\tools\rollback_antigravity_chatgpt.ps1 -Apply
```

Se um arquivo global mudou fora do escopo reconhecidamente gerenciado, o rollback para. Mudanças posteriores fora do bloco `jarvis-team-bus` do Codex são preservadas por remoção seletiva; o arquivo exclusivo de permissões do CLI só pode ser removido quando continua semanticamente igual às sete regras instaladas.

## Dados de teste

Use conteúdo sintético e pequeno. Não copie PDFs privados, trechos confidenciais ou bancos reais para fixtures. Testes que precisam de PDF devem criá-lo em diretório temporário e removê-lo ao final.

Nunca versione:

- `.env`, tokens ou credenciais;
- PDFs e uploads;
- SQLite, WAL ou SHM;
- vector stores/embeddings;
- checkpoints/tokenizadores treinados;
- relatórios de hardware que contenham caminhos/identificadores desnecessários;
- venv, caches e arquivos temporários.

## Checklist de entrega

- [ ] O escopo e os arquivos responsáveis estão claros.
- [ ] Nenhum peso/API/download semântico foi introduzido.
- [ ] Proveniência e lineage foram preservadas.
- [ ] Segurança de Host/Origin/CSRF/upload continua coberta.
- [ ] `validate.ps1` passou integralmente.
- [ ] Smoke do backend real passou quando o código neural mudou.
- [ ] Respostas strict/RAG foram verificadas com corpus sintético autorizado.
- [ ] `git diff --check` passou.
- [ ] `git status --short` não contém runtime privado.
- [ ] Limitações e partes não validadas foram registradas.
- [ ] Nenhum commit, push, deploy ou publicação foi feito sem autorização específica.

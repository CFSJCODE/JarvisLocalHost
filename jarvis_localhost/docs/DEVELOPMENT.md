# Desenvolvimento

Este repositorio foi preparado para evoluir como projeto GitHub, mantendo o codigo versionado separado de estado local, arquivos enviados pelo usuario e modelos gerados em runtime.

## Primeira Execucao

```powershell
.\jarvis_localhost\tools\setup.ps1
.\jarvis_localhost\tools\run.ps1
```

O setup cria `jarvis_localhost\.venv`, atualiza `pip` e instala `jarvis_localhost\requirements.txt`. O run usa `jarvis_localhost\.venv\Scripts\python.exe` quando ele existe; caso contrario, usa `python` do PATH.

## Validacao

```powershell
.\jarvis_localhost\tools\validate.ps1
```

Essa validacao nao instala dependencias pesadas. Ela confere a estrutura basica e executa `compileall` nos arquivos Python dentro de `jarvis_localhost/`. Isso cobre erro de sintaxe sem obrigar o GitHub Actions a baixar PyTorch em todo push.

## Padrao De Commits

Use mensagens curtas e imperativas, por exemplo:

```text
Organiza estrutura inicial do Jarvis LocalHost
Corrige contrato de upload de PDF
Documenta fluxo de treino local
```

Antes de commit:

```powershell
git status -sb
.\jarvis_localhost\tools\validate.ps1
git diff --check
```

## Arquivos Que Nao Devem Entrar No Git

Nao versionar:

- bancos SQLite;
- PDFs enviados pelo usuario;
- imagens extraidas de documentos;
- embeddings e vector stores;
- checkpoints `.pt`/`.pth`;
- `.env` e credenciais locais;
- ambientes virtuais.

Quando precisar compartilhar um exemplo de dados, crie um arquivo pequeno, anonimizado e documentado fora de `jarvis_localhost/data/` e `jarvis_localhost/uploads/`.

## Estado Do Projeto

O projeto ainda esta em desenvolvimento. Antes de tratar uma falha como regressao, confirme:

- se as dependencias opcionais estao instaladas;
- se o Tesseract esta no PATH para OCR;
- se ha modelo/tokenizer treinado em `jarvis_localhost/data/models/`;
- se os PDFs necessarios foram reenviados localmente;
- se o recurso de voz/cluster foi habilitado por variavel de ambiente.

## Auditoria Tecnica (2026-08-30)

Uma auditoria completa (arquitetura, pipeline de PDF, chunking, retrieval, geracao, seguranca, testes e um teste end-to-end real com PDFs) foi realizada e esta registrada em detalhe no relatorio entregue ao operador nessa data. Resumo do que mudou no codigo:

- `rag/grounding.py`: removida uma regex que reescrevia qualquer palavra iniciada por "compu" para uma frase fixa de robotica em QUALQUER documento (corrompia respostas extrativas), e duas regex que hardcodavam vocabulario de um livro especifico ("Robotica"/"Capitulo") — ambas violavam a regra deste projeto de nunca usar listas/nomes de dominio fixos.
- `processing/pdf_processor.py`: nova excecao `EmptyDocumentError` — um PDF que produz zero chunks canonicos nunca mais e persistido como documento "fantasma"; `minimum_words` deixou de ser fixado em `1` (sem filtro) e agora e configuravel (padrao 12, igual ao de `corpus/chunker.py`).
- `corpus/chunker.py`: `minimum_words` agora e de fato aplicado — janelas/segmentos curtos demais (ex.: um titulo de secao isolado como "1.1" que fica sozinho porque o proximo span tambem e um titulo) sao fundidos com um vizinho compativel em vez de virarem chunks permanentes de 1-2 palavras, sem nunca misturar texto com tabela/OCR nem perder conteudo real.
- `corpus/manifest.py`: novo metodo `CorpusManifest.remove_document()` e a ferramenta `tools/corpus_hygiene.py` para higienizar documentos fantasma que já existam no `corpus_manifest.json` de uma ingestao anterior a esta correcao.
- `server/app.py`: `uvicorn.run(..., reload=False)` explicito; novo `jarvis_localhost/logging_config.py` com logging estruturado (console + arquivo rotativo em `data/logs/`), usado pelo ciclo de vida do servidor.

## Continuacao Da Auditoria Tecnica (2026-08-31)

Correcoes adicionais feitas apos o relatorio inicial, com a mesma exigencia de evidencia real (comando executado + saida real, nunca simulada):

- `rag/grounding.py`: a heuristica anti-ruido (`_is_index_or_tabular_noise`) que descartava passagens numericas legitimas (ex.: "consumo < 800mW") por densidade bruta de digitos foi corrigida. Agora conta apenas "numeros nus" (`_bare_number_digit_ratio`) — um digito colado a uma letra no mesmo token ("800mw", "8MB", "5V") nao entra na contagem, so numeros isolados tipo "12" ou "3.4" (paginas/secoes/indices) entram. Sem nenhuma lista de unidades ou vocabulario fixo. Verificado reexecutando as 10 perguntas do teste E2E contra o mesmo corpus/checkpoint real: a pergunta sobre consumo de aquecimento do MQ-5 vira PASS (antes FAIL), sem nenhuma regressao nas outras 9. Uma linha de tabela achatada (memoria flash do ESP32) continua filtrada de proposito, por uma regra diferente e inalterada (excesso de virgulas+tracos) — limitacao separada, documentada, com teste proprio travando esse comportamento.
- `core/brain.py`, `processing/pdf_processor.py`, `storage/database.py`: os 16 `print()` de runtime restantes foram migrados para `logging_config` (a migracao anterior so cobria `server/app.py`). Deliberadamente NAO migrados, com motivo: as ferramentas CLI (`tools/corpus_hygiene.py`, `tools/hardware_probe.py`, `tools/directml_smoke.py`, que dependem de stdout limpo para o operador), o servidor MCP `integrations/team_bus_server.py` (que reserva stdout para o protocolo JSON-RPC e ja usa stderr para diagnostico, e e deliberadamente livre de dependencias do projeto) e `legacy/engine_AI_legacy.py` (codigo morto, sem nenhum import no resto do projeto).
- `logging_config.py`: durante a migracao acima, a verificacao com execucao real (nao so testes unitarios) revelou que `get_logger(__name__)` nao conectava um logger real como `"jarvis_localhost.core.brain"` ao logger `"jarvis"` configurado por `configure_logging()` — a checagem de prefixo antiga (`name.startswith("jarvis")`) tratava esse nome como "ja pertencente" a arvore por comecar com as mesmas letras, sem checar o ponto separador, entao as mensagens eram descartadas silenciosamente. Corrigido; validado tanto por teste (handler-sonda anexado ao logger `"jarvis"`) quanto por execucao real (linhas `[Brain]`/`[PDFProcessor]`/`[DB]` conferidas em `data/logs/jarvis.log` apos operacoes reais).
- Suite de testes: 98 (relatorio original) -> 112 (fim da sessao anterior) -> **119 passed, 0 failed** (fim desta continuacao), incluindo os novos testes de ruido numerico e de hierarquia de logging.
- Nao corrigido, permanece como estava: F7 (encoder denso sub-treinado pode piorar o ranking hibrido) — achado operacional/de processo, nao um bug pontual de codigo; ver relatorio da auditoria para a recomendacao de um "gate de qualidade pos-treino".

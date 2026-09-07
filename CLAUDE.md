# JARVIS LocalHost — orientação para sessões de IA neste repositório

## Atualização de recursos — 2026-09-07

Consulte primeiro `docs/historico/AJUSTE_RECURSOS_2026-09-07.md`. O pedido atual mantém o runtime
em E: e atualiza somente o backup oculto em D:. O LM concluído no passo 1890
foi preservado; uma parada cooperativa gravou o recuperador no lote 8521,
protegido em `D:\.JarvisLocalHost-Backup\2026-09-07\checkpoint-8521`.
A configuração agora permite orçamento explícito de 24 GiB de RAM e opções
de execução DirectML com padding limitado por lote e fusão das duas entradas,
mantendo oito pares lógicos. O orçamento não é RAM reservada nem limite RSS.
209 testes e 95 arquivos Python em validação estática passaram. Consulte o
relatório e `/api/train/status` para a medição física e o estado da retomada.
Os registros abaixo são históricos; não usar seus números como status atual.

## Atualização de recuperação — 2026-09-06

Consulte `docs/historico/RECUPERACAO_2026-09-06.md` antes de seguir o estado histórico abaixo.
Codex sincronizou diretamente com a sessão anterior do Claude Code. O corpus
atual validado tem 269 documentos e 233.223 chunks. O checkpoint íntegro mais
recente agora é `jarvis_final`: 1.890 passos de LM concluídos às 03:02 de
06/09, 30.965.760 tokens. Foi retomado originalmente de `step_1180`.
O recuperador parou às 11:24 com exceção de texto vazio. Correções adicionais
foram validadas e o servidor voltou a iniciar às 11:51; consulte a API e o
relatório de recuperação para o avanço real após essa reinicialização.
Às 12:08 a API confirmou recuperador ativo no lote 10/174820, retomada LM no
passo final 1890 e nenhum erro novo. O checkpoint contrastivo do lote 1 já foi
validado e copiado para D:. A curiosidade restaurou 96 análises já salvas.
Esse é um registro pontual; consulte `/api/train/status` para o estado atual.

E: é um disco externo USB e sofreu indisponibilidade com evento NTFS 50 em
05/09, 23:31:43. Voltou a ser acessível e os artefatos foram validados.
Por instrução expressa do usuário, `D:\.JarvisLocalHost-Backup\2026-09-06`
é uma pasta oculta usada SOMENTE como backup; o runtime continua neste projeto
em E:. Nunca iniciar uma segunda instância usando o backup.

A configuração soberana foi restaurada para 1. A linhagem histórica registra
modo desligado, mas o corpus atual foi validado sem trechos OCR e o checkpoint
usa inicialização aleatória, sem pesos externos. Não afirmar soberania contínua
no passado. Preservar os arquivos de recuperação em erros/cancelamentos.

Novas correções: carregamento restrito de estado legado DirectML; validação de
pesos/otimizador/linhagem; preservação de checkpoints durante publicação;
liberação de arquivos de log; watchdog com `--training-only` sem ingestão;
preparação linear de pares, avaliação contrastiva em lotes e checkpoint próprio
do recuperador, com lote 8 validado no DirectML; avaliações LM retomáveis com
checksum e limite do currículo de curiosidade proporcional ao corpus.
190 testes passaram, além de 93 arquivos Python em validação estática.
A ativação do watchdog foi bloqueada pela revisão automática nesta sessão;
não presumir que ele está rodando. Consulte a API para o estado vivo do treino.

O restante deste documento é histórico de 2026-09-01.

Este arquivo é lido automaticamente por sessões Claude Code abertas aqui.
Foi escrito em 2026-09-01 por uma sessão Claude (Cowork) que conduziu uma
auditoria longa deste projeto e está entregando o trabalho para continuar
em Claude Code. Leia isto primeiro; os documentos abaixo dão o histórico
completo, em ordem cronológica:

1. `docs/historico/RELATORIO_AUDITORIA_JARVIS_LOCALHOST.md` — relatório da auditoria original (achados F0-F9), entregue 2026-08-31.
2. `docs/historico/AUDITORIA_ADENDO_2026-09-01.md` — o que aconteceu depois: extensão do F1, um incidente real de queda de treino, e F10 (recurso novo de retomada de checkpoint).
3. `docs/AGENTS.md` — protocolo de colaboração multiagente já em vigor neste repo (ver "Regras não-negociáveis" abaixo — é importante).

## Regras não-negociáveis

- **Regra de Evidência**: nunca declarar algo corrigido/funcionando sem rodar e mostrar o comando + saída real. Não basta "o código parece certo" — execute.
- **Dados reais do usuário**: este repositório processa a biblioteca pessoal real do usuário. Nunca inventar/assumir conteúdo de documentos; nunca deletar dados do usuário sem autorização explícita; scripts de teste usam corpus sintético isolado, nunca o `data/` de produção diretamente sem necessidade.
- **Modo soberano é intencional, não um bug**: tokenizer/LM/retriever são treinados do zero no corpus local, sem pesos pré-treinados externos (ver `ARCHITECTURE.md`). PDFs sem camada de texto (scans) são corretamente REJEITADOS por padrão (`SovereignModeViolation`) — isso é o design funcionando, não uma falha a "corrigir" habilitando OCR de terceiros por padrão.
- **Corpus é closed-world com hash de linhagem**: qualquer mudança no conjunto de documentos muda `corpus_sha256`/`canonical_corpus_sha256`, o que invalida qualquer checkpoint de treino anterior (é assim que o F10 decide se pode retomar ou tem que treinar do zero — ver o adendo).
- **Outros agentes podem estar trabalhando neste mesmo repo** (ver `docs/AGENTS.md`: protocolo Antigravity+Codex via barramento MCP `jarvis-team-bus`). Releia o estado em disco antes de editar; um agente por arquivo por vez; nunca declarar conclusão só com base em arquivo, sempre com teste real.
- **Gotcha de ambiente** (observado usando uma ponte MCP de terminal remoto, pode ou não se aplicar ao Bash nativo do Claude Code): comandos PowerShell inline com `-Command "..."` contendo variáveis `$algo` tiveram o token silenciosamente removido em alguns casos. Se um comando com `$variavel` falhar de forma estranha, escrever um `.ps1` e rodar com `-File` resolve.

## Estado real confirmado agora (2026-09-01, ~14:30 UTC)

- Servidor rodando: `E:\SoftwareProjects\JarvisLocalHost\jarvis_localhost\.venv\Scripts\python.exe -m jarvis_localhost.server.app`, a partir de `E:\SoftwareProjects\JarvisLocalHost` (repo root), porta 8008. HUD em `http://127.0.0.1:8008/`.
- `GET /api/train/status` confirmado: `is_training:false`, `is_trained:false`, `documents:57`. Corpus atual = 57 documentos de eletrônica/robótica (higienizado — 3 "fantasmas" removidos, ver F1/adendo).
- Endpoints mutantes (`POST`) exigem sessão+CSRF: `GET /api/session` primeiro (seta cookie, devolve `csrf_token`), depois enviar cookie + header `X-Jarvis-CSRF` no POST. Upload de PDF é `POST /api/pdf/upload` (multipart). Início de treino é `POST /api/train/start` (roda em thread de segundo plano no servidor — não bloqueia).

## Tarefa em andamento agora: crescimento incremental do corpus com a biblioteca acadêmica

O usuário pediu para reiniciar o treino do zero e ir treinando "aos poucos"
com os documentos disponíveis em `E:\Acadêmico\Livros E Arquivos De Estudos`
— e explicitamente enquadrou isto como "pense que esse projeto é uma IA
equiparada a um bebê e com o tempo ela vai aprendendo, absorvendo
conhecimentos e ficando mais inteligente". Isso define a estratégia:

- **MESCLAR, nunca substituir.** Os 57 documentos atuais (eletrônica/robótica) ficam. Novos documentos se somam ao mesmo corpus — ele não "esquece" o que já sabe.
- **Fonte**: `E:\Acadêmico\Livros E Arquivos De Estudos` tem 28 subpastas de assuntos muito diferentes entre si (de Gastronomia e Literatura a Computação Quântica, Engenharia Aeronáutica, Cibersegurança, HPC, Redes, Bancos de Dados, etc. — listagem completa disponível rodando `list_directory` nessa pasta). Ignorar `CATÁLOGO DURATEX.pdf` solto na raiz (catálogo de produto, não é material de estudo).
- **Ordem proposta ao usuário** (ele não pediu uma ordem específica, delegou o "como"): começar pelas subpastas mais próximas do que o modelo já sabe — `Hardware_Eletrônica_Elétrica` e `Robótica` — antes de expandir para assuntos novos. Ajustar se o usuário disser algo diferente.
- **Por que em lotes, e não tudo de uma vez**: o tamanho do corpus decide automaticamente o tamanho do modelo/tempo de treino (ver `tools/hardware_probe.py`); a APU (AMD 4600G, DirectML) tem limites reais de VRAM/RAM compartilhada. Um salto grande demais pode gerar um modelo grande demais ou um treino de dias.
- **Por que não dá pra "resumir" entre lotes**: cada lote novo muda o corpus, logo muda `corpus_sha256`/`canonical_corpus_sha256` — o F10 corretamente vai recusar retomar de um lote pro outro (é o guard funcionando certo, não um bug). Cada lote é um treino do zero completo (BPE + LM + retriever). O resume do F10 serve para sobreviver a uma QUEDA dentro do MESMO lote, não para pular entre lotes.
- **PDFs sem camada de texto (scans)**: pulados automaticamente pelo modo soberano padrão (comportamento correto, não um bug — ver F1). Registrar quais foram pulados e reportar ao usuário, que pode decidir habilitar OCR (`JARVIS_ALLOW_PRETRAINED_OCR=true`) caso a caso — isso desliga uma garantia de soberania, então é decisão do usuário, nunca automática.

### Próximo passo concreto
Ainda não implementado: um script (PowerShell ou Python chamando a API) que,
para uma pasta dada, obtenha sessão+CSRF (`GET /api/session`), itere os PDFs
dessa pasta e envie cada um para `POST /api/pdf/upload`, registrando
sucesso/rejeição (documento sem camada de texto = rejeição esperada, não
erro) — depois disparar `POST /api/train/start` e acompanhar
`GET /api/train/status` até `is_training:false`. Começar por
`Hardware_Eletrônica_Elétrica` e/ou `Robótica` como primeiro lote.

# Recuperação e validação do Jarvis — 2026-09-06

## Escopo e estado

Objetivo: sincronizar com Claude Code, retomar o último checkpoint válido e
corrigir falhas observadas sem substituir o corpus. A sessão anterior do Claude
Code respondeu a uma consulta real somente leitura; suas correções existentes
de etapa LM concluída, lote do recuperador e frequência de checkpoint foram
preservadas. A coordenação foi registrada no barramento local (tarefas 2 a 6).

Servidor reiniciado no projeto original em E:, em `http://127.0.0.1:8008`,
com DirectML e RAG estrito. O início da retomada foi aceito pela API às
23:49 de 05/09. O LM retomou no passo 1181, avançou e terminou os 1.890
passos às 03:02 de 06/09. Às 11:24 o fluxo parou em 77%, na entrada do
recuperador. A exceção antiga não registrou tipo nem traceback; não é possível
atribuir retrospectivamente uma causa exata a partir da mensagem vazia.
Na inspeção, o processo usava 17,8 GiB de RAM e havia cerca de 1 GiB livre.
Depois de corrigir o caminho de processamento e preservar os estados, somente
esse servidor foi interrompido e reiniciado às 11:51 em E:.

## Comprovação da nova retomada — 06/09, 12:08

- API aceitou a retomada às 11:54. Depois da validação e preparação, identificou
  `jarvis_final` e informou `resumed_from_step=1890`, sem novos passos de LM.
- Às 12:05:49 gravou `retriever_training_state.pt`, de 115.309.329 bytes.
  Às 12:08 a API confirmou avanço do lote 1 para o lote 10 de 174.820,
  `is_training=true`, `is_trained=false` e nenhuma mensagem de erro.
- O estado contrastivo salvo no lote 1 foi carregado em CPU com
  `weights_only=True`, validação de tensores/otimizador/contadores e comparação
  dos hashes de corpus/tokenizador. Perda do lote salvo: 2,2566785812.
- Um conjunto de oito arquivos necessários à retomada foi copiado e conferido
  por hash em `D:\.JarvisLocalHost-Backup\2026-09-06\checkpoint-retriever-start`.
- A geração original da curiosidade voltou a carregar: 96 análises, ciclo 1,
  sem erro de checkpoint. O explorador está pausado durante o treino principal.
- O painel em `http://127.0.0.1:8008/` foi recarregado e mostrou conexão,
  treinamento ativo e lista de documentos. O erro de conexão recusada foi
  resolvido para esta execução.
- Claude Code confirmou a leitura dos dois documentos de continuidade em
  chamada real somente leitura; essa confirmação não foi tratada como prova
  independente da execução. O estado acima foi medido diretamente na API.

O treinamento completo continua em execução; recuperador, avaliações e índice
ainda precisam terminar. Não há prazo de conclusão validado. O checkpoint no
disco é gravado a cada 200 lotes (e no primeiro/fim de época); a interface
atualiza a cada dez. Uma queda abrupta pode perder o intervalo ainda não salvo.

## Falha de armazenamento e backup

O Windows registrou falha de gravação atrasada em E:\$Mft (NTFS 50),
05/09/2026 23:31:43. O disco USB Samsung ST1000LM024 desapareceu e depois
voltou. Não foi determinada a causa física; nenhuma formatação, reparação de
sistema de arquivos, reinicialização do Windows ou remoção de dados foi feita.

Backup oculto, SOMENTE backup conforme instrução do usuário:
`D:\.JarvisLocalHost-Backup\2026-09-06\JarvisLocalHost`.
Cópia inicial: 29.825 arquivos, aproximadamente 16,43 GiB, zero falhas ou
incompatibilidades reportadas. Os bancos copiados passaram em `quick_check`.
O checkpoint também possui cópia separada e conferida por hash em
`D:\.JarvisLocalHost-Backup\2026-09-06\checkpoint-1180`.
A etapa LM completa tem cópia separada em `checkpoint-final-1890`, cinco
arquivos conferidos por SHA-256. A geração original da curiosidade também foi
conferida e preservada em `curiosity-before-restart` (seis arquivos).
A cópia provisória desses cinco arquivos em C: foi removida após conferência.

O atributo Oculto foi confirmado no diretório pai. A pasta permanece acessível
pelo caminho direto e por quem habilitar a exibição de itens ocultos.

## Corpus e retomada

- 269 documentos, 19.341 páginas, 233.223 trechos autorizados.
- 217.508 trechos de camada textual e 15.715 trechos de tabelas; nenhum de OCR.
- Corpus, tokenizer e modelo validados com os hashes recalculados.
- Checkpoint 1180: 19.349.504 tokens; próximo passo 1181 de 1890.
- Todos os 8.366.592 parâmetros e os 200 tensores do otimizador são finitos.
- O modo soberano estava desligado na configuração histórica; foi restaurado
  para 1 sem modificar as demais opções. Não se declara garantia retroativa.

## Correções aplicadas

- `ai/trainer.py`: validação do par modelo/otimizador, carregamento restrito
  compatível com o estado antigo DirectML, histórico coerente com o checkpoint
  e liberação da trava quando a preparação falha.
- `core/brain.py`: preservação de staging em erro/cancelamento; descoberta de
  checkpoints íntegros; uso correto do hash textual na retomada; artefato de
  inferência separado do checkpoint; recuperação de publicação interrompida;
  exclusão conservadora de pipelines ativos quando seu arquivo está ilegível.
- `logging_config.py`: fechamento dos handlers substituídos no Windows.
- `retrieval/contrastive.py`: elimina varreduras quadráticas ao preparar pares;
  avaliação limitada ao lote; checkpoint atômico com pesos, otimizador, RNG,
  amostrador e posição. Rejeita incompatibilidade de corpus/configuração e
  tensores/contadores inválidos. Progresso exibido a cada dez lotes; checkpoint
  no primeiro, a cada 200 lotes e ao terminar cada época.
- `core/brain.py`: salva a amostragem e o tamanho do lote para retomar com a
  mesma configuração. Limita o lote pelo orçamento de memória; lote 8 testado
  fisicamente. Libera buffers do LM antes do recuperador. Erros passam a
  registrar tipo e traceback, mesmo quando sua mensagem estiver vazia.
- `ai/trainer.py`: avaliação por trecho passa a ter cache incremental atômico,
  com identificação do modelo/corpus e checksum. Trechos concluídos são
  reaproveitados, e formatos de tensores limitados reduzem variações na GPU.
- `curiosity/engine.py`: arquivos de currículo e amostragem têm limite derivado
  do número validado de trechos, com teto 256 MiB por arquivo. O currículo
  legítimo de 48,7 MB deixa de ser recusado pelo antigo limite fixo de 32 MiB.
  Manifesto, checksums, arquitetura e vínculo com o corpus continuam exigidos.
- `tools/watchdog.py`: preservação de artefatos sem manifesto, acompanhamento
  até conclusão efetiva, controle de duplicidade e modo `--training-only`.
- Regressões em `test_trainer_resume.py`, `test_training_recovery.py`,
  `test_watchdog.py`, `test_audit_fixes.py` e `test_logging_config.py`.

## Validação e limitações

190 testes passaram na suíte isolada final em 23,014 s; compilação sintática e
tabnanny passaram em 93 arquivos Python; `git diff --check` aprovado.
Uma revisão independente aprovou as alterações de orquestração e avaliação LM;
seu achado de checksum das medições foi corrigido e testado.
O smoke físico DirectML passou no LM, recuperador e curiosidade/PPO.
O ensaio adicional com 8 camadas, contexto 512 e lote 8 interrompeu no lote 1
e retomou no lote 2, sem repetir o primeiro, com parâmetros finitos.
Em CPU, a retomada contrastiva com dropout reproduziu exatamente pesos e
histórico do treino contínuo, para amostragem uniforme e ponderada.
233.223 trechos sintéticos geraram 699.668 pares em 3,125 s.
Há um fallback conhecido de uma operação AdamW para CPU; isso não impediu
a execução acelerada do restante do ensaio.

Evidências e executor reproduzível:
`D:\.JarvisLocalHost-Backup\2026-09-06\validation-resumed-final.txt` e
`D:\.JarvisLocalHost-Backup\2026-09-06\run-isolated-suite.py`.

A ordem exata do DataLoader legado não é restaurada integralmente; pesos,
otimizador, contadores e RNG são recuperados. O recuperador agora possui
retomada própria entre lotes. A equivalência exata de pesos foi testada em CPU;
o ensaio DirectML confirmou execução e restauração reais, sem prometer
determinismo bit a bit nessa GPU. A avaliação de incerteza usa negativos locais
ao lote quando o corpus é grande; essa escolha consta no manifesto e não é
calibrada como comparação global contra os 233 mil trechos.
A preparação de uma retomada LM ainda reconstrói tokens/datasets antes de
liberá-los; o pico inicial de RAM não foi eliminado. As medições LM da execução
anterior não tinham sido salvas e precisarão ser medidas após o recuperador;
daqui em diante terão sua própria retomada. Não se afirma ausência de todos os
bugs nem conclusão/qualidade final do treinamento completo.

O watchdog foi testado, mas sua ativação em segundo plano foi rejeitada pela
revisão automática, que informou bloqueio por política. Não foi instalada
automação alternativa para contornar a rejeição.

O erro separado do Claude Desktop correspondeu a 0x80070020 na ativação AppX;
o pacote estava registrado e não parcialmente instalado. O recurso bloqueado
não foi identificado. A nova inspeção às 11:56–11:58 encontrou recorrência
às 11:31:52/11:31:55 de 06/09; a consulta nativa de arquivos abertos foi
negada pelo Windows. Claude Code CLI funcionou; Desktop não foi reparado.

Nenhum commit, push, publicação, deploy ou ingestão adicional foi realizado.

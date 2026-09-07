# Atualização de recursos e recuperação — 2026-09-07

Pedido: atualizar o backup oculto e o checkpoint de recuperação, permitir orçamento
de 24 GiB de RAM e melhorar o uso da GPU, preservando o treinamento existente.

## Estado preservado antes da alteração

- Runtime: `E:\SoftwareProjects\JarvisLocalHost`, disco externo USB.
- Backup exclusivo: `D:\.JarvisLocalHost-Backup\2026-09-07`; pasta superior com
  atributo Windows `Hidden`. Não iniciar nenhuma instância a partir de D:.
- Corpus: 269 documentos, 233.223 trechos autorizados. Nenhuma ingestão nova.
- LM: 1.890 passos concluídos; `jarvis_final.pt` preservado, SHA-256
  `3416fa94729d0e5c55ed764ce23ee7d98c8530256a7aa9851149910dce01f02b`.
- Recuperador: cancelamento cooperativo gravou o lote **8.521** às 11:53:46
  (America/Sao_Paulo), antes da parada do servidor. Pesos, 101 estados do
  otimizador, linhagem e posição foram lidos e verificados como finitos.
- Ponto de partida protegido: `checkpoint-8521`, oito arquivos essenciais e
  `integrity.json` com tamanho e SHA-256 conferidos na origem e no destino.
- Linhagem canônica: `65f5d7f3c4be99006c64e7d89f58426995d78ddb722be64c0444ea190c4ded08`.

## Decisões de execução

O pedido explícito `JARVIS_MAX_RAM_GB=24` agora é aceito, mantendo reserva de
pelo menos 4 GiB ou 20% da RAM física visível. Os padrões continuam conservadores.
O valor é um teto de planejamento: não reserva 24 GiB, não força consumo e não
impõe limite de RSS. A GPU integrada compartilha RAM com o Windows; seus 6 GiB
de orçamento não constituem capacidade adicional independente.

A configuração local seleciona `directml` e ativa `bucketed`, mínimo 32, com
`JARVIS_RETRIEVER_FUSED_PAIRS=1`. O preenchimento utiliza comprimentos limitados
32/64/128/256/512 conforme os tokens do lote. Até 256 tokens, âncoras e positivos
passam juntos pelo encoder e são separados antes do InfoNCE. Acima disso,
as entradas passam separadamente, evitando a regressão de tempo e memória
medida na GPU com sequências de 512 tokens. Permanecem oito pares lógicos,
matriz 8 × 8, arquitetura, pesos, momentos, ordem de amostragem e meta de
174.820 lotes. Não há troca de corpus ou download de modelo.

O checkpoint registra a política de execução e qualquer transição. Mudar o
formato dos tensores altera o consumo aleatório do dropout: a continuação
otimizada não promete equivalência numérica bit a bit com a política antiga.
Retomadas com a mesma política passaram nos testes determinísticos em CPU.
Reverter as opções para `fixed` e `0` preserva a retomada e registra nova transição.

## Validação e backup

209 testes automatizados passaram em runtime isolado, incluindo recuperação após
falha sem repetir o LM concluído. Compilação e tabnanny validaram 95 arquivos
Python; `git diff --check` passou. Dois passes independentes revisaram as opções
de recursos e a compatibilidade dos checkpoints.

A cópia completa contém 31.677 arquivos, cerca de 16,68 GiB. A primeira passagem
encontrou o arquivo temporário de trava do treino ocupado; a passagem final,
com o servidor parado, terminou sem falhas. Não se apagaram backups anteriores.
Foram conferidos também hashes de 15 arquivos críticos e criados snapshots
consistentes dos dois bancos SQLite, ambos com `PRAGMA quick_check = ok`.

Uma validação independente do backup confirmou os 269 documentos, 233.223
trechos, 807 artefatos autorizados e o hash canônico esperado. Foram verificados
também tamanho e SHA-256 de 260 PDFs físicos presentes. Nove PDFs não foram
encontrados pelos nomes registrados nas pastas de upload verificadas, tanto
na origem E: como na cópia D:. A ausência física já existe na origem; os
dados canônicos desses documentos passaram. Pelos registros locais, localizaram-se
quatro originais com tamanho e SHA-256 exatos, acrescentados exclusivamente ao
backup em D:, sem sobrescrever arquivos. Assim, o backup reúne 264 PDFs físicos
verificados; cinco originais continuam sem localização confirmada. Os quatro
arquivos suplementares somam 12.391.149 bytes e ficam registrados em
`supplemental-original-pdfs.json`. Não se reingeriu nenhum documento nem se
modificou a linhagem para resolver essa limitação.

Evidências em D: incluem `full-backup-final.log`, `recovery-integrity.json`,
`sqlite-consistent`, `validation-resource-tuning-final.txt` e os registros de revisão.
O backup é um retrato datado; não é espelhamento contínuo nem servidor secundário.

## Recuperação em caso de falha

Se E: ficar indisponível, preservar D: e aguardar a unidade original voltar.
Antes de restaurar, verificar que não existe instância Jarvis ativa na porta
8008 e conferir os hashes do snapshot escolhido. Restaurar os arquivos para
o projeto original em E:, mantendo corpus, tokenizer, pesos, estado do
otimizador, `training_sampling.json` e `retriever_fit.json` do mesmo snapshot.
Para bancos, utilizar os arquivos de `sqlite-consistent` nos destinos indicados
em `recovery-integrity.json`; não misturar um banco restaurado com arquivos WAL
ou SHM de outra geração. Preservar o estado anterior até validar a restauração.

Ao iniciar uma única instância do servidor em E:, solicitar a retomada pela API
local com sessão e CSRF. Confirmar na API que o LM retoma no passo 1.890 e o
recuperador a partir do snapshot escolhido, depois verificar avanço real.
Checkpoints do recuperador são atômicos, no primeiro lote após a retomada,
a cada 200 lotes, ao fim da época e no cancelamento cooperativo. Uma falha
elétrica pode exigir repetir lotes posteriores ao último checkpoint comprometido.

O watchdog permanece sem ativação; não foi contornada a rejeição automática
registrada na recuperação anterior. O hook de resumo final é responsável pelo
elo de memória com Claude; nenhuma segunda captura manual é necessária.

## Medição física e retomada

O benchmark isolado usou DirectML real, FP32, arquitetura de 8.530.432 parâmetros,
oito pares lógicos, aquecimento e sincronização por leitura dos resultados.
Dados sintéticos curtos passaram de média 5,729 s/lote com preenchimento fixo e
entradas separadas para 0,280 s/lote com bucket mínimo 32 e fusão (20,4 vezes
nesse caso específico). Isso não constitui previsão do corpus real.

No comprimento máximo, a fusão irrestrita elevou o pico de RSS a aproximadamente
6,97 GiB. O limite de 256 tokens reduziu esse pico a 3,95 GiB, usando novamente
duas entradas separadas de oito sequências de 512 tokens. Foram medidos média
7,126 s e mediana 6,719 s/lote; o baseline teve mediana 6,098 s. Três amostras
por caso não isolam variações de carga; não se alega ganho para sequências longas.
Pesos, perdas, gradientes e momentos permaneceram finitos. RSS não representa
sozinho a memória do driver ou toda a alocação da GPU compartilhada.

Retomada de produção em validação; registrar o avanço real antes da entrega.

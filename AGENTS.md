# Colaboracao Antigravity e ChatGPT Desktop/Codex

## Objetivo

O Google Antigravity atua como orquestrador, navegador e executor de QA. O
Codex fornecido pelo ChatGPT Desktop atua como analista, implementador e
revisor especializado deste repositorio.

## Permissoes autorizadas

O proprietario autorizou acesso local e de internet sem prompts. Ao chamar o
servidor MCP `codex`, o Antigravity pode usar:

- `cwd`: `E:\SoftwareProjects\JarvisLocalHost`
- `sandbox`: `danger-full-access`
- `approval-policy`: `never`

Essa autorizacao de ferramenta nao amplia o objetivo do trabalho. Nao excluir
dados do usuario, publicar PDFs, transmitir credenciais, fazer push, deploy,
release, alteracao remota ou instalar servico persistente sem que a tarefa
concreta exija isso. Nunca incluir segredos em prompts, codigo, logs ou commits.

## Protocolo de trabalho

1. Antigravity e Codex registram identidades estaveis no servidor MCP
   `jarvis-team-bus` antes de delegar trabalho.
2. O orquestrador cria a tarefa no barramento; o executor a reivindica
   atomicamente, publica atualizacoes e registra o resultado final.
3. Mensagens de coordenacao usam `post_message`/`fetch_messages`. Chamadas
   imediatas podem continuar usando o MCP `codex` e o CLI `agy`, mas a decisao,
   o responsavel e o resultado ficam no barramento auditavel.
4. Apenas um agente escreve em determinado arquivo por vez.
5. Cada agente rele o estado atual em disco antes de editar e preserva mudancas de
   outros agentes.
6. Tarefas iniciais ao Codex devem ser autocontidas. Use `codex-reply` no mesmo
   thread para revisao/continuidade e evite abrir varias sessoes iniciais em
   paralelo sem associacao explicita.
7. Depois da implementacao, outro passe revisa o diff em modo somente leitura.
8. A entrega exige testes automatizados, validacao estatica, smoke test e
   `git diff --check`. Nao declarar conclusao com base apenas em arquivos.

O barramento MCP usa SQLite local em modo WAL. As ferramentas de execucao de
comandos sao intencionalmente auditadas, limitam a saida retida e exigem um
ator identificado. Comandos potencialmente destrutivos, publicacao, deploy,
push e transmissao de segredos continuam fora do escopo sem instrucao concreta
do proprietario, mesmo quando a politica global permite execucao sem prompts.

## Fronteira de rede do Jarvis

A internet pode ser usada durante bootstrap para clonar o repositorio, instalar
dependencias e consultar documentacao primaria. O runtime do Jarvis em
`JARVIS_SOVEREIGN_MODE=1` deve rejeitar downloads, APIs externas, checkpoints,
tokenizers, embeddings e OCR pre-treinados, mantendo todo estado semantico
aprendido vinculado exclusivamente ao corpus autorizado.

## Participação do Claude (Cowork / Claude Code)

Uma sessão Claude (Cowork) conduziu uma auditoria técnica completa deste
repositório entre 2026-08-30 e 2026-09-01 (ver `RELATORIO_AUDITORIA_JARVIS_LOCALHOST.md`,
`AUDITORIA_ADENDO_2026-09-01.md` e `CLAUDE.md`) e está passando a tarefa em
andamento para uma sessão Claude Code continuar neste computador. As mesmas
regras deste documento se aplicam: reler o estado em disco antes de editar,
um agente por arquivo por vez, nunca declarar conclusão sem teste real, e
nada de exclusão de dados do usuário, publicação, deploy ou instalação de
serviço persistente sem necessidade concreta da tarefa. `CLAUDE.md` tem a
orientação completa e o estado/tarefa atuais.

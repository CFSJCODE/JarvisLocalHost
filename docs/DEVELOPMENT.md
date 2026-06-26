# Desenvolvimento

Este repositorio foi preparado para evoluir como projeto GitHub, mantendo o codigo versionado separado de estado local, arquivos enviados pelo usuario e modelos gerados em runtime.

## Primeira Execucao

```powershell
.\scripts\setup.ps1
.\scripts\run.ps1
```

O setup cria `.venv`, atualiza `pip` e instala `requirements.txt`. O run usa `.venv\Scripts\python.exe` quando ele existe; caso contrario, usa `python` do PATH.

## Validacao

```powershell
.\scripts\validate.ps1
```

Essa validacao nao instala dependencias pesadas. Ela confere a estrutura basica e executa `compileall` nos arquivos Python da raiz. Isso cobre erro de sintaxe sem obrigar o GitHub Actions a baixar PyTorch em todo push.

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
.\scripts\validate.ps1
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

Quando precisar compartilhar um exemplo de dados, crie um arquivo pequeno, anonimizado e documentado fora de `data/` e `uploads/`.

## Estado Do Projeto

O projeto ainda esta em desenvolvimento. Antes de tratar uma falha como regressao, confirme:

- se as dependencias opcionais estao instaladas;
- se o Tesseract esta no PATH para OCR;
- se ha modelo/tokenizer treinado em `data/models/`;
- se os PDFs necessarios foram reenviados localmente;
- se o recurso de voz/cluster foi habilitado por variavel de ambiente.

param(
    [switch]$SkipTests,
    [switch]$DirectMLSmoke
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$appRoot = Split-Path -Parent $PSScriptRoot
$repoRoot = Split-Path -Parent $appRoot
Set-Location $repoRoot

$requiredPaths = @(
    "README.md",
    "ARCHITECTURE.md",
    "DEVELOPMENT.md",
    "AGENTS.md",
    "LICENSE",
    ".env.example",
    ".github\workflows\validate.yml",
    ".agents\mcp_config.json",
    "jarvis_localhost\.env.example",
    "jarvis_localhost\sovereign.py",
    "jarvis_localhost\paths.py",
    "jarvis_localhost\server\app.py",
    "jarvis_localhost\core\brain.py",
    "jarvis_localhost\ai\tokenizer.py",
    "jarvis_localhost\ai\language_model.py",
    "jarvis_localhost\ai\dataset.py",
    "jarvis_localhost\ai\trainer.py",
    "jarvis_localhost\corpus\provenance.py",
    "jarvis_localhost\corpus\chunker.py",
    "jarvis_localhost\corpus\manifest.py",
    "jarvis_localhost\retrieval\encoder.py",
    "jarvis_localhost\retrieval\retriever.py",
    "jarvis_localhost\rag\engine.py",
    "jarvis_localhost\curiosity\icm.py",
    "jarvis_localhost\curiosity\ppo.py",
    "jarvis_localhost\hardware\device.py",
    "jarvis_localhost\hardware\profiles.py",
    "jarvis_localhost\integrations\team_bus_server.py",
    "jarvis_localhost\processing\pdf_processor.py",
    "jarvis_localhost\storage\database.py",
    "jarvis_localhost\requirements.txt",
    "jarvis_localhost\requirements-directml.txt",
    "jarvis_localhost\tools\directml_smoke.py",
    "jarvis_localhost\tools\hardware_probe.py",
    "jarvis_localhost\tests",
    "jarvis_localhost\web\static\index.html"
)

foreach ($relativePath in $requiredPaths) {
    if (-not (Test-Path -LiteralPath (Join-Path $repoRoot $relativePath))) {
        throw "Arquivo ou diretorio obrigatorio ausente: $relativePath"
    }
}

$rootEnvironmentExample = Join-Path $repoRoot ".env.example"
$appEnvironmentExample = Join-Path $appRoot ".env.example"
if (
    (Get-FileHash -LiteralPath $rootEnvironmentExample -Algorithm SHA256).Hash -ne
    (Get-FileHash -LiteralPath $appEnvironmentExample -Algorithm SHA256).Hash
) {
    throw ".env.example da raiz diverge do baseline canônico do aplicativo."
}

$pythonExecutable = $null
$pythonPrefix = @()
$venvPython = Join-Path $appRoot ".venv\Scripts\python.exe"

if (Test-Path -LiteralPath $venvPython -PathType Leaf) {
    $pythonExecutable = $venvPython
}
elseif (Get-Command "py.exe" -ErrorAction SilentlyContinue) {
    $pythonExecutable = (Get-Command "py.exe").Source
    $pythonPrefix = @("-3.10")
}
elseif (Get-Command "python" -ErrorAction SilentlyContinue) {
    $pythonExecutable = (Get-Command "python").Source
}
else {
    throw "Python 3.10 nao encontrado. Execute jarvis_localhost\tools\setup.ps1."
}

& $pythonExecutable @pythonPrefix -c `
    "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 10) else 1)"
if ($LASTEXITCODE -ne 0) {
    throw "A validacao exige Python 3.10."
}

Write-Host "[1/4] Compilando o pacote Python..."
& $pythonExecutable @pythonPrefix -m compileall -q $appRoot
if ($LASTEXITCODE -ne 0) {
    throw "A compilacao sintatica falhou."
}

if ($SkipTests) {
    Write-Host "[2/4] Testes ignorados por solicitacao explicita (-SkipTests)."
}
else {
    Write-Host "[2/4] Executando a suite completa de testes..."
    & $pythonExecutable @pythonPrefix -W "error::ResourceWarning" -m unittest discover `
        -s (Join-Path $appRoot "tests") `
        -t $repoRoot `
        -v
    if ($LASTEXITCODE -ne 0) {
        throw "A suite de testes falhou."
    }
}

if ($DirectMLSmoke) {
    Write-Host "[extra] Exigindo a pilha neural acelerada no DirectML..."
    & $pythonExecutable @pythonPrefix (Join-Path $appRoot "tools\directml_smoke.py") `
        --require-backend directml `
        --require-accelerator
    if ($LASTEXITCODE -ne 0) {
        throw "O smoke da pilha neural falhou."
    }
}

Write-Host "[3/4] Verificando artefatos privados rastreados..."
if (Get-Command "git" -ErrorAction SilentlyContinue) {
    $trackedRuntime = @(
        & git ls-files -- "jarvis_localhost/data" "jarvis_localhost/uploads" 2>$null |
            Where-Object { $_ -notmatch "(?:^|/)\.gitkeep$" }
    )
    if ($LASTEXITCODE -ne 0) {
        throw "Nao foi possivel consultar os arquivos rastreados pelo Git."
    }
    if ($trackedRuntime.Count -gt 0) {
        $trackedRuntime | ForEach-Object {
            Write-Host "Runtime privado rastreado indevidamente: $_"
        }
        throw "Remova artefatos de runtime do Git."
    }

    Write-Host "[4/4] Verificando whitespace do diff..."
    & git diff --check
    if ($LASTEXITCODE -ne 0) {
        throw "git diff --check encontrou erros."
    }
}
else {
    Write-Host "[4/4] Git indisponivel; verificacoes de tracking/diff nao executadas."
}

Write-Host "Validacao concluida sem erros."

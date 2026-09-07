$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = Split-Path -Parent $PSScriptRoot
$appRoot = Join-Path $repoRoot "jarvis_localhost"
$serverScript = Join-Path $appRoot "integrations\team_bus_server.py"
$databasePath = Join-Path $appRoot "data\integrations\team_bus.sqlite"
$venvPython = Join-Path $appRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $serverScript -PathType Leaf)) {
    [Console]::Error.WriteLine("Servidor MCP do barramento nao encontrado: $serverScript")
    exit 1
}

$env:JARVIS_TEAM_BUS_DB = $databasePath

if (Test-Path -LiteralPath $venvPython -PathType Leaf) {
    & $venvPython -u $serverScript
    exit $LASTEXITCODE
}

$pyLauncher = Get-Command "py.exe" -ErrorAction SilentlyContinue
if ($pyLauncher) {
    & $pyLauncher.Source -3.10 -u $serverScript
    exit $LASTEXITCODE
}

$python = Get-Command "python.exe" -ErrorAction SilentlyContinue
if ($python) {
    & $python.Source -u $serverScript
    exit $LASTEXITCODE
}

[Console]::Error.WriteLine(
    "Python nao foi encontrado. Execute jarvis_localhost\tools\setup.ps1 e tente novamente."
)
exit 1

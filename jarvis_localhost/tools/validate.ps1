$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$AppRoot = Split-Path -Parent $PSScriptRoot
$RepoRoot = Split-Path -Parent $AppRoot
Set-Location $RepoRoot

$requiredPaths = @(
    "README.md",
    "LICENSE",
    "jarvis_localhost\server\app.py",
    "jarvis_localhost\core\brain.py",
    "jarvis_localhost\ai\neural.py",
    "jarvis_localhost\storage\database.py",
    "jarvis_localhost\processing\pdf_processor.py",
    "jarvis_localhost\requirements.txt",
    "jarvis_localhost\web\static\index.html"
)

foreach ($path in $requiredPaths) {
    if (-not (Test-Path $path)) {
        throw "Arquivo obrigatorio ausente: $path"
    }
}

$Python = "python"
if (Test-Path "$AppRoot\.venv\Scripts\python.exe") {
    $Python = "$AppRoot\.venv\Scripts\python.exe"
}

$pythonFiles = Get-ChildItem -Path $AppRoot -Filter "*.py" -Recurse -File |
    Where-Object { $_.FullName -notmatch "\\.venv\\" } |
    Sort-Object FullName |
    ForEach-Object { $_.FullName }
if ($pythonFiles.Count -gt 0) {
    & $Python -m compileall -q @pythonFiles
}

if (Get-Command git -ErrorAction SilentlyContinue) {
    $trackedRuntime = git ls-files jarvis_localhost/data jarvis_localhost/uploads 2>$null | Where-Object { $_ -notmatch "\.gitkeep$" }
    if ($trackedRuntime) {
        $trackedRuntime | ForEach-Object { Write-Host "Runtime rastreado indevidamente: $_" }
        throw "Remova artefatos de runtime do Git."
    }
}

Write-Host "Validacao concluida sem erros."

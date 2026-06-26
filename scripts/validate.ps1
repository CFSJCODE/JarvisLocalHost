$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$requiredPaths = @(
    "app.py",
    "brain.py",
    "neural.py",
    "database.py",
    "pdf_processor.py",
    "requirements.txt",
    "README.md",
    "static\index.html"
)

foreach ($path in $requiredPaths) {
    if (-not (Test-Path $path)) {
        throw "Arquivo obrigatorio ausente: $path"
    }
}

$Python = "python"
if (Test-Path ".\.venv\Scripts\python.exe") {
    $Python = ".\.venv\Scripts\python.exe"
}

$pythonFiles = Get-ChildItem -Path . -Filter "*.py" -File | Sort-Object Name | ForEach-Object { $_.FullName }
if ($pythonFiles.Count -gt 0) {
    & $Python -m compileall -q @pythonFiles
}

if (Get-Command git -ErrorAction SilentlyContinue) {
    $trackedRuntime = git ls-files data uploads 2>$null | Where-Object { $_ -notmatch "\.gitkeep$" }
    if ($trackedRuntime) {
        $trackedRuntime | ForEach-Object { Write-Host "Runtime rastreado indevidamente: $_" }
        throw "Remova artefatos de runtime do Git."
    }
}

Write-Host "Validacao concluida sem erros."

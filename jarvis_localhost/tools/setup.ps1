param(
    [switch]$NoDirectML
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$AppRoot = Split-Path -Parent $PSScriptRoot
Set-Location $AppRoot

if (-not (Get-Command py.exe -ErrorAction SilentlyContinue)) {
    throw "O Python Launcher (py.exe) nao foi encontrado. Instale Python 3.10 x64."
}

& py.exe -3.10 -c "import sys; assert sys.version_info[:2] == (3, 10), sys.version"
if ($LASTEXITCODE -ne 0) {
    throw "Python 3.10 x64 nao esta disponivel no py.exe."
}

$VenvPython = Join-Path $AppRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $VenvPython)) {
    & py.exe -3.10 -m venv (Join-Path $AppRoot ".venv")
    if ($LASTEXITCODE -ne 0) {
        throw "Nao foi possivel criar o ambiente Python 3.10."
    }
}

& $VenvPython -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) {
    throw "Falha ao atualizar as ferramentas de instalacao."
}
& $VenvPython -m pip install --requirement (Join-Path $AppRoot "requirements.txt")
if ($LASTEXITCODE -ne 0) {
    throw "Falha ao instalar as dependencias-base."
}

$IsNativeWindows = [System.Environment]::OSVersion.Platform -eq [System.PlatformID]::Win32NT
if ($IsNativeWindows -and -not $NoDirectML) {
    & $VenvPython -m pip install --pre --requirement (Join-Path $AppRoot "requirements-directml.txt")
    if ($LASTEXITCODE -ne 0) {
        throw "Falha ao instalar a pilha DirectML. Use -NoDirectML apenas para CPU."
    }
}

& $VenvPython (Join-Path $AppRoot "tools\hardware_probe.py") --compact
if ($LASTEXITCODE -ne 0) {
    throw "O diagnostico de hardware falhou."
}

Write-Host "Ambiente soberano preparado em $AppRoot\.venv"

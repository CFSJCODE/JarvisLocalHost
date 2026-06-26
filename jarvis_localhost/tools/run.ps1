$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$AppRoot = Split-Path -Parent $PSScriptRoot
$RepoRoot = Split-Path -Parent $AppRoot
Set-Location $RepoRoot

$Python = "python"
if (Test-Path "$AppRoot\.venv\Scripts\python.exe") {
    $Python = "$AppRoot\.venv\Scripts\python.exe"
}

$env:PYTHONPATH = $RepoRoot
& $Python -m jarvis_localhost.server.app

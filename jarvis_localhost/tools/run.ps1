$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$AppRoot = Split-Path -Parent $PSScriptRoot
$RepoRoot = Split-Path -Parent $AppRoot
Set-Location $RepoRoot

$EnvFile = Join-Path $RepoRoot ".env"
if (Test-Path -LiteralPath $EnvFile) {
    Get-Content $EnvFile | ForEach-Object {
        $line = $_.Trim()
        if ($line -and -not $line.StartsWith("#") -and $line.Contains("=")) {
            $parts = $line.Split("=", 2)
            $name = $parts[0].Trim()
            $val = $parts[1].Trim()
            [Environment]::SetEnvironmentVariable($name, $val, "Process")
        }
    }
}

$Python = "python"
if (Test-Path "$AppRoot\.venv\Scripts\python.exe") {
    $Python = "$AppRoot\.venv\Scripts\python.exe"
}

$env:PYTHONPATH = $RepoRoot
& $Python -m jarvis_localhost.server.app

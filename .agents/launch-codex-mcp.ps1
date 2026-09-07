$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

# Resolve the Codex executable shipped by the currently installed ChatGPT
# Desktop application. The hashed directory changes when the app updates, so
# the MCP configuration intentionally points to this stable launcher instead
# of pinning an installation-specific absolute executable path.
$codexBinRoot = Join-Path $env:LOCALAPPDATA "OpenAI\Codex\bin"
$codexExecutable = $null

if (Test-Path -LiteralPath $codexBinRoot) {
    $codexExecutable = Get-ChildItem -LiteralPath $codexBinRoot -Filter "codex.exe" -File -Recurse |
        Sort-Object LastWriteTimeUtc -Descending |
        Select-Object -First 1
}

if (-not $codexExecutable) {
    $fallback = Get-Command "codex.exe" -ErrorAction SilentlyContinue
    if ($fallback) {
        $codexExecutable = Get-Item -LiteralPath $fallback.Source
    }
}

if (-not $codexExecutable) {
    [Console]::Error.WriteLine(
        "Codex nao foi encontrado. Instale ou atualize o ChatGPT Desktop/Codex e tente novamente."
    )
    exit 1
}

& $codexExecutable.FullName mcp-server
exit $LASTEXITCODE

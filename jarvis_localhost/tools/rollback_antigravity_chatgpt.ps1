param(
    [switch]$Apply
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$appRoot = Split-Path -Parent $PSScriptRoot
$manifestPath = Join-Path $appRoot "data\integrations\antigravity-chatgpt-manifest.json"
$codexConfig = Join-Path $env:USERPROFILE ".codex\config.toml"
$cliSettings = Join-Path $env:USERPROFILE ".gemini\antigravity-cli\settings.json"
$managedBlockBegin = "# BEGIN jarvis-team-bus (managed)"
$managedBlockEnd = "# END jarvis-team-bus (managed)"
$managedAllowRules = @(
    "command(*)",
    "execute_url(*)",
    "mcp(*)",
    "read_file(*)",
    "read_url(*)",
    "unsandboxed(*)",
    "write_file(*)"
)

function Get-Sha256([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) {
        return $null
    }
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash
}

function Test-CodexManagedBlock([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return $false
    }
    $content = [IO.File]::ReadAllText($Path, [Text.UTF8Encoding]::new($false))
    $beginCount = ([regex]::Matches(
        $content, "(?m)^" + [regex]::Escape($managedBlockBegin) + "\r?$"
    )).Count
    $endCount = ([regex]::Matches(
        $content, "(?m)^" + [regex]::Escape($managedBlockEnd) + "\r?$"
    )).Count
    return ($beginCount -eq 1 -and $endCount -eq 1)
}

function Test-ManagedCliSettings([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return $false
    }
    try {
        $settings = [IO.File]::ReadAllText(
            $Path, [Text.UTF8Encoding]::new($false)
        ) | ConvertFrom-Json
    }
    catch {
        return $false
    }
    $topKeys = @($settings.PSObject.Properties.Name)
    if ($topKeys.Count -ne 1 -or $topKeys[0] -ne "permissions") {
        return $false
    }
    $permissionKeys = @($settings.permissions.PSObject.Properties.Name)
    if ($permissionKeys.Count -ne 1 -or $permissionKeys[0] -ne "allow") {
        return $false
    }
    $actual = @($settings.permissions.allow | Sort-Object)
    $expected = @($managedAllowRules | Sort-Object)
    return (($actual -join "`n") -ceq ($expected -join "`n"))
}

function Remove-CodexManagedBlock([string]$Path) {
    $content = [IO.File]::ReadAllText($Path, [Text.UTF8Encoding]::new($false))
    $pattern = "(?ms)^" + [regex]::Escape($managedBlockBegin) +
        ".*?^" + [regex]::Escape($managedBlockEnd) + "(?:\r?\n)?"
    $updated = [regex]::Replace($content, $pattern, "")
    if ($updated -eq $content) {
        throw "Bloco gerenciado do Codex nao foi encontrado para rollback."
    }
    $temporary = "$Path.jarvis-rollback-$PID.tmp"
    try {
        [IO.File]::WriteAllText(
            $temporary, $updated, [Text.UTF8Encoding]::new($false)
        )
        Move-Item -LiteralPath $temporary -Destination $Path -Force
    }
    finally {
        Remove-Item -LiteralPath $temporary -Force -ErrorAction SilentlyContinue
    }
}

if (-not (Test-Path -LiteralPath $manifestPath)) {
    throw "Manifesto de instalacao nao encontrado: $manifestPath"
}

$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
if ($manifest.state -ne "committed") {
    throw "Estado de manifesto inesperado: $($manifest.state)"
}

$mergeModes = @{}
foreach ($target in $manifest.targets) {
    if ((Get-Sha256 $target.path) -eq $target.installedSha256) {
        continue
    }
    $resolvedTarget = [IO.Path]::GetFullPath([string]$target.path)
    if (
        $resolvedTarget -eq [IO.Path]::GetFullPath($codexConfig) -and
        (Test-CodexManagedBlock $target.path)
    ) {
        $mergeModes[$resolvedTarget] = "remove-managed-codex-block"
        continue
    }
    if (
        $resolvedTarget -eq [IO.Path]::GetFullPath($cliSettings) -and
        -not $target.backup -and
        (Test-ManagedCliSettings $target.path)
    ) {
        $mergeModes[$resolvedTarget] = "remove-managed-cli-file"
        continue
    }
    throw "Rollback interrompido: $($target.path) mudou fora do escopo gerenciado."
}

Write-Host "Rollback validado para $($manifest.targets.Count) destinos."
if ($mergeModes.Count -gt 0) {
    Write-Host "Alteracoes posteriores fora dos blocos gerenciados serao preservadas."
}
if (-not $Apply) {
    Write-Host "Dry-run concluido. Nenhum arquivo foi alterado. Use -Apply para restaurar."
    exit 0
}

$rollbackTargets = @($manifest.targets)
[array]::Reverse($rollbackTargets)
foreach ($target in $rollbackTargets) {
    $resolvedTarget = [IO.Path]::GetFullPath([string]$target.path)
    $mergeMode = $mergeModes[$resolvedTarget]
    if ($mergeMode -eq "remove-managed-codex-block") {
        Remove-CodexManagedBlock $target.path
        continue
    }
    if ($mergeMode -eq "remove-managed-cli-file") {
        Remove-Item -LiteralPath $target.path -Force
        continue
    }
    if ($target.backup) {
        if (-not (Test-Path -LiteralPath $target.backup)) {
            throw "Backup ausente: $($target.backup)"
        }
        Copy-Item -LiteralPath $target.backup -Destination $target.path -Force
        if ((Get-Sha256 $target.path) -ne $target.beforeSha256) {
            throw "Hash restaurado divergente em $($target.path)"
        }
    }
    elseif (Test-Path -LiteralPath $target.path) {
        Remove-Item -LiteralPath $target.path -Force
    }
}

$manifest.state = "rolled_back"
$manifest.rolledBackAt = (Get-Date).ToUniversalTime().ToString("o")
[IO.File]::WriteAllText(
    $manifestPath,
    (($manifest | ConvertTo-Json -Depth 100) + [Environment]::NewLine),
    [Text.UTF8Encoding]::new($false)
)
Write-Host "Rollback aplicado com sucesso. Reinicie o Antigravity."

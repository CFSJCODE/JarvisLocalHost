param(
    [switch]$Apply
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$appRoot = Split-Path -Parent $PSScriptRoot
$repoRoot = Split-Path -Parent $appRoot
$sourceMcp = Join-Path $repoRoot ".agents\mcp_config.json"
$globalMcp = Join-Path $env:USERPROFILE ".gemini\config\mcp_config.json"
$antigravityConfig = Join-Path $env:USERPROFILE ".gemini\config\config.json"
$cliSettings = Join-Path $env:USERPROFILE ".gemini\antigravity-cli\settings.json"
$codexConfig = Join-Path $env:USERPROFILE ".codex\config.toml"
$manifestPath = Join-Path $appRoot "data\integrations\antigravity-chatgpt-manifest.json"
$managedBlockBegin = "# BEGIN jarvis-team-bus (managed)"
$managedBlockEnd = "# END jarvis-team-bus (managed)"

function Get-Sha256([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) {
        return $null
    }
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash
}

function Read-JsonObject([string]$Path, [bool]$CreateIfMissing) {
    if (-not (Test-Path -LiteralPath $Path)) {
        if ($CreateIfMissing) {
            return [pscustomobject]@{}
        }
        throw "Arquivo JSON nao encontrado: $Path"
    }
    $raw = [IO.File]::ReadAllText($Path, [Text.UTF8Encoding]::new($false))
    try {
        return $raw | ConvertFrom-Json
    }
    catch {
        throw "JSON invalido em $Path`: $($_.Exception.Message)"
    }
}

function Set-Property([object]$Object, [string]$Name, [object]$Value) {
    $propertyNames = @($Object.PSObject.Properties | ForEach-Object { $_.Name })
    if ($propertyNames -contains $Name) {
        $Object.$Name = $Value
    }
    else {
        $Object | Add-Member -NotePropertyName $Name -NotePropertyValue $Value
    }
}

function Convert-ToStableJson([object]$Object) {
    return (($Object | ConvertTo-Json -Depth 100) + [Environment]::NewLine)
}

function Read-Text([string]$Path, [bool]$CreateIfMissing) {
    if (-not (Test-Path -LiteralPath $Path)) {
        if ($CreateIfMissing) {
            return ""
        }
        throw "Arquivo de texto nao encontrado: $Path"
    }
    return [IO.File]::ReadAllText($Path, [Text.UTF8Encoding]::new($false))
}

function Convert-ToTomlBasicString([string]$Value) {
    $escaped = $Value.Replace("\", "\\").Replace('"', '\"')
    return '"' + $escaped + '"'
}

function Set-ManagedTomlBlock([string]$Content, [string]$Block) {
    $newline = if ($Content.Contains("`r`n")) { "`r`n" } else { "`n" }
    $normalizedBlock = ($Block -replace "`r?`n", $newline).TrimEnd("`r", "`n")
    $pattern = "(?ms)^" + [regex]::Escape($managedBlockBegin) +
        ".*?^" + [regex]::Escape($managedBlockEnd) + "(?:`r?`n)?"

    if ([regex]::IsMatch($Content, $pattern)) {
        return [regex]::Replace(
            $Content,
            $pattern,
            [Text.RegularExpressions.MatchEvaluator]{ param($match) $normalizedBlock + $newline },
            1
        )
    }

    if ($Content.Length -eq 0) {
        return $normalizedBlock + $newline
    }
    return $Content.TrimEnd("`r", "`n") + $newline + $newline + $normalizedBlock + $newline
}

function Write-AtomicUtf8([string]$Path, [string]$Content, [string]$Kind) {
    $directory = Split-Path -Parent $Path
    [IO.Directory]::CreateDirectory($directory) | Out-Null
    $temporary = Join-Path $directory ("." + [IO.Path]::GetFileName($Path) + ".tmp." + [guid]::NewGuid().ToString("N"))
    try {
        [IO.File]::WriteAllText($temporary, $Content, [Text.UTF8Encoding]::new($false))
        if ($Kind -eq "json") {
            $null = Read-JsonObject -Path $temporary -CreateIfMissing $false
        }
        elseif ($Kind -ne "text") {
            throw "Tipo de destino desconhecido: $Kind"
        }
        Move-Item -LiteralPath $temporary -Destination $Path -Force
    }
    finally {
        if (Test-Path -LiteralPath $temporary) {
            Remove-Item -LiteralPath $temporary -Force
        }
    }
}

if (-not (Test-Path -LiteralPath $sourceMcp)) {
    throw "Configuracao MCP do repositorio ausente: $sourceMcp"
}

$source = Read-JsonObject -Path $sourceMcp -CreateIfMissing $false
if (-not $source.mcpServers -or -not $source.mcpServers.codex -or -not $source.mcpServers.'jarvis-team-bus') {
    throw "A configuracao MCP local deve conter codex e jarvis-team-bus."
}

$global = Read-JsonObject -Path $globalMcp -CreateIfMissing $true
if (-not (@($global.PSObject.Properties | ForEach-Object { $_.Name }) -contains "mcpServers")) {
    Set-Property -Object $global -Name "mcpServers" -Value ([pscustomobject]@{})
}
Set-Property -Object $global.mcpServers -Name "codex" -Value $source.mcpServers.codex
Set-Property -Object $global.mcpServers -Name "jarvis-team-bus" -Value $source.mcpServers.'jarvis-team-bus'

$appConfig = Read-JsonObject -Path $antigravityConfig -CreateIfMissing $false
if (-not $appConfig.userSettings) {
    throw "userSettings ausente em $antigravityConfig"
}

Set-Property -Object $appConfig.userSettings -Name "autoExecutionPolicy" -Value "CASCADE_COMMANDS_AUTO_EXECUTION_EAGER"
Set-Property -Object $appConfig.userSettings -Name "enableTerminalSandbox" -Value $false
Set-Property -Object $appConfig.userSettings -Name "nonWorkspaceFileAccessPolicy" -Value "AGENT_SETTING_POLICY_ALLOW"

if (-not $appConfig.userSettings.globalPermissionGrants) {
    Set-Property -Object $appConfig.userSettings -Name "globalPermissionGrants" -Value ([pscustomobject]@{})
}
$grants = $appConfig.userSettings.globalPermissionGrants
$existingAllow = @()
if (@($grants.PSObject.Properties | ForEach-Object { $_.Name }) -contains "allow") {
    $existingAllow = @($grants.allow)
}
$fullAccessRules = @(
    "read_file(*)",
    "write_file(*)",
    "read_url(*)",
    "execute_url(*)",
    "command(*)",
    "unsandboxed(*)",
    "mcp(*)"
)
$mergedAllow = @($existingAllow + $fullAccessRules | Sort-Object -Unique)
Set-Property -Object $grants -Name "allow" -Value $mergedAllow

$cli = Read-JsonObject -Path $cliSettings -CreateIfMissing $true
Set-Property -Object $cli -Name "enableTerminalSandbox" -Value $false
if (-not (@($cli.PSObject.Properties | ForEach-Object { $_.Name }) -contains "permissions")) {
    Set-Property -Object $cli -Name "permissions" -Value ([pscustomobject]@{})
}
$cliPermissions = $cli.permissions
$cliExistingAllow = @()
if (@($cliPermissions.PSObject.Properties | ForEach-Object { $_.Name }) -contains "allow") {
    $cliExistingAllow = @($cliPermissions.allow)
}
Set-Property -Object $cliPermissions -Name "allow" -Value @($cliExistingAllow + $fullAccessRules | Sort-Object -Unique)

$existingCodexConfig = Read-Text -Path $codexConfig -CreateIfMissing $true
$teamBusLauncher = Join-Path $repoRoot ".agents\launch-team-bus.ps1"
$tomlArgs = @(
    "-NoProfile",
    "-ExecutionPolicy",
    "Bypass",
    "-File",
    $teamBusLauncher
) | ForEach-Object { Convert-ToTomlBasicString $_ }
$teamBusBlock = @"
$managedBlockBegin
[mcp_servers.jarvis_team_bus]
command = "powershell.exe"
args = [$($tomlArgs -join ', ')]
cwd = $(Convert-ToTomlBasicString $repoRoot)
startup_timeout_sec = 60
tool_timeout_sec = 3600
enabled = true
$managedBlockEnd
"@
$updatedCodexConfig = Set-ManagedTomlBlock -Content $existingCodexConfig -Block $teamBusBlock

$planned = @(
    [pscustomobject]@{ path = $globalMcp; beforeSha256 = Get-Sha256 $globalMcp; content = Convert-ToStableJson $global; kind = "json" },
    [pscustomobject]@{ path = $antigravityConfig; beforeSha256 = Get-Sha256 $antigravityConfig; content = Convert-ToStableJson $appConfig; kind = "json" },
    [pscustomobject]@{ path = $cliSettings; beforeSha256 = Get-Sha256 $cliSettings; content = Convert-ToStableJson $cli; kind = "json" },
    [pscustomobject]@{ path = $codexConfig; beforeSha256 = Get-Sha256 $codexConfig; content = $updatedCodexConfig; kind = "text" }
)

Write-Host "Plano da integracao Antigravity <-> ChatGPT Desktop/Codex:"
foreach ($item in $planned) {
    $state = if ($item.beforeSha256) { "atualizar" } else { "criar" }
    Write-Host " - $state $($item.path)"
}

if (-not $Apply) {
    Write-Host "Dry-run concluido. Nenhum arquivo foi alterado. Use -Apply para instalar."
    exit 0
}

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$backups = @()
try {
    foreach ($item in $planned) {
        $backup = $null
        if (Test-Path -LiteralPath $item.path) {
            $backup = "$($item.path).jarvis-$timestamp.bak"
            Copy-Item -LiteralPath $item.path -Destination $backup
            if ((Get-Sha256 $backup) -ne $item.beforeSha256) {
                throw "Falha ao verificar backup de $($item.path)"
            }
        }
        $backups += [pscustomobject]@{
            path = $item.path
            backup = $backup
            beforeSha256 = $item.beforeSha256
        }
    }

    foreach ($item in $planned) {
        Write-AtomicUtf8 -Path $item.path -Content $item.content -Kind $item.kind
    }

    $codexBinRoot = Join-Path $env:LOCALAPPDATA "OpenAI\Codex\bin"
    $codexExecutable = $null
    if (Test-Path -LiteralPath $codexBinRoot) {
        $codexExecutable = Get-ChildItem -LiteralPath $codexBinRoot -Filter "codex.exe" -File -Recurse |
            Sort-Object LastWriteTimeUtc -Descending |
            Select-Object -First 1
    }
    if (-not $codexExecutable) {
        $codexCommand = Get-Command "codex.exe" -ErrorAction SilentlyContinue
        if ($codexCommand) {
            $codexExecutable = Get-Item -LiteralPath $codexCommand.Source
        }
    }
    if (-not $codexExecutable) {
        throw "Codex nao foi encontrado para validar o config.toml instalado."
    }
    $null = & $codexExecutable.FullName mcp list 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "O Codex rejeitou o config.toml instalado."
    }

    $manifestDir = Split-Path -Parent $manifestPath
    [IO.Directory]::CreateDirectory($manifestDir) | Out-Null
    $manifest = [pscustomobject]@{
        schema = 1
        state = "committed"
        installedAt = (Get-Date).ToUniversalTime().ToString("o")
        repoRoot = $repoRoot
        targets = @(
            foreach ($entry in $backups) {
                [pscustomobject]@{
                    path = $entry.path
                    backup = $entry.backup
                    beforeSha256 = $entry.beforeSha256
                    installedSha256 = Get-Sha256 $entry.path
                }
            }
        )
    }
    Write-AtomicUtf8 -Path $manifestPath -Content (Convert-ToStableJson $manifest) -Kind "json"
}
catch {
    $restoreEntries = @($backups)
    [array]::Reverse($restoreEntries)
    foreach ($entry in $restoreEntries) {
        if ($entry.backup -and (Test-Path -LiteralPath $entry.backup)) {
            Copy-Item -LiteralPath $entry.backup -Destination $entry.path -Force
        }
        elseif (-not $entry.beforeSha256 -and (Test-Path -LiteralPath $entry.path)) {
            Remove-Item -LiteralPath $entry.path -Force
        }
    }
    throw
}

Write-Host "Integracao instalada com backups verificados."
Write-Host "Atualize os servidores MCP no Antigravity ou reinicie o aplicativo para carregar 'codex' e 'jarvis-team-bus'."

[CmdletBinding()]
param([ValidateSet('serve', 'doctor', 'pilot')][string]$Action = 'serve')

$ErrorActionPreference = 'Stop'
$bridgeRoot = $PSScriptRoot
$bridgePython = Join-Path $bridgeRoot '.venv\Scripts\python.exe'
$bridgeLocal = Join-Path $bridgeRoot '.local\settings.json'
if (Test-Path -LiteralPath $bridgeLocal) {
    $bridgeSettings = Get-Content -LiteralPath $bridgeLocal -Raw | ConvertFrom-Json
    if ($bridgeSettings.python) { $bridgePython = $bridgeSettings.python }
}
if (-not (Test-Path -LiteralPath $bridgePython)) {
    throw 'Python environment is missing. Run uv sync or select the existing Python 3.12 environment in PyCharm.'
}
Push-Location -LiteralPath $bridgeRoot
try {
    if ($Action -eq 'pilot') {
        & $bridgePython -X utf8 (Join-Path $bridgeRoot 'scripts\run_pilot.py')
    } else {
        & $bridgePython -X utf8 (Join-Path $bridgeRoot 'main.py') $Action
    }
    if ($LASTEXITCODE -ne 0) { throw "Bridge Edge Vision exited with code $LASTEXITCODE" }
} finally {
    Pop-Location
}

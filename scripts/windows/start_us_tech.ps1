$ErrorActionPreference = "Stop"
$RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location -LiteralPath $RepositoryRoot

python run_us_monitor.py build
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
python run_us_monitor.py serve

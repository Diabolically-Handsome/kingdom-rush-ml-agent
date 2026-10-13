param([switch]$CheckOnly)
$ErrorActionPreference='Stop'
$env:PYTHONUTF8='1'
$taskWorkspace=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$taskPython=Join-Path $taskWorkspace '.venv\Scripts\python.exe'
Push-Location -LiteralPath $taskWorkspace
try {
    if ($CheckOnly) {
        & $taskPython -m alpharush_rl.cli status
        if ($LASTEXITCODE -ne 0) { throw 'Deployment status check failed' }
        Write-Output 'CHECK-ONLY: no comparison directory or GPU job created; per-arm GPU checks run after a comparison is prepared.'
        exit 0
    }
    $prepared=& $taskPython -m alpharush_rl.cli prepare-comparison
    if ($LASTEXITCODE -ne 0) { throw 'Native comparison preparation refused' }
    $taskDirectory=(($prepared -join "`n") | ConvertFrom-Json).output_dir
    & (Join-Path $PSScriptRoot 'run-model.ps1') -Model 8b -Directory $taskDirectory
    & (Join-Path $PSScriptRoot 'run-model.ps1') -Model 24b -Directory $taskDirectory
    & $taskPython -m alpharush_rl.cli finish-comparison $taskDirectory
    if ($LASTEXITCODE -ne 0) { throw 'Native continuation/result comparison failed' }
}
finally { Pop-Location }

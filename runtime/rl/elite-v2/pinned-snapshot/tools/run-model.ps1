param(
    [Parameter(Mandatory=$true)][ValidateSet('8b','24b')][string]$Model,
    [Parameter(Mandatory=$true)][string]$Directory,
    [switch]$CheckOnly
)
$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$taskWorkspace = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$taskPython = Join-Path $taskWorkspace '.venv\Scripts\python.exe'
$taskState = Join-Path $taskWorkspace 'runtime\rl'
$taskPreflight = Join-Path $PSScriptRoot 'model-preflight.py'
function WslPath([string]$Value) {
    $absolute = [IO.Path]::GetFullPath($Value)
    if ($absolute -notmatch '^([A-Za-z]):\\') { throw 'A local drive path is required' }
    return '/mnt/' + $Matches[1].ToLowerInvariant() + '/' + $absolute.Substring(3).Replace('\','/')
}
function Argument([string]$Value) {
    # WSL parses its leading switches from the raw Windows command line.
    # Quoting '-d' makes it a Linux shell command instead of a WSL switch.
    if ($Value -match '^[A-Za-z0-9_/.\:=+,\-]+$') { return $Value }
    return '"' + $Value.Replace('"','\"') + '"'
}
function AppendJson([string]$Path, $Value) {
    [IO.File]::AppendAllText($Path, (($Value | ConvertTo-Json -Depth 30 -Compress) + "`n"), [Text.UTF8Encoding]::new($false))
}
function GetPlan {
    $raw = & $taskPython $taskPreflight --model $Model --directory $Directory
    if ($LASTEXITCODE -ne 0) { throw 'Local comparison preflight refused; inspect the explanation above' }
    return (($raw -join "`n") | ConvertFrom-Json)
}
$taskPlan = GetPlan
$taskWorkerWsl = WslPath (Join-Path $PSScriptRoot 'model-worker.py')
$gpuRaw = & wsl.exe -d Ubuntu --exec $taskPlan.venv_python_wsl -B $taskWorkerWsl --model $Model --preflight-only
if ($LASTEXITCODE -ne 0) { throw 'WSL GPU busy/device preflight refused; other jobs were left untouched' }
$gpuStatus = ($gpuRaw -join "`n") | ConvertFrom-Json
if ($gpuStatus.uuid -ne $taskPlan.gpu_uuid -or $gpuStatus.existing_compute_processes.Count -ne 0 -or
    $gpuStatus.memory_used_mib -gt $taskPlan.busy_memory_used_mib -or
    $gpuStatus.utilization_percent -gt $taskPlan.busy_utilization_percent) {
    throw 'The assigned GPU is busy or its physical identity differs; no job started'
}
$taskLockPath = Join-Path $taskState 'model-gpu.lock'
if (Test-Path -LiteralPath $taskLockPath) { throw 'An AlphaRush GPU lock exists; inspect the running or interrupted job' }
if ($CheckOnly) {
    Write-Output 'CHECK-ONLY passed: bounded inference, unchanged inputs, STOP/pins/evidence/authorization and GPU busy checks; no job started.'
    exit 0
}
$taskLock = $null
$taskOpened = $false
$taskStatus = 'failed'
$taskError = $null
$taskExitCode = -1
$taskJobProcess = $null
$taskRunId = 'infer-' + $Model + '-' + [Guid]::NewGuid().ToString('N')
$taskLedger = Join-Path $taskState 'model-inference-ledger.jsonl'
$taskRunDir = Join-Path $taskState ('model-runs\' + $taskRunId)
$taskWatch = [Diagnostics.Stopwatch]::new()
try {
    $taskLock = [IO.File]::Open($taskLockPath, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
    $lockBytes = [Text.Encoding]::UTF8.GetBytes((@{run_id=$taskRunId; model=$Model; pid=$PID; gpu_uuid=$taskPlan.gpu_uuid} | ConvertTo-Json -Compress))
    $taskLock.Write($lockBytes,0,$lockBytes.Length)
    $taskLock.Flush($true)
    $taskPlan = GetPlan
    $rows = @()
    if (Test-Path -LiteralPath $taskLedger) { $rows = @(Get-Content -LiteralPath $taskLedger | Where-Object { $_.Trim() } | ForEach-Object { $_ | ConvertFrom-Json }) }
    $opens = @($rows | Where-Object { $_.event -eq 'open' })
    $closes = @($rows | Where-Object { $_.event -eq 'close' })
    if (@($opens.run_id | Select-Object -Unique).Count -ne $opens.Count) { throw 'Duplicate inference ledger open' }
    foreach ($row in $opens) {
        if (@($closes | Where-Object { $_.run_id -eq $row.run_id }).Count -ne 1) { throw 'Unclosed/duplicated inference ledger: inspect it before launching' }
    }
    $voidSlots = @{}
    foreach ($correction in @($rows | Where-Object { $_.event -eq 'engineering_attempt_void_for_gpu_slots' })) {
        $closed = @($closes | Where-Object { $_.run_id -eq $correction.run_id })
        $stderrPath = [IO.Path]::GetFullPath((Join-Path $taskWorkspace $correction.stderr_path))
        if (-not $stderrPath.StartsWith($taskWorkspace + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase) -or
            $correction.pre_gpu_verified -ne $true -or $correction.reason_code -ne 'wsl_option_parse_before_worker' -or
            $closed.Count -ne 1 -or $closed[0].exit_code -ne 127 -or
            (Get-FileHash -LiteralPath $stderrPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $correction.stderr_sha256 -or
            -not (Get-Content -LiteralPath $stderrPath -Raw).Contains('/bin/bash: line 1: -d: command not found')) {
            throw 'Pre-GPU slot correction is not backed by its original failure evidence'
        }
        $voidSlots[$correction.run_id]=$true
    }
    $gpuSlotCount=@($opens | Where-Object { -not $voidSlots.ContainsKey($_.run_id) }).Count
    $usedSeconds = ($closes | Measure-Object -Property wall_seconds -Sum).Sum
    if ($gpuSlotCount -ge $taskPlan.max_jobs -or ($usedSeconds + $taskPlan.max_wall_seconds) -gt $taskPlan.total_wall_seconds) { throw 'Bounded comparison cumulative GPU slot/time budget is exhausted' }
    New-Item -ItemType Directory -Path $taskRunDir -ErrorAction Stop | Out-Null
    AppendJson $taskLedger @{event='open'; run_id=$taskRunId; model=$Model; gpu_uuid=$taskPlan.gpu_uuid; time=[DateTimeOffset]::UtcNow.ToString('o'); owner_words=$taskPlan.owner_words; pins_sha256=$taskPlan.pins_sha256; evidence_sha256=$taskPlan.evidence_sha256; plan_sha256=$taskPlan.plan_sha256; batch_sha256=$taskPlan.batch_sha256; cap_seconds=$taskPlan.max_wall_seconds}
    $taskOpened = $true
    $taskWatch.Start()
    $taskSupervisorWsl = WslPath (Join-Path $PSScriptRoot 'model-job.py')
    $taskBatchWsl = WslPath $taskPlan.batch_path
    $taskOutputWsl = WslPath $taskPlan.output_path
    $taskArgs = @('-d','Ubuntu','--exec','/usr/bin/timeout','--signal=TERM','--kill-after=5s',($taskPlan.max_wall_seconds.ToString()+'s'),
                  $taskPlan.venv_python_wsl,'-B',$taskSupervisorWsl,'--model',$Model,'--requests',$taskBatchWsl,
                  '--output',$taskOutputWsl,'--seconds',$taskPlan.max_wall_seconds.ToString())
    $taskArgs = @($taskArgs | ForEach-Object { Argument $_ })
    $taskStartOptions = @{FilePath='wsl.exe'; ArgumentList=$taskArgs; WindowStyle='Hidden'; PassThru=$true;
                          RedirectStandardOutput=(Join-Path $taskRunDir 'stdout.log');
                          RedirectStandardError=(Join-Path $taskRunDir 'stderr.log')}
    $taskJobProcess = Start-Process @taskStartOptions
    $taskJobProcess.WaitForExit()
    $taskJobProcess.Refresh()
    $taskExitCode = $taskJobProcess.ExitCode
    if ($taskExitCode -ne 0) { throw "Inference stopped/failed, exit $taskExitCode; inspect $taskRunDir" }
    $verified = & $taskPython $taskPreflight --model $Model --directory $Directory --verify-result
    if ($LASTEXITCODE -ne 0) { throw 'Complete native-model result verification failed' }
    $taskVerification = ($verified -join "`n") | ConvertFrom-Json
    $taskStatus = 'verified_scoped_inference'
    Write-Output ($verified -join "`n")
}
catch {
    $taskError = $_.Exception.Message
    throw
}
finally {
    $taskWatch.Stop()
    if ($taskOpened) {
        AppendJson $taskLedger @{event='close'; run_id=$taskRunId; model=$Model; time=[DateTimeOffset]::UtcNow.ToString('o'); wall_seconds=$taskWatch.Elapsed.TotalSeconds; exit_code=$taskExitCode; status=$taskStatus; error=$taskError}
        $receipt = @{schema_version=1; run_id=$taskRunId; status=$taskStatus; error=$taskError; exit_code=$taskExitCode; wall_seconds=$taskWatch.Elapsed.TotalSeconds; scope='first_level_train_pool_inference_only'; learning_updates=0; formal_training_started=$false; pins_sha256=$taskPlan.pins_sha256; evidence_sha256=$taskPlan.evidence_sha256; verified=($taskStatus -eq 'verified_scoped_inference')}
        [IO.File]::WriteAllText((Join-Path $taskRunDir 'receipt.json'), ($receipt | ConvertTo-Json -Depth 20), [Text.UTF8Encoding]::new($false))
    }
    if ($null -ne $taskLock) { $taskLock.Dispose(); Remove-Item -LiteralPath $taskLockPath }
}

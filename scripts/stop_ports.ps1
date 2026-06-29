param(
  [int[]]$Ports = @(8000, 5173)
)

foreach ($port in $Ports) {
  $connections = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
  foreach ($connection in $connections) {
    $pidToStop = [int]$connection.OwningProcess
    if ($pidToStop -gt 0) {
      Write-Host "Stopping PID $pidToStop on port $port"
      Stop-Process -Id $pidToStop -Force -ErrorAction SilentlyContinue
    }
  }
}

try {
  $projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
  $processes = Get-CimInstance Win32_Process -ErrorAction Stop
  $byPid = @{}
  foreach ($process in $processes) {
    $byPid[[int]$process.ProcessId] = $process
  }

  $keep = New-Object 'System.Collections.Generic.HashSet[int]'
  $currentProcessId = [int]$PID
  while ($currentProcessId -gt 0 -and $byPid.ContainsKey($currentProcessId) -and -not $keep.Contains($currentProcessId)) {
    [void]$keep.Add($currentProcessId)
    $currentProcessId = [int]$byPid[$currentProcessId].ParentProcessId
  }

  $remainingListeners = Get-NetTCPConnection -LocalPort $Ports -State Listen -ErrorAction SilentlyContinue
  foreach ($listener in $remainingListeners) {
    $listenerProcessId = [int]$listener.OwningProcess
    while ($listenerProcessId -gt 0 -and $byPid.ContainsKey($listenerProcessId) -and -not $keep.Contains($listenerProcessId)) {
      [void]$keep.Add($listenerProcessId)
      $listenerProcessId = [int]$byPid[$listenerProcessId].ParentProcessId
    }
  }

  $shells = $processes | Where-Object {
    $_.Name -eq "cmd.exe" -and
    $_.CommandLine -like "*$projectRoot*" -and
    $_.CommandLine -match "start_project\.cmd|run_backend\.cmd|run_frontend\.cmd" -and
    -not $keep.Contains([int]$_.ProcessId)
  }

  foreach ($shell in $shells) {
    Write-Host "Stopping stale project shell PID $($shell.ProcessId)"
    Stop-Process -Id ([int]$shell.ProcessId) -Force -ErrorAction SilentlyContinue
  }
} catch {
  Write-Host "Skipping stale shell cleanup: $($_.Exception.Message)"
}

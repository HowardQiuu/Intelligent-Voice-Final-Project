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

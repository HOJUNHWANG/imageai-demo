param(
    [Parameter(Mandatory = $true)]
    [string]$ProjectRoot
)

$ErrorActionPreference = "Stop"
$cleanProjectRoot = $ProjectRoot.Trim().Trim('"')
if ([string]::IsNullOrWhiteSpace($cleanProjectRoot)) {
    throw "ProjectRoot cannot be empty."
}
$root = [System.IO.Path]::GetFullPath($cleanProjectRoot).TrimEnd('\')
$ports = @(8000, 3000)

foreach ($port in $ports) {
    $listeners = @(Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue)
    foreach ($listener in $listeners) {
        $pidValue = [int]$listener.OwningProcess
        $processInfo = Get-CimInstance Win32_Process -Filter "ProcessId = $pidValue"
        $commandLine = [string]$processInfo.CommandLine
        $executablePath = [string]$processInfo.ExecutablePath
        $isProjectProcess = (
            $commandLine.IndexOf($root, [StringComparison]::OrdinalIgnoreCase) -ge 0 -or
            $executablePath.StartsWith($root + '\', [StringComparison]::OrdinalIgnoreCase)
        )
        $isExpectedServer = $commandLine -match '(uvicorn|backend\.main|next|npm-cli\.js)'

        if (-not ($isProjectProcess -and $isExpectedServer)) {
            Write-Error "Port $port is used by PID $pidValue outside this Morrow workspace. Stop it manually or change the port."
            exit 2
        }

        Write-Host "       Stopping previous Morrow server on port $port (PID $pidValue)..."
        Stop-Process -Id $pidValue -Force
    }
}

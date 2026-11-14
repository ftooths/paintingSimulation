param(
    [int]$Port = 8765,
    [string[]]$Url = @(),
    [ValidateSet("manual", "auto")]
    [string]$Vision = "manual",
    [switch]$NoBrowser = $false
)

$ErrorActionPreference = "Stop"
$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RootDir

$PyCandidates = @(
    "C:\ComfyUI-Easy-Install\ComfyUI-Easy-Install\python_embeded\python.exe",
    "python"
)
$PyExe = $null
foreach ($cand in $PyCandidates) {
    if ($cand -eq "python") {
        $cmd = Get-Command python -ErrorAction SilentlyContinue
        if ($cmd -and $cmd.Source -notmatch "WindowsApps") {
            $PyExe = $cmd.Source
            break
        }
    } elseif (Test-Path $cand) {
        $PyExe = $cand
        break
    }
}

if (-not $PyExe) {
    throw "Python interpreter not found; install Python 3.11+ or ensure ComfyUI python_embeded is present."
}

$env:PYTHONPATH = Join-Path $RootDir "python"

if ($Url.Count -gt 0) {
    $CliArgs = @("-m", "paint_sim.collector.cli", "--vision", $Vision, "--port", "$Port")
    foreach ($u in $Url) {
        $CliArgs += @("--url", $u)
    }
    & $PyExe @CliArgs
} else {
    $SrvArgs = @("-m", "paint_sim.collector.server", "--port", "$Port")
    if ($NoBrowser) {
        $SrvArgs += "--no-browser"
    }
    & $PyExe @SrvArgs
}

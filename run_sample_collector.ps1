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

$PyDir = Join-Path $RootDir "python"
$env:PYTHONPATH = $PyDir

if ($Url.Count -gt 0) {
    $Bootstrap = "import sys, runpy; sys.path.insert(0, sys.argv.pop(1)); runpy.run_module('paint_sim.collector.cli', run_name='__main__')"
    $CliArgs = @("-c", $Bootstrap, $PyDir, "--vision", $Vision, "--port", "$Port")
    foreach ($u in $Url) {
        $CliArgs += @("--url", $u)
    }
    & $PyExe @CliArgs
} else {
    $Bootstrap = "import sys, runpy; sys.path.insert(0, sys.argv.pop(1)); runpy.run_module('paint_sim.collector.server', run_name='__main__')"
    $SrvArgs = @("-c", $Bootstrap, $PyDir, "--port", "$Port")
    if ($NoBrowser) {
        $SrvArgs += "--no-browser"
    }
    & $PyExe @SrvArgs
}

param(
    [int]$Port = 8000,
    [string]$HostAddr = "127.0.0.1",
    [switch]$NoBrowser = $false,
    [switch]$CheckOnly = $false
)

$ErrorActionPreference = "Stop"
$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RootDir

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host " 2D Plastic-Model Paint & Finish Simulator — AI SAM 2.1 Studio Server Launcher" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan

# 1. Locate Python runtime (prefer ComfyUI embedded Python with CUDA + sam2)
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
    throw "Python interpreter not found. Ensure ComfyUI python_embeded or Python 3.11+ is installed."
}

# 2. Ensure C++20 paintcore.dll exists
$DllPath = Join-Path $RootDir "build\paintcore.dll"
if (-not (Test-Path $DllPath)) {
    Write-Host "[Init] build\paintcore.dll not found; compiling via run_local_tests.ps1..." -ForegroundColor Yellow
    powershell -ExecutionPolicy Bypass -File (Join-Path $RootDir "run_local_tests.ps1")
}

# 3. Ensure default SAM 2.1 base_plus checkpoint exists
$ChkPath = Join-Path $RootDir "checkpoints\sam2.1_hiera_base_plus.pt"
if (-not (Test-Path $ChkPath)) {
    Write-Host "[Init] SAM 2.1 checkpoint missing; running scripts\setup_hq_sam2_local.ps1..." -ForegroundColor Yellow
    powershell -ExecutionPolicy Bypass -File (Join-Path $RootDir "scripts\setup_hq_sam2_local.ps1") -Variant base_plus
}

# 4. Print GPU / SAM 2.1 / Native DLL Diagnostics
$DiagScript = @"
import sys
sys.path.insert(0, 'python')
import torch
from paint_sim.collector.vision.manual_segment import SAM2PointBoxSegmenter

seg = SAM2PointBoxSegmenter()
cuda_ok = torch.cuda.is_available()
gpu_name = torch.cuda.get_device_name(0) if cuda_ok else 'CPU ONLY'
vram_gb = (torch.cuda.get_device_properties(0).total_memory / (1024**3)) if cuda_ok else 0.0
print(f'[Diagnostics] Python     : {sys.executable}')
print(f'[Diagnostics] Torch/CUDA : {torch.__version__} | CUDA={cuda_ok} | Device={gpu_name} ({vram_gb:.1f} GB)')
print(f'[Diagnostics] SAM2 Engine: variant={seg.model_variant} | available={seg.is_available()} | native_dll={seg.bridge.has_native_dll}')
print(f'[Diagnostics] Checkpoint : {seg.checkpoint_path}')
"@
$DiagScript | & $PyExe -

if ($CheckOnly) {
    Write-Host "`n[OK] Pre-flight check completed (-CheckOnly)." -ForegroundColor Green
    exit 0
}

$CacheBust = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
$StudioUrl = "http://${HostAddr}:${Port}/studio/local_test_studio.html?v=${CacheBust}"
Write-Host "`n[Server] Starting FastAPI + SAM 2.1 Studio Backend..." -ForegroundColor Green
Write-Host "  -> Studio UI URL : $StudioUrl" -ForegroundColor Yellow
Write-Host "  -> Health API    : http://${HostAddr}:${Port}/api/v1/health" -ForegroundColor DarkGray
Write-Host "  -> Press Ctrl+C in this window to stop the server.`n" -ForegroundColor Gray

if (-not $NoBrowser) {
    # Open the browser automatically once the server port begins listening
    $null = Start-Job -ScriptBlock {
        param($Url, $HostName, $PortNum)
        for ($i = 0; $i -lt 30; $i++) {
            Start-Sleep -Milliseconds 300
            try {
                $tcp = New-Object System.Net.Sockets.TcpClient
                $tcp.Connect($HostName, $PortNum)
                $tcp.Close()
                Start-Process $Url
                break
            } catch {
                # Wait until server binds port
            }
        }
    } -ArgumentList $StudioUrl, $HostAddr, $Port
}

$RunCode = "import sys, uvicorn; sys.path.insert(0, 'python'); from paint_sim.api.main import create_fastapi_app; uvicorn.run(create_fastapi_app(), host='$HostAddr', port=$Port)"
& $PyExe -c $RunCode

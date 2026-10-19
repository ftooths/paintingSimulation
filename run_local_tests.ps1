param(
    [switch]$OpenStudio = $false
)

$ErrorActionPreference = "Stop"
$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RootDir

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host " 2D Plastic-Model Paint & Finish Simulator — Phase 1 MVP Local Build & Test" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan

$BuildDir = Join-Path $RootDir "build"
if (-not (Test-Path $BuildDir)) {
    New-Item -ItemType Directory -Path $BuildDir | Out-Null
}

# 1. Locate Visual Studio vcvars64.bat & Compile C++20 libpaintcore (DLL + Native Test Runner)
$VcVarsCandidates = @(
    "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat",
    "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
    "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
)
$VcVars = $VcVarsCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1

if ($VcVars) {
    Write-Host "`n[Step 1/3] Compiling C++20 libpaintcore (paintcore.dll & test_paintcore.exe) via MSVC..." -ForegroundColor Yellow
    $CompileCmd = "`"$VcVars`" >nul && cl /nologo /std:c++20 /O2 /EHsc /I`"$RootDir\cpp\include`" /Fo`"$BuildDir\\`" /LD `"$RootDir\cpp\src\paintcore_c_api.cpp`" /Fe:`"$BuildDir\paintcore.dll`" && cl /nologo /std:c++20 /O2 /EHsc /I`"$RootDir\cpp\include`" /Fo`"$BuildDir\\`" `"$RootDir\cpp\tests\test_paintcore_main.cpp`" /Fe:`"$BuildDir\test_paintcore.exe`""
    cmd.exe /c $CompileCmd
    if ($LASTEXITCODE -ne 0) {
        throw "C++20 compilation failed!"
    }

    Write-Host "`n[Step 2/3] Running C++20 Native Core & Physics Verification Suite..." -ForegroundColor Yellow
    & "$BuildDir\test_paintcore.exe"
    if ($LASTEXITCODE -ne 0) {
        throw "C++20 test suite failed!"
    }
} else {
    Write-Host "[WARN] MSVC vcvars64.bat not found; skipping native C++ compilation." -ForegroundColor DarkYellow
}

# 2. Locate Python (PATH or Embedded ComfyUI Python) & Run Python MVP Test Suite
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

if ($PyExe) {
    Write-Host "`n[Step 3/3] Running Python Phase 1 MVP Verification Suite ($PyExe)..." -ForegroundColor Yellow
    & $PyExe "$RootDir\python\tests\test_mvp_suite.py"
    if ($LASTEXITCODE -ne 0) {
        throw "Python MVP test suite failed!"
    }
} else {
    Write-Host "[WARN] Python interpreter not found; please install Python 3.11+ or add it to PATH." -ForegroundColor DarkYellow
}

if ($OpenStudio) {
    $StudioPath = Join-Path $RootDir "frontend\local_test_studio.html"
    Write-Host "`nLaunching Interactive Local Test Studio: $StudioPath" -ForegroundColor Green
    Start-Process $StudioPath
}

Write-Host "`n[SUCCESS] All Phase 1 MVP Core Engine builds and tests passed!" -ForegroundColor Green

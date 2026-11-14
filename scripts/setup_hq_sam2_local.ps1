<#
.SYNOPSIS
    Downloads and verifies official SAM 2 / HQ-SAM 2 model checkpoints for PaintSim.

.DESCRIPTION
    Safely downloads checkpoint weights using curl.exe with resume support (-L -C -).
    Verifies free disk space (>= 8.0 GB) and exact byte size upon completion.
    Protects ComfyUI embedded Python from package shadowing by prohibiting pip install.

.PARAMETER Variant
    Model variant: base_plus (default), small, tiny, or hq_large.

.PARAMETER CleanCache
    Removes partial or temporary download artifacts from the checkpoints directory.

.PARAMETER DryRun
    Verifies parameters and disk space without executing network downloads.
#>
param(
    [ValidateSet("base_plus", "small", "tiny", "hq_large")]
    [string]$Variant = "base_plus",
    [switch]$CleanCache,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = Split-Path -Parent $ScriptDir
$CheckpointsDir = Join-Path $RootDir "checkpoints"

# Warning banner for HQ-SAM 2 package shadowing
if ($Variant -eq "hq_large") {
    Write-Warning "=========================================================================="
    Write-Warning "WARNING: HQ-SAM 2 ('hq_large') selected."
    Write-Warning "Installing HQ-SAM 2 into ComfyUI embedded Python would shadow the official"
    Write-Warning "'sam2' package and break ComfyUI SAM 2 nodes. HQ-SAM 2 requires an isolated"
    Write-Warning "virtual environment. This script downloads weights only and will NEVER"
    Write-Warning "execute 'pip install' into ComfyUI embedded Python."
    Write-Warning "=========================================================================="
}

# Free disk space check (>= 8.0 GB on target drive)
$DriveLetter = [System.IO.Path]::GetPathRoot($RootDir)
$FreeSpaceGB = $null
try {
    $DriveInfo = [System.IO.DriveInfo]::new($DriveLetter)
    $FreeSpaceGB = [math]::Round($DriveInfo.AvailableFreeSpace / 1GB, 2)
} catch {
    Write-Warning "Skipping free disk space check for path '$DriveLetter': $($_.Exception.Message)"
}
if ($null -ne $FreeSpaceGB) {
    if ($FreeSpaceGB -lt 8.0) {
        throw "Insufficient disk space on drive $DriveLetter. Required: >= 8.0 GB, Available: $FreeSpaceGB GB."
    }
    Write-Host "Free disk space check passed: $FreeSpaceGB GB available on $DriveLetter (required >= 8.0 GB)."
}

# Model URL and exact byte size map
$ModelMap = @{
    "hq_large"  = @{
        Url          = "https://huggingface.co/lkeab/hq-sam/resolve/main/sam2.1_hq_hiera_large.pt"
        ExpectedSize = 898844313
        FileName     = "sam2.1_hq_hiera_large.pt"
    }
    "base_plus" = @{
        Url          = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt"
        ExpectedSize = 323606802
        FileName     = "sam2.1_hiera_base_plus.pt"
    }
    "small"     = @{
        Url          = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"
        ExpectedSize = 184416285
        FileName     = "sam2.1_hiera_small.pt"
    }
    "tiny"      = @{
        Url          = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
        ExpectedSize = 156008466
        FileName     = "sam2.1_hiera_tiny.pt"
    }
}

$TargetInfo = $ModelMap[$Variant]
$TargetFile = Join-Path $CheckpointsDir $TargetInfo.FileName

# Optional cache cleanup
if ($CleanCache) {
    Write-Host "Executing cache cleanup..."
    if (Test-Path $CheckpointsDir) {
        $CachePatterns = @("*.part", "*.tmp", "*.crdownload")
        foreach ($pattern in $CachePatterns) {
            $files = Get-ChildItem -Path $CheckpointsDir -Filter $pattern -ErrorAction SilentlyContinue
            foreach ($file in $files) {
                if ($DryRun) {
                    Write-Host "[DryRun] Would remove incomplete download cache: $($file.FullName)"
                } else {
                    Write-Host "Removing incomplete download cache: $($file.FullName)"
                    Remove-Item -Path $file.FullName -Force -ErrorAction SilentlyContinue
                }
            }
        }
    }
}

if ($DryRun) {
    Write-Host "[DryRun] Selected variant: $Variant"
    Write-Host "[DryRun] Target file:      $TargetFile"
    Write-Host "[DryRun] Download URL:     $($TargetInfo.Url)"
    Write-Host "[DryRun] Expected size:    $($TargetInfo.ExpectedSize) bytes"
    Write-Host "[DryRun] Dry run completed successfully. Skipping network download."
    exit 0
}

# Ensure checkpoints directory exists
if (-not (Test-Path $CheckpointsDir)) {
    New-Item -ItemType Directory -Path $CheckpointsDir -Force | Out-Null
}

# Check if file is already present and valid
if (Test-Path $TargetFile) {
    $CurrentSize = (Get-Item $TargetFile).Length
    if ($CurrentSize -eq $TargetInfo.ExpectedSize) {
        Write-Host "Checkpoint '$($TargetInfo.FileName)' already exists and verified ($CurrentSize bytes)."
        exit 0
    } elseif ($CurrentSize -gt $TargetInfo.ExpectedSize) {
        Write-Warning "Existing file is oversized/corrupt ($CurrentSize bytes > expected $($TargetInfo.ExpectedSize) bytes). Deleting before re-download..."
        Remove-Item -Path $TargetFile -Force
    } else {
        Write-Warning "Existing file has size $CurrentSize bytes (expected $($TargetInfo.ExpectedSize) bytes). Resuming download..."
    }
}

Write-Host "Downloading $($TargetInfo.FileName) from $($TargetInfo.Url)..."
& curl.exe -L -C - --fail --retry 3 -o $TargetFile $TargetInfo.Url
if ($LASTEXITCODE -ne 0) {
    throw "Download failed with exit code $LASTEXITCODE."
}

if (-not (Test-Path $TargetFile)) {
    throw "Target file '$TargetFile' not found after download."
}

$FinalSize = (Get-Item $TargetFile).Length
if ($FinalSize -ne $TargetInfo.ExpectedSize) {
    throw "Exact byte size verification failed for '$TargetFile': got $FinalSize bytes, expected $($TargetInfo.ExpectedSize) bytes."
}

Write-Host "Checkpoint download and verification successful: $TargetFile ($FinalSize bytes)."

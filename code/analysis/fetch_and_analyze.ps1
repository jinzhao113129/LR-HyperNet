# fetch_and_analyze.ps1 - pull per-sample predictions from HPC and run paired McNemar tests.
#
# Usage (from anywhere):
#   & "<repo>\code\analysis\fetch_and_analyze.ps1"
#   & "<repo>\code\analysis\fetch_and_analyze.ps1" -Seeds "7,13"
#
# Why: FECNet's FPS picks a random start point via torch.randint
# (models/modules.py:66), so without a fixed eval seed the same checkpoint scores
# differently by ~2.6pp. All comparisons must therefore use the SAME eval seed and
# be tested per-sample (paired), not by comparing two aggregate numbers.
#
# NOTE: keep this file ASCII-only. Windows PowerShell reads BOM-less UTF-8 as GBK
# and Chinese text turns into mojibake that breaks the parser.

param(
    [string]$KeyPath = "$env:USERPROFILE\.ssh\lcr_key",
    [string]$RemoteHost = "jnguan@10.160.22.46",
    [int]$Port = 2223,
    [string]$RemoteRoot = "/home/jnguan/JiangYouxin/ZhaoJin",
    [string]$Seeds = "7,13",
    [int]$NBoot = 20000
)

# Native commands (ssh/scp) write benign warnings to stderr (e.g. a remote shell's
# "zoxide unsupported version"). With ErrorActionPreference=Stop PowerShell would treat
# that as a fatal error, so keep it at Continue and check exit codes explicitly.
$ErrorActionPreference = "Continue"
$localPreds = Join-Path $PSScriptRoot "_preds\run"
# USERPROFILE keeps this file ASCII-only while still resolving the bundled python
# (the account name itself contains non-ASCII characters).
$py = Join-Path $env:USERPROFILE ".dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe"
if (-not (Test-Path $py)) {
    $py = Join-Path $env:USERPROFILE ".dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe"
}
if (-not (Test-Path $py)) {
    Write-Warning "bundled python not found at $py; falling back to PATH python"
    $py = "python"
}

Write-Host "=== 1. list npz on HPC ===" -ForegroundColor Cyan
$listCmd = "cd $RemoteRoot && ls -1 log/classification/*/preds/*.npz 2>/dev/null"
$remoteFiles = & ssh -p $Port -i $KeyPath -o StrictHostKeyChecking=no -o BatchMode=yes $RemoteHost $listCmd 2>$null
if (-not $remoteFiles) { Write-Warning "no npz found on HPC"; exit 1 }
$remoteFiles = @($remoteFiles -split "`n" | Where-Object { $_.Trim() -ne "" })
Write-Host ("  found {0} npz" -f $remoteFiles.Count)

Write-Host "=== 2. pull to local ===" -ForegroundColor Cyan
Remove-Item $localPreds -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $localPreds | Out-Null
foreach ($rf in $remoteFiles) {
    $parts = $rf.Trim() -split "/"
    $logDir = $parts[2]
    $name = $parts[-1]
    $destDir = Join-Path $localPreds $logDir
    New-Item -ItemType Directory -Force -Path $destDir | Out-Null
    $remote = "{0}:{1}/{2}" -f $RemoteHost, $RemoteRoot, $rf
    & scp -P $Port -i $KeyPath -o StrictHostKeyChecking=no $remote (Join-Path $destDir $name) 2>&1 | Out-Null
}
$got = (Get-ChildItem $localPreds -Recurse -Filter *.npz).Count
Write-Host ("  pulled {0}" -f $got)

Write-Host "=== 3. paired McNemar analysis ===" -ForegroundColor Cyan
& $py (Join-Path $PSScriptRoot "mcnemar_test.py") --preds-dir $localPreds --seeds $Seeds --n-boot $NBoot

Write-Host ""
Write-Host "local results: $localPreds" -ForegroundColor Green
Write-Host "read the cross-seed table and the ensemble table as the trustworthy ones." -ForegroundColor Yellow

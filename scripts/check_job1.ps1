$job1 = 'D:\3D-factory-v4\3D-factory-v4\local-loops\loop-20260915T153609Z-a526b568\jobs\asset-000001-6c789c810a'

Write-Host "=== Job 1 directory structure ==="
Get-ChildItem $job1 -Directory | Select-Object Name | Format-Table -AutoSize

Write-Host "`n=== Job 1 research dir ==="
$researchDir = Join-Path $job1 'research'
if (Test-Path $researchDir) {
  Get-ChildItem $researchDir -File | Select-Object Name, Length, LastWriteTime | Format-Table -AutoSize
}

Write-Host "`n=== Job 1 seed-author dir ==="
$seedDir = Join-Path $job1 'seed-author'
if (Test-Path $seedDir) {
  Get-ChildItem $seedDir -Recurse | Select-Object FullName, Length, LastWriteTime | Format-Table -AutoSize
}

Write-Host "`n=== Job 1 discovery dirs ==="
Get-ChildItem $job1 -Directory -Filter 'discovery-*' | ForEach-Object {
  Write-Host "--- $($_.Name) ---"
  Get-ChildItem $_.FullName -Recurse | Select-Object FullName, Length | Format-Table -AutoSize
}

Write-Host "`n=== Job 1 recipe.json (first 500 chars) ==="
$recipePath = Join-Path $job1 'recipe.json'
if (Test-Path $recipePath) {
  $content = Get-Content $recipePath -Raw -Encoding UTF8
  $content.Substring(0, [Math]::Min(500, $content.Length))
}

Write-Host "`n=== Check if job 1 has generation output ==="
$genDirs = @('generation', 'generator', 'output', 'deliverable', 'asset')
foreach ($dir in $genDirs) {
  $path = Join-Path $job1 $dir
  if (Test-Path $path) {
    Write-Host "Found: $dir"
    Get-ChildItem $path -Recurse | Select-Object FullName, Length | Format-Table -AutoSize
  }
}

Write-Host "`n=== Check queue.sqlite3 for job status ==="
$dbPath = 'D:\3D-factory-v4\3D-factory-v4\local-loops\loop-20260915T153609Z-a526b568\queue.sqlite3'
if (Test-Path $dbPath) {
  Write-Host "Database exists: $((Get-Item $dbPath).Length) bytes"
  # Try to read with sqlite3 if available
  $sqlite = Get-Command sqlite3 -ErrorAction SilentlyContinue
  if ($sqlite) {
    Write-Host "sqlite3 available"
    & sqlite3 $dbPath ".tables"
    & sqlite3 $dbPath "SELECT name FROM sqlite_master WHERE type='table';"
  } else {
    Write-Host "sqlite3 not available"
  }
}

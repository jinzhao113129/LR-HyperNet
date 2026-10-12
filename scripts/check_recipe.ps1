$recipe = Get-Content 'D:\3D-factory-v4\3D-factory-v4\local-loops\loop-20260915T153609Z-a526b568\recipe.json' -Raw -Encoding UTF8 | ConvertFrom-Json
Write-Host "=== recipe.json properties ==="
$recipe.PSObject.Properties.Name

Write-Host "`n=== scout phase config ==="
if ($recipe.scout) {
  $recipe.scout | ConvertTo-Json -Depth 5
}

Write-Host "`n=== phases ==="
if ($recipe.phases) {
  $recipe.phases | ConvertTo-Json -Depth 5
}

Write-Host "`n=== Check first job research dir ==="
$researchDir = 'D:\3D-factory-v4\3D-factory-v4\local-loops\loop-20260915T153609Z-a526b568\jobs\asset-000001-6c789c810a\research'
Get-ChildItem $researchDir -File | Select-Object Name, Length | Format-Table -AutoSize

Write-Host "`n=== Check first job seed-author dir ==="
$seedDir = 'D:\3D-factory-v4\3D-factory-v4\local-loops\loop-20260915T153609Z-a526b568\jobs\asset-000001-6c789c810a\seed-author'
if (Test-Path $seedDir) {
  Get-ChildItem $seedDir -Recurse | Select-Object FullName, Length | Format-Table -AutoSize
} else {
  Write-Host "seed-author dir not found or empty"
}

Write-Host "`n=== Check latest job (123) ==="
$latestJob = Get-ChildItem 'D:\3D-factory-v4\3D-factory-v4\local-loops\loop-20260915T153609Z-a526b568\jobs' -Directory | Sort-Object Name -Descending | Select-Object -First 1
Write-Host "Latest job: $($latestJob.Name)"
Get-ChildItem $latestJob.FullName -Recurse | Select-Object FullName, Length | Format-Table -AutoSize

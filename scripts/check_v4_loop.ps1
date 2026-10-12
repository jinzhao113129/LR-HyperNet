$v4Root = "D:\3D-factory-v4\3D-factory-v4"
$latestLoop = Get-ChildItem "$v4Root\local-loops" -Directory | Where-Object { $_.Name -like "loop-*" } | Sort-Object LastWriteTime -Descending | Select-Object -First 1
Write-Host "=== 当前时间 ==="
Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Write-Host "`n=== heartbeat ==="
Write-Host "loop ID: $($latestLoop.Name)"
$hb = Get-Content "$($latestLoop.FullName)\heartbeat.json" -Raw -Encoding UTF8 | ConvertFrom-Json
Write-Host "state: $($hb.state)"
Write-Host "at: $($hb.at)"
if ($hb.active_jobs -and $hb.active_jobs.Count -gt 0) {
  $j = $hb.active_jobs[0]
  Write-Host "`n--- Active Job ---"
  Write-Host "job_id: $($j.job_id)"
  Write-Host "status: $($j.status)"
  Write-Host "phase: $($j.phase)"
  Write-Host "stage: $($j.stage)"
  Write-Host "requests: $($j.requests)/$($j.limits.requests)"
  Write-Host "tools: $($j.tools)/$($j.limits.tools)"
  Write-Host "blender_calls: $($j.blender_calls)"
  Write-Host "reason: $($j.reason)"
  Write-Host "waiting_operator: $($j.waiting_operator)"
} else {
  Write-Host "无active jobs"
}
Write-Host "`n=== console.log最后20条（非heartbeat）==="
Get-Content "$($latestLoop.FullName)\console.log" -Encoding UTF8 | Where-Object { $_ -notmatch "loop_heartbeat" } | Select-Object -Last 20
Write-Host "`n=== jobs目录 ==="
$jobs = Get-ChildItem "$($latestLoop.FullName)\jobs" -Directory -ErrorAction SilentlyContinue
Write-Host "job数量: $($jobs.Count)"
$jobs | Select-Object Name, LastWriteTime | Format-Table -AutoSize
Write-Host "=== deliveries ==="
$zips = Get-ChildItem "$v4Root\deliveries" -Filter "*.zip" -Recurse -ErrorAction SilentlyContinue
Write-Host "交付ZIP数量: $($zips.Count)"

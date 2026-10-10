# Clean temporary files for Windows PowerShell
Write-Host "=== Cleaning temporary and generated artifacts ===" -ForegroundColor Cyan
Get-ChildItem -Path . -Include __pycache__, .pytest_cache -Recurse -Directory | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Get-ChildItem -Path . -Include *.pyc -Recurse -File | Remove-Item -Force -ErrorAction SilentlyContinue
Get-ChildItem -Path results/raw -Include *.csv, *.json -File | Remove-Item -Force -ErrorAction SilentlyContinue
Write-Host "=== Clean Complete ===" -ForegroundColor Green

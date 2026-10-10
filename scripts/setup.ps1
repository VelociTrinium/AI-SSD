# Setup environment for Windows PowerShell
Write-Host "=== Setting up AI-SSD Environment ===" -ForegroundColor Cyan
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

Write-Host "=== Verifying Schemas & Mocks with Pytest ===" -ForegroundColor Cyan
pytest common/tests person1_kv_engine/tests person2_ssd/tests person3_system/tests

Write-Host "=== Setup Completed Successfully! ===" -ForegroundColor Green

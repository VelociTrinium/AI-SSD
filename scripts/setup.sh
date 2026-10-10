#!/usr/bin/env bash
set -e

echo "=== Setting up AI-SSD Environment ==="
python3 -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

echo "=== Verifying Schemas & Mocks with Pytest ==="
pytest common/tests person1_kv_engine/tests person2_ssd/tests person3_system/tests

echo "=== Setup Completed Successfully! ==="

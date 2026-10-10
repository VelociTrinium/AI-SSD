#!/usr/bin/env bash

echo "=== Cleaning temporary and generated artifacts ==="
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type d -name ".pytest_cache" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
rm -f results/raw/*.csv
rm -f results/raw/*.json
echo "=== Clean Complete ==="

#!/usr/bin/env python3
"""
AI-SSD Quick Comparison Runner.
Delegates to the bare-metal ai_ssd.cli for side-by-side Baseline vs AI-SSD benchmark execution.
"""

import sys
import os
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai_ssd.cli import main

if __name__ == "__main__":
    main()

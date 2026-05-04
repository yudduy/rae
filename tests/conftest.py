"""Shared pytest fixtures and path setup.

We install the repo's src/ on sys.path so `import rae` works without an editable
install, which keeps the test loop fast (no pip install -e during iteration).
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

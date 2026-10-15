"""Template structure smoke tests (stdlib-only)."""
import subprocess
import sys
from pathlib import Path


def test_context_validator():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, str(root / 'scripts/check_context.py')], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr

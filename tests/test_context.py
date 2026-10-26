"""Template structure smoke tests (stdlib-only)."""
import subprocess
import sys
import unittest
from pathlib import Path


class ContextValidatorTests(unittest.TestCase):
    def test_context_validator(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run([sys.executable, str(root / 'scripts/check_context.py')], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

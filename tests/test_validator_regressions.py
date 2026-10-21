"""Validator must fail when any required skill file is missing (stdlib-only)."""
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILLS = ('project-bootstrap', 'session-handoff', 'knowledge-curation', 'verification')
SKILL_FILES = [f'{harness}/skills/{skill}/SKILL.md' for harness in ('.agents', '.claude') for skill in SKILLS]


class MissingSkillTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        for name in ('scripts', 'vault', 'docs', '.agents', '.claude', '.codex'):
            shutil.copytree(ROOT / name, self.tmp / name)
        for name in ('README.md', 'AGENTS.md', 'CLAUDE.md'):
            shutil.copy2(ROOT / name, self.tmp / name)

    def run_validator(self):
        return subprocess.run([sys.executable, str(self.tmp / 'scripts/check_context.py')],
                              capture_output=True, text=True)

    def test_copy_passes(self):
        result = self.run_validator()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_each_missing_skill_fails(self):
        for rel in SKILL_FILES:
            with self.subTest(skill=rel):
                path = self.tmp / rel
                original = path.read_text(encoding='utf-8')
                path.unlink()
                try:
                    result = self.run_validator()
                finally:
                    path.write_text(original, encoding='utf-8')
                self.assertEqual(result.returncode, 1, result.stdout)
                self.assertIn(f'ERROR: Missing or empty: {rel}', result.stdout)


if __name__ == '__main__':
    unittest.main()

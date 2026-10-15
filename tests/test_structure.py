import unittest
from pathlib import Path


class StructureTests(unittest.TestCase):
    def test_core_structure(self):
        root = Path(__file__).resolve().parents[1]
        for path in ('AGENTS.md', 'vault/INDEX.md', 'vault/CURRENT_STATE.md',
                     'vault/TODOS.md', '.agents/skills/session-handoff/SKILL.md'):
            self.assertTrue((root / path).is_file(), path)

#!/usr/bin/env python3
"""Basic stdlib-only structure and local reference validator for the template."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    'README.md', 'AGENTS.md', 'CLAUDE.md', 'docs/AGENT_GUIDELINES.md',
    '.codex/config.toml', '.claude/settings.json', 'vault/INDEX.md', 'vault/projects/PROJECT.md',
    'vault/CURRENT_STATE.md', 'vault/TODOS.md', 'vault/projects/PLANS.md',
    'vault/kb/decisions.md', 'vault/worklogs/README.md',
    '.agents/skills/session-handoff/SKILL.md',
    '.agents/skills/project-bootstrap/SKILL.md',
    '.claude/skills/project-bootstrap/SKILL.md',
    '.claude/skills/session-handoff/SKILL.md',
]
errors = []
for name in REQUIRED:
    path = ROOT / name
    if not path.is_file() or not path.read_text(encoding='utf-8').strip():
        errors.append(f'Missing or empty: {name}')
for p in list((ROOT / '.agents/skills').glob('*/SKILL.md')) + list((ROOT / '.claude/skills').glob('*/SKILL.md')):
    data = p.read_text(encoding='utf-8')
    if not data.startswith('---\n') or '\nname:' not in data or '\ndescription:' not in data:
        errors.append(f'Invalid skill front matter: {p.relative_to(ROOT)}')
if errors:
    for e in errors:
        print('ERROR:', e)
    sys.exit(1)
print(f'OK: {len(REQUIRED)} core files and skill metadata present')

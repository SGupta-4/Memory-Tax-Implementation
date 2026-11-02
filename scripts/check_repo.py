"""Check the repo rules in DONE.md that linters can't see.

Purpose: universal gate check 4; fails on missing docstring labels, src/ commits without
FLOW.md or BUGS.md, undocumented dependencies, or stale "Currently modifying" files.
Called by: CI, pre-commit, and people, as `python scripts/check_repo.py`; tests import it.
Assumes: run from inside the git repo; CHECK_BASE may name the commit the range starts from,
otherwise only HEAD and uncommitted changes are checked.
"""

import ast
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCSTRING_LABELS = ("Purpose:", "Called by:", "Assumes:")
DOC_FILES = {"FLOW.md", "BUGS.md"}


def check_docstrings(root: Path) -> list[str]:
    problems = []
    for path in sorted((root / "src").rglob("*.py")):
        doc = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8"))) or ""
        missing = [label for label in DOCSTRING_LABELS if label not in doc]
        if missing:
            problems.append(f"{path.relative_to(root)}: docstring lacks {', '.join(missing)}")
    return problems


def _git(root: Path, *args: str) -> list[str]:
    out = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
    return out.stdout.split() if out.returncode == 0 else []


def check_commits(root: Path, base: str | None) -> list[str]:
    # WHY: an unknown base (e.g. a force-push "before" sha, or all zeros) falls back to HEAD.
    if base and _git(root, "rev-parse", "--verify", "--quiet", f"{base}^{{commit}}"):
        commits = _git(root, "rev-list", f"{base}..HEAD")
    else:
        commits = _git(root, "rev-parse", "--verify", "--quiet", "HEAD")
    problems = []
    for sha in commits:
        files = set(_git(root, "diff-tree", "--root", "--no-commit-id", "--name-only", "-r", sha))
        if any(f.startswith("src/") for f in files) and not files & DOC_FILES:
            problems.append(f"commit {sha[:8]} changes src/ without FLOW.md or BUGS.md")
    pending = {line[3:] for line in _porcelain(root)}
    if any(f.startswith("src/") for f in pending) and not pending & DOC_FILES:
        problems.append("uncommitted src/ changes without FLOW.md or BUGS.md")
    return problems


def _porcelain(root: Path) -> list[str]:
    out = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    return out.stdout.splitlines() if out.returncode == 0 else []


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def check_dependencies(root: Path) -> list[str]:
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    specs = list(project.get("project", {}).get("dependencies", []))
    for group in project.get("dependency-groups", {}).values():
        specs += [s for s in group if isinstance(s, str)]
    specs += project.get("build-system", {}).get("requires", [])
    decisions = (root / "DECISION.md").read_text(encoding="utf-8")
    entries = [_normalize(e) for e in decisions.split("\n## D-")[1:]]
    problems = []
    for spec in specs:
        match = re.match(r"[A-Za-z0-9_.-]+", spec)
        name = _normalize(match.group(0)) if match else spec
        if not any(re.search(rf"(?<![a-z0-9-]){re.escape(name)}(?![a-z0-9-])", e) for e in entries):
            problems.append(f"dependency {name!r} has no D-NNN entry that mentions it")
    return problems


def check_currently_modifying(root: Path) -> list[str]:
    flow = (root / "FLOW.md").read_text(encoding="utf-8")
    row = next((ln for ln in flow.splitlines() if ln.startswith("| Files on the path |")), "")
    return [
        f"FLOW.md 'Currently modifying' names missing file {name}"
        for name in re.findall(r"`([^`]+)`", row)
        if not (root / name).exists()
    ]


def main() -> int:
    problems = (
        check_docstrings(ROOT)
        + check_commits(ROOT, os.environ.get("CHECK_BASE"))
        + check_dependencies(ROOT)
        + check_currently_modifying(ROOT)
    )
    for problem in problems:
        print(problem)
    print("OK" if not problems else f"FAIL: {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

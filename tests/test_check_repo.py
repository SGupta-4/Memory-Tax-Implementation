import importlib.util
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).parents[1]


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_repo", ROOT / "scripts" / "check_repo.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check_repo = _load()


def test_docstring_labels_required(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "good.py").write_text(
        '"""X.\n\nPurpose: a\nCalled by: b\nAssumes: c\n"""\n'
    )
    (tmp_path / "src" / "bad.py").write_text('"""X.\n\nPurpose: a\n"""\n')
    assert check_repo.check_docstrings(tmp_path) == [
        "src/bad.py: docstring lacks Called by:, Assumes:"
    ]


def test_dependency_needs_decision(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\ndependencies = ["numpy>=1", "requests"]\n'
        '[dependency-groups]\ndev = ["types-pyyaml"]\n'
    )
    (tmp_path / "DECISION.md").write_text(
        "# Log\n\n## D-001: Toolchain\n- Uses NumPy (`numpy`) and `types-PyYAML`.\n"
    )
    assert check_repo.check_dependencies(tmp_path) == [
        "dependency 'requests' has no D-NNN entry that mentions it"
    ]


def test_currently_modifying_files_must_exist(tmp_path: Path) -> None:
    (tmp_path / "real.py").write_text("")
    (tmp_path / "FLOW.md").write_text("| Files on the path | `real.py`, `gone.py` |\n")
    assert check_repo.check_currently_modifying(tmp_path) == [
        "FLOW.md 'Currently modifying' names missing file gone.py"
    ]

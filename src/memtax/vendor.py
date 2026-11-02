"""Load the authors' prompt modules from vendor/memorytax at the pinned commit.

Purpose: one place that imports the vendored, hash-checked code (D-006), so prompts are never
retyped or edited.
Called by: answer.py, judge.py, and pipeline.py (dev subset ids).
Assumes: memtax runs from a checkout (editable install), with vendor/ at the repo root;
tests/test_vendor_hash.py guards the files.
"""

import importlib.util
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2] / "vendor" / "memorytax"
DEV_SUBSET = ROOT / "common" / "bench" / "subset_ids" / "longmemeval_s_sub100.json"


def load(relpath: str) -> ModuleType:
    path = ROOT / relpath
    spec = importlib.util.spec_from_file_location(f"memorytax_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load vendored module {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

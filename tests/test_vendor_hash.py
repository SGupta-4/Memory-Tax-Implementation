import hashlib
from pathlib import Path

VENDOR = Path(__file__).parents[1] / "vendor" / "memorytax"


def test_vendored_files_match_pinned_hashes() -> None:
    # PAPER: D-006; files copied from hexuan21/MemoryTax at 179e6a7 and must stay verbatim.
    expected = {}
    for line in (VENDOR / "HASHES.sha256").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        expected[name.removeprefix("./")] = digest
    actual = {
        str(p.relative_to(VENDOR)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in VENDOR.rglob("*")
        if p.is_file() and p.name != "HASHES.sha256" and "__pycache__" not in p.parts
    }
    assert actual == expected

from collections.abc import Iterator
from functools import partial
from pathlib import Path
from typing import Any

import pytest
import yaml

from memtax import pipeline
from memtax.llm import Client
from tests.fake_server import FakeServer

ROOT = Path(__file__).parents[1]


@pytest.fixture
def server() -> Iterator[FakeServer]:
    with FakeServer() as s:
        yield s


@pytest.fixture
def fixture_raw(
    server: FakeServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    """configs/fixture.yaml pointed at the fake server and tmp paths, with no retry delay."""
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(pipeline, "Client", partial(Client, backoff=0.0))
    raw: dict[str, Any] = yaml.safe_load((ROOT / "configs" / "fixture.yaml").read_text())
    for ep in raw["endpoints"].values():
        ep["base_url"] = server.base_url
    raw.update(
        runs_dir=str(tmp_path / "runs"),
        cache=str(tmp_path / "cache.sqlite"),
        index_dir=str(tmp_path / "index"),
    )
    return raw

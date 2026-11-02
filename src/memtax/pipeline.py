"""Run one stage (retrieve, answer, judge) of one config, resumably and in shards.

Purpose: turns a config into immutable per-question records under runs/<run_id>/, so a dead
Kaggle session loses at most the questions in flight.
Called by: cli.run() and cli.smoke().
Assumes: configs validate here (the trust boundary for config); each stage reads the previous
stage's records; records are appended, never rewritten, except dropping a crash-truncated tail.
"""

import asyncio
import hashlib
import json
import os
import subprocess
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from memtax import answer as answer_mod
from memtax import data, vendor
from memtax import judge as judge_mod
from memtax.llm import Cache, Client, Endpoint, LLMError
from memtax.schema import Answer, Question, Retrieved
from memtax.systems.base import MemorySystem
from memtax.systems.baselines import GoldContextSystem, NoContextSystem
from memtax.systems.rag import RagSystem

STAGE_FILES = {"retrieve": "retrieval.jsonl", "answer": "answers.jsonl", "judge": "verdicts.jsonl"}
STAGE_INPUT = {"answer": "retrieve", "judge": "answer"}
SUBSETS = ("all", "dev100", "heldout400")
SYSTEMS = ("rag", "no_context", "gold")
ENDPOINT_ROLES = ("answer", "embed", "rerank", "judge")


@dataclass(frozen=True)
class Config:
    identity: dict[str, Any]  # every setting that can change a result; hashed into run_id
    name: str
    questions: Path
    questions_sha256: str
    subset: str
    system: dict[str, Any]
    endpoints: dict[str, Endpoint]
    concurrency: int
    limit: int | None
    use_cache: bool
    cache: Path
    runs_dir: Path
    index_dir: Path

    @property
    def run_id(self) -> str:
        blob = json.dumps(self.identity, sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:12]

    @property
    def run_dir(self) -> Path:
        return self.runs_dir / self.run_id


def load_config(path: Path) -> Config:
    return parse_config(yaml.safe_load(path.read_text(encoding="utf-8")))


def parse_config(raw: Any) -> Config:
    def need(cond: bool, msg: str) -> None:
        if not cond:
            raise ValueError(f"config: {msg}")

    need(isinstance(raw, dict), "top level must be a mapping")
    for key in ("name", "questions", "questions_sha256", "subset", "system", "endpoints"):
        need(key in raw, f"missing {key!r}")
    need(raw["subset"] in SUBSETS, f"subset must be one of {SUBSETS}")
    system = raw["system"]
    need(isinstance(system, dict) and system.get("name") in SYSTEMS, f"system.name in {SYSTEMS}")
    if system["name"] == "rag":
        for key in ("rounds", "bm25_k", "dense_k", "rerank_k"):
            need(isinstance(system.get(key), int) and system[key] > 0, f"system.{key} > 0")
        need(system.get("embeddings") in ("own", "authors"), "system.embeddings own|authors")
        if system["embeddings"] == "authors":
            need(isinstance(system.get("authors_embeddings"), str), "system.authors_embeddings")
    endpoints: dict[str, Endpoint] = {}
    for role, ep in (raw["endpoints"] or {}).items():
        need(role in ENDPOINT_ROLES, f"unknown endpoint role {role!r}")
        need(
            isinstance(ep, dict) and all(isinstance(ep.get(k), str) for k in ("base_url", "model")),
            f"endpoints.{role} needs base_url and model",
        )
        # WHY: a revision is required, so a model update can never reuse old cached replies.
        need(isinstance(ep.get("revision"), str) and ep["revision"], f"endpoints.{role}.revision")
        endpoints[role] = Endpoint(
            ep["base_url"], ep["model"], ep["revision"], ep.get("api_key_env", "")
        )
    concurrency = raw.get("concurrency", 8)
    need(isinstance(concurrency, int) and concurrency > 0, "concurrency > 0")
    limit = raw.get("limit")
    need(limit is None or (isinstance(limit, int) and limit > 0), "limit > 0 when set")
    use_cache = raw.get("use_cache", True)
    need(isinstance(use_cache, bool), "use_cache must be true or false")
    # WHY: paths, concurrency and base URLs don't change results, so they stay out of run_id;
    # a session can resume a run from another machine or with another concurrency.
    identity = {
        "questions_sha256": raw["questions_sha256"],
        "subset": raw["subset"],
        "system": system,
        "models": {r: [e.model, e.revision] for r, e in sorted(endpoints.items())},
        # WHY: limit and use_cache change which replies a run sees, so they get their own run_id;
        # the G1 repeat check is the same config with limit 20 and use_cache false.
        "limit": limit,
        "use_cache": use_cache,
    }
    return Config(
        identity=identity,
        name=str(raw["name"]),
        questions=Path(raw["questions"]),
        questions_sha256=str(raw["questions_sha256"]),
        subset=raw["subset"],
        system=system,
        endpoints=endpoints,
        concurrency=concurrency,
        limit=limit,
        use_cache=use_cache,
        cache=Path(raw.get("cache", "runs/cache.sqlite")),
        runs_dir=Path(raw.get("runs_dir", "runs")),
        index_dir=Path(raw.get("index_dir", "data/index")),
    )


def select_questions(
    all_questions: list[Question], subset: str, limit: int | None = None
) -> list[Question]:
    chosen = all_questions
    if subset != "all":
        dev = {q["question_id"] for q in json.loads(vendor.DEV_SUBSET.read_text())["questions"]}
        # PAPER: D-003, the authors' 100-question subset is dev; the other 400 are held out.
        want_dev = subset == "dev100"
        chosen = [q for q in all_questions if (q.question_id in dev) == want_dev]
    if limit is not None:
        chosen = sorted(chosen, key=lambda q: q.question_id)[:limit]
    return chosen


def shard(questions: list[Question], index: int, count: int) -> list[Question]:
    if not 0 <= index < count:
        raise ValueError(f"shard {index}/{count} is out of range")
    ordered = sorted(questions, key=lambda q: q.question_id)
    return [q for k, q in enumerate(ordered) if k % count == index]


def read_records(path: Path) -> dict[str, dict[str, Any]]:
    """Records by question_id. A truncated final line (crash mid-write) is cut off the file."""
    if not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    offset = 0
    content = path.read_bytes()
    lines = content.split(b"\n")
    for n, line in enumerate(lines):
        if line.strip():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                if any(rest.strip() for rest in lines[n + 1 :]):
                    raise ValueError(f"{path}: corrupt line {n + 1} is not the last") from None
                os.truncate(path, offset)
                break
            qid = record["question_id"]
            if qid in out:
                raise ValueError(f"{path}: second record for {qid}")
            out[qid] = record
        offset += len(line) + 1
    return out


def append_record(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _git(*args: str) -> str:
    try:
        out = subprocess.run(
            ["git", *args], cwd=vendor.ROOT.parents[1], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def ensure_manifest(cfg: Config) -> dict[str, Any]:
    path = cfg.run_dir / "manifest.json"
    if path.exists():
        manifest: dict[str, Any] = json.loads(path.read_text())
        return manifest
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": cfg.run_id,
        "name": cfg.name,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "identity": cfg.identity,
        "dataset": {"path": str(cfg.questions), "sha256": cfg.questions_sha256},
        "judge_id": judge_mod.judge_id(cfg.endpoints["judge"])
        if "judge" in cfg.endpoints
        else None,
        "stage_runs": [],
    }
    (cfg.run_dir / "config.yaml").write_text(yaml.safe_dump(cfg.identity, sort_keys=True))
    path.write_text(json.dumps(manifest, indent=1))
    return manifest


def make_system(cfg: Config, client: Client, dataset_index: dict[str, int]) -> MemorySystem:
    s = cfg.system
    if s["name"] == "no_context":
        return NoContextSystem()
    if s["name"] == "gold":
        return GoldContextSystem()
    embed = cfg.endpoints["embed"]
    return RagSystem(
        client,
        embed=embed,
        rerank=cfg.endpoints["rerank"],
        chat=cfg.endpoints.get("answer"),
        rounds=s["rounds"],
        bm25_k=s["bm25_k"],
        dense_k=s["dense_k"],
        rerank_k=s["rerank_k"],
        # FLOW: own indexes are per embedding model and revision, shared by every run.
        index_dir=cfg.index_dir / f"{embed.model.replace('/', '__')}@{embed.revision}",
        authors_embeddings=Path(s["authors_embeddings"]) if s["embeddings"] == "authors" else None,
        dataset_index=dataset_index,
    )


async def run_stage(
    cfg: Config, stage: str, shard_index: int = 0, shard_count: int = 1
) -> dict[str, Any]:
    if stage not in STAGE_FILES:
        raise ValueError(f"unknown stage {stage!r}")
    data.verify_sha256(cfg.questions, cfg.questions_sha256)
    all_questions = data.load_questions(cfg.questions)
    selected = select_questions(all_questions, cfg.subset, cfg.limit)
    questions = shard(selected, shard_index, shard_count)
    manifest = ensure_manifest(cfg)
    out_path = cfg.run_dir / STAGE_FILES[stage]
    done = read_records(out_path)
    inputs = (
        read_records(cfg.run_dir / STAGE_FILES[STAGE_INPUT[stage]]) if stage in STAGE_INPUT else {}
    )
    pending = [q for q in questions if q.question_id not in done]
    client = Client(Cache(cfg.cache) if cfg.use_cache else None, cfg.concurrency)
    started = time.time()
    work = _stage_work(cfg, stage, client, {q.question_id: i for i, q in enumerate(all_questions)})
    counts = {"done": 0, "failed": 0, "missing_input": 0}
    gate = asyncio.Semaphore(cfg.concurrency)

    async def one(q: Question) -> None:
        if stage in STAGE_INPUT and q.question_id not in inputs:
            counts["missing_input"] += 1
            return
        async with gate:  # WHY: bounds memory too; each RAG question loads a ~4 MB matrix
            try:
                record = await work(q, inputs.get(q.question_id, {}))
            except LLMError:
                counts["failed"] += 1  # FLOW: not written, so a rerun retries it
                return
        append_record(out_path, record)
        counts["done"] += 1

    await asyncio.gather(*(one(q) for q in pending))
    summary = {
        "stage": stage,
        "shard": f"{shard_index}/{shard_count}",
        "started": started,
        "finished": time.time(),
        "already_done": len(questions) - len(pending),
        **counts,
        "usage": {op: dict(c) for op, c in client.usage.items()},
    }
    manifest["stage_runs"].append(summary)
    (cfg.run_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return summary


Work = Callable[[Question, dict[str, Any]], Awaitable[dict[str, Any]]]


def _stage_work(cfg: Config, stage: str, client: Client, dataset_index: dict[str, int]) -> Work:
    if stage == "retrieve":
        system = make_system(cfg, client, dataset_index)

        async def retrieve(q: Question, _: dict[str, Any]) -> dict[str, Any]:
            await system.build(q)
            return asdict(await system.retrieve(q))

        return retrieve
    if stage == "answer":

        async def answer(q: Question, prev: dict[str, Any]) -> dict[str, Any]:
            retrieved = Retrieved(**prev)
            return asdict(await answer_mod.answer(client, cfg.endpoints["answer"], q, retrieved))

        return answer

    async def judge(q: Question, prev: dict[str, Any]) -> dict[str, Any]:
        return asdict(await judge_mod.judge(client, cfg.endpoints["judge"], q, Answer(**prev)))

    return judge


async def judge_external(
    cfg: Config, questions: list[Question], answers: dict[str, str], out_path: Path
) -> dict[str, dict[str, Any]]:
    """Judge answers from elsewhere (the authors') with this config's judge; resumable."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = read_records(out_path)
    client = Client(Cache(cfg.cache), cfg.concurrency)
    gate = asyncio.Semaphore(cfg.concurrency)

    async def one(q: Question) -> None:
        ans = Answer(q.question_id, answers[q.question_id], "external", False)
        async with gate:
            try:
                result = await judge_mod.judge(client, cfg.endpoints["judge"], q, ans)
            except LLMError:
                return  # FLOW: left unjudged; rerunning the command retries it
        append_record(out_path, asdict(result))

    await asyncio.gather(*(one(q) for q in questions if q.question_id not in done))
    return read_records(out_path)

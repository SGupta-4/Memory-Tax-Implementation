# Rollback plan

Use this when a change breaks something. Before a large or risky edit, add an entry below first.

## Before any risky change

```bash
git tag pre-<change-name>
cp runs/cache.sqlite runs/cache.sqlite.bak-$(date +%Y%m%d)
```

Risky means changing `metrics.py`, `schema.py`, or the cache key, vendored prompts, serving setup, dependencies, or anything that touches many files.

## Known-good points

| Tag | Means | Use when |
| --- | --- | --- |
| `phase-N-pass` | Gate N passed | A later phase breaks earlier results |
| `prereg-h2` | H2 config frozen | Checking what the held-out run used |
| `prereg-med` | Medication hypotheses frozen | Same, for the extension |
| `pre-<change>` | Just before a risky change | That change fails |

## Undo by situation

| Situation | Do this | Then re-check |
| --- | --- | --- |
| A commit broke tests | `git revert <sha>` | Universal gate |
| A branch went wrong, not merged | Delete the branch and restart from `main` | `git status` |
| `metrics.py` changed numbers | `git revert <sha>` | `memtax audit`, G0 numbers |
| Chunker broke | `git revert <sha>`; new config gets a new `run_id` | `memtax check-chunks --against authors` |
| Cache key changed by mistake | `git revert <sha>`; restore cache backup | `pytest tests/test_cache.py -q`; a known call makes no request |
| Cache file corrupted | Restore the latest HF checkpoint | `sqlite3 runs/cache.sqlite 'PRAGMA integrity_check;'` returns `ok` |
| Dependency update broke the build | `git checkout phase-N-pass -- pyproject.toml uv.lock`, then `uv sync` | Universal gate |
| vLLM fails after an upgrade | Reinstall the version in `configs/serve.lock` | `scripts/kaggle_serve.sh health` |
| Vendored prompt changed | `git checkout phase-N-pass -- vendor/` | `pytest tests/test_vendor_hash.py -q` |
| Bad run produced | Mark it `invalid` in `runs/index.csv` and record it in `BUGS.md`; don't delete it | `memtax metrics` refuses invalid runs |
| Wrong data published to HF | Point the release at the previous revision and record it | `make reproduce` from that revision |
| Secret committed | Revoke the token first; then ask a human about rewriting history | Secret scan is clean |

## Rules

- Prefer `git revert`. It keeps history and works after pushing.
- Don't use `git reset --hard` on shared branches, or force-push, without human approval.
- Don't delete finished runs. A wrong run is still evidence of what went wrong.
- Rollback doesn't reopen the held-out set. Any rerun after `prereg-h2` is a logged deviation.

## Entries for specific risky changes

```markdown
### YYYY-MM-DD: change name
- Revert to: tag or commit
- Files to restore: list
- Data to restore: cache backup or run ID
- Re-check: exact commands from DONE.md
- Outcome: not needed | used, with result
```

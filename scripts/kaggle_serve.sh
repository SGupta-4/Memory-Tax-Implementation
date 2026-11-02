#!/usr/bin/env bash
# Start, check, or stop the model servers on a Kaggle T4 x2 session.
#
#   scripts/kaggle_serve.sh answer   Qwen3.5-4B on GPU 0 (:8000); embedding (:8001) and reranker (:8002) on GPU 1
#   scripts/kaggle_serve.sh judge    Qwen3.6-27B Q4_K_M GGUF across both GPUs (:8003), in its own session (D-018)
#   scripts/kaggle_serve.sh health   one real call per running endpoint; prints "<name> OK" and no_think=true|false
#   scripts/kaggle_serve.sh stop     stops every server this script started
#
# Pinned models and revisions live in configs/serve.lock, matching configs/kaggle/*.yaml.
# UNTESTED until the Phase 1 Kaggle smoke test: vLLM flags differ across versions; record the
# versions health prints in serve.lock, and log any flag change in DECISION.md.
set -euo pipefail
cd "$(dirname "$0")/.."
source configs/serve.lock
LOG_DIR=${LOG_DIR:-/kaggle/working/logs}
mkdir -p "$LOG_DIR"

start_vllm() {  # gpu port model revision [extra vllm flags...]
  local gpu=$1 port=$2 model=$3 rev=$4
  shift 4
  # WHY: T4 (compute 7.5) has no bf16, so fp16; served name = model id, as configs expect.
  CUDA_VISIBLE_DEVICES=$gpu nohup vllm serve "$model" --revision "$rev" --port "$port" \
    --dtype half --served-model-name "$model" "$@" >"$LOG_DIR/server_$port.log" 2>&1 &
  echo $! >>"$LOG_DIR/pids"
}

wait_ready() {  # port...
  for port in "$@"; do
    for _ in $(seq 180); do
      curl -sf "http://localhost:$port/health" >/dev/null && continue 2
      sleep 5
    done
    echo "server on :$port not ready after 15 minutes; see $LOG_DIR/server_$port.log" >&2
    exit 1
  done
}

case "${1:-}" in
  answer)
    # Qwen3.5-4B is a ConditionalGeneration (vision) model; text-only keeps image memory free.
    start_vllm 0 8000 "$ANSWER_MODEL" "$ANSWER_REV" --max-model-len 16384 \
      --gpu-memory-utilization 0.90 --limit-mm-per-prompt '{"image": 0, "video": 0}'
    start_vllm 1 8001 "$EMBED_MODEL" "$EMBED_REV" $EMBED_FLAGS --max-model-len 8192 \
      --gpu-memory-utilization 0.40
    # PAPER: Qwen3-Reranker as a sequence classifier over the "no"/"yes" tokens (vLLM docs).
    start_vllm 1 8002 "$RERANK_MODEL" "$RERANK_REV" --max-model-len 8192 \
      --gpu-memory-utilization 0.40 --hf-overrides \
      '{"architectures": ["Qwen3ForSequenceClassification"], "classifier_from_token": ["no", "yes"], "is_original_qwen3_reranker": true}'
    wait_ready 8000 8001 8002
    ;;
  judge)
    # D-009: 4-bit 27B across both T4s with llama.cpp. Needs llama-server built with CUDA.
    gguf=$(hf download "$JUDGE_GGUF_REPO" "$JUDGE_GGUF_FILE" --revision "$JUDGE_GGUF_REV")
    nohup "${LLAMA_SERVER:-llama-server}" -m "$gguf" --port 8003 --alias "${JUDGE_GGUF_REPO}:Q4_K_M" \
      -ngl 999 --tensor-split 1,1 -c 8192 --jinja >"$LOG_DIR/server_8003.log" 2>&1 &
    echo $! >>"$LOG_DIR/pids"
    wait_ready 8003
    ;;
  health)
    vllm --version 2>/dev/null | sed 's/^/vllm version: /' || true
    "${LLAMA_SERVER:-llama-server}" --version 2>&1 | head -1 | sed 's/^/llama.cpp version: /' || true
    # FLOW: uses memtax.llm, so health sends exactly the request shapes the pipeline sends.
    python - <<'EOF'
import asyncio, urllib.request
from memtax.llm import Client, Endpoint, EMBED_QUERY_PREFIX
from memtax.pipeline import load_config

cfg = load_config(__import__("pathlib").Path("configs/kaggle/rag1_dev100.yaml"))
client = Client(None, retries=0)
leak = False

def up(ep: Endpoint) -> bool:
    try:
        urllib.request.urlopen(ep.base_url.rsplit("/v1", 1)[0] + "/health", timeout=5)
        return True
    except Exception:
        return False

async def main() -> None:
    global leak
    for role in ("answer", "judge"):
        ep = cfg.endpoints[role]
        if up(ep):
            msg = [{"role": "user", "content": "What is 2 + 2? Answer with one number."}]
            reply = await client.chat(ep, msg, op="health", prompt_version="health", max_tokens=16)
            leak |= "<think>" in reply
            print(f"{role} OK: {reply!r}")
    if up(cfg.endpoints["embed"]):
        vec = await client.embed(cfg.endpoints["embed"], [EMBED_QUERY_PREFIX + "hello"], op="health")
        print(f"embed OK: dim {len(vec[0])}")
    if up(cfg.endpoints["rerank"]):
        s = await client.rerank(cfg.endpoints["rerank"], "capital of France",
                                ["Paris is the capital of France.", "Bananas are yellow."], op="health")
        print(f"rerank OK: {s[0]:.3f} > {s[1]:.3f}" if s[0] > s[1] else f"rerank WRONG ORDER: {s}")
    print(f"no_think={'false' if leak else 'true'}")

asyncio.run(main())
EOF
    ;;
  stop)
    [ -f "$LOG_DIR/pids" ] && xargs -r kill <"$LOG_DIR/pids" || true
    rm -f "$LOG_DIR/pids"
    ;;
  *)
    sed -n '2,8p' "$0"
    exit 2
    ;;
esac

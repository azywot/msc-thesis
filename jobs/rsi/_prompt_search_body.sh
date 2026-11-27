#!/bin/bash
# Body of the RSI phase 1 prompt-search jobs. Sourced-by-exec from the .job wrappers.
#
# Inputs (environment):
#   CONFIG                  experiment YAML (required)
#   DOMAIN                  search | math (required)
#   EXTRA_ARGS              extra CLI args for run_prompt_search.py (optional)
#   REFLECTOR_SERVER_MODEL  if set, start `vllm serve` for the reflector on port 8001
#   REFLECTOR_GPUS          CUDA_VISIBLE_DEVICES for the reflector server (e.g. 1,2)
#   REFLECTOR_TP            tensor parallel size of the reflector server

set -euo pipefail

: "${CONFIG:?CONFIG must be set}"
: "${DOMAIN:?DOMAIN must be set}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "=========================================="
echo "MSC Thesis - RSI phase 1 prompt search"
echo "=========================================="
echo "Start time: $(date)"
echo "Job ID: ${SLURM_JOB_ID:-local}  Node: $(hostname)"
echo "Config: $CONFIG  Domain: $DOMAIN"
echo ""

module purge
module load 2025
module load Miniconda3/25.5.1-1

PROJECT_DIR="${PROJECT_DIR:-$HOME/azywot/msc-thesis}"
cd "$PROJECT_DIR" || exit 1
echo "Project directory: $PROJECT_DIR"

if [[ -f ".env" ]]; then
    echo "Loading environment variables from .env..."
    set -a
    source <(grep -v '^#' .env | grep -v '^$' | grep -v '^[[:space:]]*#')
    set +a
fi

if [[ -z "${SERPER_API_KEY:-}" && -z "${TAVILY_API_KEY:-}" ]]; then
    echo "ERROR: SERPER_API_KEY or TAVILY_API_KEY must be set for web search during rollouts."
    exit 1
fi

ENV_NAME="${ENV_NAME:-agent_engine}"
echo "Activating environment: $ENV_NAME"
source activate "$ENV_NAME"
export PYTHONNOUSERSITE=1

export VLLM_LOGGING_LEVEL=WARNING
export TOKENIZERS_PARALLELISM=false

mkdir -p out/rsi

REFLECTOR_PID=""
cleanup() {
    local pid="$REFLECTOR_PID"
    if [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1; then
        echo "Stopping reflector vLLM (pid=$pid)..."
        kill -TERM "$pid" 2>/dev/null || true
        for _i in $(seq 1 30); do
            sleep 1
            kill -0 "$pid" >/dev/null 2>&1 || break
        done
        kill -KILL "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT

if [[ -n "${REFLECTOR_SERVER_MODEL:-}" ]]; then
    : "${REFLECTOR_GPUS:?REFLECTOR_GPUS must be set with REFLECTOR_SERVER_MODEL}"
    : "${REFLECTOR_TP:?REFLECTOR_TP must be set with REFLECTOR_SERVER_MODEL}"
    REFLECTOR_LOG="out/rsi/prompt_search_${SLURM_JOB_ID:-local}_reflector.log"
    echo "Starting reflector vLLM ($REFLECTOR_SERVER_MODEL) on GPUs $REFLECTOR_GPUS, port 8001..."
    CUDA_VISIBLE_DEVICES="$REFLECTOR_GPUS" vllm serve "$REFLECTOR_SERVER_MODEL" \
        --port 8001 \
        --tensor-parallel-size "$REFLECTOR_TP" \
        --max-model-len 40960 \
        --enable-prefix-caching \
        > "$REFLECTOR_LOG" 2>&1 &
    REFLECTOR_PID=$!

    _deadline=$((SECONDS + 900))
    until curl -fsS --max-time 5 "http://127.0.0.1:8001/v1/models" >/dev/null 2>&1; do
        if ! kill -0 "$REFLECTOR_PID" >/dev/null 2>&1; then
            echo "FAIL - reflector vLLM exited during startup. See $REFLECTOR_LOG"
            exit 1
        fi
        if (( SECONDS >= _deadline )); then
            echo "FAIL - reflector not ready after 15 min. See $REFLECTOR_LOG"
            exit 1
        fi
        sleep 10
    done
    echo "Reflector vLLM is ready."
fi

export CUDA_VISIBLE_DEVICES=0

echo ""
echo "Running prompt search..."
# shellcheck disable=SC2086
python scripts/run_prompt_search.py --config "$CONFIG" --domain "$DOMAIN" $EXTRA_ARGS

echo ""
echo "End time: $(date)"
echo "=========================================="

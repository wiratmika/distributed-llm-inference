#!/usr/bin/env bash
set -euo pipefail

NUM_NODES="${1:-3}"
MODEL_NAME="${2:-gpt2}"
GATEWAY_PORT=8000
WORKER_BASE_PORT=8001

echo "Model:        $MODEL_NAME"
echo "Worker nodes: $NUM_NODES"
echo "Gateway:      http://localhost:$GATEWAY_PORT"
echo ""

PIDS=()

cleanup() {
    echo ""
    echo "Shutting down..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null
    echo "Done."
}
trap cleanup EXIT INT TERM

# Start workers
for (( rank = NUM_NODES - 1; rank >= 0; rank-- )); do
    port=$(( WORKER_BASE_PORT + rank ))

    if (( rank < NUM_NODES - 1 )); then
        next_port=$(( WORKER_BASE_PORT + rank + 1 ))
        next_url="http://localhost:$next_port"
    else
        next_url=""
    fi

    echo "Starting worker rank=$rank on port=$port (next=$next_url)"

    MODEL_NAME="$MODEL_NAME" \
    NUM_NODES="$NUM_NODES" \
    RANK="$rank" \
    NEXT_NODE_URL="$next_url" \
        uvicorn src.worker:app --port "$port" --log-level info &
    PIDS+=($!)

    sleep 1
done

# Start gateway
echo "Starting gateway on port=$GATEWAY_PORT (worker=http://localhost:$WORKER_BASE_PORT)"

MODEL_NAME="$MODEL_NAME" \
WORKER_URL="http://localhost:$WORKER_BASE_PORT" \
    uvicorn src.gateway:app --port "$GATEWAY_PORT" --log-level info &
PIDS+=($!)

echo "All nodes started"
wait

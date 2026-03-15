"""Worker node – serves a single shard of the model via HTTP.

Each worker owns a contiguous slice of transformer layers determined by its rank within the pipeline.
It exposes a /forward endpoint that:

1. Receives serialized activation tensors (or input_ids for rank 0).
2. Runs them through its local ModelShard.
3. Forwards the output to the next worker, or returns the final logits if this is the last stage.
"""

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Optional

import httpx
import torch
from fastapi import FastAPI, HTTPException
from transformers import AutoModelForCausalLM, AutoTokenizer

from .helpers import _b64_to_tensor, _tensor_to_b64, count_model_layers
from .models import (
    ModelShard,
    partition_layers,
)
from .schemas import ForwardRequest, ForwardResponse, NodeTiming

logger = logging.getLogger(__name__)

MODEL_NAME: str = os.environ.get("MODEL_NAME", "gpt2")
NUM_NODES: int = int(os.environ.get("NUM_NODES", "1"))
RANK: int = int(os.environ.get("RANK", "0"))
# Empty / unset for the last node
NEXT_NODE_URL: str = os.environ.get("NEXT_NODE_URL", "")

shard: Optional[ModelShard] = None
tokenizer = None
http_client: Optional[httpx.AsyncClient] = None


def _build_shard() -> ModelShard:
    """Load the full model, partition it, and keep only this node's shard."""
    logger.info("Loading model %s (rank %d / %d nodes)...", MODEL_NAME, RANK, NUM_NODES)
    full_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    full_model.eval()

    num_layers = count_model_layers(full_model)
    partition = partition_layers(num_layers, NUM_NODES)[RANK]
    is_first = RANK == 0
    is_last = RANK == NUM_NODES - 1

    node_shard = ModelShard(
        full_model,
        partition,
        is_first=is_first,
        is_last=is_last,
    )
    node_shard.eval()

    logger.info(
        "Shard ready: %s  (first=%s, last=%s)",
        partition,
        is_first,
        is_last,
    )
    return node_shard


@asynccontextmanager
async def lifespan(app: FastAPI):
    global shard, tokenizer, http_client

    torch.set_num_threads(2)
    shard = _build_shard()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    http_client = httpx.AsyncClient(timeout=httpx.Timeout(120.0))

    yield
    await http_client.aclose()


app = FastAPI(title=f"Worker node (rank {RANK})", lifespan=lifespan)


def _get_peak_memory() -> int:
    """Return peak RSS in bytes."""
    try:
        import resource as _resource

        usage = _resource.getrusage(_resource.RUSAGE_SELF)
        if sys.platform == "darwin":
            return usage.ru_maxrss  # bytes on macOS
        return usage.ru_maxrss * 1024  # KB -> bytes on Linux
    except Exception:
        return 0


@app.post("/forward", response_model=ForwardResponse)
async def forward(req: ForwardRequest):
    if shard is None:
        raise HTTPException(status_code=503, detail="Model shard not loaded yet")

    deserialization_start = time.perf_counter()

    with torch.inference_mode():
        if shard.is_first:
            if req.input_ids is None:
                raise HTTPException(
                    status_code=400,
                    detail="First worker (rank 0) requires 'input_ids'",
                )
            input_ids = torch.tensor(req.input_ids, dtype=torch.long)
            deserialization_ms = (time.perf_counter() - deserialization_start) * 1000

            compute_start = time.perf_counter()
            hidden = shard(hidden_states=torch.empty(0), input_ids=input_ids)
            compute_ms = (time.perf_counter() - compute_start) * 1000
        else:
            if req.hidden_b64 is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"Worker rank {RANK} requires 'hidden_b64'",
                )
            hidden = _b64_to_tensor(req.hidden_b64)
            deserialization_ms = (time.perf_counter() - deserialization_start) * 1000

            compute_start = time.perf_counter()
            hidden = shard(hidden_states=hidden)
            compute_ms = (time.perf_counter() - compute_start) * 1000

    # Serialize output tensor
    serialization_start = time.perf_counter()
    output_b64 = _tensor_to_b64(hidden)
    serialization_ms = (time.perf_counter() - serialization_start) * 1000
    total_serialization_ms = deserialization_ms + serialization_ms

    logger.info(
        "Rank %d forward: compute=%.1f ms  serialization=%.1f ms",
        RANK,
        compute_ms,
        total_serialization_ms,
    )

    my_timing = NodeTiming(
        node_id=RANK,
        compute_ms=compute_ms,
        serialization_ms=total_serialization_ms,
        peak_memory_bytes=_get_peak_memory(),
    )

    if shard.is_last:
        return ForwardResponse(logits_b64=output_b64, node_timings=[my_timing])

    if not NEXT_NODE_URL:
        raise HTTPException(
            status_code=500,
            detail="NEXT_NODE_URL not configured but this is not the last node",
        )

    next_req = ForwardRequest(hidden_b64=output_b64)
    resp = await http_client.post(
        f"{NEXT_NODE_URL}/forward",
        json=next_req.model_dump(),
    )

    if resp.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Next node returned {resp.status_code}: {resp.text}",
        )

    body = resp.json()
    downstream_timings = [NodeTiming(**nt) for nt in body.get("node_timings", [])]

    return ForwardResponse(
        logits_b64=body.get("logits_b64"),
        node_timings=[my_timing] + downstream_timings,
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "rank": RANK,
        "num_nodes": NUM_NODES,
        "partition": str(shard.partition) if shard else None,
        "is_first": shard.is_first if shard else None,
        "is_last": shard.is_last if shard else None,
    }

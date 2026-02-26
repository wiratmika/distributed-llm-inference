"""Worker node – serves a single shard of the model via HTTP.

Each worker owns a contiguous slice of transformer layers determined by its rank within the pipeline.
It exposes a /forward endpoint that:

1. Receives serialized activation tensors (or input_ids for rank 0).
2. Runs them through its local ModelShard.
3. Forwards the output to the next worker, or returns the final logits if this is the last stage.
"""

import logging
import os
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
from .schemas import ForwardRequest, ForwardResponse

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

    shard = _build_shard()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    http_client = httpx.AsyncClient(timeout=httpx.Timeout(120.0))

    yield
    await http_client.aclose()


app = FastAPI(title=f"Worker node (rank {RANK})", lifespan=lifespan)


@app.post("/forward", response_model=ForwardResponse)
async def forward(req: ForwardRequest):
    if shard is None:
        raise HTTPException(status_code=503, detail="Model shard not loaded yet")

    time_start = time.perf_counter()

    with torch.inference_mode():
        if shard.is_first:
            # First node expects token IDs
            if req.input_ids is None:
                raise HTTPException(
                    status_code=400,
                    detail="First worker (rank 0) requires 'input_ids'",
                )
            input_ids = torch.tensor(req.input_ids, dtype=torch.long)
            hidden = shard(hidden_states=torch.empty(0), input_ids=input_ids)
        else:
            # Later nodes expect serialized hidden states
            if req.hidden_b64 is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"Worker rank {RANK} requires 'hidden_b64'",
                )
            hidden = _b64_to_tensor(req.hidden_b64)
            hidden = shard(hidden_states=hidden)

    elapsed_ms = (time.perf_counter() - time_start) * 1000
    logger.info("Rank %d forward pass: %.1f ms", RANK, elapsed_ms)

    if shard.is_last:
        return ForwardResponse(logits_b64=_tensor_to_b64(hidden))

    if not NEXT_NODE_URL:
        raise HTTPException(
            status_code=500,
            detail="NEXT_NODE_URL not configured but this is not the last node",
        )

    next_req = ForwardRequest(hidden_b64=_tensor_to_b64(hidden))
    resp = await http_client.post(
        f"{NEXT_NODE_URL}/forward",
        json=next_req.model_dump(),
    )
    if resp.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Next node returned {resp.status_code}: {resp.text}",
        )

    return ForwardResponse(**resp.json())


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

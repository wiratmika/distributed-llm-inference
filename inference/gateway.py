"""Gateway node – user-facing API that drives auto-regressive generation. It:

1. Tokenises the user's prompt.
2. Iteratively sends token IDs (first step) or updated token IDs (subsequent steps) to worker rank 0 via its /forward endpoint.
3. Worker rank 0 propagates through the pipeline and the last worker returns logits.
4. The gateway greedy-decodes the next token from those logits, appends it, and repeats until max_new_tokens or EOS.
5. Returns the decoded text to the user.
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import List, Optional

import httpx
import torch
from fastapi import FastAPI, HTTPException
from transformers import AutoTokenizer

from .helpers import _b64_to_tensor
from .schemas import GenerateRequest, GenerateResponse, NodeTiming

logger = logging.getLogger(__name__)

MODEL_NAME: str = os.environ.get("MODEL_NAME", "gpt2")
WORKER_URL: str = os.environ.get("WORKER_URL", "http://localhost:8001")
MAX_NEW_TOKENS: int = int(os.environ.get("MAX_NEW_TOKENS", "50"))

tokenizer = None
http_client: Optional[httpx.AsyncClient] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, http_client

    logger.info("Gateway starting – model=%s, worker=%s", MODEL_NAME, WORKER_URL)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    http_client = httpx.AsyncClient(timeout=httpx.Timeout(300.0))

    yield
    await http_client.aclose()


app = FastAPI(title="Distributed LLM Gateway", lifespan=lifespan)

async def _pipeline_forward(
    input_ids: List[List[int]],
) -> tuple[torch.Tensor, List[NodeTiming]]:
    payload = {"input_ids": input_ids}
    resp = await http_client.post(f"{WORKER_URL}/forward", json=payload)
    if resp.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Worker pipeline returned {resp.status_code}: {resp.text}",
        )

    body = resp.json()
    logits_b64 = body.get("logits_b64")
    if logits_b64 is None:
        raise HTTPException(
            status_code=502,
            detail="Worker pipeline did not return logits",
        )

    node_timings = [NodeTiming(**nt) for nt in body.get("node_timings", [])]
    return _b64_to_tensor(logits_b64), node_timings


async def generate_tokens(
    prompt: str,
    max_new_tokens: int | None = None,
) -> tuple[str, int, float, List[NodeTiming]]:
    """Auto-regressive generation driven by the gateway.

    Returns ``(text, tokens_generated, ttft_ms, accumulated_node_timings)``.
    """
    gen_limit = max_new_tokens if max_new_tokens is not None else MAX_NEW_TOKENS
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids: torch.Tensor = encoded["input_ids"]
    eos_token_id = tokenizer.eos_token_id

    time_start = time.perf_counter()
    ttft_ms = 0.0
    tokens_generated = 0
    accumulated: dict[int, dict] = {}

    for step in range(gen_limit):
        ids_list = input_ids.tolist()
        logits, step_timings = await _pipeline_forward(ids_list)

        if step == 0:
            ttft_ms = (time.perf_counter() - time_start) * 1000

        # Accumulate per-node timings across generation steps
        for nt in step_timings:
            nid = nt.node_id
            if nid not in accumulated:
                accumulated[nid] = {
                    "node_id": nid,
                    "compute_ms": 0.0,
                    "serialization_ms": 0.0,
                    "peak_memory_bytes": 0,
                }
            accumulated[nid]["compute_ms"] += nt.compute_ms
            accumulated[nid]["serialization_ms"] += nt.serialization_ms
            accumulated[nid]["peak_memory_bytes"] = max(
                accumulated[nid]["peak_memory_bytes"], nt.peak_memory_bytes
            )

        next_logits = logits[:, -1, :]
        next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=1)
        tokens_generated += 1

        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

    text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    total_timings = [
        NodeTiming(**v)
        for v in sorted(accumulated.values(), key=lambda x: x["node_id"])
    ]
    return text, tokens_generated, ttft_ms, total_timings


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    time_start = time.perf_counter()
    output, tokens_generated, ttft_ms, node_timings = await generate_tokens(
        req.prompt, req.max_new_tokens
    )
    elapsed = (time.perf_counter() - time_start) * 1000
    tps = (tokens_generated / (elapsed / 1000)) if elapsed > 0 else 0.0

    return GenerateResponse(
        output=output,
        elapsed_ms=round(elapsed, 1),
        time_to_first_token_ms=round(ttft_ms, 1),
        tokens_generated=tokens_generated,
        tokens_per_second=round(tps, 2),
        node_timings=node_timings,
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "role": "gateway",
        "model": MODEL_NAME,
        "worker_url": WORKER_URL,
    }

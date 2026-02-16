"""Gateway node – user-facing API that drives auto-regressive generation. It:

1. Tokenises the user's prompt.
2. Iteratively sends token IDs (first step) or updated token IDs (subsequent steps) to worker rank 0 via its /forward endpoint.
3. Worker rank 0 propagates through the pipeline and the last worker returns logits.
4. The gateway samples / greedy-decodes the next token from those logits, appends it, and repeats until max_new_tokens or EOS.
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
from .schemas import GenerateRequest, GenerateResponse

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

async def _pipeline_forward(input_ids: List[List[int]]) -> torch.Tensor:
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

    return _b64_to_tensor(logits_b64)


async def generate_tokens(prompt: str) -> str:
    """Auto-regressive generation driven by the gateway.

    Each iteration sends the full token sequence through the pipeline and greedily picks the next token.
    """
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids: torch.Tensor = encoded["input_ids"]
    eos_token_id = tokenizer.eos_token_id

    for _ in range(MAX_NEW_TOKENS):
        # Send current sequence through the pipeline
        ids_list = input_ids.tolist()
        logits = await _pipeline_forward(ids_list)

        # Take logits for the last position
        next_logits = logits[:, -1, :]
        next_token = torch.argmax(next_logits, dim=-1, keepdim=True)

        # Append and check EOS
        input_ids = torch.cat([input_ids, next_token], dim=1)
        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    time_start = time.perf_counter()
    output = await generate_tokens(req.prompt)
    elapsed = (time.perf_counter() - time_start) * 1000

    return GenerateResponse(
        output=output,
        elapsed_ms=round(elapsed, 1),
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "role": "gateway",
        "model": MODEL_NAME,
        "worker_url": WORKER_URL,
    }

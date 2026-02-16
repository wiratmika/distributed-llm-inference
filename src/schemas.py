from typing import List, Optional

from pydantic import BaseModel


class GenerateRequest(BaseModel):
    prompt: str


class GenerateResponse(BaseModel):
    output: str
    elapsed_ms: float


class ForwardRequest(BaseModel):
    # Only one of these is set
    # input_ids: for the first node (list of token ID lists, one per batch item)
    # hidden_b64: base64-encoded serialized activation tensor for later stages
    input_ids: Optional[List[List[int]]] = None
    hidden_b64: Optional[str] = None


class ForwardResponse(BaseModel):
    logits_b64: Optional[str] = None

from typing import List, Optional

from pydantic import BaseModel


class NodeTiming(BaseModel):
    """Timing breakdown reported by a single pipeline stage (worker node)."""

    node_id: int = 0
    compute_ms: float = 0.0
    serialization_ms: float = 0.0
    peak_memory_bytes: int = 0


class NetworkHopTiming(BaseModel):
    """Estimated network timing for one directed hop pair (request + response)."""

    from_node_id: int
    to_node_id: int
    request_network_ms: float = 0.0
    response_network_ms: float = 0.0
    total_network_ms: float = 0.0


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: Optional[int] = None
    nodes: int = 1


class GenerateResponse(BaseModel):
    output: str
    elapsed_ms: float
    time_to_first_token_ms: float = 0.0
    tokens_generated: int = 0
    tokens_per_second: float = 0.0
    total_network_time_ms: float = 0.0
    node_timings: List[NodeTiming] = []


class ForwardRequest(BaseModel):
    # Only one of these is set
    # input_ids: for the first node (list of token ID lists, one per batch item)
    # hidden_b64: base64-encoded serialized activation tensor for later stages
    input_ids: Optional[List[List[int]]] = None
    hidden_b64: Optional[str] = None
    upstream_sent_wall_ns: Optional[int] = None


class ForwardResponse(BaseModel):
    logits_b64: Optional[str] = None
    node_timings: List[NodeTiming] = []
    ingress_network_ms: float = 0.0
    response_sent_wall_ns: int = 0
    network_hop_timings: List[NetworkHopTiming] = []

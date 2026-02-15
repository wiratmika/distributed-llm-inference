import base64
import io

import torch


def _b64_to_tensor(s: str) -> torch.Tensor:
    raw = base64.b64decode(s)
    buf = io.BytesIO(raw)
    return torch.load(buf, weights_only=True)


def _tensor_to_b64(t: torch.Tensor) -> str:
    buf = io.BytesIO()
    torch.save(t, buf)
    return base64.b64encode(buf.getvalue()).decode("ascii")

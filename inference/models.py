from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from .helpers import get_transformer_layers


@dataclass(frozen=True)
class LayerPartition:
    """Describes a contiguous range of transformer layers assigned to a node."""

    node_id: int
    start_layer: int  # inclusive
    end_layer: int  # exclusive
    num_layers: int


def partition_layers(
    num_layers: int,
    num_nodes: int,
) -> List[LayerPartition]:
    """Layer partitioning logic for distributing transformer model layers across K nodes.

    Divides num_layers transformer layers into num_nodes roughly-equal contiguous chunks.
    Extracts each chunk into a standalone model shard that can run forward passes independently.
    """
    if num_layers < 1:
        raise ValueError(f"num_layers must be >= 1, got {num_layers}")
    if num_nodes < 1:
        raise ValueError(f"num_nodes must be >= 1, got {num_nodes}")
    if num_nodes > num_layers:
        raise ValueError(
            f"num_nodes ({num_nodes}) cannot exceed num_layers ({num_layers})"
        )

    base_size = num_layers // num_nodes
    remainder = num_layers % num_nodes

    partitions: List[LayerPartition] = []
    start = 0
    for node_id in range(num_nodes):
        chunk_size = base_size + (1 if node_id < remainder else 0)
        end = start + chunk_size
        partitions.append(
            LayerPartition(
                node_id=node_id,
                start_layer=start,
                end_layer=end,
                num_layers=chunk_size,
            )
        )
        start = end

    assert start == num_layers, "Partition did not cover all layers"
    return partitions


class ModelShard(nn.Module):
    """A contiguous slice of a transformer model's layers.

    This wraps the embedding / head stages so that:
    - Node 0 runs the token embedding + positional encoding, then its assigned layers.
    - Intermediate nodes run only their assigned layers.
    - Last node runs its assigned layers, then the final layer-norm and language-model head.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        partition: LayerPartition,
        is_first: bool = False,
        is_last: bool = False,
    ) -> None:
        super().__init__()
        self.partition = partition
        self.is_first = is_first
        self.is_last = is_last

        all_layers = get_transformer_layers(model)
        self.layers = nn.ModuleList(
            all_layers[partition.start_layer : partition.end_layer]
        )

        # GPT-2 specific
        if is_first:
            self.wte = model.transformer.wte
            self.wpe = model.transformer.wpe
            self.drop = model.transformer.drop
        if is_last:
            self.ln_f = model.transformer.ln_f
            self.lm_head = model.lm_head

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        if self.is_first:
            if input_ids is None:
                raise ValueError("First shard requires input_ids")

            seq_len = input_ids.size(1)
            if position_ids is None:
                position_ids = torch.arange(
                    seq_len, dtype=torch.long, device=input_ids.device
                ).unsqueeze(0)

            hidden_states = self.wte(input_ids) + self.wpe(position_ids)
            hidden_states = self.drop(hidden_states)

        for layer in self.layers:
            outputs = layer(hidden_states)
            # HuggingFace layers return tuples; hidden state is first element
            hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs

        if self.is_last:
            hidden_states = self.ln_f(hidden_states)
            hidden_states = self.lm_head(hidden_states)

        return hidden_states

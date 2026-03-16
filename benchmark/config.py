from dataclasses import dataclass, field
from itertools import product

MODEL: str = "gpt2-medium"
GENERATION_LENGTH: int = 50
WARMUP_RUNS: int = 1
MEASUREMENT_RUNS: int = 3


@dataclass(frozen=True)
class RunConfig:
    nodes: int
    input_length: int
    concurrent_clients: int
    model: str = MODEL
    generation_length: int = GENERATION_LENGTH
    warmup_runs: int = WARMUP_RUNS
    measurement_runs: int = MEASUREMENT_RUNS

    @property
    def name(self) -> str:
        return f"n{self.nodes}_i{self.input_length}_c{self.concurrent_clients}"


@dataclass
class Experiment:
    id: int
    name: str
    question: str
    configs: list[RunConfig] = field(default_factory=list)


EXPERIMENTS: dict[int, Experiment] = {
    1: Experiment(
        id=1,
        name="Scaling",
        question="At what point does adding nodes help or hurt latency?",
        configs=[
            RunConfig(nodes=n, input_length=il, concurrent_clients=1)
            for n, il in product([1, 2, 4], [32, 128, 512])
        ],
    ),
    2: Experiment(
        id=2,
        name="Concurrency",
        question=(
            "Does pipeline parallelism actually utilize idle stages when "
            "multiple requests arrive concurrently?"
        ),
        configs=[
            RunConfig(nodes=n, input_length=128, concurrent_clients=c)
            for n, c in product([1, 2, 4], [1, 2, 3])
        ],
    ),
}

PROMPT_32 = (
    "Distributed inference splits a language model across several machines. "
    "Each node computes part of the network and forwards activations to the "
    "next stage. This design can reduce memory pressure but adds communication "
    "delay between stages."
)

PROMPT_128 = (
    "Distributed systems coordinate independent machines to solve one task. "
    "In this benchmark, a transformer model is partitioned across workers, "
    "and intermediate activations flow through a pipeline over HTTP. "
    "Sharding can reduce per-node memory and make larger models practical on "
    "CPU-only virtual machines. The trade-off is communication overhead from "
    "serialization, transfer, and synchronization at stage boundaries. "
    "Short prompts often expose this overhead because compute is small compared "
    "to network cost. Longer prompts increase prefill compute and can improve "
    "the compute-to-communication ratio. Under concurrent load, different "
    "requests may occupy different stages at the same time, raising throughput "
    "even if single-request latency does not improve. Careful measurement of "
    "stage compute time, queue wait, and network hops helps identify stragglers "
    "and guides practical partition tuning decisions."
)

PROMPT_512 = (
    " ".join([PROMPT_128] * 4)
)

PROMPTS: dict[int, str] = {
    32: PROMPT_32,
    128: PROMPT_128,
    512: PROMPT_512,
}

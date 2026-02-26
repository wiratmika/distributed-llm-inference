from dataclasses import dataclass, field
from itertools import product

MODEL: str = "gpt2-xl"
GENERATION_LENGTH: int = 50
SAMPLING: str = "greedy"
WARMUP_RUNS: int = 3
MEASUREMENT_RUNS: int = 20
TORCH_THREADS: int = 4


@dataclass(frozen=True)
class RunConfig:
    nodes: int
    input_length: int
    concurrent_clients: int
    model: str = MODEL
    generation_length: int = GENERATION_LENGTH
    sampling: str = SAMPLING
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
    # If set, this experiment reuses data from another experiment
    reuses: int | None = None


EXPERIMENTS: dict[int, Experiment] = {
    1: Experiment(
        id=1,
        name="Scaling",
        question="At what point does adding nodes help or hurt latency?",
        configs=[
            RunConfig(nodes=n, input_length=il, concurrent_clients=1)
            for n, il in product([1, 2, 4, 8], [32, 256, 1024])
        ],
    ),
    2: Experiment(
        id=2,
        name="Time breakdown",
        question=(
            "What fraction of wall-clock time is compute vs. serialization vs. network transfer vs. idle?"
        ),
        configs=[
            RunConfig(nodes=n, input_length=256, concurrent_clients=1)
            for n in [1, 2, 4, 8]
        ],
    ),
    3: Experiment(
        id=3,
        name="Concurrency",
        question=(
            "Does pipeline parallelism actually utilize idle stages when "
            "multiple requests arrive concurrently?"
        ),
        configs=[
            RunConfig(nodes=n, input_length=256, concurrent_clients=c)
            for n, c in product([1, 2, 4, 8], [1, 4, 16])
        ],
    ),
    4: Experiment(
        id=4,
        name="Input sensitivity",
        question="How does input length affect the distribution trade-off?",
        reuses=1,
        configs=[],  # populated below
    ),
    5: Experiment(
        id=5,
        name="Memory",
        question="Does sharding actually reduce per-node memory?",
        configs=[
            RunConfig(nodes=n, input_length=1024, concurrent_clients=1)
            for n in [1, 2, 4, 8]
        ],
    ),
}

# Experiment 4 reuses Experiment 1 configs exactly
EXPERIMENTS[4].configs = list(EXPERIMENTS[1].configs)

PROMPT_32 = (
    "The study of distributed systems is concerned with how independent "
    "computers communicate and coordinate to achieve a common goal in "
    "modern computing"
)

PROMPT_256 = (
    "The study of distributed systems is concerned with how independent "
    "computers communicate and coordinate to achieve a common goal. "
    "In the context of machine learning, distributing inference across "
    "multiple nodes allows large models to be split into smaller shards, "
    "each running on a different machine. This pipeline parallelism "
    "approach sends intermediate activation tensors between stages over "
    "the network. While this introduces communication overhead, it "
    "enables running models that would not fit on a single machine. "
    "The key trade-off is between the computation saved per node and "
    "the latency added by serialization and network transfer. "
    "Factors such as input sequence length, number of pipeline stages, "
    "and degree of concurrent requests all influence the overall "
    "throughput and end-to-end latency of the system. "
    "Understanding these dynamics is essential for designing efficient "
    "distributed inference architectures. When a user sends a prompt "
    "to the gateway, it is tokenized and forwarded to the first worker "
    "in the pipeline. Each worker processes its assigned layers and "
    "passes the resulting hidden states to the next worker. The final "
    "worker produces the output logits, which are sent back to the "
    "gateway for decoding. This process repeats for each generated "
    "token in the autoregressive loop. The primary sources of overhead "
    "are tensor serialization at each hop and the network round trip "
    "time between nodes. For small inputs, this overhead can dominate "
    "the total latency, making single node inference faster."
)

PROMPT_1024 = (
    "The study of distributed systems is concerned with how independent "
    "computers communicate and coordinate to achieve a common goal. "
    "In the context of machine learning, distributing inference across "
    "multiple nodes allows large models to be split into smaller shards, "
    "each running on a different machine. This pipeline parallelism "
    "approach sends intermediate activation tensors between stages over "
    "the network. While this introduces communication overhead, it "
    "enables running models that would not fit on a single machine. "
    "The key trade-off is between the computation saved per node and "
    "the latency added by serialization and network transfer. "
    "Factors such as input sequence length, number of pipeline stages, "
    "and degree of concurrent requests all influence the overall "
    "throughput and end-to-end latency of the system. "
    "Understanding these dynamics is essential for designing efficient "
    "distributed inference architectures. When a user sends a prompt "
    "to the gateway, it is tokenized and forwarded to the first worker "
    "in the pipeline. Each worker processes its assigned layers and "
    "passes the resulting hidden states to the next worker. The final "
    "worker produces the output logits, which are sent back to the "
    "gateway for decoding. This process repeats for each generated "
    "token in the autoregressive loop. The primary sources of overhead "
    "are tensor serialization at each hop and the network round trip "
    "time between nodes. For small inputs, this overhead can dominate "
    "the total latency, making single node inference faster. However, "
    "as input length grows, the computational cost of the attention "
    "mechanism scales quadratically, and the benefits of distributing "
    "this work across multiple machines begin to outweigh the "
    "communication costs. The attention mechanism in transformer models "
    "computes pairwise interactions between all tokens in the input "
    "sequence. For a sequence of length n, this requires order n squared "
    "operations, which becomes significant for longer contexts. By "
    "splitting the model layers across nodes, each machine only needs "
    "to perform attention over its subset of layers, reducing the "
    "per node memory footprint and computation time. The activation "
    "tensors passed between nodes have a fixed size determined by the "
    "hidden dimension of the model, not the number of layers on each "
    "node. This means the communication cost per hop remains constant "
    "regardless of how many layers each node handles. The total "
    "communication cost scales linearly with the number of pipeline "
    "stages, since each additional split introduces one more network "
    "hop. In practice, the optimal number of nodes depends on the "
    "balance between computation savings and communication overhead. "
    "For GPT-2 XL with 48 layers and a hidden dimension of 1600, the "
    "activation tensor at each stage boundary is a matrix of shape "
    "batch size by sequence length by 1600, stored as 32 bit floating "
    "point values. A single activation for a sequence of 1024 tokens "
    "is approximately 6.25 megabytes. At network speeds typical of "
    "cloud virtual machines within the same availability zone, "
    "transferring this amount of data takes only a few milliseconds. "
    "The serialization overhead of converting PyTorch tensors to bytes "
    "and back adds additional latency, typically on the order of one "
    "to five milliseconds depending on the serialization method used. "
    "Base64 encoding of the raw tensor bytes is a simple approach that "
    "works well with JSON based HTTP APIs, though more efficient "
    "binary protocols could reduce this overhead further. Beyond "
    "simple latency measurements, understanding the pipeline behavior "
    "under concurrent load is crucial for evaluating the practical "
    "utility of distributed inference. When multiple requests arrive "
    "simultaneously, the pipeline can process different requests at "
    "different stages concurrently. This means that while node one is "
    "processing the first layers for request B, node two can be "
    "processing the middle layers for request A. This overlap "
    "increases throughput even though individual request latency may "
    "not improve. The degree of overlap depends on the relative "
    "duration of each pipeline stage and the arrival pattern of "
    "requests. If stages are well balanced, meaning each node takes "
    "roughly the same time to process its layers, the pipeline "
    "efficiency is maximized. Imbalanced stages lead to bubbles where "
    "some nodes sit idle waiting for others to finish. Memory "
    "consumption is another critical metric in distributed inference. "
    "The primary motivation for sharding a model across nodes is often "
    "that the full model does not fit in the memory of a single "
    "machine. Even when it does fit, reducing per node memory usage "
    "allows using smaller and cheaper instances. For GPT-2 XL, the "
    "model parameters alone consume approximately six gigabytes in "
    "single precision. Adding the key value cache for long sequences "
    "and the intermediate activations during the forward pass can "
    "easily double this requirement."
)

PROMPTS: dict[int, str] = {
    32: PROMPT_32,
    256: PROMPT_256,
    1024: PROMPT_1024,
}

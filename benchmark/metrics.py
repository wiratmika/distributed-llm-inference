import statistics
import time
from dataclasses import dataclass, field


@dataclass
class NodeMetrics:
    """Timing breakdown for a single pipeline stage (worker node)."""

    node_id: int
    compute_time: float = 0.0
    serialization_time: float = 0.0
    peak_memory_rss: float = 0.0  # MB


@dataclass
class RunMetrics:
    """Metrics captured from a single benchmark run."""

    run_index: int
    end_to_end_latency: float = 0.0
    time_to_first_token: float = 0.0
    total_network_time: float = 0.0
    tokens_generated: int = 0
    tokens_per_second: float = 0.0
    node_metrics: list[NodeMetrics] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class ConfigResult:
    """All measurement runs for one RunConfig with summary stats."""

    config_name: str
    nodes: int
    input_length: int
    concurrent_clients: int
    model: str
    generation_length: int
    experiment_id: int = 0
    experiment_name: str = ""

    runs: list[RunMetrics] = field(default_factory=list)
    latency_median: float = 0.0
    ttft_median: float = 0.0
    overall_throughput: float = 0.0

    peak_memory_per_node: dict[int, float] = field(default_factory=dict)  # MB

    def compute_summary(self) -> None:
        if not self.runs:
            return

        self.latency_median = statistics.median(r.end_to_end_latency for r in self.runs)
        self.ttft_median = statistics.median(r.time_to_first_token for r in self.runs)

        min_start = min(r.timestamp - r.end_to_end_latency for r in self.runs)
        max_end = max(r.timestamp for r in self.runs)
        
        total_tokens = sum(r.tokens_generated for r in self.runs)
        duration = max_end - min_start
        self.overall_throughput = total_tokens / duration

        mem: dict[int, float] = {}

        for run in self.runs:
            for nm in run.node_metrics:
                if nm.node_id not in mem or nm.peak_memory_rss > mem[nm.node_id]:
                    mem[nm.node_id] = nm.peak_memory_rss
        self.peak_memory_per_node = mem

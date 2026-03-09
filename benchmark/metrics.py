import time
from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class NodeMetrics:
    """Timing breakdown for a single pipeline stage (worker node)."""

    node_id: int
    compute_time: float = 0.0
    serialization_time: float = 0.0
    network_transfer_time: float = 0.0
    idle_time: float = 0.0
    peak_memory_rss: int = 0


@dataclass
class RunMetrics:
    """Metrics captured from a single benchmark run."""

    run_index: int
    end_to_end_latency: float = 0.0
    time_to_first_token: float = 0.0
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
    latency_p95: float = 0.0
    ttft_median: float = 0.0
    ttft_p95: float = 0.0
    throughput_median: float = 0.0
    peak_memory_per_node: dict[int, int] = field(default_factory=dict)

    def compute_summary(self) -> None:
        if not self.runs:
            return

        latencies = [r.end_to_end_latency for r in self.runs]
        ttfts = [r.time_to_first_token for r in self.runs]
        tps_values = [r.tokens_per_second for r in self.runs]

        self.latency_median = self._percentile(latencies, 50)
        self.latency_p95 = self._percentile(latencies, 95)

        self.ttft_median = self._percentile(ttfts, 50)
        self.ttft_p95 = self._percentile(ttfts, 95)

        self.throughput_median = self._percentile(tps_values, 50)

        mem: dict[int, int] = {}
        for run in self.runs:
            for nm in run.node_metrics:
                if nm.node_id not in mem or nm.peak_memory_rss > mem[nm.node_id]:
                    mem[nm.node_id] = nm.peak_memory_rss
        self.peak_memory_per_node = mem

    def _percentile(self, values: Sequence[float], pct: float) -> float:
        if not values:
            return 0.0
        sorted_v = sorted(values)
        k = int(pct / 100 * (len(sorted_v) - 1) + 0.5)
        return sorted_v[k]

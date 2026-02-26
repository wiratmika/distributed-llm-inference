import json
import pathlib
from dataclasses import asdict
from typing import Any

from benchmark.metrics import ConfigResult, NodeMetrics, RunMetrics

RESULTS_DIR = pathlib.Path("results")


def save_result(
    result: ConfigResult, directory: pathlib.Path = RESULTS_DIR
) -> pathlib.Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{result.config_name}.json"
    with open(path, "w") as f:
        json.dump(asdict(result), f, indent=2)
    return path


def load_result(path: pathlib.Path) -> ConfigResult:
    with open(path) as f:
        return from_dict(json.load(f))


def load_all_results(directory: pathlib.Path = RESULTS_DIR) -> list[ConfigResult]:
    if not directory.exists():
        return []
    return [load_result(p) for p in sorted(directory.glob("*.json"))]


def from_dict(data: dict[str, Any]) -> ConfigResult:
    runs = [
        RunMetrics(
            run_index=r["run_index"],
            end_to_end_latency=r.get("end_to_end_latency", 0.0),
            time_to_first_token=r.get("time_to_first_token", 0.0),
            tokens_generated=r.get("tokens_generated", 0),
            tokens_per_second=r.get("tokens_per_second", 0.0),
            node_metrics=[NodeMetrics(**nm) for nm in r.get("node_metrics", [])],
            timestamp=r.get("timestamp", 0.0),
        )
        for r in data.get("runs", [])
    ]

    result = ConfigResult(
        config_name=data["config_name"],
        nodes=data["nodes"],
        input_length=data["input_length"],
        concurrent_clients=data["concurrent_clients"],
        model=data["model"],
        generation_length=data["generation_length"],
        runs=runs,
        latency_median=data.get("latency_median", 0.0),
        latency_p95=data.get("latency_p95", 0.0),
        ttft_median=data.get("ttft_median", 0.0),
        ttft_p95=data.get("ttft_p95", 0.0),
        throughput_median=data.get("throughput_median", 0.0),
        peak_memory_per_node={
            int(k): v for k, v in data.get("peak_memory_per_node", {}).items()
        },
    )
    return result

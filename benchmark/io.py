import json
import pathlib
from dataclasses import asdict
from typing import Any

import pandas as pd

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
    return [load_result(p) for p in sorted(directory.rglob("*.json"))]


def from_dict(data: dict[str, Any]) -> ConfigResult:
    runs = [
        RunMetrics(
            run_index=r["run_index"],
            end_to_end_latency=r.get("end_to_end_latency", 0.0),
            time_to_first_token=r.get("time_to_first_token", 0.0),
            total_network_time=r.get("total_network_time", 0.0),
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
        ttft_median=data.get("ttft_median", 0.0),
        overall_throughput=data.get("overall_throughput", 0.0),
        peak_memory_per_node={
            int(k): v for k, v in data.get("peak_memory_per_node", {}).items()
        },
    )
    return result


def results_to_dataframe(
    results: list[ConfigResult],
) -> pd.DataFrame:
    """Flatten a list of :class:`ConfigResult` into a single DataFrame."""
    rows: list[dict] = []
    for result in results:
        for i, run in enumerate(result.runs):
            rows.append(
                {
                    "name": result.config_name,
                    "nodes": result.nodes,
                    "sequence_length": result.input_length,
                    "batch_size": result.concurrent_clients,
                    "generation_length": result.generation_length,
                    "run": i,
                    "total_time": run.end_to_end_latency,
                    "tokens_per_second": run.tokens_per_second,
                    "tokens_generated": run.tokens_generated,
                    "time_to_first_token": run.time_to_first_token,
                    "total_network_time": run.total_network_time,
                }
            )
    return pd.DataFrame(rows)

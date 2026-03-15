import logging
import os
from typing import Callable

import httpx

from benchmark.config import RunConfig
from benchmark.helpers import _fire_requests, _response_to_run_metrics
from benchmark.metrics import ConfigResult, RunMetrics

logger = logging.getLogger(__name__)

GATEWAY_URL: str = os.environ.get("GATEWAY_URL", "http://localhost:8000")


def run_config(
    gateway_url: str,
    config: RunConfig,
    prompt: str,
    experiment_id: int = 0,
    experiment_name: str = "",
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> ConfigResult:
    total_steps = config.warmup_runs + config.measurement_runs

    with httpx.Client(timeout=httpx.Timeout(300.0)) as client:
        # Warmup
        for i in range(config.warmup_runs):
            if progress_callback:
                progress_callback(
                    i, total_steps, f"Gateway: {gateway_url} - Warmup {i + 1}/{config.warmup_runs}"
                )
            _fire_requests(
                client,
                gateway_url,
                prompt,
                config.generation_length,
                config.concurrent_clients,
            )

        # Measurement
        all_runs: list[RunMetrics] = []
        run_idx = 0

        for i in range(config.measurement_runs):
            if progress_callback:
                progress_callback(
                    config.warmup_runs + i,
                    total_steps,
                    f"Gateway: {gateway_url} - Measurement {i + 1}/{config.measurement_runs}",
                )
            responses = _fire_requests(
                client,
                gateway_url,
                prompt,
                config.generation_length,
                config.concurrent_clients,
            )
            for resp in responses:
                all_runs.append(_response_to_run_metrics(resp, run_idx))
                run_idx += 1

    result = ConfigResult(
        config_name=config.name,
        nodes=config.nodes,
        input_length=config.input_length,
        concurrent_clients=config.concurrent_clients,
        model=config.model,
        generation_length=config.generation_length,
        experiment_id=experiment_id,
        experiment_name=experiment_name,
        runs=all_runs,
    )
    result.compute_summary()
    return result

import logging
import os
import pathlib
from typing import Callable

import httpx

from benchmark.config import (
    EXPERIMENTS,
    MEASUREMENT_RUNS,
    PROMPTS,
    WARMUP_RUNS,
    RunConfig,
)
from benchmark.helpers import _fire_requests, _response_to_run_metrics
from benchmark.io import save_result
from benchmark.metrics import ConfigResult, RunMetrics

logger = logging.getLogger(__name__)

GATEWAY_URL: str = os.environ.get("GATEWAY_URL", "http://localhost:8000")


def run_config(
    gateway_url: str,
    config: RunConfig,
    prompt: str,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> ConfigResult:
    total_steps = config.warmup_runs + config.measurement_runs

    with httpx.Client(timeout=httpx.Timeout(300.0)) as client:
        # Warmup
        for i in range(config.warmup_runs):
            if progress_callback:
                progress_callback(
                    i, total_steps, f"Warmup {i + 1}/{config.warmup_runs}"
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
                    f"Measurement {i + 1}/{config.measurement_runs}",
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
        runs=all_runs,
    )
    result.compute_summary()
    return result


def main() -> None:
    """Run benchmarks from the command line."""
    import argparse

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    parser = argparse.ArgumentParser(
        description="Distributed LLM Inference Benchmark Runner"
    )
    parser.add_argument(
        "--gateway-url",
        default=GATEWAY_URL,
        help="Gateway base URL (env: GATEWAY_URL, default: http://localhost:8000)",
    )
    parser.add_argument(
        "--experiment",
        type=int,
        choices=sorted(EXPERIMENTS.keys()),
        help="Experiment ID to run (runs all its configs)",
    )
    parser.add_argument(
        "--nodes", type=int, default=1, help="Node count (single-config mode)"
    )
    parser.add_argument(
        "--input-length",
        type=int,
        default=32,
        choices=sorted(PROMPTS.keys()),
        help="Input prompt token length",
    )
    parser.add_argument(
        "--concurrent-clients", type=int, default=1, help="Concurrent clients"
    )
    parser.add_argument(
        "--warmup", type=int, default=WARMUP_RUNS, help="Warmup iterations"
    )
    parser.add_argument(
        "--measurement",
        type=int,
        default=MEASUREMENT_RUNS,
        help="Measurement iterations",
    )
    parser.add_argument(
        "--output-dir", default="results", help="Results output directory"
    )

    args = parser.parse_args()

    def _log_progress(step: int, total: int, msg: str) -> None:
        logger.info("  [%d/%d] %s", step + 1, total, msg)

    output_dir = pathlib.Path(args.output_dir)

    if args.experiment is not None:
        experiment = EXPERIMENTS[args.experiment]
        logger.info(
            "Running experiment %d: %s (%d configs)",
            experiment.id,
            experiment.name,
            len(experiment.configs),
        )
        for idx, base_cfg in enumerate(experiment.configs):
            cfg = RunConfig(
                nodes=base_cfg.nodes,
                input_length=base_cfg.input_length,
                concurrent_clients=base_cfg.concurrent_clients,
                model=base_cfg.model,
                generation_length=base_cfg.generation_length,
                warmup_runs=args.warmup,
                measurement_runs=args.measurement,
            )
            prompt = PROMPTS.get(cfg.input_length, PROMPTS[32])
            logger.info("Config %d/%d: %s", idx + 1, len(experiment.configs), cfg.name)
            result = run_config(
                args.gateway_url, cfg, prompt, progress_callback=_log_progress
            )
            path = save_result(result, output_dir)
            logger.info(
                "  -> saved %s  (median latency=%.3fs, throughput=%.1f tok/s)",
                path,
                result.latency_median,
                result.throughput_median,
            )
    else:
        cfg = RunConfig(
            nodes=args.nodes,
            input_length=args.input_length,
            concurrent_clients=args.concurrent_clients,
            warmup_runs=args.warmup,
            measurement_runs=args.measurement,
        )
        prompt = PROMPTS.get(cfg.input_length, PROMPTS[32])
        logger.info("Running single config: %s", cfg.name)
        result = run_config(
            args.gateway_url, cfg, prompt, progress_callback=_log_progress
        )
        path = save_result(result, output_dir)
        logger.info("Saved -> %s", path)
        logger.info("  Median latency : %.3f s", result.latency_median)
        logger.info("  P95 latency    : %.3f s", result.latency_p95)
        logger.info("  Median TTFT    : %.3f s", result.ttft_median)
        logger.info("  Throughput     : %.1f tok/s", result.throughput_median)


if __name__ == "__main__":
    main()

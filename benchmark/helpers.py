from concurrent.futures import ThreadPoolExecutor

import httpx

from benchmark.metrics import NodeMetrics, RunMetrics


def _fire_requests(
    client: httpx.Client,
    gateway_url: str,
    prompt: str,
    max_new_tokens: int,
    concurrent_clients: int,
) -> list[dict]:
    """Fire concurrent_clients requests; use a thread pool when > 1."""
    if concurrent_clients <= 1:
        return [_send_request(client, gateway_url, prompt, max_new_tokens)]

    with ThreadPoolExecutor(max_workers=concurrent_clients) as pool:
        futures = [
            pool.submit(_send_request, client, gateway_url, prompt, max_new_tokens)
            for _ in range(concurrent_clients)
        ]
        return [f.result() for f in futures]


def _send_request(
    client: httpx.Client,
    gateway_url: str,
    prompt: str,
    max_new_tokens: int,
) -> dict:
    """Send a single /generate request and return the parsed JSON body."""
    resp = client.post(
        f"{gateway_url.rstrip('/')}/generate",
        json={"prompt": prompt, "max_new_tokens": max_new_tokens},
    )
    resp.raise_for_status()
    return resp.json()


def _response_to_run_metrics(response: dict, run_index: int) -> RunMetrics:
    """Convert a gateway JSON response into a RunMetrics instance."""
    raw_timings: list[dict] = response.get("node_timings", [])

    node_metrics_list: list[NodeMetrics] = []
    for i, nt in enumerate(raw_timings):
        compute_s = nt.get("compute_ms", 0.0) / 1000
        ser_s = nt.get("serialization_ms", 0.0) / 1000
        fwd_s = nt.get("network_ms", 0.0) / 1000  # includes downstream processing

        # Derive pure network transfer time by subtracting downstream totals.
        downstream_s = sum(
            (
                raw_timings[j].get("compute_ms", 0.0)
                + raw_timings[j].get("serialization_ms", 0.0)
                + raw_timings[j].get("network_ms", 0.0)
            )
            / 1000
            for j in range(i + 1, len(raw_timings))
        )
        net_transfer_s = max(0.0, fwd_s - downstream_s)

        node_metrics_list.append(
            NodeMetrics(
                node_id=nt.get("node_id", i),
                compute_time=compute_s,
                serialization_time=ser_s,
                network_transfer_time=net_transfer_s,
                idle_time=0.0,
                peak_memory_rss=nt.get("peak_memory_bytes", 0),
            )
        )

    return RunMetrics(
        run_index=run_index,
        end_to_end_latency=response.get("elapsed_ms", 0.0) / 1000,
        time_to_first_token=response.get("time_to_first_token_ms", 0.0) / 1000,
        tokens_generated=response.get("tokens_generated", 0),
        tokens_per_second=response.get("tokens_per_second", 0.0),
        node_metrics=node_metrics_list,
    )

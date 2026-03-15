import pathlib

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from benchmark.config import (
    EXPERIMENTS,
    PROMPTS,
)
from benchmark.io import (
    load_all_results,
    load_result,
    results_to_dataframe,
    save_result,
)
from benchmark.metrics import ConfigResult
from benchmark.runner import run_config

RESULTS_DIR = pathlib.Path("results")


def page_overview(df: pd.DataFrame) -> None:
    st.header("Distributed LLM Inference Benchmark Dashboard")
    st.text("High-level KPIs and summary table.")

    if df.empty:
        st.info(
            "No benchmark results found. Run benchmarks first and place JSON outputs in the **results/** directory."
        )
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Total configs", df["name"].nunique())
    col2.metric("Total runs", len(df))
    col3.metric("Avg tokens/s", f"{df['tokens_per_second'].mean():.1f}")

    summary = (
        df.groupby("name")
        .agg(
            nodes=("nodes", "first"),
            seq_len=("sequence_length", "first"),
            batch=("batch_size", "first"),
            mean_time=("total_time", "mean"),
            std_time=("total_time", "std"),
            mean_tps=("tokens_per_second", "mean"),
        )
        .reset_index()
        .sort_values("mean_time")
    )
    st.dataframe(summary, use_container_width=True)


def page_run_benchmark(df: pd.DataFrame) -> None:
    st.header("Run Benchmark")

    default_gateway_url = st.text_input(
        "Default Gateway URL",
        "http://localhost:8000",
    )
    with st.expander("Gateway URLs", expanded=True):
        url_1 = st.text_input(
            "1 worker node", "", placeholder="http://gateway-1node:8000"
        )
        url_2 = st.text_input(
            "2 worker nodes", "", placeholder="http://gateway-2node:8000"
        )
        url_4 = st.text_input(
            "4 worker nodes", "", placeholder="http://gateway-4node:8000"
        )

    gateway_urls_map: dict[int, str] = {}
    if url_1:
        gateway_urls_map[1] = url_1
    if url_2:
        gateway_urls_map[2] = url_2
    if url_4:
        gateway_urls_map[4] = url_4

    exp_labels = {
        f"{e.id}: {e.name} - {e.question}": e.id for e in EXPERIMENTS.values()
    }
    exp_label = st.selectbox("Experiment", list(exp_labels.keys()))
    experiment = EXPERIMENTS[exp_labels[exp_label]]

    st.caption(f"**{len(experiment.configs)}** configs in this experiment")

    if experiment.configs:
        all_names = [c.name for c in experiment.configs]
        selected = st.multiselect("Filter configs (leave empty for all)", all_names)
    else:
        selected = []

    configs = experiment.configs
    if selected:
        configs = [c for c in configs if c.name in selected]

    if configs:
        preview = pd.DataFrame(
            {
                "Config": c.name,
                "Nodes": c.nodes,
                "Input length": c.input_length,
                "Concurrent clients": c.concurrent_clients,
                "Generation length": c.generation_length,
                "Warmup": c.warmup_runs,
                "Runs": c.measurement_runs,
            }
            for c in configs
        )
        st.dataframe(preview, use_container_width=True, hide_index=True)

    if not st.button("Start benchmark", type="primary"):
        return

    if not configs:
        st.warning("No configurations selected.")
        return

    progress = st.progress(0.0)
    status = st.empty()
    log_area = st.empty()

    configs_to_run = []
    for cfg in configs:
        if not (RESULTS_DIR / f"{cfg.name}.json").exists():
            configs_to_run.append(cfg)

    total_steps = sum(c.warmup_runs + c.measurement_runs for c in configs_to_run)
    completed_steps = 0

    all_results: list[ConfigResult] = []

    for i, cfg in enumerate(configs):
        if (RESULTS_DIR / f"{cfg.name}.json").exists():
            all_results.append(load_result(RESULTS_DIR / f"{cfg.name}.json"))
            log_area.caption(f":fast_forward: {cfg.name} — skipped (already exists)")
            continue

        prompt = PROMPTS.get(cfg.input_length, PROMPTS[32])

        def _on_progress(
            step: int,
            step_total: int,
            msg: str,
            _cfg_name: str = cfg.name,
            _cfg_idx: int = i,
        ) -> None:
            nonlocal completed_steps
            completed_steps += 1
            if total_steps > 0:
                progress.progress(min(completed_steps / total_steps, 1.0))
            status.text(f"Config {_cfg_idx + 1}/{len(configs)} **{_cfg_name}** — {msg}")

        try:
            gateway_url = gateway_urls_map.get(cfg.nodes, default_gateway_url)
            result = run_config(
                gateway_url,
                cfg,
                prompt,
                experiment_id=experiment.id,
                experiment_name=experiment.name,
                progress_callback=_on_progress,
            )
            save_result(result)

            all_results.append(result)
            log_area.caption(
                f":white_check_mark: {cfg.name} — "
                f"median {result.latency_median:.3f}s, "
                f"{result.overall_throughput:.1f} tok/s"
            )
        except Exception as exc:
            st.error(f"**{cfg.name}** failed: {exc}")

    status.text("Done!")
    progress.progress(1.0)

    if not all_results:
        return

    st.subheader("Results")
    rows = []
    for r in all_results:
        rows.append(
            {
                "Config": r.config_name,
                "Nodes": r.nodes,
                "Input length": r.input_length,
                "Clients": r.concurrent_clients,
                "Latency (median)": f"{r.latency_median:.3f} s",
                "TTFT (median)": f"{r.ttft_median:.3f} s",
                "System Throughput": f"{r.overall_throughput:.1f} tok/s",
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


def page_latency(df: pd.DataFrame) -> None:
    st.header("Latency Analysis")
    st.text("Latency analysis across node counts and sequence lengths.")

    if df.empty:
        st.info("No data available.")
        return

    st.subheader("Latency vs. Node Count")
    latency_by_nodes = (
        df.groupby("nodes")["total_time"].agg(["mean", "std"]).reset_index()
    )
    fig = px.bar(
        latency_by_nodes,
        x="nodes",
        y="mean",
        error_y="std",
        labels={"mean": "Mean latency (s)", "nodes": "Node count"},
        title="Mean generation latency by node count",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Latency vs. Sequence Length")
    latency_by_seq = (
        df.groupby("sequence_length")["total_time"].agg(["mean", "std"]).reset_index()
    )
    fig2 = px.line(
        latency_by_seq,
        x="sequence_length",
        y="mean",
        markers=True,
        labels={
            "mean": "Mean latency (s)",
            "sequence_length": "Sequence length (tokens)",
        },
        title="Mean generation latency by input sequence length",
    )
    st.plotly_chart(fig2, use_container_width=True)


def page_throughput(df: pd.DataFrame) -> None:
    st.header("Throughput Analysis")
    st.text("Throughput analysis across configurations.")

    if df.empty:
        st.info("No data available.")
        return

    st.subheader("Throughput vs. Batch Size")
    throughput_by_batch = (
        df.groupby("batch_size")["tokens_per_second"].agg(["mean", "std"]).reset_index()
    )
    fig = px.bar(
        throughput_by_batch,
        x="batch_size",
        y="mean",
        error_y="std",
        labels={"mean": "Mean tokens/s", "batch_size": "Batch size"},
        title="Throughput by batch size",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Throughput vs. Node Count")
    throughput_by_nodes = (
        df.groupby("nodes")["tokens_per_second"].agg(["mean", "std"]).reset_index()
    )
    fig2 = px.bar(
        throughput_by_nodes,
        x="nodes",
        y="mean",
        error_y="std",
        labels={"mean": "Mean tokens/s", "nodes": "Node count"},
        title="Throughput by node count",
    )
    st.plotly_chart(fig2, use_container_width=True)


def page_scaling(df: pd.DataFrame) -> None:
    st.header("Scaling Efficiency")
    st.text("Scaling efficiency: speedup relative to single-node baseline.")

    if df.empty:
        st.info("No data available.")
        return

    baseline = df.loc[df["nodes"] == 1, "total_time"].mean()
    if pd.isna(baseline) or baseline == 0:
        st.warning("No single-node baseline found — cannot compute scaling efficiency.")
        return

    scaling = df.groupby("nodes")["total_time"].mean().reset_index()
    scaling["speedup"] = baseline / scaling["total_time"]
    scaling["ideal_speedup"] = scaling["nodes"]
    scaling["efficiency"] = (scaling["speedup"] / scaling["ideal_speedup"]) * 100

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=scaling["nodes"],
            y=scaling["speedup"],
            mode="lines+markers",
            name="Actual speedup",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=scaling["nodes"],
            y=scaling["ideal_speedup"],
            mode="lines",
            name="Ideal (linear)",
            line=dict(dash="dash"),
        )
    )
    fig.update_layout(
        title="Scaling: actual vs. ideal speedup",
        xaxis_title="Node count",
        yaxis_title="Speedup",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Efficiency Table")
    st.dataframe(
        scaling[["nodes", "speedup", "ideal_speedup", "efficiency"]].round(2),
        use_container_width=True,
    )


def page_compare(df: pd.DataFrame) -> None:
    st.header("Configuration Comparison")
    st.text("Side-by-side comparison of selected configurations.")

    if df.empty:
        st.info("No data available.")
        return

    configs = df["name"].unique().tolist()
    selected = st.multiselect(
        "Select configurations to compare", configs, default=configs[:4]
    )

    if not selected:
        return

    subset = df[df["name"].isin(selected)]

    fig = px.box(
        subset,
        x="name",
        y="total_time",
        labels={"total_time": "Total time (s)", "name": "Configuration"},
        title="Distribution of generation time by configuration",
    )
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.box(
        subset,
        x="name",
        y="tokens_per_second",
        labels={"tokens_per_second": "Tokens / s", "name": "Configuration"},
        title="Distribution of throughput by configuration",
    )
    st.plotly_chart(fig2, use_container_width=True)


PAGES = {
    "Overview": page_overview,
    "Run Benchmark": page_run_benchmark,
    "Latency": page_latency,
    "Throughput": page_throughput,
    "Scaling Efficiency": page_scaling,
    "Compare Configs": page_compare,
}


def main() -> None:
    st.set_page_config(page_title="LLM Inference Benchmark", layout="wide")

    results = load_all_results(RESULTS_DIR)
    df = results_to_dataframe(results)

    page = st.sidebar.radio("Page", list(PAGES.keys()))
    PAGES[page](df)


if __name__ == "__main__":
    main()

import pathlib
import statistics

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
FIG5_REP_INPUT_LEN = 128
FIG5_REP_CONCURRENCY = 1
PAPER_FONT_STACK = "Times New Roman, Times, serif"


def _apply_paper_figure_style(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        font=dict(family=PAPER_FONT_STACK),
        title_font=dict(family=PAPER_FONT_STACK),
        legend_title_font=dict(family=PAPER_FONT_STACK),
        legend_font=dict(family=PAPER_FONT_STACK),
    )
    fig.update_xaxes(title_font=dict(family=PAPER_FONT_STACK), tickfont=dict(family=PAPER_FONT_STACK))
    fig.update_yaxes(title_font=dict(family=PAPER_FONT_STACK), tickfont=dict(family=PAPER_FONT_STACK))
    return fig


def _plot(fig: go.Figure) -> None:
    st.plotly_chart(_apply_paper_figure_style(fig), use_container_width=True)


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

    gateway_url = st.text_input(
        "Gateway URL",
        "http://localhost:8000",
    )

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
            result = run_config(
                gateway_url,
                cfg,
                prompt,
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
    _plot(fig)

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
    _plot(fig2)


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
    _plot(fig)

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
    _plot(fig2)


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
    _plot(fig)

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
    _plot(fig)

    fig2 = px.box(
        subset,
        x="name",
        y="tokens_per_second",
        labels={"tokens_per_second": "Tokens / s", "name": "Configuration"},
        title="Distribution of throughput by configuration",
    )
    _plot(fig2)


def page_paper_figures(df: pd.DataFrame, results: list[ConfigResult]) -> None:
    st.header("Paper Figures")
    st.text("Figures aligned with the Evaluation section in the report.")

    if df.empty or not results:
        st.info("No data available.")
        return

    # Fig. 1: Latency vs node count (c=1), one line per input length, with std error bars.
    st.subheader("Fig. 1 - Latency vs Node Count")
    fig1_df = (
        df[df["batch_size"] == 1]
        .groupby(["sequence_length", "nodes"])["total_time"]
        .agg(["mean", "std"])
        .reset_index()
    )
    fig1_df["input_label"] = fig1_df["sequence_length"].astype(str)
    fig1 = px.line(
        fig1_df,
        x="nodes",
        y="mean",
        color="input_label",
        error_y="std",
        markers=True,
        labels={
            "nodes": "Node count",
            "mean": "Latency (s)",
            "input_label": "Input length",
        },
        title="Latency vs Node Count (concurrency = 1)",
    )
    _plot(fig1)

    # Fig. 2: TTFT vs node count (c=1), one line per input length, with std error bars.
    st.subheader("Fig. 2 - TTFT vs Node Count")
    fig2_df = (
        df[df["batch_size"] == 1]
        .groupby(["sequence_length", "nodes"])["time_to_first_token"]
        .agg(["mean", "std"])
        .reset_index()
    )
    fig2_df["input_label"] = fig2_df["sequence_length"].astype(str)
    fig2 = px.line(
        fig2_df,
        x="nodes",
        y="mean",
        color="input_label",
        error_y="std",
        markers=True,
        labels={
            "nodes": "Node count",
            "mean": "TTFT (s)",
            "input_label": "Input length",
        },
        title="TTFT vs Node Count (concurrency = 1)",
    )
    _plot(fig2)

    # Fig. 3: Stacked time-share bars for each {input length, nodes} at concurrency=1.
    st.subheader("Fig. 3 - Time-Share Decomposition")
    share_rows: list[dict] = []
    for res in results:
        if res.concurrent_clients != 1:
            continue

        run_net: list[float] = []
        run_comp: list[float] = []
        run_ser: list[float] = []

        for run in res.runs:
            if run.end_to_end_latency <= 0:
                continue
            lat = run.end_to_end_latency
            run_net.append((run.total_network_time / lat) * 100)
            comp = sum(nm.compute_time for nm in run.node_metrics)
            ser = sum(nm.serialization_time for nm in run.node_metrics)
            run_comp.append((comp / lat) * 100)
            run_ser.append((ser / lat) * 100)

        if not run_net:
            continue

        net_med = statistics.median(run_net)
        comp_med = statistics.median(run_comp)
        ser_med = statistics.median(run_ser)
        residual = max(0.0, 100.0 - net_med - comp_med - ser_med)
        label = f"input={res.input_length}, nodes={res.nodes}"

        share_rows.extend(
            [
                {
                    "config": label,
                    "input_tokens": res.input_length,
                    "nodes": res.nodes,
                    "component": "compute",
                    "value": comp_med,
                },
                {
                    "config": label,
                    "input_tokens": res.input_length,
                    "nodes": res.nodes,
                    "component": "serialization",
                    "value": ser_med,
                },
                {
                    "config": label,
                    "input_tokens": res.input_length,
                    "nodes": res.nodes,
                    "component": "network",
                    "value": net_med,
                },
                {
                    "config": label,
                    "input_tokens": res.input_length,
                    "nodes": res.nodes,
                    "component": "residual",
                    "value": residual,
                },
            ]
        )

    if share_rows:
        share_df = pd.DataFrame(share_rows).sort_values(["input_tokens", "nodes"])

        # Build an explicit ordered category for each bar: one entry per {input, nodes}
        inputs_sorted = sorted(share_df["input_tokens"].unique())
        nodes_sorted = sorted(share_df["nodes"].unique())
        categories = [f"{inp}_{n}" for inp in inputs_sorted for n in nodes_sorted]

        # Add a helper x-category column to keep the original data intact
        share_df = share_df.copy()
        share_df["x_cat"] = share_df.apply(lambda r: f"{r['input_tokens']}_{r['nodes']}", axis=1)

        # Primary tick labels should show node counts (e.g. '1n','2n','4n') for each bar
        node_tick_labels = [f"{n}n" for _ in inputs_sorted for n in nodes_sorted]

        fig3 = px.bar(
            share_df,
            x="x_cat",
            y="value",
            color="component",
            category_orders={"x_cat": categories},
            labels={"value": "Time share (%)"},
            title="Time-share breakdown per input length and node count (concurrency = 1)",
        )
        fig3.update_layout(barmode="stack")
        fig3.update_layout(legend=dict(traceorder="normal"))

        # Replace the default x tick labels with node-count labels (primary ticks)
        fig3.update_xaxes(
            tickmode="array",
            tickvals=categories,
            ticktext=node_tick_labels,
            title_text=None,
            categoryorder="array",
            categoryarray=categories,
        )

        # Add bracket-like annotations spanning the three node bars for each input length
        # Increase bottom margin to make room for the secondary labels
        fig3.update_layout(margin=dict(b=170))

        # Position brackets and labels further below the axis so they don't overlap ticks
        bracket_y = -0.15
        text_y = -0.30

        # Slightly reduce x-axis tick font to improve legibility at small widths
        fig3.update_xaxes(tickfont=dict(size=10))
        for inp in inputs_sorted:
            start_cat = f"{inp}_{nodes_sorted[0]}"
            mid_cat = f"{inp}_{nodes_sorted[1]}"
            end_cat = f"{inp}_{nodes_sorted[-1]}"

            # horizontal bracket line
            fig3.add_shape(
                type="line",
                xref="x",
                yref="paper",
                x0=start_cat,
                x1=end_cat,
                y0=bracket_y,
                y1=bracket_y,
                line=dict(color="rgba(0,0,0,0.45)", width=0.8),
            )
            # small vertical end ticks
            fig3.add_shape(
                type="line",
                xref="x",
                yref="paper",
                x0=start_cat,
                x1=start_cat,
                y0=bracket_y,
                y1=bracket_y + 0.03,
                line=dict(color="rgba(0,0,0,0.45)", width=0.8),
            )
            fig3.add_shape(
                type="line",
                xref="x",
                yref="paper",
                x0=end_cat,
                x1=end_cat,
                y0=bracket_y,
                y1=bracket_y + 0.03,
                line=dict(color="rgba(0,0,0,0.45)", width=0.8),
            )

            # centered input-length label below the bracket (secondary grouping)
            fig3.add_annotation(
                x=mid_cat,
                xref="x",
                y=text_y,
                yref="paper",
                text=str(inp),
                showarrow=False,
                font=dict(size=12),
                xanchor="center",
            )

        _plot(fig3)
    else:
        st.info("Not enough data to render Fig. 3.")

    # Fig. 4: Throughput vs concurrency at input length 128, one line per node count.
    st.subheader("Fig. 4 - Throughput vs Concurrency (Input Length = 128)")
    fig4_rows: list[dict] = []
    for res in results:
        if res.input_length != 128:
            continue
        fig4_rows.append(
            {
                "nodes": res.nodes,
                "concurrency": res.concurrent_clients,
                "throughput": res.overall_throughput,
            }
        )
    if fig4_rows:
        fig4_df = pd.DataFrame(fig4_rows).sort_values(["nodes", "concurrency"])
        fig4_df["nodes_label"] = fig4_df["nodes"].astype(str)
        fig4 = px.line(
            fig4_df,
            x="concurrency",
            y="throughput",
            color="nodes_label",
            markers=True,
            labels={
                "concurrency": "Concurrent clients",
                "throughput": "Throughput (tokens/s)",
                "nodes_label": "Node count",
            },
            title="Throughput vs Concurrency at Input Length 128",
        )
        _plot(fig4)
    else:
        st.info("No input_length=128 results available for Fig. 4.")

    # Fig. 5: Per-node peak RSS bar chart for node counts 1, 2, 4.
    st.subheader("Fig. 5 - Per-Node Peak RSS")
    st.caption(
        "Default representative setting: input length 128, concurrency 1. "
        "Use override only for sensitivity checks."
    )

    use_mem_override = st.checkbox(
        "Override Fig. 5 parameters",
        value=False,
        help="Keep disabled for the paper's canonical memory comparison.",
    )

    if use_mem_override:
        col_a, col_b = st.columns(2)
        mem_input_len = col_a.selectbox("Input length for memory figure", [32, 128, 512], index=1)
        mem_concurrency = col_b.selectbox("Concurrency for memory figure", [1, 2, 3], index=0)
    else:
        mem_input_len = FIG5_REP_INPUT_LEN
        mem_concurrency = FIG5_REP_CONCURRENCY

    mem_rows: list[dict] = []
    for res in results:
        if res.input_length != mem_input_len or res.concurrent_clients != mem_concurrency:
            continue
        for node_id, rss_mb in sorted(res.peak_memory_per_node.items()):
            mem_rows.append(
                {
                    "nodes": res.nodes,
                    "worker": f"rank-{node_id}",
                    "peak_rss_mb": rss_mb,
                }
            )

    if mem_rows:
        mem_df = pd.DataFrame(mem_rows)
        mem_df["nodes_label"] = mem_df["nodes"].astype(str)
        fig5 = px.bar(
            mem_df,
            x="worker",
            y="peak_rss_mb",
            color="nodes_label",
            barmode="group",
            labels={
                "worker": "Worker rank",
                "peak_rss_mb": "Peak RSS (MB)",
                "nodes_label": "Node count",
            },
            title=(
                "Per-node Peak RSS by Pipeline Size "
                f"(input={mem_input_len}, concurrency={mem_concurrency})"
            ),
        )
        _plot(fig5)
    else:
        st.info("No matching results available for Fig. 5 with the selected filters.")


PAGES = {
    "Overview": page_overview,
    "Run Benchmark": page_run_benchmark,
    "Latency": page_latency,
    "Throughput": page_throughput,
    "Scaling Efficiency": page_scaling,
    "Compare Configs": page_compare,
    "Paper Figures": page_paper_figures,
}


def main() -> None:
    st.set_page_config(page_title="LLM Inference Benchmark", layout="centered")

    # Constrain content width so paper figures are easier to read and screenshot consistently.
    st.markdown(
        """
        <style>
            html, body, [class*="css"], .stApp {
                font-family: Times New Roman, Times, serif;
            }
            .block-container {
                max-width: 1050px;
                padding-top: 1.25rem;
                padding-bottom: 2rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    results = load_all_results(RESULTS_DIR)
    df = results_to_dataframe(results)

    page = st.sidebar.radio("Page", list(PAGES.keys()))
    # Paper Figures needs raw ConfigResult objects (peak_memory_per_node, per-run
    # timing breakdowns) that don't survive the dataframe flattening, so it's
    # dispatched separately.
    if page == "Paper Figures":
        page_paper_figures(df, results)
    else:
        PAGES[page](df)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Render fixed-window flow graphs as PNG images."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render flow-window graph previews.")
    parser.add_argument("--input-csv", type=Path, required=True, help="Cleaned flow CSV from gcn_ids/data_graph.py.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for PNG images.")
    parser.add_argument("--max-windows", type=int, default=16, help="Maximum number of windows to render.")
    parser.add_argument("--layout-seed", type=int, default=42, help="NetworkX layout seed.")
    parser.add_argument("--dpi", type=int, default=180, help="PNG resolution.")
    return parser


def render_window(window_df: pd.DataFrame, output_path: Path, title: str, layout_seed: int, dpi: int) -> None:
    graph = nx.DiGraph()
    edge_counts: dict[tuple[str, str], int] = {}

    for row in window_df.itertuples(index=False):
        src = str(getattr(row, "Source_IP"))
        dst = str(getattr(row, "Destination_IP"))
        is_malicious = int(getattr(row, "binary_label")) == 1

        for node in (src, dst):
            if node not in graph:
                graph.add_node(node, malicious=False, flow_count=0)

        graph.nodes[src]["flow_count"] += 1
        graph.nodes[dst]["flow_count"] += 1
        graph.nodes[src]["malicious"] = graph.nodes[src]["malicious"] or is_malicious
        graph.nodes[dst]["malicious"] = graph.nodes[dst]["malicious"] or is_malicious

        edge_key = (src, dst)
        edge_counts[edge_key] = edge_counts.get(edge_key, 0) + 1
        graph.add_edge(src, dst)

    if not graph:
        return

    node_colors = ["#d62728" if graph.nodes[n]["malicious"] else "#2ca02c" for n in graph.nodes]
    node_sizes = [min(120 + graph.nodes[n]["flow_count"] * 8, 900) for n in graph.nodes]
    widths = [min(0.5 + edge_counts[(u, v)] * 0.08, 4.0) for u, v in graph.edges]

    pos = nx.spring_layout(graph, k=0.9, iterations=100, seed=layout_seed, weight=None)

    plt.figure(figsize=(13, 9))
    plt.title(title, fontsize=15, fontweight="bold", pad=14)
    nx.draw_networkx_edges(
        graph,
        pos,
        alpha=0.28,
        width=widths,
        edge_color="#555555",
        arrows=True,
        arrowsize=10,
        connectionstyle="arc3,rad=0.08",
    )
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors="#111111",
        linewidths=0.8,
    )

    benign_patch = mpatches.Patch(color="#2ca02c", label="Benign-only node")
    malicious_patch = mpatches.Patch(color="#d62728", label="Node touched by malicious flow")
    plt.legend(handles=[benign_patch, malicious_patch], loc="upper right", framealpha=0.92)
    plt.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()


def main() -> None:
    args = build_parser().parse_args()
    df = pd.read_csv(args.input_csv, low_memory=False)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Timestamp", "window_start"]).copy()
    df = df.rename(columns={"Source IP": "Source_IP", "Destination IP": "Destination_IP"})

    windows = sorted(df["window_start"].dropna().unique())[: args.max_windows]
    for idx, window_start in enumerate(windows, start=1):
        window_df = df[df["window_start"] == window_start]
        malicious_flows = int((window_df["binary_label"] == 1).sum())
        title = (
            f"UNSW Graph Window {idx:02d} | {window_start} | "
            f"Flows: {len(window_df):,} | Malicious: {malicious_flows:,}"
        )
        output_path = args.output_dir / f"unsw_window_{idx:02d}.png"
        render_window(window_df, output_path, title, args.layout_seed, args.dpi)
        print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build node-level windowed dataset from flow records for GNN/baseline training."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple


TIME_FORMATS = [
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M",
    "%m/%d/%Y %I:%M:%S %p",
    "%m/%d/%Y %I:%M %p",
    "%d/%m/%Y %I:%M:%S %p",
    "%d/%m/%Y %I:%M %p",
]


@dataclass
class NodeAgg:
    flow_count: int = 0
    malicious_flow_count: int = 0
    benign_flow_count: int = 0
    inbound_flow_count: int = 0
    outbound_flow_count: int = 0
    protocol_tcp_count: int = 0
    protocol_udp_count: int = 0
    protocol_other_count: int = 0
    total_packets_sum: float = 0.0
    total_bytes_sum: float = 0.0
    flow_duration_sum: float = 0.0
    flow_bytes_per_s_sum: float = 0.0
    flow_packets_per_s_sum: float = 0.0
    avg_packet_size_sum: float = 0.0


def parse_ts(raw: str) -> datetime:
    value = raw.strip()
    for fmt in TIME_FORMATS:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unsupported timestamp format: {raw}")


def safe_float(s: str) -> float:
    try:
        v = float(s)
        if v != v or v in (float("inf"), float("-inf")):
            return 0.0
        return v
    except Exception:
        return 0.0


def floor_window(ts: datetime, minutes: int) -> datetime:
    discard = timedelta(minutes=ts.minute % minutes, seconds=ts.second, microseconds=ts.microsecond)
    return ts - discard


def add_row(agg: NodeAgg, row: Dict[str, str], is_source: bool, is_malicious: bool) -> None:
    proto = int(safe_float(row.get("Protocol", "0")))
    total_fwd_pkts = safe_float(row.get("Total Fwd Packets", "0"))
    total_bwd_pkts = safe_float(row.get("Total Backward Packets", "0"))
    total_fwd_bytes = safe_float(row.get("Total Length of Fwd Packets", "0"))
    total_bwd_bytes = safe_float(row.get("Total Length of Bwd Packets", "0"))

    agg.flow_count += 1
    agg.malicious_flow_count += int(is_malicious)
    agg.benign_flow_count += int(not is_malicious)
    agg.outbound_flow_count += int(is_source)
    agg.inbound_flow_count += int(not is_source)

    if proto == 6:
        agg.protocol_tcp_count += 1
    elif proto == 17:
        agg.protocol_udp_count += 1
    else:
        agg.protocol_other_count += 1

    agg.total_packets_sum += total_fwd_pkts + total_bwd_pkts
    agg.total_bytes_sum += total_fwd_bytes + total_bwd_bytes
    agg.flow_duration_sum += safe_float(row.get("Flow Duration", "0"))
    agg.flow_bytes_per_s_sum += safe_float(row.get("Flow Bytes/s", "0"))
    agg.flow_packets_per_s_sum += safe_float(row.get("Flow Packets/s", "0"))
    agg.avg_packet_size_sum += safe_float(row.get("Average Packet Size", "0"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build node-level windowed dataset from flow CSV.")
    parser.add_argument("--input-csv", type=Path, default=Path("data/processed/merged_flow_unscaled.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--window-minutes", type=int, default=5)
    parser.add_argument("--label-rule", choices=["any_malicious_wins", "majority"], default="any_malicious_wins")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    node_aggs: Dict[Tuple[datetime, str], NodeAgg] = defaultdict(NodeAgg)
    peers: Dict[Tuple[datetime, str], set] = defaultdict(set)

    total_flows = 0
    skipped_rows = 0

    with args.input_csv.open("r", newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            src = row.get("Source IP", "").strip()
            dst = row.get("Destination IP", "").strip()
            ts_raw = row.get("Timestamp", "")
            if not src or not dst or not ts_raw:
                skipped_rows += 1
                continue

            try:
                ts = parse_ts(ts_raw)
            except ValueError:
                skipped_rows += 1
                continue

            w = floor_window(ts, args.window_minutes)
            is_malicious = row.get("binary_label", "0").strip() == "1"

            add_row(node_aggs[(w, src)], row, is_source=True, is_malicious=is_malicious)
            add_row(node_aggs[(w, dst)], row, is_source=False, is_malicious=is_malicious)

            peers[(w, src)].add(dst)
            peers[(w, dst)].add(src)

            total_flows += 1

    rows: List[Dict[str, str]] = []
    windows = sorted({w for (w, _node) in node_aggs.keys()})

    split_map = {}
    n = len(windows)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    for i, w in enumerate(windows):
        if i < train_end:
            split_map[w] = "train"
        elif i < val_end:
            split_map[w] = "val"
        else:
            split_map[w] = "test"

    label_counter = Counter()
    split_counter = Counter()

    for (w, node), a in sorted(node_aggs.items(), key=lambda x: (x[0][0], x[0][1])):
        if args.label_rule == "any_malicious_wins":
            label = 1 if a.malicious_flow_count > 0 else 0
        else:
            label = 1 if a.malicious_flow_count > a.benign_flow_count else 0

        flow_count = max(a.flow_count, 1)
        out = {
            "window_start": w.strftime("%Y-%m-%d %H:%M:%S"),
            "node_ip": node,
            "label": str(label),
            "flow_count": str(a.flow_count),
            "malicious_flow_count": str(a.malicious_flow_count),
            "benign_flow_count": str(a.benign_flow_count),
            "malicious_flow_ratio": f"{a.malicious_flow_count / flow_count:.6f}",
            "unique_peer_count": str(len(peers[(w, node)])),
            "inbound_flow_count": str(a.inbound_flow_count),
            "outbound_flow_count": str(a.outbound_flow_count),
            "protocol_tcp_count": str(a.protocol_tcp_count),
            "protocol_udp_count": str(a.protocol_udp_count),
            "protocol_other_count": str(a.protocol_other_count),
            "total_packets_sum": f"{a.total_packets_sum:.6f}",
            "total_bytes_sum": f"{a.total_bytes_sum:.6f}",
            "avg_flow_duration": f"{a.flow_duration_sum / flow_count:.6f}",
            "avg_flow_bytes_per_s": f"{a.flow_bytes_per_s_sum / flow_count:.6f}",
            "avg_flow_packets_per_s": f"{a.flow_packets_per_s_sum / flow_count:.6f}",
            "avg_packet_size": f"{a.avg_packet_size_sum / flow_count:.6f}",
            "split": split_map[w],
        }

        rows.append(out)
        label_counter[label] += 1
        split_counter[split_map[w]] += 1

    fields = [
        "window_start",
        "node_ip",
        "label",
        "flow_count",
        "malicious_flow_count",
        "benign_flow_count",
        "malicious_flow_ratio",
        "unique_peer_count",
        "inbound_flow_count",
        "outbound_flow_count",
        "protocol_tcp_count",
        "protocol_udp_count",
        "protocol_other_count",
        "total_packets_sum",
        "total_bytes_sum",
        "avg_flow_duration",
        "avg_flow_bytes_per_s",
        "avg_flow_packets_per_s",
        "avg_packet_size",
        "split",
    ]

    out_all = args.output_dir / "node_window_dataset.csv"
    out_train = args.output_dir / "node_window_train.csv"
    out_val = args.output_dir / "node_window_val.csv"
    out_test = args.output_dir / "node_window_test.csv"
    out_manifest = args.output_dir / "node_window_manifest.json"

    with out_all.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    for split_name, out_path in [("train", out_train), ("val", out_val), ("test", out_test)]:
        with out_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for row in rows:
                if row["split"] == split_name:
                    w.writerow(row)

    manifest = {
        "input_csv": str(args.input_csv),
        "window_minutes": args.window_minutes,
        "label_rule": args.label_rule,
        "time_split": {"train": 0.70, "val": 0.15, "test": 0.15},
        "total_input_flows": total_flows,
        "skipped_input_rows": skipped_rows,
        "num_windows": len(windows),
        "num_node_window_rows": len(rows),
        "label_distribution": {"benign_0": label_counter[0], "malicious_1": label_counter[1]},
        "split_distribution": dict(split_counter),
        "outputs": {
            "all": str(out_all),
            "train": str(out_train),
            "val": str(out_val),
            "test": str(out_test),
        },
        "features": fields,
    }

    out_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote: {out_all}")
    print(f"Wrote: {out_train}")
    print(f"Wrote: {out_val}")
    print(f"Wrote: {out_test}")
    print(f"Wrote: {out_manifest}")
    print(f"Node-window rows: {len(rows)}")


if __name__ == "__main__":
    main()

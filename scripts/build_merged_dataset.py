#!/usr/bin/env python3
"""Build a merged flow dataset with common features and scaling.

This script is dependency-free (stdlib only) so it can run in minimal environments.
It performs:
1) Header normalization and common feature alignment.
2) Reservoir sampling per source dataset to keep balanced class/source coverage.
3) Cleaning (NaN/Inf handling), binary label mapping, z-score scaling.
4) Export of unscaled + scaled CSVs and a JSON manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Tuple


PREFERRED_FEATURES = [
    "Protocol",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Fwd IAT Mean",
    "Bwd IAT Mean",
    "Average Packet Size",
]

IDENTITY_COLUMNS = ["Source IP", "Destination IP", "Timestamp", "Label"]

ALIASES = {
    "Src IP": "Source IP",
    "srcip": "Source IP",
    "Dst IP": "Destination IP",
    "dstip": "Destination IP",
    "stime": "Timestamp",
    "dur": "Flow Duration",
    "spkts": "Total Fwd Packets",
    "dpkts": "Total Backward Packets",
    "sbytes": "Total Length of Fwd Packets",
    "dbytes": "Total Length of Bwd Packets",
    "Tot Fwd Pkts": "Total Fwd Packets",
    "Tot Bwd Pkts": "Total Backward Packets",
    "TotLen Fwd Pkts": "Total Length of Fwd Packets",
    "TotLen Bwd Pkts": "Total Length of Bwd Packets",
    "Flow Byts/s": "Flow Bytes/s",
    "rate": "Flow Packets/s",
    "Flow Pkts/s": "Flow Packets/s",
    "Pkt Size Avg": "Average Packet Size",
    "label": "Label",
    "proto": "Protocol",
}


def normalize_col(name: str) -> str:
    key = re.sub(r"\s+", " ", name).strip()
    return ALIASES.get(key, ALIASES.get(key.lower(), key))


def parse_float(value: str) -> float | None:
    if value is None:
        return None
    x = value.strip()
    if x == "":
        return None
    x_lower = x.lower()
    if x_lower in {"nan", "na", "none", "null"}:
        return None
    if x_lower in {"inf", "+inf", "infinity", "+infinity"}:
        return None
    if x_lower in {"-inf", "-infinity"}:
        return None
    try:
        v = float(x)
    except ValueError:
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def label_to_binary(label: str) -> int:
    token = label.strip().lower()
    return 0 if token in {"0", "benign", "normal", "background"} else 1


@dataclass
class DatasetStats:
    name: str
    sampled_rows: int
    benign_rows: int
    malicious_rows: int


def read_headers(
    csv_files: Iterable[Path],
) -> Tuple[Dict[Path, Dict[str, str]], List[str], List[Dict[str, str]]]:
    per_file_map: Dict[Path, Dict[str, str]] = {}
    skipped_files: List[Dict[str, str]] = []
    common: set[str] | None = None

    for path in csv_files:
        with path.open("r", newline="", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            raw = next(reader)

        mapping = {normalize_col(c): c for c in raw}
        required_missing = [c for c in IDENTITY_COLUMNS if c not in mapping]
        if required_missing:
            skipped_files.append(
                {
                    "file": str(path),
                    "reason": "missing_required_identity_columns",
                    "missing": required_missing,
                }
            )
            continue

        per_file_map[path] = mapping
        keys = set(mapping.keys())
        common = keys if common is None else common & keys

    if common is None:
        return {}, [], skipped_files

    ordered_common = sorted(common)
    return per_file_map, ordered_common, skipped_files


def reservoir_sample_rows(
    path: Path,
    header_map: Dict[str, str],
    keep_columns: List[str],
    per_file_limit: int,
    seed: int,
) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    reservoir: List[Dict[str, str]] = []

    with path.open("r", newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        n = 0
        for row in reader:
            extracted = {col: row.get(header_map[col], "") for col in keep_columns}
            if len(reservoir) < per_file_limit:
                reservoir.append(extracted)
            else:
                j = rng.randint(0, n)
                if j < per_file_limit:
                    reservoir[j] = extracted
            n += 1

    return reservoir


def compute_impute_and_scale(rows: List[Dict[str, str]], numeric_features: List[str]):
    # Collect numeric values for median imputation.
    values_by_feature: Dict[str, List[float]] = {f: [] for f in numeric_features}
    for row in rows:
        for feat in numeric_features:
            v = parse_float(row.get(feat, ""))
            if v is not None:
                values_by_feature[feat].append(v)

    medians: Dict[str, float] = {}
    means: Dict[str, float] = {}
    stds: Dict[str, float] = {}

    for feat in numeric_features:
        vals = values_by_feature[feat]
        if not vals:
            medians[feat] = 0.0
            means[feat] = 0.0
            stds[feat] = 1.0
            continue
        med = float(median(vals))
        medians[feat] = med
        mean = sum(vals) / len(vals)
        means[feat] = mean
        var = sum((x - mean) ** 2 for x in vals) / len(vals)
        std = math.sqrt(var)
        stds[feat] = std if std > 1e-12 else 1.0

    cleaned_unscaled = []
    cleaned_scaled = []

    for row in rows:
        out_u = dict(row)
        out_s = dict(row)

        for feat in numeric_features:
            raw = parse_float(row.get(feat, ""))
            value = medians[feat] if raw is None else raw
            out_u[feat] = f"{value:.6f}"
            z = (value - means[feat]) / stds[feat]
            out_s[feat] = f"{z:.6f}"

        out_u["binary_label"] = str(label_to_binary(row.get("Label", "")))
        out_s["binary_label"] = out_u["binary_label"]

        cleaned_unscaled.append(out_u)
        cleaned_scaled.append(out_s)

    return cleaned_unscaled, cleaned_scaled, medians, means, stds


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge and scale CSV flow datasets.")
    parser.add_argument(
        "--input-dirs",
        nargs="+",
        type=Path,
        default=[Path("data/raw/unsw_complete")],
        help="One or more directories containing CSV files",
    )
    parser.add_argument(
        "--input-glob",
        type=str,
        default="*.csv",
        help="Glob pattern used inside each input directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory to write merged outputs",
    )
    parser.add_argument(
        "--per-file-limit",
        type=int,
        default=50000,
        help="Reservoir sample size per source file",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    csv_files: List[Path] = []
    for d in args.input_dirs:
        csv_files.extend(sorted(d.glob(args.input_glob)))
    csv_files = sorted(set(csv_files))

    if not csv_files:
        joined = ", ".join(str(x) for x in args.input_dirs)
        raise SystemExit(f"No CSV files found in input dirs: {joined}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    header_maps, common, skipped_files = read_headers(csv_files)
    csv_files = sorted(header_maps.keys())
    if not csv_files:
        raise SystemExit("No node-compatible CSV files found after schema checks.")
    common_set = set(common)

    keep_features = [c for c in PREFERRED_FEATURES if c in common_set]
    keep_columns = [c for c in IDENTITY_COLUMNS if c in common_set] + keep_features

    missing_id = [c for c in IDENTITY_COLUMNS if c not in common_set]
    if missing_id:
        raise SystemExit(f"Required identity columns missing from common schema: {missing_id}")

    all_rows: List[Dict[str, str]] = []
    per_dataset_stats: List[DatasetStats] = []

    for idx, path in enumerate(csv_files):
        rows = reservoir_sample_rows(
            path=path,
            header_map=header_maps[path],
            keep_columns=keep_columns,
            per_file_limit=args.per_file_limit,
            seed=args.seed + idx,
        )

        benign = 0
        malicious = 0
        for r in rows:
            if label_to_binary(r.get("Label", "")) == 0:
                benign += 1
            else:
                malicious += 1
            r["source_dataset"] = path.name

        all_rows.extend(rows)
        per_dataset_stats.append(
            DatasetStats(path.name, len(rows), benign_rows=benign, malicious_rows=malicious)
        )

    random.Random(args.seed).shuffle(all_rows)

    unscaled, scaled, medians, means, stds = compute_impute_and_scale(all_rows, keep_features)

    field_order = [
        "source_dataset",
        "Source IP",
        "Destination IP",
        "Timestamp",
        "Label",
        "binary_label",
    ] + keep_features

    unscaled_path = args.output_dir / "merged_flow_unscaled.csv"
    scaled_path = args.output_dir / "merged_flow_scaled.csv"
    manifest_path = args.output_dir / "merged_flow_manifest.json"

    with unscaled_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=field_order)
        writer.writeheader()
        writer.writerows(unscaled)

    with scaled_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=field_order)
        writer.writeheader()
        writer.writerows(scaled)

    manifest = {
        "input_dirs": [str(x) for x in args.input_dirs],
        "num_source_files": len(csv_files),
        "num_skipped_files": len(skipped_files),
        "source_files": [p.name for p in csv_files],
        "skipped_files": skipped_files,
        "rows_total": len(all_rows),
        "per_file_limit": args.per_file_limit,
        "seed": args.seed,
        "selected_features": keep_features,
        "imputation": "median",
        "scaling": "zscore",
        "binary_label_rule": "0 if Label == BENIGN else 1",
        "per_dataset_stats": [s.__dict__ for s in per_dataset_stats],
        "feature_stats": {
            feat: {
                "median": medians[feat],
                "mean": means[feat],
                "std": stds[feat],
            }
            for feat in keep_features
        },
        "output_files": {
            "unscaled_csv": str(unscaled_path),
            "scaled_csv": str(scaled_path),
        },
    }

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote: {unscaled_path}")
    print(f"Wrote: {scaled_path}")
    print(f"Wrote: {manifest_path}")
    print(f"Total rows: {len(all_rows)}")


if __name__ == "__main__":
    main()

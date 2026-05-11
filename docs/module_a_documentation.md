# Module A Documentation

Owner: Dhanush

Module A is the data engineering stage of the project. Its job is to take raw intrusion-detection flow records, clean and normalize them, convert them into fixed-window graph datasets, and export reproducible artifacts for Module B (GCN modeling) and Module C (graph visualization).

## Purpose

Module A prepares raw network flow data for graph neural network training by:

- loading one or more CSV files,
- normalizing dataset-specific column names into one common schema,
- cleaning timestamps, labels, protocol fields, and numeric traffic features,
- grouping flows into fixed time windows,
- building one directed graph per window,
- generating node features, edge features, and node labels,
- splitting graph windows into train, validation, and test sets,
- fitting scalers on train graphs only,
- exporting `.npz` graph files and metadata files.

## Main Implementation

Main code:

- `gcn_ids/data_graph.py`

Related tests:

- `tests/test_data_graph.py`

## Supported Inputs

Module A supports:

- UNSW-NB15 raw shard files such as `UNSW-NB15_1.csv` through `UNSW-NB15_4.csv`
- UNSW-style CSVs with named columns such as `srcip`, `dstip`, `stime`, `proto`, `dur`, `spkts`, `dpkts`, `sbytes`, `dbytes`, `rate`, `label`
- earlier IDS/IoT-style CSVs with columns already close to the canonical schema

For headerless raw UNSW files, Module A applies the official 49-column UNSW schema automatically.

## Canonical Flow Schema

After loading and normalization, Module A keeps these cleaned flow fields:

- `source_file`
- `Source IP`
- `Destination IP`
- `Timestamp`
- `Label`
- `binary_label`
- `Protocol`
- `Flow Duration`
- `Total Fwd Packets`
- `Total Backward Packets`
- `Total Length of Fwd Packets`
- `Total Length of Bwd Packets`
- `Flow Bytes/s`
- `Flow Packets/s`
- `Average Packet Size`

These cleaned fields are then aggregated into graph features.

## Graph Construction

Each time window becomes one directed graph:

- Nodes: IP addresses
- Directed edges: observed flows from source IP to destination IP in that window
- Node labels: derived from flows touching the node

Current node label rule:

- `any_malicious_wins`
  - a node is labeled malicious in a window if it appears in at least one malicious flow in that window

## Feature Design

### Node features

Each node has 14 features:

- `flow_count`
- `malicious_flow_ratio`
- `inbound_flow_count`
- `outbound_flow_count`
- `unique_peer_count`
- `protocol_tcp_ratio`
- `protocol_udp_ratio`
- `protocol_other_ratio`
- `total_packets_sum`
- `total_bytes_sum`
- `avg_flow_duration`
- `avg_flow_bytes_per_s`
- `avg_flow_packets_per_s`
- `avg_packet_size`

### Edge features

Each directed edge has 10 features:

- `flow_count`
- `malicious_ratio`
- `total_packets_sum`
- `total_bytes_sum`
- `avg_flow_duration`
- `avg_flow_bytes_per_s`
- `avg_flow_packets_per_s`
- `protocol_tcp_ratio`
- `protocol_udp_ratio`
- `protocol_other_ratio`

## Split Strategies

Module A supports two split strategies for graph windows:

- `temporal`
  - chronological split by window order
- `stratified_attack_presence`
  - stratified split based on whether each window contains attack traffic

Both use the same train/validation/test ratio, but the second strategy helps when classes are imbalanced across time windows.

## Reproducibility Choices

Module A is designed to be reproducible:

- fixed random seed supported through `--seed`
- split ratio explicitly provided through `--split-ratio`
- graph scaling fit only on the training split
- deterministic schema normalization and feature generation
- all output artifacts recorded in a manifest file

## Output Interface

Module A writes a graph dataset directory with this structure:

```text
output_dir/
  graphs/
    train/window_*.npz
    val/window_*.npz
    test/window_*.npz
  manifest.json
  schema.json
  node_mapping.json
  scaler_node.pkl
  scaler_edge.pkl
  merged_cleaned_flows.csv   # optional
```

Each graph file contains:

- `node_features`: shape `(num_nodes, 14)`
- `node_labels`: shape `(num_nodes,)`
- `edge_index`: shape `(2, num_edges)`
- `edge_features`: shape `(num_edges, 10)`

Module B should consume:

- `graphs/train/*.npz`
- `graphs/val/*.npz`
- `graphs/test/*.npz`
- `schema.json`
- `node_mapping.json`
- scaler files if feature preprocessing needs to be inspected or reused

## Key Parameters

Important command-line parameters:

- `--input-files`
- `--input-glob`
- `--output-dir`
- `--window` or `--window-size` plus `--window-unit`
- `--split-ratio`
- `--split-strategy`
- `--node-label-rule`
- `--seed`
- `--max-rows-per-file`
- `--save-cleaned-flows`

## Current Project Runs

### Main mentor dataset: UNSW-NB15

Mentor-provided dataset:

- Kaggle dataset: `harshwardhanbhangale/unsw-complete-dataset`

Latest clean stratified UNSW build:

- output folder: `data/graph_unsw_full_10min_stratified_clean`
- rows cleaned: `2,540,047`
- benign flows: `2,218,764`
- malicious flows: `321,283`
- windows: `149`
- split: `89 train / 30 val / 30 test`
- attack-window distribution: `50 / 17 / 17`
- benign-only windows: `39 / 13 / 13`
- global nodes: `49`
- node feature dimension: `14`
- edge feature dimension: `10`

### Alternate modeling-check dataset: IoT

Updated IoT build for Andrew:

- output folder: `data/graph_10min_moduleA_stratified`
- split strategy: `stratified_attack_presence`
- ratio: `60 / 20 / 20`
- windows: `250`
- attack-window distribution: `39 / 13 / 13`

## Testing and Verification

Module A currently has unit and pipeline tests covering:

- window parsing
- split-ratio normalization
- temporal splitting
- stratified splitting
- binary label mapping
- end-to-end pipeline artifact creation
- UNSW named-column input
- raw headerless UNSW shard input

Test command:

```bash
python3 -m pytest -q tests/test_data_graph.py
```

Current expected result:

```text
8 passed
```

## How Module A Connects to the Full Workflow

Module A sits at the front of the integrated pipeline:

1. Module A cleans raw flow data and builds graph datasets
2. Module B reads graph artifacts and trains/evaluates the GCN
3. Module C visualizes graph structure or prediction results using selected graphs

Without Module A, the later modules do not have standardized graph inputs, feature definitions, labels, or split metadata.

## Presentation Summary

If presenting Module A briefly, the simplest summary is:

> Module A converts raw intrusion-detection flow records into reproducible graph-learning datasets. It cleans and normalizes the input schema, builds fixed-window directed IP graphs, creates node and edge features, assigns node labels, and exports train/validation/test graph artifacts for downstream GCN modeling and visualization.

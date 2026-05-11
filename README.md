# GNN Intrusion Detection

An end-to-end graph neural network workflow for intrusion detection in network traffic.

This project converts raw network-flow CSV data into graph datasets, trains a graph convolutional network (GCN) for node-level malicious IP detection, and renders test-set graph visualizations from model predictions.

## Team Modules

- `gcn_ids/data_graph.py` - Module A: data cleaning, graph building, split generation
- `gcn_ids/learning.py` - Module B: GCN training, validation, testing, metrics, plots
- `gcn_ids/graph_viz.py` - Module C: test-set network graph visualizations
- `main.py` - orchestration entry point for Module A -> Module B -> Module C

## Repository Structure

```text
gcn_ids/
  data_graph.py
  learning.py
  graph_viz.py
docs/
  module_a_documentation.md
  integration_report.md
scripts/
  run_module_a_windows.ps1
  run_module_a_mac.sh
tests/
  test_data_graph.py
main.py
requirements.txt
pyproject.toml
CHANGELOG.md
RELEASE_NOTES.md
```

## Supported Datasets

This repo was used with two graph-dataset builds:

1. `UNSW-NB15`
   - mentor-selected main dataset
   - full raw shard support
   - latest clean build: `data/graph_unsw_full_10min_stratified_clean`

2. `ids-custom` / IoT-style dataset
   - used for a secondary modeling check
   - latest build: `data/graph_10min_moduleA_stratified`

## Workflow Overview

1. Raw CSV flow data is cleaned and normalized.
2. Flows are grouped into fixed time windows.
3. Each window becomes one directed graph:
   - nodes = IP addresses
   - edges = observed flows
4. Node and edge features are aggregated per window.
5. Graph windows are split into train, validation, and test sets.
6. A 2-layer GCN is trained on Module A outputs.
7. Test predictions are visualized as TP/TN/FP/FN network graphs.

## Module A Summary

Module A creates graph artifacts with:

- 14 node features
- 10 edge features
- per-window `.npz` graph files
- `schema.json`
- `node_mapping.json`
- scaler files
- `manifest.json`

Module A supports both:

- `temporal` splitting
- `stratified_attack_presence` splitting

For full documentation, see:

- `docs/module_a_documentation.md`

## Main Results Included In This Workspace

### UNSW-NB15 stratified build

- cleaned flows: `2,540,047`
- windows: `149`
- split: `89 train / 30 val / 30 test`
- attack-window distribution: `50 / 17 / 17`

### IoT stratified build

- split ratio: `60 / 20 / 20`
- attack-window distribution: `39 / 13 / 13`
- downstream GCN improvement confirmed by Module B:
  - recall improved from `0.4144` to `0.6488`
  - F1 improved from `0.5529` to `0.7268`

## Installation

Create a virtual environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
```

## Quickstart

### Run the full integrated workflow

```bash
python3 main.py \
  --dataset-dir data/raw/unsw_complete \
  --graph-dir data/graph_unsw_full_10min_stratified_clean \
  --results-dir data/graph_unsw_full_10min_stratified_clean/module_b_results \
  --viz-dir data/graph_unsw_full_10min_stratified_clean/module_c_test_viz \
  --epochs 50 \
  --seed 42
```

### Run Module A only

```bash
python3 gcn_ids/data_graph.py \
  --input-files data/raw/unsw_complete \
  --input-glob "UNSW-NB15_*.csv" \
  --output-dir data/graph_unsw_full_10min_stratified_clean \
  --window 10min \
  --split-ratio 0.6,0.2,0.2 \
  --split-strategy stratified_attack_presence \
  --node-label-rule any_malicious_wins \
  --seed 42
```

### Run Module B only

```bash
python3 gcn_ids/learning.py \
  --graph-dir data/graph_unsw_full_10min_stratified_clean \
  --output-dir data/graph_unsw_full_10min_stratified_clean/module_b_results \
  --epochs 50 \
  --seed 42
```

### Run Module C only

```bash
python3 gcn_ids/graph_viz.py \
  --graph-dir data/graph_unsw_full_10min_stratified_clean \
  --predictions-csv data/graph_unsw_full_10min_stratified_clean/module_b_results/test_predictions.csv \
  --output-dir data/graph_unsw_full_10min_stratified_clean/module_c_test_viz \
  --max-graphs 16 \
  --layout-seed 42
```

## Testing

Run the current Module A test suite:

```bash
python3 -m pytest -q tests/test_data_graph.py
```

Expected result:

```text
8 passed
```

## Reproducibility Notes

- all major commands support a fixed `--seed`
- graph scalers are fit on train split only
- artifact metadata is recorded in `manifest.json`
- model metrics are recorded in `metrics.json`
- train/validation curves and confusion matrix are saved as PNGs

## Documentation

- `docs/module_a_documentation.md` - detailed Module A documentation
- `docs/module_a_handoff.md` - teammate handoff instructions
- `docs/integration_report.md` - integrated workflow summary

## Figures

Representative output figures are included in `docs/figures/`:

- `training_curves.png` - 50-epoch training and validation curves
- `confusion_matrix.png` - test-set confusion matrix
- `test_graph_example.png` - Module C test-set graph visualization

## Release Readiness

This workspace now includes:

- `CHANGELOG.md`
- `RELEASE_NOTES.md`
- `pyproject.toml`

These support the course rubric categories around documentation, release notes, and packaging readiness.

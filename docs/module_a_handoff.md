# Module A Handoff: Data Cleaning, Graph Building, and Splitting

Owner: Dhanush

This module prepares the UNSW-NB15 Kaggle dataset for the GCN modeling module. It loads the raw UNSW CSV files, cleans and normalizes the schema, builds fixed-window graph artifacts, and writes train/validation/test graph files.

## What to Send Teammates

Send these project files/folders:

- `gcn_ids/data_graph.py`
- `gcn_ids/__init__.py`
- `requirements.txt`
- `scripts/run_module_a_windows.ps1`
- `scripts/run_module_a_mac.sh`
- `docs/module_a_handoff.md`
- `tests/test_data_graph.py`

Optional, if you do not want them to regenerate Module A:

- `data/graph_unsw_full_10min/`

That artifact folder is about 412 MB and includes the graph `.npz` files, manifest, schema, node mapping, scalers, and cleaned flow CSV. If you send the artifact folder, Andrew can start Module B directly from `data/graph_unsw_full_10min/graphs`.

Do not send Mac cache files such as `.DS_Store`, `.pytest_cache`, `.matplotlib-cache`, or your personal Kaggle cache path.

## Dataset Placement

Each teammate should download the mentor's Kaggle dataset:

`https://www.kaggle.com/datasets/harshwardhanbhangale/unsw-complete-dataset`

After extracting it, place these files here:

```text
data/raw/unsw_complete/
  UNSW-NB15_1.csv
  UNSW-NB15_2.csv
  UNSW-NB15_3.csv
  UNSW-NB15_4.csv
  NUSW-NB15_features.csv
```

Only the four `UNSW-NB15_*.csv` shard files are required by Module A.

## Windows Setup

Open PowerShell in the project root folder:

```powershell
cd path\to\Intrusion_Detection
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Run Module A on Windows

From the project root:

```powershell
.\scripts\run_module_a_windows.ps1
```

Equivalent manual command:

```powershell
py gcn_ids\data_graph.py `
  --input-files data\raw\unsw_complete `
  --input-glob "UNSW-NB15_*.csv" `
  --output-dir data\graph_unsw_full_10min `
  --window 10min `
  --split-ratio "0.6,0.2,0.2" `
  --node-label-rule any_malicious_wins `
  --save-cleaned-flows `
  --seed 42
```

## Run Module A on Mac/Linux

```bash
bash scripts/run_module_a_mac.sh
```

Equivalent manual command:

```bash
python3 gcn_ids/data_graph.py \
  --input-files data/raw/unsw_complete \
  --input-glob "UNSW-NB15_*.csv" \
  --output-dir data/graph_unsw_full_10min \
  --window 10min \
  --split-ratio "0.6,0.2,0.2" \
  --node-label-rule any_malicious_wins \
  --save-cleaned-flows \
  --seed 42
```

## Module A Output Interface

Module A writes:

```text
data/graph_unsw_full_10min/
  graphs/
    train/window_*.npz
    val/window_*.npz
    test/window_*.npz
  manifest.json
  schema.json
  node_mapping.json
  scaler_node.pkl
  scaler_edge.pkl
  merged_cleaned_flows.csv
```

Module B should read:

- `data/graph_unsw_full_10min/graphs/train/*.npz`
- `data/graph_unsw_full_10min/graphs/val/*.npz`
- `data/graph_unsw_full_10min/graphs/test/*.npz`
- `data/graph_unsw_full_10min/schema.json`
- `data/graph_unsw_full_10min/node_mapping.json`

Each `.npz` graph contains:

- `node_features`: shape `(num_nodes, 14)`
- `node_labels`: shape `(num_nodes,)`
- `edge_index`: shape `(2, num_edges)`
- `edge_features`: shape `(num_edges, 10)`

## Current Full UNSW Run Summary

Using the full UNSW dataset:

- Rows cleaned: `2,540,047`
- Benign flows: `2,218,764`
- Malicious flows: `321,283`
- Window size: `10 minutes`
- Split: chronological `60% train / 20% validation / 20% test`
- Train windows: `89`
- Validation windows: `30`
- Test windows: `30`
- Global nodes: `49`
- Node feature dimension: `14`
- Edge feature dimension: `10`

## Reproducibility Notes

- Seed: `42`
- Split is chronological by time window, not random.
- Scalers are fit only on the train split and reused for validation/test.
- Labels use `any_malicious_wins`: a node is malicious in a window if it appears in at least one malicious flow in that window.
- Raw UNSW files are headerless; `data_graph.py` applies the official 49-column schema automatically for `UNSW-NB15_*.csv`.

## Quick Verification

Run:

```powershell
py -m pytest -q
```

Expected result:

```text
7 passed
```

# Integrated UNSW GCN IDS Workflow

## End-to-end module connection

The integrated workflow follows the mentor outline:

1. **Module A - Data cleaning, graph building, splitting**
   - Entry point: `gcn_ids/data_graph.py`
   - Input: raw UNSW-NB15 shard CSVs, `UNSW-NB15_1.csv` through `UNSW-NB15_4.csv`
   - Output: windowed graph artifacts in `data/graph_unsw_full_10min/`
   - Interface to Module B:
     - `graphs/train/window_*.npz`
     - `graphs/val/window_*.npz`
     - `graphs/test/window_*.npz`
     - `schema.json`
     - `node_mapping.json`
     - `manifest.json`

2. **Module B - GCN modeling**
   - Entry point: `gcn_ids/learning.py`
   - Input: Module A `.npz` graph files
   - Model: two-layer GCN using normalized adjacency with self-loops
   - Output: `data/graph_unsw_full_10min/module_b_results/`
   - Artifacts:
     - `gcn_model.pt`
     - `training_history.csv`
     - `training_curves.png`
     - `confusion_matrix.png`
     - `test_predictions.csv`
     - `metrics.json`

3. **Module C - Network graph visualizations**
   - Entry point: `gcn_ids/graph_viz.py`
   - Input: Module A test graphs plus Module B `test_predictions.csv`
   - Output: `data/graph_unsw_full_10min/module_c_test_viz/`
   - Scope: test-set graphs only, with node colors showing TP/TN/FP/FN outcomes.

`main.py` provides a single orchestration command for Module A -> Module B -> Module C.

## Reproducibility and RSE choices

- Fixed seed: `42`
- Chronological window split: `60% train / 20% validation / 20% test`
- Window size: `10 minutes`
- Scalers fit on train split only and reused for validation/test
- Stable module interfaces through `.npz`, JSON manifests, and CSV prediction files
- Parameter and artifact paths recorded in `manifest.json` and `metrics.json`

## Tuning adjustments applied

- Updated Module A to read headerless raw UNSW-NB15 shard files with the official 49-column schema.
- Added UNSW aliases for source IP, destination IP, timestamp, protocol, duration, packets, bytes, and labels.
- Used the full UNSW dataset with no row cap.
- Added class-weighted cross entropy for class imbalance.
- Used a 2-layer GCN with:
  - hidden dimension: `32`
  - dropout: `0.25`
  - learning rate: `0.01`
  - weight decay: `0.0005`
  - epochs: `50`
- Module C now uses test-set predictions only.

## Preliminary integrated results

Module A full UNSW graph build:

- Cleaned flows: `2,540,047`
- Benign flows: `2,218,764`
- Malicious flows: `321,283`
- Global nodes: `49`
- Graph windows: `149`
- Split: `89 train`, `30 validation`, `30 test`
- Node features: `14`
- Edge features: `10`

Module B 50-epoch GCN:

- Best validation accuracy: `0.9973`
- Final train accuracy: `0.9979`
- Final validation accuracy: `0.9959`
- Test accuracy: `0.9966`
- Malicious precision: `0.9882`
- Malicious recall: `1.0000`
- Malicious F1: `0.9941`

Test confusion matrix:

| Actual \\ Predicted | Benign | Malicious |
| --- | ---: | ---: |
| Benign | 1045 | 5 |
| Malicious | 0 | 420 |

## Key figures

- Training and validation curves: `data/graph_unsw_full_10min/module_b_results/training_curves.png`
- Confusion matrix: `data/graph_unsw_full_10min/module_b_results/confusion_matrix.png`
- Test-set graph visualizations: `data/graph_unsw_full_10min/module_c_test_viz/`

## Commands used

Build full UNSW graphs:

```bash
python3 gcn_ids/data_graph.py \
  --input-files /Users/dhanushmarreddi/.cache/kagglehub/datasets/harshwardhanbhangale/unsw-complete-dataset/versions/1 \
  --input-glob 'UNSW-NB15_*.csv' \
  --output-dir data/graph_unsw_full_10min \
  --window 10min \
  --split-ratio 0.6,0.2,0.2 \
  --node-label-rule any_malicious_wins \
  --save-cleaned-flows
```

Train/evaluate GCN:

```bash
python3 gcn_ids/learning.py \
  --graph-dir data/graph_unsw_full_10min \
  --output-dir data/graph_unsw_full_10min/module_b_results \
  --epochs 50 \
  --seed 42
```

Render test-set prediction visualizations:

```bash
python3 gcn_ids/graph_viz.py \
  --graph-dir data/graph_unsw_full_10min \
  --predictions-csv data/graph_unsw_full_10min/module_b_results/test_predictions.csv \
  --output-dir data/graph_unsw_full_10min/module_c_test_viz \
  --max-graphs 16
```

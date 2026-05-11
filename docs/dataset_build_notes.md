# Merged Flow Dataset (for Node-Level IDS GCN)

## Outputs
- `data/processed/merged_flow_unscaled.csv`
- `data/processed/merged_flow_scaled.csv`
- `data/processed/merged_flow_manifest.json`

## Source datasets merged
1. Benign-Monday-WorkingHours.pcap_ISCX.csv
2. Botnet-Friday-WorkingHours-Morning.pcap_ISCX.csv
3. Bruteforce-Tuesday-WorkingHours.pcap_ISCX.csv
4. DDoS-Friday-WorkingHours-Afternoon.pcap_ISCX.csv
5. DoS-Wednesday-WorkingHours.pcap_ISCX.csv
6. Infiltration-Thursday-WorkingHours-Afternoon.pcap_ISCX.csv
7. Portscan-Friday-WorkingHours-Afternoon.pcap_ISCX.csv
8. WebAttacks-Thursday-WorkingHours-Morning.pcap_ISCX.csv

## New mentor dataset: UNSW-NB15
The new Kaggle dataset link is:
- `https://www.kaggle.com/datasets/harshwardhanbhangale/unsw-complete-dataset`

Download/extract the CSV files into:
- `data/raw/unsw_complete/`

The graph pipeline needs source IP, destination IP, timestamp, and label columns. UNSW original-flow CSVs commonly use `srcip`, `dstip`, `stime`, `proto`, `dur`, `spkts`, `dpkts`, `sbytes`, `dbytes`, `rate`, and `label`; these aliases are now supported by `gcn_ids/data_graph.py`.

If the downloaded Kaggle train/test files only contain feature-table columns and do not include `srcip`, `dstip`, and `stime`, use the original UNSW-NB15 shard CSVs for the GCN graph build.

## Common feature schema used
Identity columns:
- `Source IP`
- `Destination IP`
- `Timestamp`
- `Label`
- `binary_label` (`0=BENIGN`, `1=malicious`)

Model-ready numeric features:
- `Protocol`
- `Flow Duration`
- `Total Fwd Packets`
- `Total Backward Packets`
- `Total Length of Fwd Packets`
- `Total Length of Bwd Packets`
- `Flow Bytes/s`
- `Flow Packets/s`
- `Fwd IAT Mean`
- `Bwd IAT Mean`
- `Average Packet Size`

## Cleaning and scaling
- Column names normalized to handle spacing differences across files.
- Invalid numeric values (`NaN`, `Inf`, empty) are imputed with feature median.
- Scaled version uses z-score normalization: `(x - mean) / std`.

## Current run configuration
- Per-file reservoir sample: `20,000`
- Total rows: `160,000`
- Binary label distribution: `129,415 benign` / `30,585 malicious`
- Script: `scripts/build_merged_dataset.py`

## Re-run command
```bash
python3 scripts/build_merged_dataset.py --per-file-limit 20000
```

## Multi-dataset command
```bash
python3 scripts/build_merged_dataset.py \
  --input-dirs data/raw/unsw_complete data/raw/cse_cic_ids2018_samples \
  --per-file-limit 20000
```

## For a larger dataset
Increase `--per-file-limit`.
Examples:
- `--per-file-limit 50000` (about 400k rows)
- `--per-file-limit 100000` (about 800k rows, slower/larger)

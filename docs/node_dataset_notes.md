# Node-Window Dataset Package (Tomorrow Ready)

## Generated files
- `data/processed/node_window_dataset.csv`
- `data/processed/node_window_train.csv`
- `data/processed/node_window_val.csv`
- `data/processed/node_window_test.csv`
- `data/processed/node_window_manifest.json`

## How it was built
Input:
- `data/processed/merged_flow_unscaled.csv`

Windowing:
- Fixed non-overlapping `5-minute` windows.

Node aggregation:
- Each flow contributes to both endpoints (`Source IP`, `Destination IP`).
- Per node per window, aggregate counts/rates/bytes/packets/protocol mixes.

Node label rule:
- `any_malicious_wins` (label `1` if node appears in at least one malicious flow in that window; else `0`).

Time split:
- Earliest 70% windows -> `train`
- Next 15% windows -> `val`
- Latest 15% windows -> `test`

## Current output stats
- Input flows used: `135,000`
- Node-window rows: `39,474`
- Windows: `515`
- Labels: `39,202 benign` / `272 malicious`

## Re-run
```bash
python3 scripts/build_node_window_dataset.py --window-minutes 5
```

## Change label rule (optional)
```bash
python3 scripts/build_node_window_dataset.py --window-minutes 5 --label-rule majority
```

## Windows note
Run commands from the project root folder. On Windows PowerShell, replace `python3` with `py` if needed:

```powershell
py scripts/build_node_window_dataset.py --window-minutes 5
```

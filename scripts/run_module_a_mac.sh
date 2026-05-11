#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="${1:-data/raw/unsw_complete}"
OUTPUT_DIR="${2:-data/graph_unsw_full_10min}"
WINDOW="${3:-10min}"

echo "Module A: UNSW data cleaning, graph building, and splitting"
echo "Input directory: ${INPUT_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Window size: ${WINDOW}"

if [ ! -d "${INPUT_DIR}" ]; then
  echo "Input directory not found: ${INPUT_DIR}. Put UNSW-NB15_1.csv through UNSW-NB15_4.csv there first." >&2
  exit 1
fi

python3 -m pip install -r requirements.txt

python3 gcn_ids/data_graph.py \
  --input-files "${INPUT_DIR}" \
  --input-glob "UNSW-NB15_*.csv" \
  --output-dir "${OUTPUT_DIR}" \
  --window "${WINDOW}" \
  --split-ratio "0.6,0.2,0.2" \
  --node-label-rule any_malicious_wins \
  --save-cleaned-flows \
  --seed 42

echo "Done. Module A graph artifacts are in ${OUTPUT_DIR}"

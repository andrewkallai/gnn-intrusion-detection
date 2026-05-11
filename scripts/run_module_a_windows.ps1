param(
    [string]$InputDir = "data\raw\unsw_complete",
    [string]$OutputDir = "data\graph_unsw_full_10min",
    [string]$Window = "10min"
)

$ErrorActionPreference = "Stop"

Write-Host "Module A: UNSW data cleaning, graph building, and splitting"
Write-Host "Input directory: $InputDir"
Write-Host "Output directory: $OutputDir"
Write-Host "Window size: $Window"

if (-not (Test-Path $InputDir)) {
    Write-Error "Input directory not found: $InputDir. Put UNSW-NB15_1.csv through UNSW-NB15_4.csv there first."
}

py -m pip install -r requirements.txt

py gcn_ids\data_graph.py `
    --input-files $InputDir `
    --input-glob "UNSW-NB15_*.csv" `
    --output-dir $OutputDir `
    --window $Window `
    --split-ratio "0.6,0.2,0.2" `
    --node-label-rule any_malicious_wins `
    --save-cleaned-flows `
    --seed 42

Write-Host "Done. Module A graph artifacts are in $OutputDir"

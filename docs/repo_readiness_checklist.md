# Repo Readiness Checklist

This note summarizes what is now ready to show in the project repository and what still needs confirmation before the final presentation or final GitHub submission.

## Ready Now

- top-level project README describing the full Module A -> Module B -> Module C workflow
- Module A documentation in `docs/module_a_documentation.md`
- integrated workflow summary in `docs/integration_report.md`
- `requirements.txt` for reproducible Python installation
- `pyproject.toml` for packaging metadata
- `CHANGELOG.md`
- `RELEASE_NOTES.md`
- Module A test file `tests/test_data_graph.py`
- integrated entry point `main.py`

## Evidence To Show

Repository-level evidence:

- `README.md`
- `CHANGELOG.md`
- `RELEASE_NOTES.md`
- `pyproject.toml`
- `requirements.txt`

Module-level evidence:

- `gcn_ids/data_graph.py`
- `gcn_ids/learning.py`
- `gcn_ids/graph_viz.py`
- `docs/module_a_documentation.md`
- `tests/test_data_graph.py`

Artifact-level evidence:

- `data/graph_unsw_full_10min_stratified_clean/manifest.json`
- `data/graph_10min_moduleA_stratified/manifest.json`
- Module B outputs in `data/graph_unsw_full_10min/module_b_results/`
- Module C outputs in `data/graph_unsw_full_10min/module_c_test_viz/`

## Questions To Confirm With Andrew

1. Is the final GitHub repo using `graph_viz.py` or an older `baselines_viz.py` file for Module C?
2. Are Module B and Module C tests being added, or are we presenting only Module A tests?
3. Is the GitHub root README being replaced with the updated project README?
4. Will the final repo include a release tag on GitHub before the final deadline?
5. Are the setup commands in the presentation being updated to match the actual repo paths?

## Presentation Risk Flags

- do not claim CUDA, DDP, RMM, or NeighborLoader unless the final repo clearly implements them
- do not describe traditional ML baseline comparison if the scope is now GCN-only
- make sure slide commands match the actual repo paths
- show tests, documentation, and reproducibility evidence, not only accuracy

## Best Show-and-Tell Order

If showing the repo live, open these in order:

1. `README.md`
2. `docs/module_a_documentation.md`
3. `main.py`
4. `gcn_ids/data_graph.py`
5. `tests/test_data_graph.py`
6. one `manifest.json`
7. one result figure such as the confusion matrix or training curves

# Legacy and diagnostic scripts

These scripts were used during the project but are **not** part of the final
pipeline. They are kept for transparency, because some of the intermediate
results mentioned in the thesis were produced with them.

| Script | Status |
|---|---|
| `sample_graphs.py` | First version of the 250-node sampling. **Contains a bug**: for every graph with more than 250 nodes it replaces the node features with a constant (`torch.ones`). With the reaction-ID dataset this erased the reaction IDs of 7,362 of the 7,644 graphs. The "constant node features" configuration in the results comes from this file. Replaced by `scripts/sample_graphs_v2.py`. |
| `check_features.py` | Diagnostic that detected the bug above: counts the graphs whose node features are all identical after sampling. |
| `classify_embeddings.py` | First classification script for Graphormer embeddings (scaler fitted on the whole dataset). Replaced by `scripts/evaluate_all.py`. |
| `grakel_classification.py` | First WL kernel run (node degree labels). Replaced by `scripts/evaluate_all.py`. |
| `grakel_reactions.py` | WL kernel with reaction-ID labels, standalone version. Replaced by `scripts/evaluate_all.py`. |
| `fix_confusion_matrix.py` | Early helper from the preliminary archaea experiment (cross-validated confusion matrix). |

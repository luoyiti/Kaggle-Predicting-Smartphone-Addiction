# MLE workflow (Kaggle-calibrated)

This folder is the mle-workflow skill applied to playground-series-s6e8.

| Skill artifact | File |
| --- | --- |
| Iteration Compact | [iteration_compact.md](iteration_compact.md) |
| Prediction contract | [prediction_contract.md](prediction_contract.md) |
| Data contract | [data_contract.md](data_contract.md) |
| Decision Brain | [decision_brain.md](decision_brain.md) |
| Observation ledger | [observation_ledger.md](observation_ledger.md) |
| Review checklist | [review_checklist.md](review_checklist.md) |
| Path coverage | [path_coverage.md](path_coverage.md) |
| Machine-readable schema | `s6e8/contracts.py` |
| Human HTML report | `reports/mle_modeling_report.html` |

Production lanes mapped here: batch Kernel submission, StratifiedKFold OOF, experiment ledger. Not mapped: Kubernetes, MLflow Server, feature store, online serving, canary traffic.

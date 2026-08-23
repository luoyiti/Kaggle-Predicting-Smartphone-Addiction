# Review checklist — S6E8 (mle-workflow, Kaggle-calibrated)

- [x] Prediction contract is explicit and testable (`docs/mle/prediction_contract.md`, submission schema tests)
- [x] Data contract defines grain, label timing, split, snapshot (`docs/mle/data_contract.md`, `s6e8/contracts.py`)
- [x] Leakage risks checked vs prediction-time availability (no future joins; TE is fold-safe; `id` dropped)
- [x] Training is reproducible from YAML + seed + optional git SHA (`scripts/train.py`)
- [x] Metrics compare against baseline `lgbm_nocat` (`scripts/promote.py`)
- [x] Slice metrics for gender / stress / academic / missingness (`s6e8/metrics.py`)
- [x] Promotion gates automated and fail closed (`s6e8/promotion.py`)
- [x] Train and “serving” transforms are the same function (`s6e8.features.transform`)
- [x] Artifact bundle: OOF/test npy+parquet, metrics.json, submission.csv
- [ ] Online serving timeout/fallback — **N/A** (batch Kernel). Rollback = old experiment name
- [x] Monitoring = experiment ledger (`experiments/LOG.md`, `docs/mle/observation_ledger.md`), not a prod dashboard
- [x] Secrets excluded (`.gitignore`: data/raw, oof, submissions, .env, kaggle.json)

Serving-path items from the generic skill (feature-store freshness, p95 latency dashboards, canary traffic) are **intentionally out of scope** for this competition repo.

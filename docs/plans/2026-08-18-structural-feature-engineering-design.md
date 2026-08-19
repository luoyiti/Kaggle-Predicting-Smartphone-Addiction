# Structural feature engineering design

Date: 2026-08-18

Status: approved direction; implementation pending

Pull request: #11 (`agent/histgb-nocat-long-v1`)

## Context

The current best fixed five-fold OOF ROC AUC is `0.9640872438`, produced by a
60% `lgbm_nocat` / 40% `histgb_nocat_long_v1` blend. The next iteration should
test representations that match the data-generating structure instead of adding
more generic ratios.

Public evidence suggests three useful facts:

1. Apparently continuous values are sampled on repeated grids. Copying their
   canonical exact values into CatBoost categorical columns lets CatBoost learn
   ordered target statistics for those grid points.
2. `daily_screen_time` contains a budget constraint relative to social, gaming,
   and work/study screen time. The unallocated remainder is informative.
3. The public 7,500-row source dataset is distributionally related but is not a
   safe direct extension of the competition training set.

Public notebook scores are treated only as research priors because they use
different folds, seeds, and model budgets. No public prediction file will be
downloaded, blended, or committed.

## Goals

- Preserve the existing config-driven runner and fixed
  `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` protocol.
- Isolate the value of exact categorical copies, screen-budget features, and
  external reference features with controlled ablations.
- Use CatBoost's native categorical handling rather than precomputed target
  encoding for the primary exact-value experiment.
- Save OOF predictions, test predictions, per-fold metrics, configuration, and
  provenance for every formal experiment.
- Evaluate single-model and fold-stable blend gains against the current
  LightGBM, HistGB, and 60/40 blend baselines.
- Run full-data formal experiments in Kaggle cloud compute and keep leaderboard
  submission manual.
- Add a fixed-five-fold Lookup-Transformer only if the tree experiments plateau.

## Non-goals

- No automatic Kaggle submission.
- No training-row concatenation with the 7,500-row source data.
- No use of another competitor's prediction or submission CSV.
- No broad search over generic ratios, target-encoding smoothers, or blend
  weights on the same full OOF vector.
- No comparison of this repository's five-fold AUC with public notebook AUC as
  if they were measured under the same protocol.

## Configuration surface

Feature behavior remains fully controlled by YAML. The runner will accept
blocks with the following shape; disabled or absent blocks preserve current
behavior.

```yaml
features:
  exact_categorical:
    enabled: true
    columns: auto_numeric
    suffix: __exact
    missing_token: __MISSING__
    decimal_places:
      age: 0
      daily_screen_time: 2
      sleep_hours: 2
      weekend_screen_time: 2
  screen_budget:
    enabled: true
  decimal_lattice:
    enabled: false
    columns: []
external_reference:
  enabled: false
  path: /kaggle/input/.../Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv
  mode: distribution
  remove_query_overlaps: true
```

`external_reference` is top-level because it controls an attached data source
and its provenance as well as the derived feature block. Row-local feature
families remain nested under `features`.

Formal experiment configs must repeat the fixed validation seed and folds. A
config validator will reject formal configs that change those values.

## Exact-value categorical copies

Raw numeric columns remain available as numeric features. For each configured
column, the feature pipeline also creates a categorical copy.

Canonicalization rules:

- Parse through the existing numeric loader.
- Round to the configured source-grid precision, avoiding floating-point string
  artifacts such as `3.1000000000000001`.
- Serialize deterministically without scientific notation.
- Map missing values to the explicit `__MISSING__` token.
- Prefix values with the source column name so categories from different fields
  cannot collide.

The generated columns are appended to CatBoost's `cat_features`. They are not
OOF target encoded before training. Within each outer fold, CatBoost receives
only the fold-training labels and computes its categorical statistics internally.
The primary configuration uses native CTRs, a low `one_hot_max_size`, and a
bounded `max_ctr_complexity`; it does not force GPU `boosting_type=Ordered`,
which is a separate algorithm choice from ordered categorical statistics.

## Screen-budget features

The budget block deliberately separates complete-case and observed-component
semantics. Missing components are never silently treated as known zero in the
complete-case features.

Let:

- `D` = daily screen time,
- `S`, `G`, `W` = social, gaming, and work/study screen time,
- `C = S + G + W`,
- `K` = number of observed components,
- `E` = weekend screen time,
- `H` = sleep hours.

The block creates:

- `screen_component_sum_complete = C`, only when all three components exist;
- `screen_component_sum_observed`, the sum of observed components;
- `screen_component_count = K`;
- `screen_remainder_complete = D - C`, only for complete rows;
- `screen_remainder_observed = D - observed_sum`, accompanied by `K`;
- `screen_component_share_complete = C / D`;
- `screen_remainder_share_complete = (D - C) / D`;
- `weekend_minus_component_sum = E - C`;
- `weekend_minus_remainder = E - (D - C)`;
- `awake_non_screen_hours = 24 - H - D`;
- boundary and violation indicators computed with a small numeric tolerance.

Division uses the shared safe-divide helper and emits missing values for invalid
denominators. These features are one coherent structural block and are tested as
a block rather than being multiplied into a large ratio family.

## Decimal-lattice ablation

A small secondary ablation may expose the first decimal grid digit for the
hour-valued fields:

- fractional part;
- first decimal digit.

It is disabled in the main budget experiment and enabled only in its own config.
Second decimal digits and generic modulo families are excluded unless the first
ablation produces a repeatable five-fold gain.

## External reference data

The source dataset is auxiliary reference data, never appended to competition
training rows. Before fitting reference transforms:

1. normalize the twelve shared predictor columns;
2. deduplicate source rows by their full predictor-row hash;
3. remove every source row that exactly matches any competition train or test
   row;
4. record raw rows, unique rows, removed overlaps, retained rows, file hash, and
   source URL in experiment metadata.

Two modes are evaluated independently:

### Distribution mode

Target-free features only:

- empirical CDF position for selected numeric variables;
- signed and absolute distance from source medians and robust scales;
- optional source-grid frequency, normalized only by retained source rows.

This mode is fitted once from the external source and is fold invariant. It does
not inspect the competition target.

### Label-aware mode

This is a separate, clearly labelled experiment that may additionally use the
source dataset's `addicted_label` to create smoothed reference summaries such as
class-conditional CDF gaps and coarse source-bin target means.

It is not considered competition-target leakage because it uses an independently
published label, but it is external supervision and therefore receives an
`external_supervision: true` marker. Exact train and test overlaps are removed to
prevent direct label lookup. It is not promoted unless the gain is stable across
folds and exceeds the target-free reference alternative.

## Experiment sequence and promotion rules

Formal experiments run in this order:

1. `catboost_numeric_v1` — numeric/canonical-category control without exact
   numeric copies or budget features;
2. `catboost_exactcat_v1` — add exact categorical copies only;
3. `catboost_exactcat_budget_v1` — add the structural budget block;
4. `catboost_exactcat_budget_noage_v1` — one preregistered targeted ablation
   that retains numeric age but excludes only `age__exact`, motivated by the
   public negative age-category ablation;
5. an isolated first-decimal lattice child of the promoted internal parent;
6. a target-free source-reference child of the promoted internal parent;
7. a separately marked label-aware source-reference child — external
   supervision, only if distribution mode is sound.

No other per-column backward selection is performed. If the no-age child is
promoted, downstream lattice/reference configs must inherit it; otherwise the
pre-existing `catboost_exactcat_budget_*` children remain the controlled path.

Each stage reports:

- overall OOF ROC AUC;
- five fold AUCs, mean, standard deviation, minimum, and maximum;
- delta by fold against its immediate parent and against `lgbm_nocat`;
- runtime, best iteration by fold, feature counts, and categorical feature count;
- OOF Pearson and rank correlation with existing model predictions;
- fixed-weight and leave-one-fold-out blend results against current baselines.

A stage is promoted when it improves overall OOF, does not rely on one exceptional
fold, and either improves at least three folds or provides a positive honest blend
gain. Tiny full-OOF blend-weight optima are not treated as evidence: blend weights
are selected on four folds and evaluated on the held-out fold, then aggregated for
test prediction.

## Lookup-Transformer phase gate

Lookup-Transformer implementation begins only after the CatBoost and target-free
reference ablations finish. It is triggered when exact-grid signal is established
and two successive successful target-free tree candidates improve the best-so-far
frontier by less than `0.00015`, or when a slightly weaker tree improves honest
LOFO by at least `0.00010` with at least three positive folds. Label-aware source
features do not count toward this tree frontier.

The implementation will use the same five outer folds and expose exact lookup
embeddings, smooth periodic numeric embeddings, budget tokens, random feature
masking, EMA, and all optimization parameters through YAML. Vocabulary and
numeric preprocessing use predictors only; no validation target is visible outside
the fold trainer. A tiny CPU smoke test will verify shapes and fold isolation, while
formal training runs on Kaggle GPU.

## Artifacts and provenance

Every experiment directory must contain:

- `oof_predictions.parquet` with row id, target, prediction, and fold id;
- `test_predictions.parquet` with row id and prediction;
- `metrics.json` with overall and per-fold metrics;
- `experiment.json` with resolved config, git commit, data provenance, runtime,
  feature list, and dependency versions;
- `submission.csv` as a local artifact only, never submitted automatically.

Failed or interrupted cloud runs must not be entered into the score table as
completed experiments. Metrics copied into README or the experiment log must be
traceable to committed `metrics.json`/experiment files and a successful cloud run.

## Kaggle cloud execution

The existing GitHub Actions to Kaggle kernel path remains the formal training
route. Backend-specific optional dependencies are installed only inside the cloud
runner when required. CatBoost stages are run sequentially so an invalid early
ablation can stop later dependent stages and avoid wasted compute.

The 7,500-row source dataset is attached to the Kaggle kernel as an input dataset;
it is not committed to Git. Cloud artifacts are downloaded for comparison, while
large OOF and test files remain release/workflow artifacts rather than normal Git
history.

No workflow or script may call Kaggle's competition submit command.

## Testing and acceptance

Implementation follows tests-first development. Required coverage includes:

- deterministic exact-category canonicalization and missing tokens;
- train/test schema parity and categorical-column routing into CatBoost;
- complete/observed budget semantics, zero denominators, and missing inputs;
- external row hashing, deduplication, overlap removal, and target-free mode;
- explicit provenance for label-aware reference mode;
- fixed five-fold config validation and fold ids in OOF artifacts;
- backend-specific Kaggle dependency handling;
- tiny end-to-end CatBoost smoke training when CatBoost is available;
- Lookup-Transformer tensor/shape/fold-isolation tests if its phase is triggered.

Before updating the Draft PR, the full unit-test suite, config validation, Python
compilation, experiment JSON parsing, and `git diff --check` must pass. The PR body
will include the experiment matrix, per-fold table, blend analysis, cloud run links,
limitations, and an explicit statement that no leaderboard submission was made.

## GitHub delivery

Work continues on the existing Draft PR #11 branch so the documented `0.964087`
baseline remains in the same review chain. Commits are kept separable by concern:
design, tested feature infrastructure, formal configs, cloud-result records, and
documentation. The PR remains Draft until cloud metrics and independent review are
complete.

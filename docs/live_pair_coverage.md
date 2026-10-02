# DKPS dataset–model–budget coverage

The [pair-coverage card](../jhu_ta1/cards/jhu_run_predict_pair_coverage_kwdagger.yaml) tests:

> For more than 95% of (dataset, model, query budget) combinations, DKPS has lower
> expected absolute score-estimation error than the same-budget sample mean.

For each combination, the evaluator repeatedly samples queries, computes response
embeddings, and fits eight-dimensional DKPS with OLS. It excludes the target's
provider family from the reference models and uses all remaining references by
default. The score estimate is `0.8 * sample_mean + 0.2 * clipped_dkps_prediction`.
Both estimators receive the same sampled queries and target labels. Only reference
models supply full benchmark scores to the predictor; the target's full score is
used to measure error afterward.

Absolute errors are averaged **over replicates within each combination**. Coverage
is the fraction of combinations where DKPS's average error is strictly smaller.
Each combination has one vote; ties do not improve coverage. No raw MAEs are
averaged across datasets or budgets. The card passes when pooled coverage exceeds
95%.

## Inputs

Supply a directory of HELM runs and a [dataset manifest](../jhu_ta1/magnet/pair_coverage_datasets.example.json):

```json
{
  "datasets": [
    {"dataset": "med_qa", "metric": "quasi_exact_match"},
    {"dataset": "legalbench:subset=abercrombie", "metric": "quasi_exact_match"},
    {"dataset": "legalbench:subset=corporate_lobbying", "metric": "quasi_exact_match"}
  ]
}
```

Each run contains `scenario_state.json` and either `per_instance_stats.json` or
`display_predictions.json`. These are saved language-model responses and item
scores: the evaluator computes DKPS predictions anew but does not call the target
language models again. Dataset names select matching run-directory prefixes.

Optional fields in each dataset entry are `runs` (explicit paths relative to the
suite), `split`, and `train_trial_index` (default 0). The manifest can also declare
`target_models` and a `model_families` mapping. Otherwise, targets are all models
common to the datasets, and family is the model ID's provider prefix before `/`.

Items are aligned by their intersection across each dataset's model runs. Input
text and references must agree. The benchmark truth is the item-score mean on
that aligned pool. Scores must be finite and in [0, 1]; each item must have one
completion. Duplicate responses, missing scores, or insufficient reference models
stop the evaluation. Other HELM response layouts need a dedicated adapter.

## Run with MAGNET and Docker

Use the [standard containerization guide](containerized_evaluation.md) to build
`Dockerfile` and run the card through MAGNET's materialize-and-evaluate pipeline.
For an already assembled suite, the helper skips materialization:

```bash
python scripts/run_pair_coverage.py \
  --helm_suite_path /absolute/path/to/helm/runs/v1.0.0 \
  --dataset_manifest "$PWD/jhu_ta1/magnet/pair_coverage_datasets.example.json" \
  --out_dpath "$PWD/results/pair-coverage"
```

The default image is `jhu-magnet-dkps-gpu`, with DKPS installed inside it. Optional
`--dkps_root /path/to/dkps` overrides that installed library for development.
The default is 1,024 replicates at budgets 1, 2, 4, and 8, in one evidence row.
Use `--num_replicates 32` for a smaller run or `--queries 1 2` to select budgets.
The helper assigns a unique run ID to force new fits without changing the seed.

For text embeddings, the image includes sentence-transformers. Supply local
weights with `--embed_model /path/to/weights --embedding_mount /path/to/weights`,
or let the embedding provider download them. A completed `FALSIFIED` verdict is
a valid experimental outcome. The helper prints its log path and verdict.

## Run directly and inspect results

With DKPS and the scientific dependencies installed:

```bash
PYTHONPATH="$PWD:/absolute/path/to/dkps" python -m jhu_ta1.magnet.pair_coverage \
  --helm_suite_path /absolute/path/to/helm/runs/v1.0.0 \
  --dataset_manifest jhu_ta1/magnet/pair_coverage_datasets.example.json \
  --query_budgets 1,2,4,8 --num_replicates 128 \
  --out_fpath /absolute/path/to/results/pair_coverage.json
```

This computes evidence; the MAGNET runner also executes the card's assertions.
The JSON contains pooled metrics, `per_setting` errors and gains, `per_budget`
coverage, and input provenance. The neighboring `.trials.jsonl` records sampled
query IDs, reference models, predictions, sample scores, and evaluation truth.
The worker also accepts `--base_seed` and `--reference_models` (0 means all eligible
references; a positive count samples references per target and replicate).

Run the focused tests in an environment with DKPS and MAGNET installed:

```bash
python -m unittest discover -s tests -v
```

For the historical 18-dataset pools and cached Google response embeddings, see
[the full-run preparation and execution guide](full_pair_coverage.md).

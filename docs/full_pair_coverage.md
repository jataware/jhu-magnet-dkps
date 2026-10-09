# Run all 18 datasets with cached Google embeddings

This runs the same pair-coverage card with fresh query samples, DKPS spaces, OLS
fits, and score predictions. Only response embeddings are reused. No saved DKPS
predictions are read, and no Google API key or embedding API calls are required in the
worker.

The [full manifest](../jhu_ta1/magnet/pair_coverage_datasets.full.json) contains
seven MATH subjects, five WMT14 language pairs, five LegalBench subsets, and MedQA.
MATH and WMT use cached `google / gemini-embedding-001` vectors. LegalBench and
MedQA compute one-hot embeddings during the run.

## Prepare the inputs once

Use an environment with NumPy, pandas, and the local DKPS checkout available.
From this repository, set `DKPS_CHECKOUT` to the checkout containing `dkps/` and
`examples/helm/`:

```bash
export DKPS_CHECKOUT=/absolute/path/to/dkps
export PAIR_COVERAGE_INPUTS=/absolute/path/to/prepared-pair-coverage

PYTHONPATH="$PWD:$DKPS_CHECKOUT" python scripts/prepare_pair_coverage_18.py \
  --helm_runs_root "$DKPS_CHECKOUT/examples/helm/crfm-helm-public/lite/benchmark_output/runs" \
  --tables_dir "$DKPS_CHECKOUT/examples/helm/data" \
  --google_cache_dir "$DKPS_CHECKOUT/examples/helm/.cache/embed/google" \
  --out_dpath "$PAIR_COVERAGE_INPUTS"
```

The output must be a new directory. Preparation reads the existing extracted
item tables and checks their selected responses, references, and native scores
against the source HELM runs. It creates:

- `suite/`: HELM response files for the selected datasets, models, and item pools.
- `dataset_manifest.json`: the 18 datasets and score metrics.
- `embeddings/`: numerical Google vectors and an index keyed by dataset, model,
  and item, with response and chunk checksums.
- `preparation.json`: source hashes, item/model counts, and selection provenance.

The historical Google cache uses pickle files keyed by an entire batch's input
texts and position. Preparation reconstructs those batches from the extracted
tables, reads the trusted local cache, and exports NumPy arrays. The worker reads
these arrays with pickle loading disabled. Missing batches, mismatched responses,
or invalid scores fail preparation; missing or changed vectors fail evaluation.

To preserve the historical pools, WMT uses the same 20% item selection with seed
123 and the METEOR item scores already computed in `wmt_14.tsv`. Those scores are
matched to the original HELM responses and references; they are not DKPS estimates.
MATH's source metric is `math_equiv_chain_of_thought`. The original parser's model
and item selections are retained through the extracted tables. The common target
panel is selected by availability before evaluation.

The old embedding code converted pandas NA values to the string `"nan"`. Reusing
its vectors preserves that behavior for empty/NA responses. The exported index
records both the raw-response hash and embedding-input hash, and preparation
reports how many rows had this transformation.

## Run the card

Install the host controller as described in the
[containerization guide](containerized_evaluation.md). Use the full image:

```bash
docker build -t jhu-magnet-dkps-gpu .

python scripts/run_pair_coverage.py \
  --helm_suite_path "$PAIR_COVERAGE_INPUTS/suite" \
  --dataset_manifest "$PAIR_COVERAGE_INPUTS/dataset_manifest.json" \
  --embedding_cache_path "$PAIR_COVERAGE_INPUTS/embeddings" \
  --embed_provider google \
  --embed_model gemini-embedding-001 \
  --out_dpath "$PWD/results/pair-coverage-18" \
  --num_replicates 1024
```

All four budgets (1, 2, 4, 8) are evaluated together. With the local historical
inputs, there are 92 common target models and 6,624 combinations. At 1,024
replicates, that is 6,782,976 fresh DKPS fits. For an execution smoke test, use
`--num_replicates 2`; that is too few replicates to assess the expected-error claim.

The prepared directory can be copied to another machine; it does not depend on
the original Google cache at evaluation time. Source paths in `preparation.json`
are provenance only. The standard MAGNET container wrapper mounts the supplied directories.
Evaluation only reads the prepared inputs. The output protocol records the cache
index checksum; metrics distinguish cached embedding batches from all batches.

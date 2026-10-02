# Precompute nomic embeddings once, then run the card against them

Datasets that are not one-hot (MATH, WMT) are embedded with a neural model. Left
to itself, the card embeds a fresh batch of sampled responses on every replicate,
and `dkps.embed` builds a new embedder, reloading the model, on each call. The
embedding cache avoids both: every response is embedded once, and the card reads
vectors from disk (`--embedding_cache_path`). Only embeddings are reused; query
sampling, DKPS fits, and OLS predictions are still computed fresh.

`jhu_ta1.magnet.precompute_embeddings` builds that cache for
`sentence-transformers / nomic-ai/nomic-embed-text-v2-moe`. MedQA and LegalBench
use one-hot embeddings and are skipped.

## Rebuild the image once

The nomic embedder needs `transformers<5`; its remote modeling code calls
`get_extended_attention_mask`, which 5.x removed. Images built before the pin
carry transformers 5 and fail with
`AttributeError: 'NomicBertModel' object has no attribute 'get_extended_attention_mask'`
the first time text is encoded, with or without this cache. Rebuild:

```bash
docker build -t jhu-magnet-dkps-gpu .
```

## Build the cache (incremental)

```bash
export REPO="$PWD"
export DATA="$REPO/data/crfm-helm-public"
export SUITE="$DATA/lite/benchmark_output/runs/_all"
export CACHE="$REPO/results/embedding-cache-nomic"
export HF_HOME="$REPO/.cache/huggingface"     # keeps the model weights between runs
mkdir -p "$HF_HOME"

# The manifest names the datasets to cache. Add entries to it over time.
cat > "$REPO/results/manifest_math.json" <<'EOF'
{"datasets": [
  {"dataset": "math:subject=algebra", "metric": "math_equiv_chain_of_thought"}
]}
EOF

docker run --rm --network host --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e HF_HOME -e PYTHONPATH="$REPO" \
  -v "$REPO:$REPO" -w "$REPO" jhu-magnet-dkps-gpu \
  python -m jhu_ta1.magnet.precompute_embeddings \
    --helm_suite_path "$SUITE" \
    --dataset_manifest "$REPO/results/manifest_math.json" \
    --cache_dpath "$CACHE"
```

It prints progress every 1% with throughput and a time estimate. To add datasets,
append them to the manifest and run the same command: datasets already complete in
the index are skipped. `--datasets NAME [NAME ...]` limits a run to some of the
manifest's datasets, and `--force` re-embeds datasets that are already cached.

- Every model and every pool item of a dataset is embedded, because the card checks
  coverage of the whole panel before it starts.
- Chunk files are named by a hash of their provider, model, and texts. An
  interrupted run resumes without redoing finished chunks (the index is only written
  when a whole dataset is done), and ctrl-C is safe.
- If the HELM responses for a dataset change, it is detected by response hash and
  the dataset is re-embedded.
- One cache holds one provider and model. Changing `--embed_model` against an
  existing cache directory is refused; use a new directory.

## Run the card against it

`$REPO` is mounted at its own path, so a cache under it needs no extra mount.
Point the card at the same manifest the cache was built from (or any manifest whose
non-one-hot datasets are all in the cache):

```bash
python -m magnet.evaluation_new \
  "$REPO/jhu_ta1/cards/jhu_run_predict_pair_coverage_kwdagger.yaml" \
  --backend serial \
  --container_image jhu-magnet-dkps-gpu \
  --container_mounts "$REPO:$DATA" \
  --output_path "$REPO/results/pair-coverage-math-algebra" \
  --params "matrix:
    materialize_lite.version: '_all'
    materialize_lite.precomputed_roots: '$DATA'
    materialize_lite.download: never
    materialize_lite.runs: 'regex:^math:subject=algebra,.*'
    pair_coverage.dataset_manifest: '$REPO/results/manifest_math.json'
    pair_coverage.embedding_cache_path: '$CACHE'
    pair_coverage.num_replicates: 32"
```

`pair_coverage.embedding_cache_path` is an input path, so the container sees it as
long as it lies under a mounted directory. For a cache elsewhere, add it to
`--container_mounts` (colon-separated: `"$REPO:$DATA:/path/to/cache"`).

The result's metrics report `cached_embedding_batches` next to `embedding_batches`;
with a full cache the two are equal. The protocol block records the index checksum.

## Notes

- `$DATA` is the parent of `lite/`, not the `runs/_all` directory: the materializer
  looks for `<root>/lite/benchmark_output/runs/<version>`.
- On a machine with no GPU the container runs nomic on the CPU. Precomputing is the
  slow step, and it happens once per dataset.
- The cache's `index.json` records the embedding provider and model, and the card
  fails fast if they differ from `--embed_provider` / `--embed_model`.

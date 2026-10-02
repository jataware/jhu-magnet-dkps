docker build -t jhu-magnet-dkps-gpu .

# --

export REPO="$PWD"
export DATA="$PWD/data/crfm-helm-public/"
export PYTHONPATH="$REPO"

# smoke test - med_qa only
mkdir -p $REPO/results
cat > $REPO/results/manifest_med_qa.json <<'EOF'
{
  "datasets": [
    {"dataset": "med_qa", "metric": "quasi_exact_match"}
  ]
}
EOF

python -m magnet.evaluation_new \
  "$REPO/jhu_ta1/cards/jhu_run_predict_pair_coverage_kwdagger.yaml" \
  --backend           serial \
  --container_image   jhu-magnet-dkps-gpu \
  --container_mounts  "$REPO:$DATA" \
  --output_path       "$REPO/results/pair-coverage-all-versions" \
  --params "matrix:
    materialize_lite.version: '_all'
    materialize_lite.precomputed_roots: '$DATA'
    materialize_lite.download: never
    materialize_lite.runs: 'regex:^med_qa[:,].*'
    pair_coverage.dataset_manifest: '$REPO/results/manifest_med_qa.json'
    pair_coverage.num_replicates: 16"

# --

cat > $REPO/results/manifest_math_algebra.json <<'EOF'
{
  "datasets": [
    {"dataset": "math:subject=algebra", "metric": "math_equiv_chain_of_thought"}
  ]
}
EOF

  {"datasets": [
    {"dataset": "med_qa",                                      "metric": "quasi_exact_match"},
    {"dataset": "legalbench:subset=abercrombie",               "metric": "quasi_exact_match"},
    {"dataset": "legalbench:subset=corporate_lobbying",        "metric": "quasi_exact_match"},
    {"dataset": "legalbench:subset=function_of_decision_section", "metric": "quasi_exact_match"},
    {"dataset": "legalbench:subset=international_citizenship_questions", "metric": "quasi_exact_match"},
    {"dataset": "legalbench:subset=proa",                      "metric": "quasi_exact_match"},
    {"dataset": "math:subject=number_theory",            "metric": "math_equiv_chain_of_thought"},
    {"dataset": "math:subject=geometry",                 "metric": "math_equiv_chain_of_thought"},
    {"dataset": "math:subject=counting_and_probability", "metric": "math_equiv_chain_of_thought"},
    {"dataset": "math:subject=intermediate_algebra",     "metric": "math_equiv_chain_of_thought"},
    {"dataset": "math:subject=precalculus",              "metric": "math_equiv_chain_of_thought"},
    {"dataset": "math:subject=prealgebra",               "metric": "math_equiv_chain_of_thought"},
    {"dataset": "math:subject=algebra",                  "metric": "math_equiv_chain_of_thought"},
    {"dataset": "wmt_14:language_pair=cs-en", "metric": "bleu_4"},
    {"dataset": "wmt_14:language_pair=de-en", "metric": "bleu_4"},
    {"dataset": "wmt_14:language_pair=fr-en", "metric": "bleu_4"},
    {"dataset": "wmt_14:language_pair=hi-en", "metric": "bleu_4"},
    {"dataset": "wmt_14:language_pair=ru-en", "metric": "bleu_4"}
  ]}


# --
# math:subject=algebra: precompute nomic embeddings once, then run the card on them.
# (med_qa / legalbench are one-hot and need no cache; math/wmt use the nomic embedder.)
# Needs the image above rebuilt after the transformers<5 pin: nomic's remote code
# breaks on transformers 5. Full detail: docs/precomputed_embeddings.md

# 1. Build the cache. ~12.8k responses for algebra (95 models x 135 items); ~45 min on
#    CPU (measured ~3-5 responses/s). Progress prints every 1%. Safe to ctrl-C and
#    rerun: finished chunks are reused. To add datasets later, add them to the
#    manifest (or a new one) and rerun: datasets already cached are skipped, and
#    `--datasets NAME ...` limits a run to some of them. Needs `newgrp docker` if
#    docker says permission denied.
export SUITE="$DATA/lite/benchmark_output/runs/_all"
export CACHE="$REPO/results/embedding-cache-nomic"
export HF_HOME="$REPO/.cache/huggingface"    # keeps the nomic weights between runs
mkdir -p "$HF_HOME"

# >>
# docker run --rm --gpus all --network host --user "$(id -u):$(id -g)" \
#   -e HOME=/tmp -e HF_HOME -e PYTHONPATH="$REPO" \
#   -v "$REPO:$REPO" -w "$REPO" jhu-magnet-dkps-gpu \
#   python -m jhu_ta1.magnet.precompute_embeddings \
#     --helm_suite_path  "$SUITE" \
#     --dataset_manifest "$REPO/results/manifest_math_algebra.json" \
#     --cache_dpath      "$CACHE"
# --
export CACHE_CL="$REPO/results/embedding-cache-nomic-clustering"

docker run --rm --gpus all --network host --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e HF_HOME -e PYTHONPATH="$REPO" \
  -v "$REPO:$REPO" -w "$REPO" \
  -v "$REPO/.cache/dkps-patch/embed.py:/opt/conda/lib/python3.11/site-packages/dkps/embed.py:ro" \
  jhu-magnet-dkps-gpu \
  python -m jhu_ta1.magnet.precompute_embeddings \
    --helm_suite_path  "$SUITE" \
    --dataset_manifest "$REPO/results/manifest_math_algebra.json" \
    --cache_dpath      "$CACHE_CL"
# <<


# 2. Run the card against the cache. Same command as without a cache, plus
#    embedding_cache_path; metrics should then report cached_embedding_batches ==
#    embedding_batches. Use a fresh --output_path: kwdagger skips a node whose output
#    already exists.
python -m magnet.evaluation_new \
  "$REPO/jhu_ta1/cards/jhu_run_predict_pair_coverage_kwdagger.yaml" \
  --backend           serial \
  --container_image   jhu-magnet-dkps-gpu \
  --container_mounts  "$REPO:$DATA" \
  --output_path       "$REPO/results/pair-coverage-math-algebra" \
  --params "matrix:
    materialize_lite.version: '_all'
    materialize_lite.precomputed_roots: '$DATA'
    materialize_lite.download: never
    materialize_lite.runs: 'regex:^math:subject=algebra,.*'
    pair_coverage.dataset_manifest: '$REPO/results/manifest_math_algebra.json'
    pair_coverage.embedding_cache_path: '$CACHE_CL'
    pair_coverage.num_replicates: 128"
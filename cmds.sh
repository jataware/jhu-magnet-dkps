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
    materialize_lite.version           : '_all'
    materialize_lite.precomputed_roots : '$DATA'
    materialize_lite.download          : never
    materialize_lite.runs              : 'regex:^med_qa[:,].*'
    pair_coverage.dataset_manifest     : '$REPO/results/manifest_med_qa.json'
    pair_coverage.num_replicates       : 16"

# --

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

docker run --rm --gpus all --network host --user "$(id -u):$(id -g)" \
  -e HOME=/tmp                                                       \
  -e HF_HOME                                                         \
  -e PYTHONPATH="$REPO"                                              \
  -v "$REPO:$REPO"                                                   \
  -w "$REPO"                                                         \
  jhu-magnet-dkps-gpu                                                \
  python -m jhu_ta1.magnet.precompute_embeddings                     \
    --helm_suite_path  "$SUITE"                                      \
    --dataset_manifest "$REPO/jhu_ta1/cards/manifest_full.json"      \
    --cache_dpath      "$CACHE"


# 2. Run the card against the cache. Same command as without a cache, plus
#    embedding_cache_path; metrics should then report cached_embedding_batches ==
#    embedding_batches. Use a fresh --output_path: kwdagger skips a node whose output
#    already exists.
python -m magnet.evaluation_new \
  "$REPO/jhu_ta1/cards/jhu_run_predict_pair_coverage_kwdagger.yaml" \
  --backend           serial \
  --container_image   jhu-magnet-dkps-gpu \
  --container_mounts  "$REPO:$DATA" \
  --output_path       "$REPO/results/pair-coverage-full" \
  --params "matrix:
    materialize_lite.version           : '_all'
    materialize_lite.precomputed_roots : '$DATA'
    materialize_lite.download          : never
    materialize_lite.runs              : 'regex:^(med_qa|legalbench|math|wmt_14)[:,].*'
    pair_coverage.dataset_manifest     : "$REPO/jhu_ta1/cards/manifest_full.json" 
    pair_coverage.embedding_cache_path : '$CACHE'
    pair_coverage.num_replicates       : 10"

# --
# Full manifest, split into three parallel pair_coverage nodes (one-hot, MATH,
# WMT) that share one materialize step. With the cache, pair_coverage is CPU-only,
# so set --tmux_workers explicitly ('auto' caps it at the GPU count).

python -m magnet.evaluation_new \
  "$REPO/jhu_ta1/cards/jhu_run_predict_pair_coverage_kwdagger.yaml" \
  --backend           tmux \
  --tmux_workers      3 \
  --container_image   jhu-magnet-dkps-gpu \
  --container_mounts  "$REPO:$DATA" \
  --output_path       "$REPO/results/pair-coverage-full-split" \
  --params "matrix:
    materialize_lite.version           : '_all'
    materialize_lite.precomputed_roots : '$DATA'
    materialize_lite.download          : never
    materialize_lite.runs              : 'regex:^(med_qa|legalbench|math|wmt_14)[:,].*'
    pair_coverage.dataset_manifest     :
      - '$REPO/results/manifest_onehot.json'
      - '$REPO/results/manifest_math.json'
      - '$REPO/results/manifest_wmt.json'
    pair_coverage.embedding_cache_path : '$CACHE'
    pair_coverage.num_replicates       : 10"

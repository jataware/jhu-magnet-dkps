This repository contains an example TA1 algorithm integrated with the
MAGNET evaluation framework.

The example algorithm, provided by the JHU team, predicts whether or
not a model will produce the correct answer for a given question based
on the performance of similar models in Data Kernel Perspective Space (DKPS) [1]

## Setup

### Python environment

We use the `uv` tool for environment management.  Installation instructions:
https://docs.astral.sh/uv/#installation

```
uv venv --python 3.11 --seed .venv-311-example
source .venv-311-example/bin/activate
uv pip install .
```

### Downloading HELM results

The example evaluation card requires precomputed HELM results on the
helm-lite benchmark.  If you do not already have these downloaded, the magnet 
framework provides a download utility for these results. You can run the following
commands to download and link them into a single `_all` directory.

```bash
function download_dataset {
  DST=$1
  DATASET=$2
  mkdir -p $DST
  magnet download helm --benchmark=lite --list-versions | while read version; do
      magnet download helm data/crfm-helm-public --benchmark=lite --version="$version" --runs "regex:${DATASET}.*"
      (cd $DST && ln -s "../$version"/* .)
  done
}

DST=data/crfm-helm-public/lite/benchmark_output/runs/_all
download_dataset $DST "wmt"
download_dataset $DST "math"
download_dataset $DST "med_qa"
download_dataset $DST "legalbench"

find $(dirname $DST) -type d | fgrep anthropic_claude-3-5-haiku-20241022 |\
  fgrep ',stop=none' | xargs -I {} rm -r {}

find $(dirname $DST) -type d | fgrep google_gemini-2.0-flash-exp |\
  fgrep ',stop=none' | xargs -I {} rm -r {}
```

### Linking pre-downloaded HELM results

If you do already have them downloaded but not linked into a single
`_all` directory you can run the following:

```
mkdir -p data/crfm-helm-public/lite/benchmark_output/runs/_all
cd data/crfm-helm-public/lite/benchmark_output/runs/_all
ln -s /path/to/existing/helm/lite/runs/*/wmt* .
ln -s /path/to/existing/helm/lite/runs/*/math* .
ln -s /path/to/existing/helm/lite/runs/*/med_qa* .
ln -s /path/to/existing/helm/lite/runs/*/legalbench* .
cd -
```

(Note: as there are some duplicate runs across versions, it's safe to
ignore warnings about files / links already existing when running the
`ln` command)

## Running the card

```bash
# --
# Build embedding cache

export REPO="$PWD"
export DATA="$PWD/data/crfm-helm-public/"
export PYTHONPATH="$REPO"
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


# --
# Full manifest, serial

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
    pair_coverage.num_replicates       : 128"

# --
# Full manifest, parallel

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
    pair_coverage.dataset_manifest     :
      - '$REPO/results/manifest_onehot.json'
      - '$REPO/results/manifest_math.json'
      - '$REPO/results/manifest_wmt.json'
    pair_coverage.embedding_cache_path : '$CACHE'
    pair_coverage.num_replicates       : 128"
```

(Note: it's safe to ignore warnings about "dkps.embed: unable to load google-genai")

Once the card has been fully evaluated, you should see the following:

```
================================
Settings Evaluated: 3
  Verified:     1.00
  Falsified:    0.00
  Inconclusive: 0.00
================================


Title:       JHU DKPS based per-instance metric prediction
Description: We can predict whether a particular model will produce the correct output based on the performance of similar models in Data Kernel Perspective Space (DKPS)

================================
CLAIM:       
assert computed_auc > auc_threshold, assert_failed_msg

================================
RESULT:      VERIFIED
================================
CARD STATUS: EVALUATED
```

This output indicates that three variations of the evaluation card
have been evaluated (the example card sweeps over three different seed
values for random evaluation set selection).  In this case all three
variations have been verified (claim passed), so the final `RESULT` of
the card is that it is `"VERIFIED"`.

## Dataset–model–budget coverage

The `jhu_run_predict_pair_coverage_kwdagger.yaml` card tests whether DKPS has
lower expected absolute score-estimation error than the same-budget sample mean
for more than 95% of dataset/model/query-budget combinations. It computes response
embeddings and DKPS predictions from a supplied HELM suite, using a fixed sample
weight of 0.8 and eight DKPS dimensions.

See [the pair-coverage guide](docs/live_pair_coverage.md) for the dataset manifest,
Docker runner, output format, and tests.

## Citations

[1] Hayden Helm, Aranyak Acharyya, Youngser Park, Brandon Duderstadt, and Carey Priebe. 2025. Statistical inference on black-box generative models in the data kernel perspective space. In Findings of the Association for Computational Linguistics: ACL 2025, pages 3955–3970, Vienna, Austria. Association for Computational Linguistics.

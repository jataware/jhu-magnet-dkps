This repository contains an example TA1 algorithm integrated with the
MAGNET evaluation framework.

The example algorithm, provided by the JHU team, predicts whether or
not a model will produce the correct answer for a given question based
on the performance of similar models in Data Kernel Perspective Space (DKPS) [1].
The pair-coverage card estimates a model's benchmark score from a small number
of queries, following the query-efficient evaluation approach of [2].

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
```

Once the card has been fully evaluated, you should see something like the following:

```
Estimator: sample weight=0.8, DKPS dimensions=8
Evaluation: 18 datasets, 94 models, budgets [1, 2, 4, 8], 10 replicates per combination
Completed: 6768/6768 combinations, 67680 DKPS fits
Claim: 6474/6768 combinations improve (95.66%); required: more than 95%
...
INFO     ================================  evaluation.py:448
INFO     RESULT:      VERIFIED             evaluation.py:449
INFO     ================================  evaluation.py:454
INFO     CARD STATUS: EVALUATED            evaluation.py:455

```

*Note:* If you want to run faster, you can set `pair_coverage.num_replicates` to 8, 16, 32, etc.  That will make the evaluation run faster but increases noise / reduces statistical significance of results.

*Also Note:* I tried to increase the parallelism here, but because of the way the aggregation works, the manifests were getting aggregated independently, and the 95% check being applied to each on their own.  The likelihood that _one_ of the individal splits doesn't meet the 95% mark is increased, so probability to reject _one_ of the splits and thus the _whole_ claim is increased.  So - I'm sure there are better ways to parallelize - but need to make sure that everything gets re-combined before the final 95% test.

## Citations

[1] Hayden Helm, Aranyak Acharyya, Youngser Park, Brandon Duderstadt, and Carey Priebe. 2025. Statistical inference on black-box generative models in the data kernel perspective space. In Findings of the Association for Computational Linguistics: ACL 2025, pages 3955–3970, Vienna, Austria. Association for Computational Linguistics.

[2] Hayden Helm, Ben Johnson, and Carey Priebe. 2026. Query-efficient model evaluation using cached responses. arXiv preprint arXiv:2605.07096. https://arxiv.org/abs/2605.07096

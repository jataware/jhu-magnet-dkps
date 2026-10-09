# Running the pair-coverage card in a container

The card is
[`jhu_run_predict_pair_coverage_kwdagger.yaml`](../jhu_ta1/cards/jhu_run_predict_pair_coverage_kwdagger.yaml).
It tests whether DKPS has lower expected absolute error than the same-budget
sample mean for more than 95% of evaluated (dataset, model, query budget)
combinations. See [the evaluation description](live_pair_coverage.md) for the
estimator and aggregation details.

## How Kitware runs it

The host runs `python -m magnet.evaluation_new`. MAGNET reads the card's
`kwdagger:` block and schedules two nodes, each in the image built from the
repository's **Dockerfile**:

1. `materialize_lite` assembles the selected HELM runs. It links runs from a local
   bucket mirror, or downloads missing runs from the public HELM bucket.
2. `pair_coverage` reads that suite, computes response embeddings, repeatedly fits
   DKPS, and writes the errors and coverage. MAGNET then evaluates the claim and
   writes `verdict.json`.

Both nodes use Kitware's `magnet.process_node.MagnetProcessNode`. The checkout and
inputs are bind-mounted at their host paths; `PYTHONPATH` selects the checkout's
code. The image supplies MAGNET, DKPS, PyTorch, and the embedding dependencies.
The default manifest selects MedQA and two LegalBench subsets, which use one-hot
embeddings and need no GPU or embedding service. Saved HELM responses and labels
are inputs; DKPS predictions are computed during evaluation. Target language
models are not called again, so inference leasing is not needed.

## Build and set up the host

Run from this repository:

```bash
export REPO="$PWD"
docker build -t jhu-magnet-dkps-gpu .

python -m pip install 'aiq-magnet[optional]==0.1.0' 'kwdagger==0.4.1'
export PYTHONPATH="$REPO"
```

On Apple Silicon, build with `docker build --platform linux/amd64
-t jhu-magnet-dkps-gpu .` and add
`--container_docker_args '--platform linux/amd64'` to the evaluation command.
This uses the full CUDA image under emulation; a Mac cannot test CUDA execution.
On macOS, install GNU coreutils and put its commands on PATH for kwdagger:

```bash
brew install coreutils
export PATH="$(brew --prefix coreutils)/libexec/gnubin:$PATH"
```

## Smoke test

This command downloads the three default datasets and runs two replicates at
budgets 1, 2, 4, and 8:

```bash
python -m magnet.evaluation_new \
  "$REPO/jhu_ta1/cards/jhu_run_predict_pair_coverage_kwdagger.yaml" \
  --output_path "$REPO/results/pair-coverage-smoke" \
  --backend serial \
  --container_image jhu-magnet-dkps-gpu \
  --container_mounts "$REPO" \
  --params "matrix:
    pair_coverage.num_replicates: 2
    pair_coverage.run_id: 'smoke-$(date +%s)'"
```

Two replicates test execution, not the stability of expected-error estimates.
Either a completed `VERIFIED` or `FALSIFIED` verdict is a valid smoke-test outcome;
a failed node or missing evidence is not.

To use an existing bucket mirror, set `DATA` to a directory containing
`lite/benchmark_output/runs/v1.0.0`, add it to the mounts, and select it in params:

```bash
export DATA=/absolute/path/to/crfm-helm-public
python -m magnet.evaluation_new \
  "$REPO/jhu_ta1/cards/jhu_run_predict_pair_coverage_kwdagger.yaml" \
  --output_path "$REPO/results/pair-coverage" \
  --backend serial \
  --container_image jhu-magnet-dkps-gpu \
  --container_mounts "$REPO:$DATA" \
  --params "matrix:
    materialize_lite.precomputed_roots: '$DATA'
    materialize_lite.download: never
    pair_coverage.num_replicates: 32"
```

`download: never` restricts the run to the local mirror. Omit it to allow missing
runs to be fetched. The default is 1,024 replicates when not overridden. Use
`--backend tmux` for a detached workstation run (requires tmux), or the configured
Slurm backend on the cluster. For neural embeddings on an NVIDIA host, add
`--container_docker_args '--gpus device=0'`.

Results are under `--output_path`, including the native MAGNET verdict and the
worker's `pair_coverage.json` and `.trials.jsonl`. Identical node configurations
reuse completed artifacts. Set a new `pair_coverage.run_id` to force new DKPS
fits while retaining the materialized suite; `base_seed` controls query sampling.

## Other datasets and the prepared 18-dataset suite

For another HELM release, change `materialize_lite.version` and
`materialize_lite.runs`, and supply `pair_coverage.dataset_manifest` selecting the
matching datasets and score metrics. Mount external input directories too. Text
embeddings can be computed by the image's local embedding model, or read from an
explicitly supplied response-embedding cache.

The historical 18-dataset comparison uses multiple HELM releases and matching
cached Google vectors. Follow [the full-run guide](full_pair_coverage.md) to
prepare it, then run `scripts/run_pair_coverage.py`. That helper uses the same
card, standard container node, and full image, with a supplied-suite pipeline
that skips downloading/materializing an already prepared suite. A local DKPS
checkout is not required for evaluation; the image installs DKPS.

The standard MAGNET container wrapper allows network access and mounts supplied
paths writable. It does not mount the Docker socket into workers. There is no
separate lightweight Dockerfile for this workflow.

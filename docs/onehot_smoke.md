# One-hot smoke test: native, then Docker

Both runs evaluate MedQA, LegalBench Abercrombie, and LegalBench Corporate
Lobbying with two query replicates at each budget (1, 2, 4, 8). These use one-hot
response embeddings, so no neural model, Google cache, credentials, or GPU is
needed. Two replicates check execution; either a completed VERIFIED or FALSIFIED
verdict is a valid outcome.

## Native execution

From a checkout containing the pair-coverage card, install the project and its
runtime dependencies into a virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e . 'aiq-magnet[optional]==0.1.0' 'kwdagger==0.4.1'
export REPO="$PWD"
export PYTHONPATH="$REPO"
```

On macOS, install GNU coreutils and put its commands on PATH:

```bash
brew install coreutils
export PATH="$(brew --prefix coreutils)/libexec/gnubin:$PATH"
```

Run the card without a container image. The first node downloads the default
HELM datasets; the second computes the embeddings and DKPS predictions:

```bash
python -m magnet.evaluation_new \
  jhu_ta1/cards/jhu_run_predict_pair_coverage_kwdagger.yaml \
  --backend serial \
  --output_path "$REPO/results/smoke-native" \
  --params "matrix:
    pair_coverage.num_replicates: 2
    pair_coverage.run_id: 'smoke-native-$(date +%s)'"
```

## Docker execution

Use the full Dockerfile and the same card:

```bash
docker build -t jhu-magnet-dkps-gpu .

python -m magnet.evaluation_new \
  jhu_ta1/cards/jhu_run_predict_pair_coverage_kwdagger.yaml \
  --backend serial \
  --output_path "$REPO/results/smoke-docker" \
  --container_image jhu-magnet-dkps-gpu \
  --container_mounts "$REPO" \
  --params "matrix:
    pair_coverage.num_replicates: 2
    pair_coverage.run_id: 'smoke-docker-$(date +%s)'"
```

For Apple Silicon, add `--platform linux/amd64` to the build and
`--container_docker_args '--platform linux/amd64'` to the evaluation command.
This runs the full image under emulation.

For Docker-only use, the host needs just Docker and the pinned MAGNET/kwdagger
controller packages; the image installs the scientific dependencies and DKPS.
See the [containerization guide](containerized_evaluation.md) for reusing a local
HELM mirror, GPU settings, and interpreting results.

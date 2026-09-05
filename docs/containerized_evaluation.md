# Running the evaluation card in a container

## How Kitware runs this card

`python -m magnet.evaluation_new` reads a card's `kwdagger:` block and turns
it into a DAG. Each node of that DAG runs as one `docker run` of the image
built from this repo's `Dockerfile`. The checkout is bind-mounted into the
container at its own absolute path, the node's working directory is that
path, and `PYTHONPATH` is forwarded, so the node runs the mounted checkout
while the image supplies the installed environment. Results land under
`--output_path`. The DAG backend is tmux on a workstation and Slurm on the
cluster; the card and the image are the same in both cases.

The pipelines start with a `materialize_lite` node that assembles the HELM
lite suite the card needs. Given a precomputed root it symlinks the matching
runs; given none it downloads them from the public `crfm-helm-public`
bucket. Either way the DKPS nodes read one suite directory produced inside
the run, so nothing about the host's data layout reaches them.

Per-node leasing does not apply: the cards score precomputed HELM runs and
do no live inference.

## Build

```bash
cd $REPO
docker build -t jhu-magnet-dkps-gpu .
```

`MAGNET_VERSION` in the Dockerfile is the aiq-magnet release the evaluator uses,
from PyPI (0.1.0, which also brings aiq-magnet-theory). To build against
another release:

```bash
docker build --build-arg MAGNET_VERSION=0.1.0 -t jhu-magnet-dkps-gpu .
```

## Reproduce the June dry run

On the host you need docker, tmux, and the same aiq-magnet pin the image
carries:

```bash
pip install "aiq-magnet[optional]==0.1.0"
export PYTHONPATH=$REPO
```

With a local mirror of the bucket at `$DATA` (so that
`$DATA/lite/benchmark_output/runs/v1.0.0` exists), the materialize node
symlinks from it:

```bash
cd $REPO
python -m magnet.evaluation_new jhu_ta1/cards/jhu_instance_predict_auc_kwdagger.yaml \
    --output_path runs/jhu_instance_predict_auc \
    --backend tmux \
    --container_image jhu-magnet-dkps-gpu \
    --container_mounts "$REPO:$DATA" \
    --container_docker_args "--gpus device=0" \
    --params "matrix: {materialize_lite.precomputed_roots: '$DATA'}"
```

Without a mirror, drop the `$DATA` mount and the `--params` line; the node
downloads the runs it needs (the card asks for the med_qa subset, a few
hundred megabytes), which needs network access from inside the container.

Expect `VERIFIED`: the claim is that the AUC exceeds 0.5 for each seed. The
per-seed AUC values are around 0.86, 0.88 and 0.89. The verdict is written to
`runs/jhu_instance_predict_auc/<hash>_<stamp>/verdict.json`, with a `latest`
symlink beside it. A second run reuses every node whose artifact exists,
including the materialized suite.

The query-efficiency cards run the same way; only the card path changes.

## Leasing

Does not apply. These cards do no live inference.

## What Kitware changes when evaluating

Our runner supplies the host-specific values: which GPU a container sees,
where the bucket mirror is mounted, whether the backend is tmux or Slurm,
and a provenance record beside the verdict. The card, this image and the
command shape above are what we run.

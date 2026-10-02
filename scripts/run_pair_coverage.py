"""Run the containerized pair-coverage card on an already assembled HELM suite."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "jhu_ta1/cards/jhu_run_predict_pair_coverage_kwdagger.yaml"
PIPELINE = ROOT / "jhu_ta1/magnet/pair_coverage_supplied_pipeline.yaml"
INFERENCE_ENV = ("OPENAI_API_KEY", "OPENAI_BASE_URL", "HF_TOKEN", "HF_HOME")


def controller_environment(output, dkps_root=None):
    """Configure local imports and the GNU tools required by kwdagger on macOS."""
    env = dict(os.environ)
    for name in INFERENCE_ENV:
        env.pop(name, None)
    env.update(
        PYTHONPATH=os.pathsep.join(
            [str(ROOT)] + ([str(dkps_root)] if dkps_root else [])
        ),
        PYTHONUNBUFFERED="1",
        XDG_CACHE_HOME=str(output / ".controller-cache"),
    )
    if sys.platform == "darwin":
        bin_directory = output / ".host-bin"
        bin_directory.mkdir(exist_ok=True)
        for name in ("chmod", "ln"):
            executable = shutil.which(f"g{name}")
            if not executable:
                raise RuntimeError(
                    f"macOS requires GNU coreutils (g{name}) for kwdagger scripts"
                )
            link = bin_directory / name
            if not link.exists():
                link.symlink_to(executable)
            if link.resolve() != Path(executable).resolve():
                raise RuntimeError(f"Unexpected host-bin/{name} symlink")
        env["PATH"] = str(bin_directory) + os.pathsep + env.get("PATH", "")
    return env


def read_verdict(evaluation_directory):
    """Require completed evidence; FALSIFIED is a valid experimental outcome."""
    paths = list(evaluation_directory.glob("*/verdict.json"))
    if len(paths) != 1:
        raise RuntimeError(f"Expected one MAGNET verdict under {evaluation_directory}")
    path = paths[0]
    verdict = json.loads(path.read_text())
    if verdict["evidence"]["available"] != 1:
        raise RuntimeError(f"Expected one completed evidence row: {path}")
    if verdict["result"] not in {"VERIFIED", "FALSIFIED"}:
        raise RuntimeError(f"Incomplete evaluation: {path}")
    return path, verdict["result"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--helm_suite_path", required=True, type=Path)
    parser.add_argument("--dataset_manifest", required=True, type=Path)
    parser.add_argument("--out_dpath", required=True, type=Path)
    parser.add_argument(
        "--dkps_root",
        type=Path,
        help="Optional development override for the DKPS installed in the image",
    )
    parser.add_argument("--image", default="jhu-magnet-dkps-gpu")
    parser.add_argument(
        "--container_docker_args", default="", help="For example: --gpus device=0"
    )
    parser.add_argument("--num_replicates", type=int, default=1024)
    parser.add_argument("--queries", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument(
        "--embed_model", help="Embedding model name or local weights path"
    )
    parser.add_argument("--embed_provider", default="sentence-transformers")
    parser.add_argument(
        "--embedding_cache_path",
        type=Path,
        help="Exported response embeddings; missing entries fail the run",
    )
    parser.add_argument(
        "--embedding_mount", type=Path, help="Local embedding model directory"
    )
    args = parser.parse_args(argv)

    suite = args.helm_suite_path.resolve()
    manifest = args.dataset_manifest.resolve()
    output = args.out_dpath.resolve()
    dkps_root = args.dkps_root.resolve() if args.dkps_root else None
    if not suite.is_dir() or not manifest.is_file():
        parser.error("Provide an existing HELM suite directory and dataset manifest")
    if dkps_root and not (dkps_root / "dkps/__init__.py").is_file():
        parser.error("dkps_root must contain dkps/__init__.py")
    if len(set(args.queries)) != len(args.queries) or not set(args.queries) <= {
        1,
        2,
        4,
        8,
    }:
        parser.error("Choose distinct query budgets from 1, 2, 4, 8")
    if args.num_replicates < 2:
        parser.error("At least two replicates are required")
    output.mkdir(parents=True, exist_ok=True)
    env = controller_environment(output, dkps_root)

    mounts = [ROOT, suite, manifest, output]
    if dkps_root:
        mounts.append(dkps_root / "dkps")
    if args.embedding_mount:
        weights = args.embedding_mount.resolve()
        if not weights.is_dir():
            parser.error("embedding_mount must be an existing directory")
        mounts.append(weights)

    # A unique execution ID forces new fits without changing the sampling seed.
    run_id = uuid.uuid4().hex
    evaluation_directory = output / "evaluation" / run_id
    matrix = {
        "pair_coverage.helm_suite_path": str(suite),
        "pair_coverage.dataset_manifest": str(manifest),
        "pair_coverage.query_budgets": ",".join(map(str, args.queries)),
        "pair_coverage.num_replicates": args.num_replicates,
        "pair_coverage.run_id": run_id,
    }
    if args.embed_model:
        matrix["pair_coverage.embed_model"] = args.embed_model
    matrix["pair_coverage.embed_provider"] = args.embed_provider
    if args.embedding_cache_path:
        cache = args.embedding_cache_path.resolve()
        if not (cache / "index.json").is_file():
            parser.error("embedding_cache_path must contain index.json")
        mounts.append(cache)
        matrix["pair_coverage.embedding_cache_path"] = str(cache)
    command = [
        sys.executable,
        "-m",
        "magnet.evaluation_new",
        str(CARD),
        "--output_path",
        str(evaluation_directory),
        "--backend",
        "serial",
        "--container_image",
        args.image,
        "--container_mounts",
        ":".join(map(str, mounts)),
        "--container_env",
        json.dumps(dict.fromkeys(INFERENCE_ENV, "")),
        "--container_docker_args",
        args.container_docker_args,
        "--params",
        json.dumps({"pipeline": str(PIPELINE), "matrix": matrix}),
    ]
    log = output / f"controller-{run_id}.log"
    print(
        f"Evaluating {len(args.queries)} query budgets in one Docker job; log: {log}",
        flush=True,
    )
    with log.open("w") as stream:
        subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )
    path, result = read_verdict(evaluation_directory)
    print(f"{result}: {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

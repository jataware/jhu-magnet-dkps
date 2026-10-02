"""Prepare the historical 18 HELM pools and export cached Google embeddings.

Reads saved HELM responses, extracted item-score tables, and trusted local DKPS
pickle caches. It never reads DKPS predictions or calls an embedding API.
"""

import argparse
import hashlib
import json
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# Legacy extracted tables omitted the colon-suffixed provider model version.
MODEL_PARAMETER = re.compile(r"(?:[:,])model=([^,:]+)")


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def select_pool(frame, dataset):
    """Reproduce the original input selection, before query sampling."""
    selected = frame[frame.dataset == dataset]
    if dataset.startswith("wmt_14:"):
        items = selected.instance_id.unique()
        keep = np.random.default_rng(123).choice(
            items, int(len(items) * 0.2), replace=False
        )
        selected = selected[selected.instance_id.isin(keep)]
    selected = selected.sort_values(["model", "instance_id"]).copy()
    selected["item_id"] = selected.instance_id.str.removeprefix(dataset + "--")
    if selected.empty or selected.duplicated(["model", "item_id"]).any():
        raise ValueError(f"Empty or duplicated source rows: {dataset}")
    return selected


def matching_requests(state, rows, dataset, metric, original_scores):
    """Check item identity, references, responses, and native scores against HELM."""
    expected = rows.set_index("item_id")
    selected = []
    seen = set()
    if dataset.startswith("legalbench:"):
        from dkps.helm import clean_legalbench_answer

        choices = sorted(
            {
                r["instance"]["references"][0]["output"]["text"]
                for r in state["request_states"]
            }
        )
        choices = {text.lower(): index + 1 for index, text in enumerate(choices)}
    for request in state["request_states"]:
        instance = request["instance"]
        item_id = instance["id"]
        if item_id not in expected.index:
            continue
        if request.get("train_trial_index", 0) != 0 or instance.get("perturbation"):
            continue
        if item_id in seen:
            raise ValueError("Duplicate source HELM item")
        seen.add(item_id)
        row = expected.loc[item_id]
        completions = request["result"]["completions"]
        if len(completions) != 1:
            raise ValueError("Expected one source HELM completion")
        text = completions[0]["text"]
        if dataset.startswith("legalbench:"):
            response = choices.get(clean_legalbench_answer(text), 0)
        elif dataset == "med_qa":
            response = text.strip()
        else:
            response = text
        if str(response) != str(row.source_response):
            raise ValueError(f"Response differs at {item_id}")
        target = instance["references"][0]["output"]["text"]
        if str(target) != str(row.source_target):
            raise ValueError(f"Reference differs at {item_id}")
        if not dataset.startswith("wmt_14:"):
            if not np.isclose(
                original_scores[item_id][metric], row.score, rtol=0, atol=1e-12
            ):
                raise ValueError(f"Score differs at {item_id}")
        selected.append(request)
    if seen != set(expected.index):
        raise ValueError("Source HELM run does not cover the selected item pool")
    return selected


def materialize_dataset(runs_root, output, config, rows):
    dataset, metric = config["dataset"], config["metric"]
    candidates = {}
    for path in sorted(runs_root.glob(f"*/{dataset}*/scenario_state.json")):
        match = MODEL_PARAMETER.search(path.parent.name)
        if match:
            candidates.setdefault(match[1], []).append(path)
    model_ids = {}
    sources = []
    for name, group in rows.groupby("model", sort=True):
        paths = candidates.get(name, [])
        # The original LegalBench parser preferred the default stop configuration.
        if dataset.startswith("legalbench:") and len(paths) > 1:
            paths = [path for path in paths if ",stop=" not in path.parent.name]
        failures = []
        for path in paths:
            state = json.loads(path.read_text())
            score_path = path.parent / "display_predictions.json"
            raw_scores = {
                entry["instance_id"]: entry["stats"]
                for entry in json.loads(score_path.read_text())
            }
            try:
                requests = matching_requests(state, group, dataset, metric, raw_scores)
            except (ValueError, KeyError) as error:
                failures.append(f"{path}: {error}")
                continue
            model_ids[name] = state["adapter_spec"]["model"]
            destination = output / "suite" / f"{dataset},model={name}"
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "scenario_state.json").write_text(
                json.dumps(
                    {
                        "adapter_spec": state["adapter_spec"],
                        "request_states": requests,
                    }
                )
            )
            selected_scores = group.set_index("item_id")
            scores = []
            for request in requests:
                item_id = request["instance"]["id"]
                value = float(
                    selected_scores.loc[
                        item_id, "meteor" if dataset.startswith("wmt_14:") else "score"
                    ]
                )
                if not np.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f"Invalid item score: {dataset}/{name}/{item_id}")
                scores.append(
                    {
                        "instance_id": item_id,
                        "train_trial_index": 0,
                        "stats": {metric: value},
                    }
                )
            (destination / "display_predictions.json").write_text(json.dumps(scores))
            sources.append(
                {
                    "model": model_ids[name],
                    "source": str(path),
                    "scenario_sha256": file_hash(path),
                    "score_sha256": file_hash(score_path),
                }
            )
            break
        else:
            raise ValueError(f"No matching HELM run for {dataset}/{name}: {failures}")
    return model_ids, sources


def export_embeddings(rows, dataset, model_ids, google_cache, output, index):
    """Recover the exact historical batches, then index vectors by response identity."""
    texts = [str(text) for text in rows.response]
    records = index["datasets"].setdefault(dataset, {})
    for chunk_id, start in enumerate(range(0, len(texts), 50)):
        chunk_texts = texts[start : start + 50]
        params = {"chunk_id": chunk_id, "chunk": chunk_texts, "model": index["model"]}
        cache_key = hashlib.md5(
            ("_aembed_google_chunk-> " + str(sorted(params.items()))).encode()
        ).hexdigest()
        source = google_cache / f"{cache_key}.pkl"
        if not source.is_file():
            raise ValueError(
                f"Missing Google cache batch {dataset}/{chunk_id}: {source}"
            )
        with source.open("rb") as stream:
            saved_id, vectors = pickle.load(stream)
        vectors = np.asarray(vectors)
        if (
            saved_id != chunk_id
            or vectors.shape != (len(chunk_texts), 3072)
            or not np.isfinite(vectors).all()
        ):
            raise ValueError(f"Invalid Google cache batch: {source}")
        name = f"{cache_key}.npy"
        destination = output / "embeddings/chunks" / name
        np.save(destination, vectors, allow_pickle=False)
        index["chunk_sha256"][name] = file_hash(destination)
        for offset, row in enumerate(rows.iloc[start : start + 50].itertuples()):
            model = model_ids[row.model]
            records.setdefault(model, {})[row.item_id] = {
                "response_sha256": hashlib.sha256(
                    row.source_response.encode()
                ).hexdigest(),
                "embedding_input_sha256": hashlib.sha256(
                    str(row.response).encode()
                ).hexdigest(),
                "chunk": name,
                "row": offset,
            }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--helm_runs_root",
        type=Path,
        required=True,
        help="Directory containing HELM version directories",
    )
    parser.add_argument(
        "--tables_dir",
        type=Path,
        required=True,
        help="Existing examples/helm/data item-score tables",
    )
    parser.add_argument(
        "--google_cache_dir",
        type=Path,
        required=True,
        help="Trusted examples/helm/.cache/embed/google directory",
    )
    parser.add_argument("--out_dpath", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.out_dpath.resolve()
    if output.exists():
        parser.error("out_dpath must be a new directory")
    (output / "embeddings/chunks").mkdir(parents=True)
    manifest_path = ROOT / "jhu_ta1/magnet/pair_coverage_datasets.full.json"
    manifest = json.loads(manifest_path.read_text())
    index = {
        "format_version": 1,
        "provider": "google",
        "model": "gemini-embedding-001",
        "dimensions": 3072,
        "datasets": {},
        "chunk_sha256": {},
        "text_preprocessing": "Historical pandas default NA parsing, then str(response); empty/NA strings become literal nan",
    }
    audit = {
        "datasets": [],
        "tables": {},
        "wmt_pool_selection": "20% of extracted item IDs; numpy default_rng(123), without replacement",
        "wmt_scores": "METEOR item scores from the existing extracted table, verified against HELM responses/references",
    }
    for family in ["math", "wmt_14", "legalbench", "med_qa"]:
        table = args.tables_dir / f"{family}.tsv"
        columns = ["dataset", "model", "instance_id", "response", "target", "score"]
        if family == "wmt_14":
            columns.append("meteor")
        frame = pd.read_csv(table, sep="\t", usecols=columns)
        raw_text = pd.read_csv(
            table,
            sep="\t",
            usecols=["response", "target"],
            dtype=str,
            keep_default_na=False,
        )
        frame["source_response"] = raw_text.response
        frame["source_target"] = raw_text.target
        audit["tables"][str(table.resolve())] = file_hash(table)
        for config in manifest["datasets"]:
            dataset = config["dataset"]
            if dataset.split(":")[0] != family:
                continue
            rows = select_pool(frame, dataset)
            model_ids, sources = materialize_dataset(
                args.helm_runs_root, output, config, rows
            )
            if family in {"math", "wmt_14"}:
                export_embeddings(
                    rows, dataset, model_ids, args.google_cache_dir, output, index
                )
            record = {
                "dataset": dataset,
                "models": len(model_ids),
                "items": int(rows.item_id.nunique()),
                "rows": len(rows),
                "sources": sources,
                "legacy_na_embedding_inputs": int(
                    (rows.response.map(str) != rows.source_response).sum()
                )
                if family in {"math", "wmt_14"}
                else 0,
            }
            audit["datasets"].append(record)
            print(
                f"{dataset}: {record['models']} models, {record['items']} items",
                flush=True,
            )
    (output / "embeddings/index.json").write_text(json.dumps(index) + "\n")
    (output / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "preparation.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(
        f"Prepared all 18 datasets and {len(index['chunk_sha256'])} Google embedding batches in {output}"
    )


if __name__ == "__main__":
    main()

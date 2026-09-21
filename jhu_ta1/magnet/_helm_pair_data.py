"""Load aligned response/score panels from actual HELM run outputs.

Accept full HELM runs (per_instance_stats.json) and public-lite exports
(display_predictions.json). Neither format contains DKPS predictions. Select
one train trial and unperturbed items, rejecting ambiguous or incomplete data.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@dataclass
class Panel:
    """Responses and item scores aligned to a common item pool for one dataset."""

    dataset: str
    metric: str
    models: list[str]
    item_ids: list[str]
    responses: dict[str, list[str]]
    references: list[list[dict]]
    scores: dict[str, np.ndarray]
    sources: list[dict]
    original_pool_sizes: dict[str, int]

    @property
    def pool_size(self):
        return len(self.item_ids)


def _scores(path, metric, trial):
    result = {}
    for entry in json.loads(path.read_text()):
        if entry.get("train_trial_index", 0) != trial:
            continue
        stats = entry["stats"]
        if isinstance(stats, dict):
            values = [(None, stats[metric])] if metric in stats else []
        else:
            values = [
                (s["name"].get("split"), s["mean"])
                for s in stats
                if s["name"]["name"] == metric
                and not s["name"].get("perturbation")
                and s.get("count", 1) > 0
                and "mean" in s
            ]
        for split, value in values:
            key = (entry["instance_id"], split)
            if key in result:
                raise ValueError(f"Duplicate metric row {key} in {path}")
            value = float(value)
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(
                    f"{metric} must have finite item scores in [0,1]: {path}"
                )
            result[key] = value
    if not result:
        raise ValueError(f"No {metric} scores for trial {trial} in {path}")
    return result


def _matches(name, dataset):
    return name == dataset or name.startswith((f"{dataset},", f"{dataset}:"))


def load_panel(suite_path, config):
    """Read selected HELM runs and align them by item identity and content."""
    root = Path(suite_path).resolve()
    dataset, metric = config["dataset"], config["metric"]
    trial = int(config.get("train_trial_index", 0))
    split = config.get("split")
    if "runs" in config:
        directories = [(root / p).resolve() for p in config["runs"]]
        if any(not p.is_relative_to(root) for p in directories):
            raise ValueError("Explicit run paths must be inside the HELM suite")
    else:
        directories = sorted(
            p for p in root.iterdir() if p.is_dir() and _matches(p.name, dataset)
        )
    if not directories:
        raise ValueError(f"No HELM runs for {dataset} in {root}")
    items, metadata, sources = {}, {}, []
    for directory in directories:
        state_path = directory / "scenario_state.json"
        score_path = directory / "per_instance_stats.json"
        if not score_path.exists():
            score_path = directory / "display_predictions.json"
        if not state_path.exists() or not score_path.exists():
            raise ValueError(f"Incomplete HELM run: {directory}")
        state = json.loads(state_path.read_text())
        model = state["adapter_spec"]["model"]
        scores = _scores(score_path, metric, trial)
        model_items = items.setdefault(model, {})
        selected = 0
        for request in state["request_states"]:
            instance = request["instance"]
            if request.get("train_trial_index", 0) != trial or instance.get(
                "perturbation"
            ):
                continue
            item_split = instance.get("split")
            if split is not None and item_split != split:
                continue
            iid = instance["id"]
            score_key = (
                (iid, item_split) if (iid, item_split) in scores else (iid, None)
            )
            if score_key not in scores:
                raise ValueError(
                    f"Missing {metric} for {model}, {iid}, split={item_split}"
                )
            key = json.dumps([item_split, iid], separators=(",", ":"))
            if key in model_items:
                raise ValueError(
                    f"Duplicate response for {model}, {key}; select one run/trial explicitly"
                )
            completions = request.get("result", {}).get("completions", [])
            if len(completions) != 1:
                raise ValueError(f"Expected one completion for {model}, {key}")
            signature = json.dumps(
                {k: instance.get(k) for k in ["input", "references", "split"]},
                sort_keys=True,
            )
            if key in metadata and metadata[key][0] != signature:
                raise ValueError(
                    f"Item content differs between models: {dataset}, {key}"
                )
            metadata[key] = (signature, instance.get("references", []))
            model_items[key] = (completions[0]["text"], scores[score_key])
            selected += 1
        if not selected:
            raise ValueError(f"No requested items for {model} in {directory}")
        sources.append(
            dict(
                model=model,
                run=str(directory),
                scenario_sha256=file_hash(state_path),
                score_file=score_path.name,
                score_sha256=file_hash(score_path),
            )
        )
    models = sorted(items)
    if len(models) < 3:
        raise ValueError(f"{dataset}: need at least three model runs")
    common = sorted(set.intersection(*(set(v) for v in items.values())))
    if not common:
        raise ValueError(f"{dataset}: no shared items across model runs")
    return Panel(
        dataset=dataset,
        metric=metric,
        models=models,
        item_ids=common,
        responses={m: [items[m][i][0] for i in common] for m in models},
        references=[metadata[i][1] for i in common],
        scores={m: np.array([items[m][i][1] for i in common]) for m in models},
        sources=sources,
        original_pool_sizes={m: len(items[m]) for m in models},
    )


def load_panels(suite_path, manifest):
    """Load datasets and fix a target panel present on every requested dataset."""
    configs = manifest["datasets"]
    names = [entry["dataset"] for entry in configs]
    if not names or len(names) != len(set(names)):
        raise ValueError("Dataset names must be nonempty and unique")
    panels = [load_panel(suite_path, entry) for entry in configs]
    common = set.intersection(*(set(p.models) for p in panels))
    targets = manifest.get("target_models", sorted(common))
    if not targets or len(targets) != len(set(targets)) or not set(targets) <= common:
        raise ValueError(
            "Target models must be distinct and present on every requested dataset"
        )
    return panels, sorted(targets)

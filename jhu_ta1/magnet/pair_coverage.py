"""Evaluate DKPS error across dataset/model/query-budget combinations.

HELM supplies saved responses and item scores. This module samples queries,
computes response embeddings, and fits DKPS for each target and replicate.
Only the evaluator uses the target's full benchmark score.
"""

import argparse
import contextlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

from jhu_ta1.magnet._helm_pair_data import file_hash, load_panels


def eligible_references(models, target, family_map):
    """Exclude the target and every model in its declared provider family."""

    def family(model):
        return family_map.get(model, model.split("/", 1)[0])

    return [model for model in models if family(model) != family(target)]


def embed_queries(panel, indices, provider, model):
    """Embed only sampled responses, sharing the batch across target fits."""
    from dkps.helm import compute_embeddings, make_embedding_dict, prepare_responses

    frame = pd.DataFrame(
        [
            {
                "model": model_name,
                "instance_id": panel.item_ids[index],
                "response": panel.responses[model_name][index],
            }
            for model_name in panel.models
            for index in indices
        ]
    )
    frame = prepare_responses(
        frame,
        panel.dataset,
        references=[panel.references[index] for index in indices],
    )
    frame = compute_embeddings(frame, panel.dataset, provider, model)
    return make_embedding_dict(frame)


def predict(
    embeddings, reference_scores, target, target_query_scores, *, dimensions, alpha
):
    """Return DKPS, sample-mean, and raw predictions without full target scores."""
    from dkps.dkps import DataKernelPerspectiveSpace
    from sklearn.linear_model import LinearRegression

    references = sorted(reference_scores)
    if target in references or len(references) < max(2, dimensions):
        raise ValueError(
            "Need sufficient disjoint reference models for the requested dimension"
        )

    selected = {model: embeddings[model] for model in sorted([*references, target])}
    coordinates = DataKernelPerspectiveSpace(
        n_components_cmds=dimensions
    ).fit_transform(selected, return_dict=True)
    regressor = LinearRegression().fit(
        np.vstack([coordinates[model] for model in references]),
        np.array([reference_scores[model] for model in references]),
    )
    raw_prediction = float(
        np.clip(regressor.predict(coordinates[target][None])[0], 0, 1)
    )
    sample_mean = float(np.mean(target_query_scores))
    prediction = alpha * sample_mean + (1 - alpha) * raw_prediction
    return prediction, sample_mean, raw_prediction


def summarize_errors(
    dkps_errors,
    sample_errors,
    datasets,
    models,
    query_budgets,
    *,
    confidence_level=0.95,
):
    """Average replicates within each combination, then count strict improvements.

    Arrays have axes (dataset, query budget, replicate, target model). The
    confidence bounds describe Monte Carlo uncertainty and do not set the verdict.
    """
    dkps_errors = np.asarray(dkps_errors)
    sample_errors = np.asarray(sample_errors)
    if dkps_errors.shape != sample_errors.shape or dkps_errors.ndim != 4:
        raise ValueError(
            "Expected paired errors indexed by dataset, budget, replicate, model"
        )

    num_datasets, num_budgets, num_replicates, num_models = dkps_errors.shape
    if (
        (num_datasets, num_budgets, num_models)
        != (len(datasets), len(query_budgets), len(models))
        or min(num_datasets, num_budgets, num_models) < 1
        or num_replicates < 2
    ):
        raise ValueError("Invalid dataset/model/budget/replicate panel")
    if not np.isfinite(dkps_errors).all() or not np.isfinite(sample_errors).all():
        raise ValueError("Non-finite prediction errors")
    if not 0 < confidence_level < 1:
        raise ValueError("Confidence level must be between zero and one")

    gains = sample_errors - dkps_errors
    expected_gains = gains.mean(axis=2)
    num_settings = expected_gains.size
    # Paired one-sided t bounds, Bonferroni-corrected across all combinations.
    critical_value = t.ppf(
        1 - (1 - confidence_level) / num_settings, num_replicates - 1
    )
    standard_errors = gains.std(axis=2, ddof=1) / np.sqrt(num_replicates)
    lower_bounds = expected_gains - critical_value * standard_errors
    dkps_mae = dkps_errors.mean(axis=2)
    sample_mae = sample_errors.mean(axis=2)

    per_setting = []
    for dataset_index, dataset in enumerate(datasets):
        for budget_index, budget in enumerate(query_budgets):
            for model_index, model in enumerate(models):
                index = (dataset_index, budget_index, model_index)
                per_setting.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "n_eval": budget,
                        "dkps_mae": float(dkps_mae[index]),
                        "sample_mae": float(sample_mae[index]),
                        "expected_gain": float(expected_gains[index]),
                        "gain_lower_confidence": float(lower_bounds[index]),
                    }
                )

    per_budget = []
    num_pairs = num_datasets * num_models
    for budget_index, budget in enumerate(query_budgets):
        improved = int((expected_gains[:, budget_index] > 0).sum())
        per_budget.append(
            {
                "n_eval": budget,
                "num_pairs": num_pairs,
                "pairs_improved": improved,
                "pair_improvement_fraction": improved / num_pairs,
            }
        )

    improved = int((expected_gains > 0).sum())
    confident = int((lower_bounds > 0).sum())
    metrics = {
        "num_datasets": num_datasets,
        "num_models": num_models,
        "num_replicates": num_replicates,
        "num_budgets": num_budgets,
        "budget_spec": ",".join(map(str, query_budgets)),
        "num_pairs": num_pairs,
        "num_settings": num_settings,
        "settings_improved": improved,
        "setting_improvement_fraction": improved / num_settings,
        "settings_improved_with_confidence": confident,
        "setting_improvement_fraction_with_confidence": confident / num_settings,
        "confidence_level": confidence_level,
        "confidence_comparisons": num_settings,
    }
    return {
        "result": {"metrics": metrics},
        "per_setting": per_setting,
        "per_budget": per_budget,
    }


def reference_banks(panels, targets, family_map, reference_models, dimensions):
    """Validate every reference bank before starting any embeddings or fits."""
    banks = {}
    for panel in panels:
        for target in targets:
            references = eligible_references(panel.models, target, family_map)
            count = reference_models or len(references)
            if not max(2, dimensions) <= count <= len(references):
                raise ValueError(
                    f"{panel.dataset}/{target}: only {len(references)} eligible references "
                    f"for count={count}, dimensions={dimensions}"
                )
            banks[panel.dataset, target] = references
    return banks


def evaluate(
    suite_path,
    manifest,
    *,
    query_budgets,
    num_replicates=1024,
    base_seed=0,
    alpha=0.8,
    dimensions=8,
    reference_models=0,
    embed_provider="sentence-transformers",
    embed_model="nomic-ai/nomic-embed-text-v2-moe",
    confidence_level=0.95,
    trial_stream=None,
):
    """Load HELM once, evaluate every requested combination, and summarize errors."""
    if (
        not query_budgets
        or len(set(query_budgets)) != len(query_budgets)
        or any(n <= 0 for n in query_budgets)
    ):
        raise ValueError("Query budgets must be distinct positive integers")
    if (
        not 0 <= alpha <= 1
        or num_replicates < 2
        or dimensions < 1
        or reference_models < 0
        or base_seed < 0
    ):
        raise ValueError("Invalid estimator or replicate settings")
    if not 0 < confidence_level < 1:
        raise ValueError("Confidence level must be between zero and one")

    panels, targets = load_panels(suite_path, manifest)
    for panel in panels:
        if max(query_budgets) > panel.pool_size:
            raise ValueError(
                f"{panel.dataset}: query budget exceeds aligned pool size {panel.pool_size}"
            )
    banks = reference_banks(
        panels,
        targets,
        manifest.get("model_families", {}),
        reference_models,
        dimensions,
    )

    # Each combination retains its own replicate errors until the final reduction.
    shape = (len(panels), len(query_budgets), num_replicates, len(targets))
    dkps_errors = np.empty(shape)
    sample_errors = np.empty(shape)
    fits = 0
    embedding_batches = 0

    for dataset_index, panel in enumerate(panels):
        full_scores = {
            model: float(panel.scores[model].mean()) for model in panel.models
        }
        for budget_index, budget in enumerate(query_budgets):
            for replicate in range(num_replicates):
                seed = base_seed + replicate
                query_rng = np.random.default_rng(
                    np.random.SeedSequence([seed, dataset_index, 0])
                )
                indices = np.sort(
                    query_rng.choice(panel.pool_size, budget, replace=False)
                )
                embeddings = embed_queries(panel, indices, embed_provider, embed_model)
                embedding_batches += 1

                for model_index, target in enumerate(targets):
                    references = banks[panel.dataset, target]
                    if reference_models:
                        reference_rng = np.random.default_rng(
                            np.random.SeedSequence(
                                [seed, dataset_index, model_index, 1]
                            )
                        )
                        references = sorted(
                            reference_rng.choice(
                                references, reference_models, replace=False
                            ).tolist()
                        )

                    # DKPS may use randomized numerical routines. Seed them separately
                    # from query/reference sampling to keep results reproducible.
                    fit_seed = np.random.SeedSequence(
                        [seed, dataset_index, model_index, 2]
                    )
                    np.random.seed(fit_seed.generate_state(1)[0])
                    prediction, sample_mean, raw_prediction = predict(
                        embeddings,
                        {model: full_scores[model] for model in references},
                        target,
                        panel.scores[target][indices],
                        dimensions=dimensions,
                        alpha=alpha,
                    )
                    fits += 1

                    # Evaluation only; never passed to predict.
                    truth = full_scores[target]
                    index = (dataset_index, budget_index, replicate, model_index)
                    dkps_errors[index] = abs(prediction - truth)
                    sample_errors[index] = abs(sample_mean - truth)
                    if trial_stream is not None:
                        trial = {
                            "dataset": panel.dataset,
                            "model": target,
                            "n_eval": budget,
                            "seed": seed,
                            "query_ids": [panel.item_ids[i] for i in indices],
                            "reference_models": references,
                            "prediction": prediction,
                            "raw_prediction": raw_prediction,
                            "sample_score": sample_mean,
                            "truth": truth,
                        }
                        trial_stream.write(json.dumps(trial, allow_nan=False) + "\n")

                if (replicate + 1) % 32 == 0 or replicate == num_replicates - 1:
                    print(
                        f"{panel.dataset}, budget {budget}: {replicate + 1}/{num_replicates} "
                        f"replicates; {fits} DKPS fits",
                        flush=True,
                    )

    result = summarize_errors(
        dkps_errors,
        sample_errors,
        [panel.dataset for panel in panels],
        targets,
        query_budgets,
        confidence_level=confidence_level,
    )
    counts = [reference_models or len(references) for references in banks.values()]
    result["result"]["metrics"].update(
        alpha=alpha,
        n_components_cmds=dimensions,
        dkps_fits=fits,
        embedding_batches=embedding_batches,
        reference_models_min=min(counts),
        reference_models_max=max(counts),
    )
    result["protocol"] = {
        "estimator": "alpha*sample_score + (1-alpha)*clipped_OLS_DKPS_prediction",
        "holdout": "provider family, or supplied model_families mapping",
        "targets": targets,
        "dataset_manifest": manifest,
        "base_seed": base_seed,
        "query_budgets": list(query_budgets),
        "query_sampling": "uniform without replacement; shared across targets within a dataset/replicate",
        "reference_sampling": "ALL eligible"
        if not reference_models
        else f"{reference_models} sampled per target/replicate",
        "target_truth": "item mean on the aligned unperturbed evaluation pool, used only for evaluation",
        "coverage_unit": "dataset/model/query-budget tuple",
        "confidence": "approximate paired one-sided t bounds, Bonferroni over all evaluated combinations",
        "embed_provider": embed_provider,
        "embed_model": embed_model,
        "sources": [
            {
                "dataset": panel.dataset,
                "metric": panel.metric,
                "pool_size": panel.pool_size,
                "original_pool_sizes": panel.original_pool_sizes,
                "models": panel.models,
                "runs": panel.sources,
            }
            for panel in panels
        ],
    }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--helm_suite_path", required=True)
    parser.add_argument("--dataset_manifest", required=True)
    parser.add_argument(
        "--query_budgets", default="1,2,4,8", help="Comma-separated query budgets"
    )
    parser.add_argument("--num_replicates", type=int, default=1024)
    parser.add_argument("--base_seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--n_components_cmds", type=int, default=8)
    parser.add_argument(
        "--reference_models",
        type=int,
        default=0,
        help="0 means ALL eligible references",
    )
    parser.add_argument("--embed_provider", default="sentence-transformers")
    parser.add_argument("--embed_model", default="nomic-ai/nomic-embed-text-v2-moe")
    parser.add_argument("--confidence_level", type=float, default=0.95)
    parser.add_argument(
        "--run_id",
        default="manual",
        help="Execution identifier; does not affect random seeds",
    )
    parser.add_argument("--out_fpath", default="pair_coverage.json")
    args = parser.parse_args(argv)
    try:
        query_budgets = [int(value) for value in args.query_budgets.split(",")]
    except ValueError:
        parser.error("query_budgets must be comma-separated integers")

    suite = Path(args.helm_suite_path).resolve()
    manifest_path = Path(args.dataset_manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    output = Path(args.out_fpath).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    trial_path = output.with_suffix(".trials.jsonl")

    # Embedding helpers may cache text embeddings. Keep their writes in the job
    # directory, including when the checkout and HELM data are read-only mounts.
    with contextlib.chdir(output.parent), trial_path.open("w") as stream:
        payload = evaluate(
            suite,
            manifest,
            query_budgets=query_budgets,
            num_replicates=args.num_replicates,
            base_seed=args.base_seed,
            alpha=args.alpha,
            dimensions=args.n_components_cmds,
            reference_models=args.reference_models,
            embed_provider=args.embed_provider,
            embed_model=args.embed_model,
            confidence_level=args.confidence_level,
            trial_stream=stream,
        )
    payload["protocol"].update(
        dataset_manifest_sha256=file_hash(manifest_path),
        run_id=args.run_id,
        trial_file=trial_path.name,
    )
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps(payload["result"]["metrics"], indent=2))


if __name__ == "__main__":
    main()

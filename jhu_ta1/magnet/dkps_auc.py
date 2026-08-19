#!/usr/bin/env python3
"""
Fit DKPS for one seed and score the per-instance predictions.

One job per seed. Each seed reads and embeds the whole suite, so a run that
reuses finished cells saves real time, and a single seed can be re-run on its
own.
"""
import json

import kwconf


class DkpsAucConfig(kwconf.Config):
    helm_suite_path: str = kwconf.Value(
        None, help='a HELM suite directory, e.g. .../benchmark_output/runs/v1.0.0')
    dataset: str = kwconf.Value('med_qa', help='HELM dataset to predict on')
    metric: str = kwconf.Value('exact_match', help='per-instance stat to score')
    seed: int = kwconf.Value(1, help='replicate; one job per value')
    num_embedding_queries: int = kwconf.Value(50, help='queries that build the space')
    num_eval_samples: int = kwconf.Value(75, help='disjoint queries predicted on')
    num_example_runs: int = kwconf.Value(
        50, help='runs sampled from the suite; must not exceed what it holds')
    n_components_cmds: int = kwconf.Value(8, help='CMDS components')
    out_fpath: str = kwconf.Value(
        'dkps_auc.json', help='where to write the score',
        tags=['out_path', 'primary'])


def main(argv=None, **kwargs):
    import numpy as np
    from sklearn.metrics import roc_auc_score

    from magnet.backends.helm.helm_outputs import HelmSuite
    from magnet.instance_predictor import InstancePrediction
    from jhu_ta1.algorithms.dkps_instance_predictor import DKPSInstancePredictor

    config = DkpsAucConfig.cli(argv=argv, data=kwargs, strict=True)

    seed = int(config['seed'])
    metric = config['metric']
    np.random.seed(seed)

    predictor = DKPSInstancePredictor(
        random_seed=seed,
        num_example_runs=int(config['num_example_runs']),
        num_eval_samples=int(config['num_eval_samples']),
        num_embedding_queries=int(config['num_embedding_queries']),
        n_components_cmds=int(config['n_components_cmds']),
        dataset=config['dataset'],
        metric=metric,
    )

    suite = HelmSuite(config['helm_suite_path'])
    train_split, test_split = predictor.prepare_all_dataframes(suite)
    eval_instance_stats = test_split.per_instance_stats

    predictions = predictor.predict(train_split, test_split.sequester())
    predicted = InstancePrediction.to_df(predictions)

    comparison = predictor.compare_predicted_to_actual(
        predicted, eval_instance_stats)
    scored = comparison[comparison['stat_name'] == metric]

    try:
        auc = float(roc_auc_score(scored['actual_mean'], scored['predicted_mean']))
        degenerate = False
    except ValueError:
        # One class only, so AUC is undefined. The card scored this 0.0, which
        # reads as a failed prediction rather than an absent one; recorded here
        # so a cell that could not be scored is distinguishable from one that
        # scored badly.
        auc = 0.0
        degenerate = True

    payload = {
        'seed': seed,
        'auc': auc,
        'degenerate': degenerate,
        'n_scored_instances': int(len(scored)),
        'n_example_runs': int(config['num_example_runs']),
        'dataset': config['dataset'],
        'metric': metric,
    }

    with open(config['out_fpath'], 'w') as file:
        json.dump(payload, file, indent=2)


__cli__ = DkpsAucConfig

if __name__ == '__main__':
    main()

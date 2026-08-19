#!/usr/bin/env python3
"""
Mean absolute error over replicates: DKPS on few queries against sampling on more.

One job per dataset_cfg. DKPS predicts from ``n_eval_small`` queries and the
baseline samples ``n_eval_large`` of them, so a DKPS error that is no worse is
the query-efficiency finding.
"""
import json
from contextlib import redirect_stdout
from io import StringIO

import kwconf

from jhu_ta1.magnet._dataset_cfg import split_dataset_cfg


class QueryEfficiencyMaeConfig(kwconf.Config):
    helm_suite_path: str = kwconf.Value(
        None, help='a HELM suite directory')
    dataset_cfg: str = kwconf.Value(
        'med_qa|quasi_exact_match', help='"<dataset>|<metric>"')
    n_eval_small: int = kwconf.Value(4, help='queries DKPS predicts from')
    n_eval_large: int = kwconf.Value(8, help='queries the baseline samples')
    num_replicates: int = kwconf.Value(32, help='replicates to average')
    base_seed: int = kwconf.Value(1, help='first seed; replicates count up')
    num_example_runs: int = kwconf.Value(
        64, help='runs sampled from the suite; must not exceed what it holds')
    n_components_cmds: int = kwconf.Value(8, help='CMDS components')
    embed_provider: str = kwconf.Value('sentence-transformers')
    embed_model: str = kwconf.Value('nomic-ai/nomic-embed-text-v2-moe')
    out_fpath: str = kwconf.Value(
        'query_efficiency_mae.json', help='where to write the averages',
        tags=['out_path', 'primary'])


def _pull_actual(test_split, metric):
    """Full-benchmark score: the mean of the per-split means."""
    rows = test_split.stats
    rows = rows[
        (rows['stats.name.name'] == metric)
        & (rows['stats.name.perturbation.name'].isna())
    ]
    return float(rows['stats.mean'].mean())


def _pull_sample_mean(test_split, metric):
    rows = test_split.per_instance_stats
    rows = rows[rows['per_instance_stats.stats.name.name'] == metric]
    return float(rows['per_instance_stats.stats.mean'].mean())


def main(argv=None, **kwargs):
    from magnet.backends.helm.helm_outputs import HelmSuite
    from jhu_ta1.algorithms.dkps_run_predictor import DKPSRunPredictor

    config = QueryEfficiencyMaeConfig.cli(argv=argv, data=kwargs, strict=True)
    dataset, metric = split_dataset_cfg(config['dataset_cfg'])
    suite = HelmSuite(config['helm_suite_path'])

    shared = dict(
        num_example_runs=int(config['num_example_runs']),
        n_components_cmds=int(config['n_components_cmds']),
        dataset=dataset,
        metric=metric,
        embed_provider=config['embed_provider'],
        embed_model=config['embed_model'],
    )

    errors_dkps, errors_sample = [], []
    for offset in range(int(config['num_replicates'])):
        seed = int(config['base_seed']) + offset
        print(f'[{dataset}] replicate {offset} of {config["num_replicates"]}')
        buf = StringIO()
        with redirect_stdout(buf):
            dkps = DKPSRunPredictor(
                num_eval_samples=int(config['n_eval_small']),
                random_seed=seed, **shared)
            train_dkps, test_dkps = dkps.prepare_all_dataframes(suite)
            p_dkps = float(
                dkps.predict(train_dkps, test_dkps.sequester())[0].mean)

            # Same seed picks the same target model; the larger n_eval_samples
            # is what gives the baseline more queries to average over.
            sampler = DKPSRunPredictor(
                num_eval_samples=int(config['n_eval_large']),
                random_seed=seed, **shared)
            _, test_sample = sampler.prepare_all_dataframes(suite)

        errors_dkps.append(abs(p_dkps - _pull_actual(test_dkps, metric)))
        errors_sample.append(abs(
            _pull_sample_mean(test_sample, metric)
            - _pull_actual(test_sample, metric)))

    avg_dkps = sum(errors_dkps) / len(errors_dkps)
    avg_sample = sum(errors_sample) / len(errors_sample)

    payload = {
        'dataset': dataset,
        'metric': metric,
        'num_replicates': int(config['num_replicates']),
        'avg_dkps_mae': avg_dkps,
        'avg_sample_mae': avg_sample,
        # Non-positive when DKPS matches or beats the baseline on more queries.
        'mae_gap': avg_dkps - avg_sample,
    }

    with open(config['out_fpath'], 'w') as file:
        json.dump(payload, file, indent=2)


__cli__ = QueryEfficiencyMaeConfig

if __name__ == '__main__':
    main()

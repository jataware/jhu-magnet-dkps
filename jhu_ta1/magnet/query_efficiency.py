#!/usr/bin/env python3
"""
One query-efficiency replicate: DKPS against a same-size sample baseline.

One job per (dataset_cfg, seed). The card asks whether DKPS predicts a held-out
run better than sampling does, so the quantity it needs is the difference
between the two errors.
"""
import json

import kwconf

from jhu_ta1.magnet._dataset_cfg import split_dataset_cfg


class QueryEfficiencyConfig(kwconf.Config):
    helm_suite_path: str = kwconf.Value(
        None, help='a HELM suite directory')
    dataset_cfg: str = kwconf.Value(
        'med_qa|quasi_exact_match', help='"<dataset>|<metric>"')
    seed: int = kwconf.Value(1, help='replicate; one job per value')
    n_eval: int = kwconf.Value(4, help='evaluation queries')
    num_example_runs: int = kwconf.Value(
        64, help='runs sampled from the suite; must not exceed what it holds')
    n_components_cmds: int = kwconf.Value(8, help='CMDS components')
    embed_provider: str = kwconf.Value('sentence-transformers')
    embed_model: str = kwconf.Value('nomic-ai/nomic-embed-text-v2-moe')
    out_fpath: str = kwconf.Value(
        'query_efficiency.json', help='where to write the errors',
        tags=['out_path', 'primary'])


def main(argv=None, **kwargs):
    from jhu_ta1.algorithms.claim_helpers import run_one_replicate

    config = QueryEfficiencyConfig.cli(argv=argv, data=kwargs, strict=True)
    dataset, metric = split_dataset_cfg(config['dataset_cfg'])

    result = run_one_replicate(
        helm_suite_path=config['helm_suite_path'],
        dataset=dataset,
        metric=metric,
        n_eval=int(config['n_eval']),
        seed=int(config['seed']),
        num_example_runs=int(config['num_example_runs']),
        n_components_cmds=int(config['n_components_cmds']),
        embed_provider=config['embed_provider'],
        embed_model=config['embed_model'],
    )

    err_dkps = abs(result['p_dkps'] - result['actual'])
    err_sample = abs(result['p_sample'] - result['actual'])

    payload = {
        'dataset': dataset,
        'metric': metric,
        'seed': int(config['seed']),
        'target_run_spec': result['target_run_spec'],
        'actual': result['actual'],
        'p_dkps': result['p_dkps'],
        'p_sample': result['p_sample'],
        'err_dkps': err_dkps,
        'err_sample': err_sample,
        # Negative when DKPS beats the baseline, which is the finding.
        'err_gap': err_dkps - err_sample,
    }

    with open(config['out_fpath'], 'w') as file:
        json.dump(payload, file, indent=2)


__cli__ = QueryEfficiencyConfig

if __name__ == '__main__':
    main()

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

# Theory annotations against theory/indexes/dkps-144de76c.yaml in the eval
# superrepo. Inert at run time; MAGNET reads them with `ast` and imports
# nothing. Must be imported as a namespace -- bare-name calls extract nothing.
import magnet.theory as theory


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


# The theorem scores a query subset directly. Pooling by SPLIT reweights
# instances by inverse split size, so what this returns is not the instance-mean
# the statement indexes over. The difference is small on HELM lite, where the
# valid/test splits are near-equal, but it is the kind of thing that only
# becomes visible once someone writes the binder down.
#
# This function is also called separately for the candidate and the baseline
# arm, from independently prepared splits. The two agree because the seed fixes
# the target model and pooling does not depend on the query budget -- but the
# proposition compares two predictors against ONE truth, and the code computes
# it twice. Faithful by construction rather than by design.
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


# What this node computes and what it stands for are two different
# propositions, and separating them is the point.
#
# `EmpiricalCrossBudgetMAEClaim` is what the loop below LITERALLY computes:
# mean absolute error of the DKPS predictor at n_eval_small queries against the
# sample mean at n_eval_large, averaged over num_replicates. The match is
# definitional, so this is `tests`.
#
# The population statement -- highProbMAE_queryEfficient_crossBudget_of_... --
# is what the card SHADOWS, and it is where every gap lives; it is linked from
# the card with `approximates` and its premises are annotated on the predictor.
#
# There is no dataset-population layer to formalize here: datasets are matrix
# cells and the claim is per cell -- `mae_gap <= threshold` for this dataset.
# Any aggregation across datasets lives in how a reader reads the per-cell
# verdicts, not in the card.
@theory.tests('DkpsQuench2026.Paper.TheoryPractice.EmpiricalCrossBudgetMAEClaim',
              note='the node computes exactly this proposition per dataset: empirical MAE of the '
                   'DKPS predictor at 4 queries against the sample mean at 8, averaged over 32 '
                   'Monte-Carlo replicates')
@theory.satisfies('DkpsQuench2026.Paper.TheoryPractice.highProbMAE_queryEfficient_crossBudget_of_affineRiskGap::hmu',
                  note='each replicate is a seeded draw -- base_seed + offset fixes the target '
                       'model, the reference runs and the query subset -- so the estimation '
                       'randomness is a genuine probability measure at every stage')
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

    # Nested under result.metrics, which is where kwdagger's generic
    # YamlProcessNode loader reads a node's metrics from (the pipeline is
    # declared in YAML, so it has no load_result of its own). A flat payload
    # loads as an empty metrics namespace and the claim dies on the name
    # `metrics` after the node has already succeeded.
    with open(config['out_fpath'], 'w') as file:
        json.dump({'result': {'metrics': payload}}, file, indent=2)


__cli__ = QueryEfficiencyMaeConfig

if __name__ == '__main__':
    main()

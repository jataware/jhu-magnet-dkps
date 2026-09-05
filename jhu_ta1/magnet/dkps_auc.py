#!/usr/bin/env python3
"""
Fit DKPS for one seed and score the per-instance predictions.

One job per seed. Each seed reads and embeds the whole suite, so a run that
reuses finished cells saves real time, and a single seed can be re-run on its
own.
"""
import json

import kwconf

# Theory annotations against theory/indexes/dkps-144de76c.yaml in the eval
# superrepo. Inert at run time; MAGNET reads them with `ast` and imports
# nothing. Must be imported as a namespace -- bare-name calls extract nothing.
import magnet.theory as theory


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


# What the card asserts and what the theorem concludes are different KINDS of
# thing, and that is the honest headline for this card. Helm2025.DKPS.Theorem1
# concludes CONVERGENCE: as the estimation budget grows, the risk of a learning
# rule on estimated embeddings tends to its risk on the true embeddings. This
# node reports a point -- AUC on one held-out split at one pool size -- and the
# card asserts that point is above 0.5. A point above chance is not a statement
# about a limit, and no hypothesis in the theorem bridges them.
#
# A4 (`h_cont_loss`) is `violates`, not `substitutes`. The theorem controls a
# risk under a JOINTLY CONTINUOUS loss; roc_auc_score is a rank statistic over
# thresholded exact-match labels and is not continuous in the predictions --
# it is piecewise constant, and it is undefined outright when one class is
# absent, which is the `degenerate` branch below. Substituting a discontinuous
# score does not weaken the conclusion, it steps outside the hypothesis. Marked
# `violates` because the requirement is explicit and demonstrably unmet, though
# there is no formal counterexample to cite.
@theory.approximates('Helm2025.DKPS.Theorem1',
                     note='the card asserts a point (AUC > 0.5 at one pool size); the theorem '
                          'concludes convergence of the estimated-embedding risk to the '
                          'true-embedding risk as the estimation budget grows')
@theory.violates('Helm2025.DKPS.Theorem1::h_cont_loss',
                 note='A4 asks for a jointly continuous loss; AUC is a piecewise-constant rank '
                      'statistic over thresholded labels, and is undefined when one class is '
                      'absent (the degenerate branch)')
@theory.satisfies('Helm2025.DKPS.Theorem1::h_bound_label',
                  note='labels are per-instance correctness indicators in {0,1}, so their support '
                       'is compact')
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

    # Nested under result.metrics, which is where kwdagger's generic
    # YamlProcessNode loader reads a node's metrics from (the pipeline is
    # declared in YAML, so it has no load_result of its own). A flat payload
    # loads as an empty metrics namespace and the claim dies on the name
    # `metrics` after the node has already succeeded.
    with open(config['out_fpath'], 'w') as file:
        json.dump({'result': {'metrics': payload}}, file, indent=2)


__cli__ = DkpsAucConfig

if __name__ == '__main__':
    main()

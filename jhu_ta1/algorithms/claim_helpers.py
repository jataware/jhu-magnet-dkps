"""
Helper for evaluation cards that sweep over (dataset, seed) and want to call
DKPSRunPredictor exactly once per sweep point.

Uses the library's standard path:
    predictor = DKPSRunPredictor(random_seed=seed, ...)
    train_split, test_split = predictor.prepare_all_dataframes(helm_suite)
    pred = predictor.predict(train_split, test_split.sequester())[0].mean

The library handles train/eval sampling internally via random_seed.

Returns a dict of per-replicate quantities:
  - actual:    the target model's full-benchmark score (ground truth)
  - p_sample:  the naive sample-mean estimator over n_eval queries
  - p_dkps:    the DKPS regression prediction
"""

from contextlib import redirect_stdout
from io import StringIO
from typing import TypedDict

from magnet.backends.helm.helm_outputs import HelmSuite

from jhu_ta1.algorithms.dkps_run_predictor import DKPSRunPredictor

# Theory annotations against theory/indexes/dkps-144de76c.yaml in the eval
# superrepo. Inert at run time; MAGNET reads them with `ast` and imports
# nothing. Must be imported as a namespace -- bare-name calls extract nothing.
import magnet.theory as theory


class ReplicateResult(TypedDict):
    actual: float
    p_sample: float
    p_dkps: float
    target_run_spec: str


# Cache HelmSuite across sweep points that share a suite path.
_SUITE_CACHE: dict = {}


def _suite_for(helm_suite_path: str) -> HelmSuite:
    if helm_suite_path not in _SUITE_CACHE:
        _SUITE_CACHE[helm_suite_path] = HelmSuite(helm_suite_path)
    return _SUITE_CACHE[helm_suite_path]


# The population statement the per-replicate win-rate card shadows, and the one
# place to record how far the shadow falls from it.
#
# TWO BUDGETS, ONE NUMBER. The theorem is CROSS-budget: the OLS predictor reads
# a small query subset and the baseline averages a larger one, and the gap
# between the two is what "query efficient" names. This helper passes a single
# `n_eval` and derives BOTH arms from it -- p_dkps predicts from those queries
# and p_sample is the mean over the same ones. So the card is an equal-budget
# accuracy comparison wearing a query-efficiency title, and it does not exercise
# the cross-budget structure the theorem is about at all. The MAE card
# (n_eval_small=4 against n_eval_large=8) is the one that does.
#
# The conclusion is also asymptotic and high-probability, in the reference-pool
# size; this runs at one fixed pool (num_example_runs) and reports a point.
@theory.approximates('DkpsQuench2026.Paper.OLS.highProb_queryEfficient_crossBudget_of_affineRiskGap',
                     note='EQUAL budgets: p_dkps and p_sample are both derived from the same '
                          'n_eval queries, so the cross-budget asymmetry the theorem is about is '
                          'absent. The card measures a per-replicate absolute-error win, not a '
                          'squared risk, and at one pool size rather than in the limit')
@theory.satisfies('DkpsQuench2026.Paper.OLS.highProb_queryEfficient_crossBudget_of_affineRiskGap::hmu',
                  note='the replicate is a seeded draw: `seed` fixes the target run, the '
                       'reference runs and the query subset through DKPSRunPredictor')
@theory.assumes('DkpsQuench2026.Paper.OLS.highProb_queryEfficient_crossBudget_of_affineRiskGap::hgap',
                note='the theorem\'s non-vacuity condition -- an affine witness in TRUE '
                     'perspective coordinates beating the baseline in population MSE. psi is '
                     'never computed here, so this is unobservable in principle from what the '
                     'card measures, not merely unchecked')
@theory.assumes('DkpsQuench2026.Paper.OLS.highProb_queryEfficient_crossBudget_of_affineRiskGap::hcompetitive',
                note='that the fitted OLS asymptotically matches the affine witness in risk. '
                     'Also stated in true coordinates, and the appeal to Lipschitz regularity '
                     'that would justify it is proved insufficient by '
                     'DkpsQuench2026.Paper.OLS.lipschitz_not_sufficient_for_affineRealizability')
# The geometry underneath: the same DKPS coordinates the MAE card uses, so the
# same three premises apply. Fuller notes live on
# DKPSRunPredictor.predict in dkps_run_predictor.py.
@theory.approximates('Acharyya2024.Consistency.fixed_models_fixed_queries_consistency_of_uniqueProfile',
                     note='DKPS classical-MDS coordinates stand in for the consistent MDS '
                          'configuration; nothing bridges "pairwise distances are recovered" to '
                          '"a regression on these coordinates predicts benchmark scores"')
@theory.substitutes('Acharyya2024.Consistency.fixed_models_fixed_queries_consistency_of_uniqueProfile::hpsihat',
                    note='classical MDS minimizes strain, not the raw stress the theorem requires; '
                         'the two agree only when the dissimilarities are exactly Euclidean and '
                         'nothing checks whether they are')
@theory.assumes('Acharyya2024.Consistency.fixed_models_fixed_queries_consistency_of_uniqueProfile::huniq',
                note='the unique-pair-profile condition is never checked; a blind '
                     'n_components_cmds truncation is exactly where near-degenerate spectra break '
                     'it')
@theory.approximates('Acharyya2024.Consistency.fixed_models_fixed_queries_consistency_of_uniqueProfile::hD',
                     note='dissimilarities come from a single draw of n_eval queries; no growing '
                          'budget and no convergence diagnostic')
def run_one_replicate(
    helm_suite_path: str,
    dataset: str,
    metric: str,
    n_eval: int,
    seed: int,
    num_example_runs: int,
    n_components_cmds: int = 8,
    split: str | None = None,
    embed_provider: str | None = None,
    embed_model: str | None = None,
) -> ReplicateResult:
    """One sampling replicate. Delegates sampling to DKPSRunPredictor.prepare_all_dataframes.

    split=None (default) pools across all HELM splits for the ground-truth score,
    matching the predictor and the private pipeline. embed_provider/embed_model
    apply to text-embedding datasets (e.g. math, wmt_14); the predictor ignores
    them for onehot datasets (med_qa, legalbench).
    """
    predictor = DKPSRunPredictor(
        num_example_runs  = num_example_runs,
        num_eval_samples  = n_eval,
        random_seed       = seed,
        n_components_cmds = n_components_cmds,
        dataset           = dataset,
        metric            = metric,
        split             = split,
        embed_provider    = embed_provider,
        embed_model       = embed_model,
    )

    buf = StringIO()
    with redirect_stdout(buf):
        train_split, test_split = predictor.prepare_all_dataframes(_suite_for(helm_suite_path))
        p_dkps = float(predictor.predict(train_split, test_split.sequester())[0].mean)

    # Ground truth: target's full-benchmark aggregate score. With split=None,
    # pool across splits (mean of the per-split means) to match the predictor.
    target_stats_row = test_split.stats
    target_stats_row = target_stats_row[
        (target_stats_row['stats.name.name'] == metric)
        & (target_stats_row['stats.name.perturbation.name'].isna())
    ]
    if split is not None:
        target_stats_row = target_stats_row[target_stats_row['stats.name.split'] == split]
    actual = float(target_stats_row['stats.mean'].mean())

    # Sample-mean over the n_eval queried instances.
    target_per_instance_rows = test_split.per_instance_stats
    target_per_instance_rows = target_per_instance_rows[
        target_per_instance_rows['per_instance_stats.stats.name.name'] == metric
    ]
    p_sample = float(target_per_instance_rows['per_instance_stats.stats.mean'].mean())

    target_run_spec = test_split.run_specs['run_spec.name'].iloc[0]

    return ReplicateResult(
        actual          = actual,
        p_sample        = p_sample,
        p_dkps          = p_dkps,
        target_run_spec = target_run_spec,
    )

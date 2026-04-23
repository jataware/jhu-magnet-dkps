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


def run_one_replicate(
    helm_suite_path: str,
    dataset: str,
    metric: str,
    split: str,
    n_eval: int,
    seed: int,
    num_example_runs: int,
    n_components_cmds: int = 8,
) -> ReplicateResult:
    """One sampling replicate. Delegates sampling to DKPSRunPredictor.prepare_all_dataframes."""
    predictor = DKPSRunPredictor(
        num_example_runs  = num_example_runs,
        num_eval_samples  = n_eval,
        random_seed       = seed,
        n_components_cmds = n_components_cmds,
        dataset           = dataset,
        metric            = metric,
        split             = split,
    )

    buf = StringIO()
    with redirect_stdout(buf):
        train_split, test_split = predictor.prepare_all_dataframes(_suite_for(helm_suite_path))
        p_dkps = float(predictor.predict(train_split, test_split.sequester())[0].mean)

    # Ground truth: target's full-benchmark aggregate score.
    target_stats_row = test_split.stats
    target_stats_row = target_stats_row[
        (target_stats_row['stats.name.name'] == metric)
        & (target_stats_row['stats.name.split'] == split)
        & (target_stats_row['stats.name.perturbation.name'].isna())
    ]
    actual = float(target_stats_row['stats.mean'].iloc[0])

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

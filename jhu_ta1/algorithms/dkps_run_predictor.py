import argparse
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from magnet.predictor import RunPredictor, RunPrediction
from magnet.data_splits import TrainSplit, SequesteredTestSplit
from dkps.dkps import DataKernelPerspectiveSpace as DKPS
from dkps.helm import (
    compute_embeddings, prepare_responses, make_embedding_dict, uses_onehot,
    DEFAULT_EMBED_PROVIDER, DEFAULT_EMBED_MODEL,
)

# Records what this predictor assumes about the DKPS formalization, against
# theory/indexes/dkps-144de76c.yaml in the eval superrepo. Inert at run time --
# the decorators return their target unchanged -- and MAGNET reads them out of
# the source with `ast`, so an audit never imports sklearn, dkps or HELM.
# Import it as a namespace: bare-name calls extract nothing, silently.
import magnet.theory as theory


class DKPSRunPredictor(RunPredictor):
    """Predict run-level aggregate score (e.g. mean quasi_exact_match over a full benchmark)
    for a held-out target model using Data Kernel Perspective Space (DKPS).

    Flow:
        1. Sample `num_eval_samples` queries shared by all runs.
        2. For each train model and the target model, embed their responses on those
           queries into DKPS coordinates.
        3. Fit a LinearRegression from train-model DKPS coordinates to each train
           model's FULL-BENCHMARK aggregate score (from `train_stats_df`, not the
           `num_eval_samples` subset).
        4. Predict the target model's full-benchmark aggregate score.

    Args:
        num_example_runs: Number of training runs (models) used for the DKPS space and LR fit.
        num_eval_samples: Number of shared queries used to compute DKPS coordinates.
        random_seed: Random seed for reproducibility.
        n_components_cmds: Number of CMDS components used by DKPS.
        dataset: HELM dataset name (e.g. 'med_qa', 'legalbench:subset=abercrombie').
            Controls run_spec filtering and embedding strategy.
        metric: HELM metric name to predict (e.g. 'quasi_exact_match'). The private
            DKPS codebase scores runs with quasi_exact_match (see parsers/*.py).
        split: HELM split to target. Default None pools across all splits (the
            full-benchmark mean), matching the private pipeline -- HELM reports
            some benchmarks (med_qa, wmt_14) on both 'valid' and 'test', and
            the eval instances are drawn from all of them. Set a split string
            to restrict the target to one split.
        embed_provider: Embedding API provider for text-embedding datasets. Ignored for
            onehot datasets (med_qa, legalbench).
        embed_model: Embedding model name. Ignored for onehot datasets.
    """

    def __init__(
        self,
        num_example_runs: int = 20,
        num_eval_samples: int = 32,
        random_seed: int = 1,
        n_components_cmds: int = 8,
        dataset: str = "med_qa",
        metric: str = "quasi_exact_match",
        split: str | None = None,
        embed_provider: str | None = None,
        embed_model: str | None = None,
    ):
        super().__init__(
            num_example_runs=num_example_runs,
            num_eval_samples=num_eval_samples,
            random_seed=random_seed,
        )
        self.n_components_cmds = n_components_cmds
        self.dataset = dataset
        self.metric = metric
        self.split = split

        if not uses_onehot(dataset):
            self.embed_provider = embed_provider or DEFAULT_EMBED_PROVIDER
            self.embed_model = embed_model or DEFAULT_EMBED_MODEL
        else:
            self.embed_provider = None
            self.embed_model = None

    def run_spec_filter(self, run_spec):
        return run_spec['name'].startswith(self.dataset)

    # What the run-level MAE card's population statement needs from this method,
    # and what it gets. Statement:
    # DkpsQuench2026.Paper.TheoryPractice.highProbMAE_queryEfficient_crossBudget_of_affineRiskGap
    #
    # `hgap` and `hcompetitive` are the two premises the OLS route added, and
    # they are the whole story. Both are stated in terms of the TRUE perspective
    # embedding psi. This method never computes psi -- only the estimate coming
    # out of DKPS below -- so neither premise is unobserved by accident. They
    # are unobservable in principle from what the card measures, and they must
    # stay `assumes` until either an estimator for the population MSE gap exists
    # or the theorem is restated in estimated coordinates.
    #
    # The estimator binds the theorem faithfully: the OLS theorem's `fit` binder
    # is an exact least-squares minimizer, which is what `LinearRegression().fit`
    # computes. What is NOT discharged is the pair of premises below.
    @theory.assumes('DkpsQuench2026.Paper.TheoryPractice.highProbMAE_queryEfficient_crossBudget_of_affineRiskGap::hgap',
                    note='the non-vacuity condition of the whole theorem: that an affine witness '
                         'in TRUE perspective coordinates beats the baseline in population MSE. '
                         'psi is never computed and the witness theta is never constructed, so '
                         'the card can exhibit the empirical inequality with this false, and vice '
                         'versa')
    @theory.assumes('DkpsQuench2026.Paper.TheoryPractice.highProbMAE_queryEfficient_crossBudget_of_affineRiskGap::hcompetitive',
                    note='that the fitted OLS asymptotically matches the affine witness in risk. '
                         'Asymptotic in the reference-pool size, and the card fits on one fixed '
                         'pool (64 runs), so there is no empirical trace of the limit. The usual '
                         'informal justification -- close in DKPS space implies close in score -- '
                         'is PROVED insufficient by '
                         'DkpsQuench2026.Paper.OLS.lipschitz_not_sufficient_for_affineRealizability, '
                         'a three-point configuration that is 2-Lipschitz in the feature and '
                         'admits no affine fit')
    @theory.satisfies('DkpsQuench2026.Paper.TheoryPractice.highProbMAE_queryEfficient_crossBudget_of_affineRiskGap::habs',
                      note='HELM scores lie in [0,1] and y_hat is clipped to [0,1] below, so the '
                           'absolute error is bounded and integrable. The clipping is proved '
                           'harmless: empiricalMAE_clipUnit_le shows pointwise clipping to [0,1] '
                           'cannot increase empirical MAE when the truth lies in [0,1]')
    @theory.satisfies('DkpsQuench2026.Paper.TheoryPractice.highProbMAE_queryEfficient_crossBudget_of_affineRiskGap::hsq',
                      note='bounded error implies square-integrability against a probability '
                           'measure; same clipping argument as habs')
    # The geometry underneath. DKPS coordinates only mean anything if the
    # embedding recovers the population geometry, and this is the statement that
    # would make them so -- MDS embeddings of consistently estimated
    # dissimilarities recovering pairwise distances in probability.
    #
    # `approximates` and not `tests`: the conclusion is recovery of PAIRWISE
    # DISTANCES along a subsequence. The jump from "distances are recovered" to
    # "a regression on these coordinates predicts benchmark scores" is itself
    # unformalized; no statement in the index bridges it.
    @theory.approximates('Acharyya2024.Consistency.fixed_models_fixed_queries_consistency_of_uniqueProfile',
                         note='DKPS classical-MDS coordinates stand in for the consistent MDS '
                              'configuration the statement is about')
    @theory.substitutes('Acharyya2024.Consistency.fixed_models_fixed_queries_consistency_of_uniqueProfile::hpsihat',
                        note='DKPS uses CLASSICAL MDS -- closed-form double-centering and '
                             'eigendecomposition -- which minimizes strain, not the raw stress '
                             'the theorem requires. The two agree when the dissimilarities are '
                             'exactly Euclidean and disagree otherwise, and nothing checks which '
                             'case obtains. The sharpest surviving estimator gap in this set')
    @theory.assumes('Acharyya2024.Consistency.fixed_models_fixed_queries_consistency_of_uniqueProfile::huniq',
                    note='the unique-pair-profile condition is never checked. Flagged in the Lean '
                         'source itself as an extra assumption beyond the paper. Near-degenerate '
                         'spectra break it, and a blind top-8 truncation (n_components_cmds) is '
                         'exactly where that happens. Checkable: a runtime `checks` on the '
                         'eigenvalue gap would be cheap')
    @theory.approximates('Acharyya2024.Consistency.fixed_models_fixed_queries_consistency_of_uniqueProfile::hD',
                         note='dissimilarities come from a single draw of num_eval_samples '
                              'queries; there is no growing budget and no convergence diagnostic, '
                              'so nothing observes the convergence in probability the premise asks '
                              'for')
    def predict(
        self,
        train_split: TrainSplit,
        sequestered_test_split: SequesteredTestSplit,
    ) -> list[RunPrediction]:

        train_run_specs_df = train_split.run_specs
        train_scenario_states_df = train_split.scenario_state
        train_stats_df = train_split.stats

        eval_run_specs_df = sequestered_test_split.run_specs
        eval_scenario_state_df = sequestered_test_split.scenario_state

        print(f'[DKPSRunPredictor] train_run_specs_df: {train_run_specs_df.shape} '
              f'({train_run_specs_df["run_spec.name"].nunique()} unique run_specs)')
        print(f'[DKPSRunPredictor] train_scenario_states_df: {train_scenario_states_df.shape} '
              f'({train_scenario_states_df["scenario_state.adapter_spec.model"].nunique()} models x '
              f'{train_scenario_states_df["scenario_state.request_states.instance.id"].nunique()} instances)')
        print(f'[DKPSRunPredictor] train_stats_df: {train_stats_df.shape}')
        print(f'[DKPSRunPredictor] eval_scenario_state_df: {eval_scenario_state_df.shape} '
              f'({eval_scenario_state_df["scenario_state.adapter_spec.model"].nunique()} models x '
              f'{eval_scenario_state_df["scenario_state.request_states.instance.id"].nunique()} instances)')

        # --
        # Target model
        eval_models = eval_scenario_state_df['scenario_state.adapter_spec.model'].unique()
        assert len(eval_models) == 1, f'Expected exactly one eval model, got {list(eval_models)}'
        target_model_full = eval_models[0]
        target_run_spec = eval_run_specs_df['run_spec.name'].unique()
        assert len(target_run_spec) == 1, f'Expected exactly one eval run_spec, got {list(target_run_spec)}'
        target_run_spec = target_run_spec[0]

        # --
        # Fetch LR targets: each train model's FULL-benchmark aggregate score.
        #
        # train_stats_df has multiple rows per run_spec (one per stat-name/split/
        # perturbation combo). Keep the unperturbed rows for `metric`; with
        # self.split=None we pool across splits (mean of the per-split means --
        # HELM's valid/test splits are ~equal size, so this is the full-benchmark
        # mean), matching the private pipeline.
        y_rows = train_stats_df[
            (train_stats_df['stats.name.name'] == self.metric)
            & (train_stats_df['stats.name.perturbation.name'].isna())
        ]
        if self.split is not None:
            y_rows = y_rows[y_rows['stats.name.split'] == self.split]
        assert len(y_rows) > 0, (
            f'No train_stats rows match (metric={self.metric}, split={self.split}, unperturbed). '
            f'Available: {train_stats_df[["stats.name.name", "stats.name.split"]].drop_duplicates().to_dict("records")}'
        )

        y_by_run_spec = y_rows.groupby('run_spec.name')['stats.mean'].mean().to_dict()

        # --
        # Build embedding dataframes for DKPS.
        #
        # We need train-model responses on the SAME instance set as the target model's
        # responses, so DKPS coordinates live in the same space. The harness already
        # subsampled eval_scenario_state_df to `num_eval_samples` instances; we filter
        # train responses to match.
        embedding_instance_ids = eval_scenario_state_df[
            'scenario_state.request_states.instance.id'
        ].unique()
        print(f'[DKPSRunPredictor] embedding_instance_ids: n={len(embedding_instance_ids)}')
        sel = train_scenario_states_df['scenario_state.request_states.instance.id'].isin(
            embedding_instance_ids
        )
        train_scenario_states_for_embedding = train_scenario_states_df[sel]
        print(f'[DKPSRunPredictor] train_scenario_states_for_embedding: {train_scenario_states_for_embedding.shape} '
              f'({train_scenario_states_for_embedding["scenario_state.adapter_spec.model"].nunique()} models x '
              f'{train_scenario_states_for_embedding["scenario_state.request_states.instance.id"].nunique()} instances)')

        def _fmt_df(scenario_states_df):
            df = scenario_states_df[[
                'run_spec.name',
                'scenario_state.adapter_spec.model',
                'scenario_state.request_states.instance.id',
                'scenario_state.request_states.result.completions',
            ]].copy()
            df['model'] = df['scenario_state.adapter_spec.model']
            df['response'] = df['scenario_state.request_states.result.completions'].apply(
                lambda x: x[0]['text']
            )
            df = df.rename(columns={
                'run_spec.name': 'run_spec',
                'scenario_state.request_states.instance.id': 'instance_id',
            })
            df = df.drop_duplicates(subset=['model', 'instance_id'], keep='first')
            df = df.sort_values(['model', 'instance_id']).reset_index(drop=True)
            return df[['run_spec', 'instance_id', 'model', 'response']]

        df_train_embed = _fmt_df(train_scenario_states_for_embedding)
        df_valid_embed = _fmt_df(eval_scenario_state_df)
        print(f'[DKPSRunPredictor] df_train_embed: {df_train_embed.shape} '
              f'({df_train_embed.model.nunique()} models x {df_train_embed.instance_id.nunique()} instances)')
        print(f'[DKPSRunPredictor] df_valid_embed: {df_valid_embed.shape} '
              f'({df_valid_embed.model.nunique()} models x {df_valid_embed.instance_id.nunique()} instances)')

        # Drop train models missing any of the embedding instance IDs
        required = set(df_valid_embed.instance_id.unique())
        models_to_drop = []
        for model, grp in df_train_embed.groupby('model'):
            if not required.issubset(set(grp.instance_id)):
                models_to_drop.append(model)
        if models_to_drop:
            print(f"Warning: dropping {len(models_to_drop)} train model(s) missing embedding instances: {models_to_drop}")
            df_train_embed = df_train_embed[~df_train_embed.model.isin(models_to_drop)].reset_index(drop=True)

        # Map run_spec -> model for the kept train runs, so we can line up LR targets
        train_run_to_model = dict(zip(df_train_embed.run_spec, df_train_embed.model))
        kept_train_run_specs = [rs for rs in y_by_run_spec if rs in train_run_to_model]
        assert len(kept_train_run_specs) >= 2, (
            f'Need at least 2 train runs with both embeddings and stats; got {len(kept_train_run_specs)}'
        )

        # --
        # Sanity
        train_models = df_train_embed.model.unique()
        assert target_model_full not in train_models, 'Target model must be disjoint from train models'

        # --
        # Compute embeddings and fit DKPS (onehot embeddings depend on joint vocabulary,
        # so embed train + target together).
        df_all = pd.concat([df_train_embed, df_valid_embed]).reset_index(drop=True)

        # Dataset-specific response normalization (e.g. legalbench answer -> gold
        # class mapping) lives in dkps.helm so it stays consistent with the
        # private DKPS pipeline. Pass references unconditionally; prepare_responses
        # ignores them for datasets that don't need them.
        ref_col = 'scenario_state.request_states.instance.references'
        all_references = pd.concat([
            train_scenario_states_for_embedding[ref_col],
            eval_scenario_state_df[ref_col],
        ])
        df_all = prepare_responses(df_all, self.dataset, references=all_references)

        df_all = compute_embeddings(df_all, self.dataset, self.embed_provider, self.embed_model)
        print(f'[DKPSRunPredictor] df_all (post-embed): {df_all.shape}; '
              f'embedding dim: {np.asarray(df_all.embedding.iloc[0]).shape}')

        embedding_dict = make_embedding_dict(df_all)
        sample_key = next(iter(embedding_dict))
        print(f'[DKPSRunPredictor] embedding_dict: {len(embedding_dict)} models; '
              f'each value shape: {embedding_dict[sample_key].shape}')
        P = DKPS(n_components_cmds=self.n_components_cmds).fit_transform(embedding_dict, return_dict=True)
        print(f'[DKPSRunPredictor] DKPS coords P: {len(P)} models; each shape: {P[sample_key].shape}')

        # --
        # LR fit: DKPS coords -> full-benchmark aggregate score
        X_train = np.vstack([P[train_run_to_model[rs]] for rs in kept_train_run_specs])
        y_train = np.array([y_by_run_spec[rs] for rs in kept_train_run_specs])
        X_valid = P[target_model_full][None]
        print(f'[DKPSRunPredictor] X_train: {X_train.shape}, y_train: {y_train.shape}, X_valid: {X_valid.shape}')
        print(f'[DKPSRunPredictor] y_train range: [{y_train.min():.3f}, {y_train.max():.3f}] mean={y_train.mean():.3f}')

        lr = LinearRegression().fit(X_train, y_train)
        y_hat = float(lr.predict(X_valid)[0])
        y_hat = float(np.clip(y_hat, 0.0, 1.0))  # metric bounded [0,1]; matches dkps run_dkps.py
        print(f'[DKPSRunPredictor] target={target_model_full} y_hat={y_hat:.3f}')

        return [RunPrediction(
            run_spec_name=target_run_spec,
            split=self.split or 'all',
            stat_name=self.metric,
            mean=y_hat,
        )]


if __name__ == "__main__":
    np.random.seed(1)

    parser = argparse.ArgumentParser()
    parser.add_argument('helm_suite_path', type=str)
    parser.add_argument('--num-example-runs', default=20, type=int)
    parser.add_argument('--num-eval-samples', default=32, type=int)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--n-components-cmds', default=8, type=int)
    parser.add_argument('--dataset', default='med_qa', type=str)
    parser.add_argument('--metric', default='quasi_exact_match', type=str)
    parser.add_argument('--split', default=None, type=str, help="HELM split; default None pools all splits.")
    parser.add_argument('--embed-provider', default=None, type=str)
    parser.add_argument('--embed-model', default=None, type=str)
    args = parser.parse_args()

    predictor = DKPSRunPredictor(
        num_example_runs=args.num_example_runs,
        num_eval_samples=args.num_eval_samples,
        random_seed=args.seed,
        n_components_cmds=args.n_components_cmds,
        dataset=args.dataset,
        metric=args.metric,
        split=args.split,
        embed_provider=args.embed_provider,
        embed_model=args.embed_model,
    )
    predictor(helm_suites=args.helm_suite_path)

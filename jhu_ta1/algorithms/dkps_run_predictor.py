import argparse
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from magnet.predictor import RunPredictor, RunPrediction
from magnet.data_splits import TrainSplit, SequesteredTestSplit
from dkps.dkps import DataKernelPerspectiveSpace as DKPS
from dkps.helm import compute_embeddings, make_embedding_dict, uses_onehot, DEFAULT_EMBED_PROVIDER, DEFAULT_EMBED_MODEL


class DKPSRunPredictor(RunPredictor):
    """Predict run-level aggregate score (e.g. mean exact_match over a full benchmark)
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
        metric: HELM metric name to predict (e.g. 'exact_match').
        split: HELM split to target (e.g. 'valid').
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
        metric: str = "exact_match",
        split: str = "valid",
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
        # train_stats_df has multiple rows per run_spec (one per stat-name/split/perturbation
        # combo). We want the row matching (metric, split, no perturbation).
        y_rows = train_stats_df[
            (train_stats_df['stats.name.name'] == self.metric)
            & (train_stats_df['stats.name.split'] == self.split)
            & (train_stats_df['stats.name.perturbation.name'].isna())
        ]
        assert len(y_rows) > 0, (
            f'No train_stats rows match (metric={self.metric}, split={self.split}, unperturbed). '
            f'Available: {train_stats_df[["stats.name.name", "stats.name.split"]].drop_duplicates().to_dict("records")}'
        )

        y_by_run_spec = dict(zip(y_rows['run_spec.name'], y_rows['stats.mean']))

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
        print(f'[DKPSRunPredictor] target={target_model_full} y_hat={y_hat:.3f}')

        return [RunPrediction(
            run_spec_name=target_run_spec,
            split=self.split,
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
    parser.add_argument('--metric', default='exact_match', type=str)
    parser.add_argument('--split', default='valid', type=str)
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

import random
import argparse
import pandas as pd
import numpy as np

from magnet.instance_predictor import InstancePredictor, InstancePrediction
from magnet.data_splits import TrainSplit, SequesteredTestSplit

from sklearn.linear_model import LinearRegression, LogisticRegression
from dkps.dkps import DataKernelPerspectiveSpace as DKPS
from dkps.helm import compute_embeddings, make_embedding_dict, uses_onehot, DEFAULT_EMBED_PROVIDER, DEFAULT_EMBED_MODEL

# --
# Predictor

class DKPSInstancePredictor(InstancePredictor):
    """Predict per-instance metrics using Data Kernel Perspective Space (DKPS).

    Args:
        num_example_runs: Number of training runs (models) to use for building the DKPS space.
        num_eval_samples: Total number of queries sampled for the target model (embedding + test).
        num_embedding_queries: Number of queries used to build the DKPS embedding. The remaining
            (num_eval_samples - num_embedding_queries) queries are used for prediction.
        random_seed: Random seed for reproducibility of instance sampling.
        n_components_cmds: Number of CMDS components used by DKPS.
        dataset: HELM dataset name (e.g. 'med_qa', 'legalbench:subset=abercrombie',
            'math:subject=algebra', 'wmt_14:language_pair=cs-en'). Controls both run_spec
            filtering and embedding strategy.
        metric: HELM metric name to predict (e.g. 'exact_match').
        embed_provider: Embedding API provider for text-embedding datasets. Defaults to
            'google' for non-onehot datasets; ignored for onehot datasets (med_qa, legalbench).
        embed_model: Embedding model name. Defaults to 'gemini-embedding-001' for non-onehot
            datasets; ignored for onehot datasets.
    """

    def __init__(
        self,
        num_example_runs: int = 3,
        num_eval_samples: int = 20,
        num_embedding_queries: int = 8,
        random_seed: int = 1,
        n_components_cmds: int = 8,
        dataset: str = "med_qa",
        metric: str = "exact_match",
        embed_provider: str | None = None,
        embed_model: str | None = None,
    ):
        super().__init__(
            num_example_runs = num_example_runs,
            num_eval_samples = num_eval_samples,
            random_seed      = random_seed
        )
        self.num_embedding_queries = num_embedding_queries
        self.n_components_cmds = n_components_cmds
        self.dataset = dataset
        self.metric = metric

        if not uses_onehot(dataset):
            self.embed_provider = embed_provider or DEFAULT_EMBED_PROVIDER
            self.embed_model    = embed_model or DEFAULT_EMBED_MODEL
        else:
            self.embed_provider = None
            self.embed_model    = None

    def run_spec_filter(self, run_spec):
        return run_spec['name'].startswith(self.dataset)

    def predict(self,
        train_split: TrainSplit,
        sequestered_test_split: SequesteredTestSplit
    ) -> list[InstancePrediction]:

        # --
        # Parse MAGNET format

        # Unpack split classes into dataframes
        train_run_specs_df       = train_split.run_specs
        train_scenario_states_df = train_split.scenario_state
        # train_stats_df           = train_split.stats
        train_instance_stats_df   = train_split.per_instance_stats

        # eval_run_specs_df = sequestered_test_split.run_specs  # NOQA
        eval_scenario_state_df = sequestered_test_split.scenario_state

        eval_models = eval_scenario_state_df['scenario_state.adapter_spec.model'].unique()
        assert len(eval_models) == 1, (
            f'Expected exactly one eval model, got {len(eval_models)}: {list(eval_models)}'
        )

        # --
        # Split eval instances into embedding queries and test queries

        all_eval_instance_ids = list(eval_scenario_state_df['scenario_state.request_states.instance.id'].unique())
        rng = random.Random(self.random_seed)
        rng.shuffle(all_eval_instance_ids)

        embedding_instance_ids = all_eval_instance_ids[:self.num_embedding_queries]
        test_instance_ids = all_eval_instance_ids[self.num_embedding_queries:]

        # --
        # Filter training scenario_states to embedding queries only (for DKPS embedding)
        # Keep training per_instance_stats unfiltered (need scores on test queries for regression targets)

        sel = train_scenario_states_df['scenario_state.request_states.instance.id'].isin(embedding_instance_ids)
        train_scenario_states_for_embedding = train_scenario_states_df[sel]

        # Filter eval scenario_state into embedding and test subsets
        sel_embed = eval_scenario_state_df['scenario_state.request_states.instance.id'].isin(embedding_instance_ids)
        eval_scenario_state_embedding = eval_scenario_state_df[sel_embed]

        # --
        # Convert to our format

        id2magnet = eval_scenario_state_df[['scenario_state.request_states.instance.id', 'magnet.instance_predict_id']]
        id2magnet = id2magnet.set_index('scenario_state.request_states.instance.id').to_dict()['magnet.instance_predict_id']

        metrics = train_run_specs_df['run_spec.metric_specs'].iloc[0][0]['args']['names']

        def _fmt_df(scenario_states_df, instance_stats_df=None):
            df = scenario_states_df[[
                'run_spec.name',
                'scenario_state.adapter_spec.model',
                'scenario_state.request_states.instance.id',
                'scenario_state.request_states.result.completions'
            ]].copy()

            df['model_family'] = df['scenario_state.adapter_spec.model'].apply(lambda x: x.split('/')[0])
            df['model']        = df['scenario_state.adapter_spec.model'].apply(lambda x: x.split('/')[-1])
            df['response']     = df['scenario_state.request_states.result.completions'].apply(lambda x: x[0]['text'])

            df = df.rename(columns={
                'run_spec.name' : 'run_spec',
                'scenario_state.request_states.instance.id': 'instance_id',
            })

            if instance_stats_df is not None:
                df_stats = instance_stats_df[instance_stats_df['per_instance_stats.stats.name.name'].isin(metrics)]
                df_stats = df_stats.pivot(
                    index   = ['run_spec.name', 'per_instance_stats.instance_id'],
                    columns = 'per_instance_stats.stats.name.name',
                    values  = 'per_instance_stats.stats.mean'
                ).reset_index()

                df_stats = df_stats.rename(columns={
                    'run_spec.name' : 'run_spec',
                    'per_instance_stats.instance_id': 'instance_id'
                })

                df = pd.merge(df, df_stats, on=['run_spec', 'instance_id'], how='left')

            # Drop duplicate (model, instance_id) rows from runs that differ only
            # in stop sequences (e.g. stop=none variants in legalbench)
            df = df.drop_duplicates(subset=['model', 'instance_id'], keep='first')
            df = df.sort_values(['model', 'instance_id']).reset_index(drop=True)

            cols = ['run_spec', 'instance_id', 'model_family', 'model', 'response']
            if instance_stats_df is not None:
                cols += metrics

            return df[cols]

        # Build embedding dataframes (embedding queries only, no scores needed for DKPS)
        df_train_embedding = _fmt_df(train_scenario_states_for_embedding)
        df_valid_embedding = _fmt_df(eval_scenario_state_embedding)

        # Drop training models missing any of the embedding instance IDs
        required_instance_ids = set(df_valid_embedding.instance_id.unique())
        models_to_drop = []
        for model, grp in df_train_embedding.groupby('model'):
            if not required_instance_ids.issubset(set(grp.instance_id)):
                models_to_drop.append(model)
        if models_to_drop:
            print(f"Warning: dropping {len(models_to_drop)} training model(s) missing embedding instances: {models_to_drop}")
            df_train_embedding = df_train_embedding[~df_train_embedding.model.isin(models_to_drop)].reset_index(drop=True)

        # Build full training dataframe with scores (for regression targets on test queries)
        df_train_full = _fmt_df(train_scenario_states_df, train_instance_stats_df)

        # --
        # Data checks

        train_models    = df_train_embedding.model.unique()
        target_model    = df_valid_embedding.model.unique()[0]
        target_run_spec = df_valid_embedding[df_valid_embedding.model == target_model].run_spec.unique()[0]

        assert df_valid_embedding[df_valid_embedding.model == target_model].run_spec.unique().shape[0] == 1, 'Only one target run_spec is supported'
        assert df_valid_embedding.model.unique().shape[0] == 1, 'Only one target model is supported'
        assert target_model not in train_models, 'Target model must be different from train models'

        # --
        # Compute embeddings from embedding queries only
        # Embed jointly so onehot levels are consistent across train/valid

        df_combined_embedding = compute_embeddings(
            pd.concat([df_train_embedding, df_valid_embedding]).reset_index(drop=True),
            self.dataset, self.embed_provider, self.embed_model
        )

        embedding_dict = make_embedding_dict(df_combined_embedding)
        P              = DKPS(n_components_cmds=self.n_components_cmds).fit_transform(embedding_dict, return_dict=True)

        # [TODO] (maybe) filter models from same model family as target model

        X_train = np.vstack([P[m] for m in train_models])
        X_valid = P[target_model][None]

        # --
        # Predict on test queries only

        predictions = []
        for instance_id in test_instance_ids:
            for metric in metrics:
                df_train_sub = df_train_full[df_train_full.instance_id == instance_id]

                # Use set_index to align robustly instead of asserting exact array equality
                df_train_sub = df_train_sub.set_index('model')
                if not all(m in df_train_sub.index for m in train_models):
                    # Skip instances where training models lack data
                    continue

                df_train_sub = df_train_sub.loc[train_models]
                y_train = df_train_sub[metric].values

                # [TODO] what if y_train only has one value?
                #        could either assume - "everything is right" or "does it match target model response"

                is_binary = (np.unique(y_train) == [0, 1]).all()
                if is_binary:
                    # if the target is binary (via a hacky check), use LogisticRegression
                    lr      = LogisticRegression().fit(X_train, y_train)
                    y_hat   = lr.predict_proba(X_valid)[0][1]
                else:
                    # otherwise, use LinearRegression
                    lr      = LinearRegression().fit(X_train, y_train)
                    y_hat   = lr.predict(X_valid)[0]

                predictions.append(
                    InstancePrediction(
                        run_spec_name       = target_run_spec,
                        instance_predict_id = id2magnet[instance_id],
                        stat_name           = metric,
                        mean                = y_hat
                    )
                )

        return predictions

if __name__ == "__main__":
    np.random.seed(1)

    parser = argparse.ArgumentParser()
    parser.add_argument('helm_suite_path',
                        type=str,
                        help="Path to HELM run outputs for a suite (usually 'something/something/benchmark_output/runs/suite_name')")
    parser.add_argument(
        "--num-example-runs", default=50, type=int, help="Number of training runs used by DKPS.",
    )
    parser.add_argument(
        "--num-eval-samples", default=20, type=int, help="Total number of queries sampled for target model (embedding + test).",
    )
    parser.add_argument(
        "--num-embedding-queries", default=8, type=int, help="Number of queries used for DKPS embedding.",
    )
    parser.add_argument("--seed", default=1, type=int, help="Random seed to use.")

    parser.add_argument("--n-components-cmds", default=8, type=int, help="Number of components used by DKPS.")

    parser.add_argument("--dataset", default="med_qa", type=str, help="HELM dataset name.")
    parser.add_argument("--embed-provider", default=None, type=str, help="Embedding provider (required for text-embedding datasets like math, wmt_14).")
    parser.add_argument("--embed-model", default=None, type=str, help="Embedding model name.")

    args = parser.parse_args()

    predictor = DKPSInstancePredictor(
        random_seed       = args.seed,
        num_example_runs  = args.num_example_runs,
        num_eval_samples  = args.num_eval_samples,
        num_embedding_queries = args.num_embedding_queries,
        n_components_cmds = args.n_components_cmds,
        dataset           = args.dataset,
        embed_provider    = args.embed_provider,
        embed_model       = args.embed_model,
    )

    predictor(args.helm_suite_path)
